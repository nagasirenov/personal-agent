"""Read-only bridge for local VS Code companion snapshots."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ..models import TimelineEvent

_MAX_DISCOVERED = 256
_MAX_WINDOWS = 32
_MAX_FILE_BYTES = 64 * 1024
_MAX_STALE = timedelta(seconds=90)
_HEARTBEAT = timedelta(seconds=60)
_MAX_OPEN_FILES = 50
_MAX_TERMINALS = 12
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b\d{5,}:[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b"),
    re.compile(r"(?i)\b(?:token|password|secret|api[_ -]?key)\s*[:=]\s*\S+"),
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:24]


def _short(value: Any, limit: int = 240) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\x00", " ").split())[:limit]


def _safe_metadata(value: Any, limit: int = 240) -> str:
    """Bound metadata without changing meaningful path whitespace; scrub common secrets."""
    if not isinstance(value, str):
        return ""
    result = value.replace("\x00", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[redacted]", result)
    return result[:limit]


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


class VscodeCompanionObserver:
    """Read small, fresh metadata snapshots written by the local VS Code extension."""

    def __init__(
        self,
        storage_root: Path | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        appdata = os.environ.get("APPDATA")
        default_root = (Path(appdata) if appdata else Path.home() / "AppData" / "Roaming")
        self.root = Path(storage_root or (default_root / "PersonalAgent" / "vscode-context")).expanduser().resolve()
        self._known: dict[str, dict[str, Any]] = {}
        self._last_poll_at: datetime | None = None
        self._files_scanned = 0
        self._events_emitted = 0
        self._last_error: str | None = None
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def poll(self) -> list[TimelineEvent]:
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)
        self._last_poll_at = now
        self._files_scanned = 0
        self._events_emitted = 0
        self._last_error = None
        events: list[TimelineEvent] = []
        seen_windows: set[str] = set()
        try:
            candidates_with_time = []
            if self.root.is_dir():
                for index, path in enumerate(self.root.glob("*.json")):
                    if index >= _MAX_DISCOVERED:
                        break
                    try:
                        if path.is_file():
                            candidates_with_time.append((path.stat().st_mtime_ns, path))
                    except OSError:
                        continue
            candidates = [path for _, path in sorted(candidates_with_time, key=lambda item: item[0], reverse=True)[:_MAX_WINDOWS]]
            for path in candidates:
                self._files_scanned += 1
                try:
                    if path.stat().st_size > _MAX_FILE_BYTES:
                        continue
                    raw = path.read_bytes()
                    snapshot = json.loads(raw)
                    normalized = self._normalize(snapshot)
                    if normalized is None:
                        continue
                    captured_at, data, workspace = normalized
                    window_id = _short(path.stem, 80) or _hash(str(path))
                    if now - captured_at > _MAX_STALE or captured_at > now + timedelta(minutes=2):
                        continue
                    seen_windows.add(window_id)
                    fingerprint = _hash(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    previous = self._known.get(window_id)
                    emit = (
                        previous is None
                        or previous.get("fingerprint") != fingerprint
                        or now - previous.get("last_event_at", now) >= _HEARTBEAT
                    )
                    self._known[window_id] = {
                        "fingerprint": fingerprint,
                        "captured_at": captured_at,
                        "last_event_at": now if emit else previous["last_event_at"],
                        "data": data,
                        "workspace": workspace,
                    }
                    if emit:
                        events.append(self._event(window_id, data, workspace, captured_at, "vscode.context_updated"))
                except (OSError, ValueError, TypeError):
                    continue

            for window_id, previous in list(self._known.items()):
                if window_id in seen_windows:
                    continue
                last_capture = previous.get("captured_at", now)
                if now - last_capture <= _MAX_STALE:
                    continue
                data = {"status": "closed", "workspace": previous.get("workspace") or ""}
                events.append(self._event(window_id, data, previous.get("workspace"), now, "vscode.workspace_closed"))
                self._known.pop(window_id, None)
        except OSError as exc:
            self._last_error = type(exc).__name__
        self._events_emitted = len(events)
        return events

    @staticmethod
    def _normalize(snapshot: Any) -> tuple[datetime, dict[str, Any], str | None] | None:
        if not isinstance(snapshot, dict):
            return None
        if snapshot.get("schema_version", snapshot.get("version")) != 1:
            return None
        captured_at = _time(snapshot.get("captured_at"))
        if captured_at is None:
            return None

        workspace = _safe_metadata(snapshot.get("workspace"), 1024)
        folders_raw = snapshot.get("workspace_folders")
        folders = [_safe_metadata(item, 1024) for item in folders_raw[:16] if isinstance(item, str)] if isinstance(folders_raw, list) else []
        active_file = _safe_metadata(snapshot.get("active_file"), 1024)
        files_raw = snapshot.get("open_files")
        open_files = list(dict.fromkeys(
            _safe_metadata(item, 512) for item in files_raw[:_MAX_OPEN_FILES]
            if isinstance(item, str) and _safe_metadata(item, 512)
        )) if isinstance(files_raw, list) else []
        selection_raw = snapshot.get("selection") if isinstance(snapshot.get("selection"), dict) else {}
        diagnostics_raw = snapshot.get("diagnostics") if isinstance(snapshot.get("diagnostics"), dict) else {}
        terminals_raw = snapshot.get("terminals")
        terminals = []
        if isinstance(terminals_raw, list):
            for item in terminals_raw[:_MAX_TERMINALS]:
                if not isinstance(item, dict):
                    continue
                terminal: dict[str, Any] = {"name": _safe_metadata(item.get("name"), 80)}
                pid = item.get("process_id")
                if type(pid) is int and pid > 0:
                    terminal["process_id"] = pid
                cwd = _safe_metadata(item.get("cwd"), 1024)
                if cwd:
                    terminal["cwd"] = cwd
                terminals.append(terminal)

        data = {
            "workspace": workspace,
            "workspace_folders": folders,
            "active_file": active_file,
            "open_files": open_files,
            "open_file_count": max(len(open_files), _bounded_int(snapshot.get("open_file_count", len(open_files)))),
            "selection_length": _bounded_int(selection_raw.get("length")),
            "selection_lines": _bounded_int(selection_raw.get("lines")),
            "diagnostics": {
                key: _bounded_int(diagnostics_raw.get(key))
                for key in ("error", "warning", "information", "hint")
            },
            "git_branch": _safe_metadata(snapshot.get("git_branch"), 120),
            "active_editor_language": _safe_metadata(
                snapshot.get("active_editor_language", snapshot.get("active_editor")), 60
            ),
            "terminals": terminals,
            "terminal_count": max(len(terminals), _bounded_int(snapshot.get("terminal_count", len(terminals)))),
            "status": "active",
        }
        return captured_at, data, workspace or (folders[0] if folders else None)

    @staticmethod
    def _event(
        window_id: str,
        data: dict[str, Any],
        workspace: str | None,
        when: datetime,
        event_type: str,
    ) -> TimelineEvent:
        data = dict(data)
        data["session_id"] = _hash(window_id)
        event_id = _hash(f"{window_id}:{event_type}:{when.isoformat()}:" + json.dumps(data, ensure_ascii=False, sort_keys=True))
        return TimelineEvent(
            source="vscode",
            type=event_type,
            data=data,
            project=workspace,
            time=when,
            id=event_id,
        )

    def health(self) -> dict[str, Any]:
        return {
            "available": self.root.is_dir() and self._last_error is None,
            "last_poll_at": self._last_poll_at.isoformat() if self._last_poll_at else None,
            "files_scanned": self._files_scanned,
            "active_workspaces": len(self._known),
            "events_emitted": self._events_emitted,
            "error_type": self._last_error,
        }


def _bounded_int(value: Any, maximum: int = 1_000_000) -> int:
    if type(value) is not int:
        return 0
    return min(max(value, 0), maximum)
