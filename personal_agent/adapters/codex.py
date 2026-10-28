"""Bounded, read-only metadata observation for local Codex JSONL sessions."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from ..models import TimelineEvent

_MAX_DISCOVERED_FILES = 4096
_MAX_FILES_PER_POLL = 24
_MAX_TOTAL_BYTES_PER_POLL = 2 * 1024 * 1024
_MAX_INITIAL_BYTES_PER_FILE = 256 * 1024
_MAX_LINE_BYTES = 96 * 1024
_MAX_EVENTS_PER_POLL = 250
_MAX_REMEMBERED_IDS = 10_000

_STATUS_VALUES = {"running", "completed", "failed", "cancelled", "pending", "in_progress"}
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b\d{5,}:[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b"),
    re.compile(r"(?i)\b(?:token|password|secret|api[_ -]?key)\s*[:=]\s*\S+"),
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:24]


def _safe_text(value: Any, limit: int = 240) -> str:
    if not isinstance(value, str):
        return ""
    value = " ".join(value.replace("\x00", " ").split())
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[redacted]", value)
    return value[:limit]


def _safe_status(value: Any) -> str:
    normalized = str(value or "").strip().casefold().replace(" ", "_")
    return normalized if normalized in _STATUS_VALUES else ""


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        try:
            number = float(value)
            if number > 10_000_000_000:
                number /= 1000
            return datetime.fromtimestamp(number, timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        except ValueError:
            return None
    return None


class CodexSessionObserver:
    """Poll recent Codex session files and emit only bounded task metadata."""

    def __init__(self, sessions_root: Path | None = None) -> None:
        self.root = Path(sessions_root or (Path.home() / ".codex" / "sessions")).expanduser().resolve()
        self._offsets: dict[Path, int] = {}
        self._file_meta: dict[Path, tuple[str, str | None]] = {}
        self._seen_ids: set[str] = set()
        self._last_poll_at: datetime | None = None
        self._files_scanned = 0
        self._events_emitted = 0
        self._skipped_oversized_lines = 0
        self._last_error: str | None = None

    def poll(self) -> list[TimelineEvent]:
        now = datetime.now(timezone.utc)
        self._last_poll_at = now
        self._last_error = None
        self._files_scanned = 0
        self._events_emitted = 0
        if not self.root.is_dir():
            return []

        events: list[TimelineEvent] = []
        total_bytes = 0
        try:
            files = self._recent_files()
            for path in files:
                if len(events) >= _MAX_EVENTS_PER_POLL or total_bytes >= _MAX_TOTAL_BYTES_PER_POLL:
                    break
                try:
                    size = path.stat().st_size
                    offset = self._offsets.get(path)
                    initial = offset is None or size < offset
                    if initial:
                        start = max(0, size - _MAX_INITIAL_BYTES_PER_FILE)
                        session_key, project = self._read_header(path)
                        if start:
                            with path.open("rb") as handle:
                                handle.seek(start)
                                if self._discard_partial_line(handle):
                                    start = handle.tell()
                                else:
                                    start = size
                        self._offsets[path] = start
                        self._file_meta[path] = (session_key, project)
                    else:
                        start = offset
                    allowance = min(_MAX_TOTAL_BYTES_PER_POLL - total_bytes, 512 * 1024)
                    fresh, new_offset, consumed = self._read_file(path, start, allowance)
                    total_bytes += consumed
                    self._offsets[path] = new_offset
                    self._files_scanned += 1
                    session_key, project = self._file_meta.get(path, (str(path), None))
                    for line_offset, row in fresh:
                        event = self._event_from_row(row, path, line_offset, session_key, project, now)
                        if event is not None and event.id not in self._seen_ids:
                            self._remember(event.id)
                            events.append(event)
                            if len(events) >= _MAX_EVENTS_PER_POLL:
                                break
                except (OSError, ValueError):
                    continue
        except Exception as exc:
            self._last_error = type(exc).__name__
        self._events_emitted = len(events)
        return events

    def _recent_files(self) -> list[Path]:
        candidates: list[tuple[int, Path]] = []
        discovered = 0
        for path in self.root.rglob("*.jsonl"):
            discovered += 1
            if discovered > _MAX_DISCOVERED_FILES:
                break
            try:
                if path.is_file():
                    candidates.append((path.stat().st_mtime_ns, path))
            except OSError:
                continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [path for _, path in candidates[:_MAX_FILES_PER_POLL]]
        keep = set(selected)
        self._offsets = {path: offset for path, offset in self._offsets.items() if path in keep}
        self._file_meta = {path: meta for path, meta in self._file_meta.items() if path in keep}
        return selected

    def _read_header(self, path: Path) -> tuple[str, str | None]:
        try:
            with path.open("rb") as handle:
                line = handle.readline(_MAX_LINE_BYTES + 1)
            if len(line) > _MAX_LINE_BYTES:
                return str(path), None
            row = json.loads(line)
            payload = row.get("payload") if isinstance(row, dict) else None
            if isinstance(payload, dict) and row.get("type") == "session_meta":
                session = str(payload.get("session_id") or payload.get("id") or path)
                cwd = payload.get("cwd")
                project = PureWindowsPath(str(cwd)).name if isinstance(cwd, str) and cwd else None
                return session, _safe_text(project, 100) or None
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return str(path), None

    @staticmethod
    def _discard_partial_line(handle) -> bool:
        # A nonzero seek can land in the middle of a UTF-8 character or JSON row.
        remaining = _MAX_INITIAL_BYTES_PER_FILE
        while remaining > 0:
            chunk = handle.readline(min(64 * 1024, remaining))
            remaining -= len(chunk)
            if not chunk:
                return False
            if chunk.endswith(b"\n"):
                return True
        return False

    def _read_file(self, path: Path, offset: int, allowance: int) -> tuple[list[tuple[int, dict[str, Any]]], int, int]:
        rows: list[tuple[int, dict[str, Any]]] = []
        consumed = 0
        with path.open("rb") as handle:
            handle.seek(offset)
            while consumed < allowance and len(rows) < _MAX_EVENTS_PER_POLL:
                line_offset = handle.tell()
                raw = handle.readline(min(_MAX_LINE_BYTES + 1, allowance - consumed))
                if not raw:
                    break
                consumed += len(raw)
                if not raw.endswith(b"\n") and len(raw) < _MAX_LINE_BYTES:
                    # The last row is still being written. Re-read it next time.
                    handle.seek(line_offset)
                    break
                if not raw.endswith(b"\n") and len(raw) >= _MAX_LINE_BYTES:
                    self._skipped_oversized_lines += 1
                    while consumed < allowance:
                        rest = handle.readline(min(4096, allowance - consumed))
                        if not rest:
                            break
                        consumed += len(rest)
                        if rest.endswith(b"\n"):
                            break
                    continue
                if len(raw) > _MAX_LINE_BYTES:
                    self._skipped_oversized_lines += 1
                    continue
                try:
                    row = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(row, dict):
                    rows.append((line_offset, row))
            return rows, handle.tell(), consumed

    def _event_from_row(
        self,
        row: dict[str, Any],
        path: Path,
        line_offset: int,
        session_key: str,
        project: str | None,
        fallback_time: datetime,
    ) -> TimelineEvent | None:
        if row.get("type") != "event_msg":
            return None
        payload = row.get("payload")
        if not isinstance(payload, dict):
            return None
        event_kind = payload.get("type")
        task = ""
        status = ""
        event_type = ""
        extra: dict[str, Any] = {}
        when = (_timestamp(payload.get("completed_at_ms"))
                or _timestamp(payload.get("started_at_ms"))
                or _timestamp(payload.get("completed_at"))
                or _timestamp(payload.get("started_at"))
                or _timestamp(row.get("timestamp"))
                or fallback_time)

        if event_kind == "task_started":
            event_type, status = "codex.task_started", "running"
        elif event_kind == "task_complete":
            event_type, status = "codex.task_completed", "completed"
        elif event_kind == "thread_goal_updated":
            goal = payload.get("goal")
            if isinstance(goal, dict):
                task = _safe_text(goal.get("objective"))
                status = _safe_status(goal.get("status"))
            event_type = "codex.task_updated"
        elif event_kind == "item_completed":
            item = payload.get("item")
            if not isinstance(item, dict):
                return None
            item_type = item.get("type")
            status = _safe_status(item.get("status"))
            if item_type == "CommandExecution":
                event_type = "codex.command_completed"
            elif item_type == "FileChange":
                event_type = "codex.file_changed"
                changes = item.get("changes")
                if isinstance(changes, dict):
                    extra["changed_files"] = min(len(changes), 100)
            elif item_type == "CollabAgentToolCall":
                event_type = "codex.agent_tool_completed"
                tool = _safe_text(item.get("tool"), 60)
                if tool:
                    extra["tool"] = tool
            else:
                return None
        else:
            return None

        event_key = f"{path.relative_to(self.root).as_posix()}:{line_offset}:{event_type}"
        data: dict[str, Any] = {"session_id": _hash(session_key), "status": status}
        if task:
            data["task"] = task
        data.update(extra)
        return TimelineEvent(
            source="codex",
            type=event_type,
            data=data,
            project=project,
            time=when,
            id=_hash(event_key),
        )

    def _remember(self, event_id: str) -> None:
        self._seen_ids.add(event_id)
        if len(self._seen_ids) > _MAX_REMEMBERED_IDS:
            self._seen_ids = set(sorted(self._seen_ids)[-_MAX_REMEMBERED_IDS:])

    def health(self) -> dict[str, Any]:
        return {
            "available": self.root.is_dir(),
            "last_poll_at": self._last_poll_at.isoformat() if self._last_poll_at else None,
            "files_scanned": self._files_scanned,
            "events_emitted": self._events_emitted,
            "skipped_oversized_lines": self._skipped_oversized_lines,
            "error_type": self._last_error,
        }
