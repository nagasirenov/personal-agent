"""Read-only observer for the active page snapshots supplied by Browser Companion."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ..browser_bridge import normalize_browser_snapshot
from ..models import TimelineEvent
from ..security import owned_path

_MAX_SNAPSHOTS = 8
_MAX_FILE_BYTES = 128 * 1024
_MAX_STALE = timedelta(minutes=30)
_HEARTBEAT = timedelta(minutes=5)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:24]


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


class BrowserContextObserver:
    """Read bounded active-page snapshots; retain page content only in process memory."""

    def __init__(self, storage_root: Path, *, clock: Callable[[], datetime] | None = None, receiver=None):
        self.root = Path(storage_root).expanduser().resolve()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.receiver = receiver
        self._known: dict[str, dict[str, Any]] = {}
        self._current: dict[str, dict[str, Any]] = {}
        self._last_poll_at: datetime | None = None
        self._files_scanned = 0
        self._events_emitted = 0
        self._last_error: str | None = None

    def poll(self) -> list[TimelineEvent]:
        now = self._clock()
        now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
        self._last_poll_at = now
        self._files_scanned = 0
        self._events_emitted = 0
        self._last_error = None
        events: list[TimelineEvent] = []
        seen: set[str] = set()
        current: dict[str, dict[str, Any]] = {}

        try:
            candidates = []
            if self.root.is_dir():
                for path in self.root.glob("*.json"):
                    if path.is_symlink() or not path.is_file():
                        continue
                    try:
                        candidates.append((path.stat().st_mtime_ns, path))
                    except OSError:
                        continue
            candidates.sort(key=lambda item: item[0], reverse=True)
            for _, path in candidates[:_MAX_SNAPSHOTS]:
                self._files_scanned += 1
                try:
                    if path.stat().st_size > _MAX_FILE_BYTES:
                        continue
                    raw = json.loads(path.read_bytes())
                    captured_at = _timestamp(raw.get("captured_at")) if isinstance(raw, dict) else None
                    if captured_at is None or now - captured_at > _MAX_STALE or captured_at > now + timedelta(minutes=2):
                        continue
                    snapshot = normalize_browser_snapshot(raw, captured_at=captured_at)
                    if snapshot is None:
                        continue
                    snapshot_time = _timestamp(snapshot.get("captured_at")) or captured_at
                    instance_id = snapshot["instance_id"]
                    if snapshot["status"] == "closed":
                        previous = self._known.pop(instance_id, None)
                        self._remove_snapshot(path)
                        if previous:
                            closed = dict(previous["event_data"], status="closed")
                            events.append(self._event(instance_id, closed, now, "browser.tab_closed"))
                        continue

                    seen.add(instance_id)
                    current[instance_id] = snapshot
                    event_data = self._event_metadata(snapshot)
                    fingerprint = _hash(json.dumps(event_data, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    previous = self._known.get(instance_id)
                    emit = (
                        previous is None
                        or previous["fingerprint"] != fingerprint
                        or now - previous["last_event_at"] >= _HEARTBEAT
                    )
                    self._known[instance_id] = {
                        "fingerprint": fingerprint,
                        "captured_at": snapshot_time,
                        "last_event_at": now if emit else previous["last_event_at"],
                        "event_data": event_data,
                    }
                    if emit:
                        events.append(self._event(instance_id, event_data, captured_at, "browser.context_updated"))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue

            for instance_id, previous in list(self._known.items()):
                if instance_id in seen or instance_id in current:
                    continue
                if now - previous["captured_at"] <= _MAX_STALE:
                    continue
                expired = dict(previous["event_data"], status="expired")
                events.append(self._event(instance_id, expired, now, "browser.context_expired"))
                self._known.pop(instance_id, None)
                current.pop(instance_id, None)

            for path in candidates:
                _, candidate = path
                if candidate.suffix == ".json" and candidate not in [self.root / f"{key}.json" for key in current]:
                    try:
                        raw = json.loads(candidate.read_bytes()) if candidate.stat().st_size <= _MAX_FILE_BYTES else {}
                        stamp = _timestamp(raw.get("captured_at")) if isinstance(raw, dict) else None
                        if stamp and now - stamp > _MAX_STALE:
                            self._remove_snapshot(candidate)
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        continue
        except OSError as exc:
            self._last_error = type(exc).__name__

        self._current = current
        self._events_emitted = len(events)
        return events

    @staticmethod
    def _event_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "session_id": _hash(snapshot["instance_id"]),
            "browser": snapshot["browser"],
            "url": snapshot["url"],
            "title": snapshot["title"],
            "page_text_length": len(snapshot["page_text"]),
            "selected_text_length": len(snapshot["selected_text"]),
            "heading_count": len(snapshot["dom_summary"]["headings"]),
            "link_count": snapshot["dom_summary"]["link_count"],
            "content_available": snapshot["content_available"],
            "status": "active",
        }

    @staticmethod
    def _event(instance_id: str, data: dict[str, Any], when: datetime, event_type: str) -> TimelineEvent:
        payload = dict(data)
        event_id = _hash(f"{instance_id}:{event_type}:{when.isoformat()}:{json.dumps(payload, ensure_ascii=False, sort_keys=True)}")
        return TimelineEvent(
            source="browser",
            type=event_type,
            data=payload,
            project=None,
            time=when,
            id=event_id,
        )

    def _remove_snapshot(self, path: Path) -> None:
        try:
            if not path.is_symlink() and path.is_file():
                owned_path(self.root, path.name).unlink(missing_ok=True)
        except (OSError, ValueError):
            return

    def current_contexts(self) -> list[dict[str, Any]]:
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)
        snapshots = []
        try:
            candidates = []
            for path in self.root.glob("*.json"):
                try:
                    if not path.is_symlink() and path.is_file():
                        candidates.append((path.stat().st_mtime_ns, path))
                except OSError:
                    continue
            for _, path in sorted(candidates, key=lambda item: item[0], reverse=True)[:_MAX_SNAPSHOTS]:
                try:
                    if path.stat().st_size > _MAX_FILE_BYTES:
                        continue
                    raw = json.loads(path.read_bytes())
                    captured = _timestamp(raw.get("captured_at")) if isinstance(raw, dict) else None
                    if captured is None or now - captured > _MAX_STALE or captured > now + timedelta(minutes=2):
                        continue
                    snapshot = normalize_browser_snapshot(raw, captured_at=captured)
                    if snapshot and snapshot["status"] == "active":
                        snapshots.append(snapshot)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
        except OSError:
            return []
        return sorted(snapshots, key=lambda item: item["last_interaction_at"], reverse=True)

    def current_context(self) -> dict[str, Any] | None:
        contexts = self.current_contexts()
        return contexts[0] if contexts else None

    def health(self) -> dict[str, Any]:
        receiver_health = self.receiver.health() if self.receiver else None
        receiver_available = receiver_health.get("available") if isinstance(receiver_health, dict) else True
        return {
            "available": self.root.is_dir() and self._last_error is None and receiver_available,
            "last_poll_at": self._last_poll_at.isoformat() if self._last_poll_at else None,
            "files_scanned": self._files_scanned,
            "active_tabs": len(self._current),
            "events_emitted": self._events_emitted,
            "error_type": self._last_error or (receiver_health.get("error_type") if isinstance(receiver_health, dict) else None),
            "receiver_host": receiver_health.get("host") if isinstance(receiver_health, dict) else None,
            "receiver_port": receiver_health.get("port") if isinstance(receiver_health, dict) else None,
            "receiver_token_path": receiver_health.get("token_path") if isinstance(receiver_health, dict) else None,
        }
