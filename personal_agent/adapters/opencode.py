"""Read-only OpenCode session metadata observer backed by SQLite."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from ..models import TimelineEvent

_MAX_SESSIONS = 100
_MAX_EVENTS_PER_POLL = 100
_MAX_REMEMBERED_SESSIONS = 5000
_STATUS_FIELDS = (
    "time_archived", "time_suspended", "time_idle", "idle_outcome",
)
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


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
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


class OpenCodeSessionObserver:
    """Poll recent OpenCode session metadata without opening transcript tables."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.path = Path(db_path or (Path.home() / ".local" / "share" / "opencode" / "opencode.db")).expanduser()
        self._fingerprints: dict[str, str] = {}
        self._last_poll_at: datetime | None = None
        self._sessions_scanned = 0
        self._events_emitted = 0
        self._last_error: str | None = None
        self._available = False

    def poll(self) -> list[TimelineEvent]:
        now = datetime.now(timezone.utc)
        self._last_poll_at = now
        self._sessions_scanned = 0
        self._events_emitted = 0
        self._last_error = None
        self._available = self.path.is_file()
        if not self._available:
            return []

        try:
            rows = self._load_rows()
            if not self._available:
                return []
            events: list[TimelineEvent] = []
            cutoff = now - timedelta(hours=24)
            for row in rows:
                self._sessions_scanned += 1
                session_id = str(row.get("id") or "")
                if not session_id:
                    continue
                updated = _timestamp(row.get("time_updated")) or _timestamp(row.get("time_created")) or now
                if updated < cutoff:
                    continue
                status = self._status(row, updated, now)
                title = _safe_text(row.get("title"))
                project = self._project_name(row)
                fingerprint = _hash("|".join((session_id, updated.isoformat(), status, title, project or "")))
                session_key = _hash(session_id)
                if self._fingerprints.get(session_key) == fingerprint:
                    continue
                self._fingerprints[session_key] = fingerprint
                if len(self._fingerprints) > _MAX_REMEMBERED_SESSIONS:
                    self._fingerprints.pop(next(iter(self._fingerprints)))
                if status == "active":
                    event_type = "opencode.task_started"
                elif status == "completed":
                    event_type = "opencode.task_completed"
                elif status == "suspended":
                    event_type = "opencode.task_suspended"
                else:
                    event_type = "opencode.task_observed"
                data: dict[str, Any] = {"session_id": session_key, "status": status}
                if title:
                    data["task"] = title
                events.append(TimelineEvent(
                    source="opencode",
                    type=event_type,
                    data=data,
                    project=project,
                    time=updated,
                    id=fingerprint,
                ))
                if len(events) >= _MAX_EVENTS_PER_POLL:
                    break
            self._events_emitted = len(events)
            self._available = True
            return events
        except (OSError, sqlite3.Error, ValueError) as exc:
            self._last_error = type(exc).__name__
            return []

    def _load_rows(self) -> list[dict[str, Any]]:
        uri = self.path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=0.25)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA busy_timeout=250")
            table_rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('session_v2','session','project')"
            ).fetchall()
            tables = {str(row[0]) for row in table_rows}
            session_table = "session_v2" if "session_v2" in tables else "session" if "session" in tables else None
            if session_table is None:
                self._available = False
                self._last_error = "session_table_missing"
                return []
            columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({session_table})")}
            selected = [name for name in (
                "id", "project_id", "directory", "title", "time_created", "time_updated",
                "time_archived", "time_suspended", "time_idle", "idle_outcome",
            ) if name in columns]
            if "id" not in selected:
                self._available = False
                self._last_error = "session_id_missing"
                return []
            if "time_updated" in selected:
                order = "time_updated DESC"
            elif "time_created" in selected:
                order = "time_created DESC"
            else:
                order = "id DESC"
            query = f"SELECT {', '.join('s.' + column for column in selected)}"
            join = ""
            if "project" in tables and "project_id" in selected:
                project_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(project)")}
                project_select = [name for name in ("name", "worktree") if name in project_columns]
                if "id" in project_columns and project_select:
                    query += ", " + ", ".join(f"p.{column} AS project_{column}" for column in project_select)
                    join = " LEFT JOIN project p ON p.id=s.project_id"
            query += f" FROM {session_table} s{join} ORDER BY s.{order} LIMIT ?"
            rows = connection.execute(query, (_MAX_SESSIONS,)).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    @staticmethod
    def _status(row: dict[str, Any], updated: datetime, now: datetime) -> str:
        if row.get("time_archived"):
            return "archived"
        if row.get("time_suspended"):
            return "suspended"
        outcome = str(row.get("idle_outcome") or "").strip().casefold()
        if outcome in {"completed", "success", "succeeded"}:
            return "completed"
        if now - updated <= timedelta(minutes=3):
            return "active"
        return "idle"

    @staticmethod
    def _project_name(row: dict[str, Any]) -> str | None:
        value = row.get("project_name") or row.get("project_worktree") or row.get("directory")
        if not isinstance(value, str) or not value.strip():
            return None
        name = PureWindowsPath(value).name or value
        return _safe_text(name, 100) or None

    def health(self) -> dict[str, Any]:
        return {
            "available": self._available,
            "last_poll_at": self._last_poll_at.isoformat() if self._last_poll_at else None,
            "sessions_scanned": self._sessions_scanned,
            "events_emitted": self._events_emitted,
            "error_type": self._last_error,
        }
