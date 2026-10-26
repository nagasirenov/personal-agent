"""Local, durable memory for the passive personal-agent milestone."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from datetime import date, datetime, time, timezone, tzinfo
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import TimelineEvent
from .security import owned_path


_MEMORY_DIRS = (
    "current",
    "projects",
    "days",
    "episodes",
    "attachments",
    "transcripts",
    "summaries",
    "people",
    "trash_pending",
)
_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_QUERY_STOPWORDS = set("а в во где для до за и из к как когда ли мне на насколько не о об от по почему с со то у чем что я ты мы вы он она это этом этой сегодня вчера последние последний последних час часа часов минут минуты делал делала занимался занималась готов готова готовность портом порта the a an is are what where when why how did i my last today yesterday".split())


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def _dt_text(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="microseconds")


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _limit(value: int, maximum: int) -> int:
    return max(0, min(int(value), maximum))


class MemoryStore:
    """SQLite-backed canonical memory plus deterministic Markdown views.

    Added helpers ``get_checkpoint``/``set_checkpoint`` expose the dedicated
    durable polling-checkpoint table without overloading general runtime state.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.memory_root = owned_path(self.root, "memory")
        self.memory_root.mkdir(parents=True, exist_ok=True)
        for name in _MEMORY_DIRS:
            owned_path(self.root, "memory", name).mkdir(parents=True, exist_ok=True)

        self.db_path = owned_path(self.root, "memory", "state.sqlite3")
        for suffix in ("-wal", "-shm"):
            owned_path(self.root, "memory", "state.sqlite3" + suffix)
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self._mark_session_started()

    def _create_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    time_utc TEXT NOT NULL,
                    source TEXT NOT NULL,
                    type TEXT NOT NULL,
                    project TEXT,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_time ON events(time_utc);
                CREATE INDEX IF NOT EXISTS idx_events_project_time ON events(project, time_utc);

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    time_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_user_time ON messages(user_id, time_utc);

                CREATE TABLE IF NOT EXISTS attachments (
                    id TEXT PRIMARY KEY,
                    time_utc TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    name TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS telemetry (
                    day_utc TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY(day_utc, name)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    kind UNINDEXED,
                    ref_id UNINDEXED,
                    project,
                    time_utc UNINDEXED,
                    text,
                    tokenize='unicode61'
                );
                """
            )

    def _mark_session_started(self) -> None:
        previous = self.get_state("runtime.status", None)
        self.set_state("runtime.previous_shutdown_clean", previous in (None, "clean"))
        self.set_state("runtime.status", "running")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemoryStore is closed")

    def record_event(self, event: TimelineEvent) -> None:
        self.record_events([event])

    def record_events(self, events: list[TimelineEvent]) -> None:
        self._ensure_open()
        with self._lock, self._conn:
            for event in events:
                when = _dt_text(event.time)
                data_json = _json_dumps(event.data)
                searchable = " ".join(part for part in (event.source, event.type, event.project or "", data_json) if part)
                cur = self._conn.execute(
                    """INSERT OR IGNORE INTO events(id,time_utc,source,type,project,data_json)
                       VALUES(?,?,?,?,?,?)""",
                    (event.id, when, event.source, event.type, event.project, data_json),
                )
                if cur.rowcount:
                    self._conn.execute(
                        "INSERT INTO memory_fts(kind,ref_id,project,time_utc,text) VALUES(?,?,?,?,?)",
                        ("event", event.id, event.project or "", when, searchable),
                    )

    def events_between(
        self, start: datetime, end: datetime, limit: int = 500
    ) -> list[TimelineEvent]:
        self._ensure_open()
        if end <= start:
            return []
        bounded = _limit(limit, 5000)
        if bounded == 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                """SELECT id,time_utc,source,type,project,data_json FROM events
                   WHERE time_utc >= ? AND time_utc < ?
                   ORDER BY time_utc ASC, id ASC LIMIT ?""",
                (_dt_text(start), _dt_text(end), bounded),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> TimelineEvent:
        return TimelineEvent(
            source=row["source"],
            type=row["type"],
            data=json.loads(row["data_json"]),
            project=row["project"],
            time=_parse_dt(row["time_utc"]),
            id=row["id"],
        )

    @staticmethod
    def _fts_query(query: str) -> str | None:
        tokens = [token for token in _FTS_TOKEN_RE.findall(query.casefold()) if token not in _QUERY_STOPWORDS][:12]
        if not tokens:
            return None
        return " OR ".join(f'"{token}"' for token in tokens)

    def retrieve(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        self._ensure_open()
        fts_query = self._fts_query(query)
        bounded = _limit(limit, 50)
        if not fts_query or bounded == 0:
            return []
        with self._lock:
            hits = self._conn.execute(
                """SELECT kind,ref_id,project,time_utc,
                          snippet(memory_fts,4,'[',']','…',18) AS match_text
                   FROM memory_fts WHERE memory_fts MATCH ?
                   ORDER BY bm25(memory_fts), time_utc DESC LIMIT ?""",
                (fts_query, bounded),
            ).fetchall()
            results: list[dict[str, Any]] = []
            for hit in hits:
                kind = hit["kind"]
                ref_id = hit["ref_id"]
                base: dict[str, Any] = {
                    "kind": kind,
                    "id": ref_id,
                    "time": hit["time_utc"],
                    "project": hit["project"] or None,
                    "match": hit["match_text"],
                    "provenance": {"table": kind + "s", "id": ref_id},
                }
                if kind == "event":
                    row = self._conn.execute(
                        "SELECT source,type,data_json FROM events WHERE id=?", (ref_id,)
                    ).fetchone()
                    if row:
                        base.update(
                            source=row["source"],
                            type=row["type"],
                            data=json.loads(row["data_json"]),
                        )
                elif kind == "message":
                    row = self._conn.execute(
                        "SELECT user_id,role,text FROM messages WHERE id=?", (ref_id,)
                    ).fetchone()
                    if row:
                        base.update(user_id=row["user_id"], role=row["role"], text=row["text"])
                elif kind == "attachment":
                    row = self._conn.execute(
                        "SELECT metadata_json FROM attachments WHERE id=?", (ref_id,)
                    ).fetchone()
                    if row:
                        base["metadata"] = json.loads(row["metadata_json"])
                results.append(base)
        return results

    def record_message(self, user_id: int, role: str, text: str) -> str:
        self._ensure_open()
        message_id = str(uuid4())
        when = _dt_text(datetime.now(timezone.utc))
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages(id,user_id,role,text,time_utc) VALUES(?,?,?,?,?)",
                (message_id, int(user_id), str(role), str(text), when),
            )
            self._conn.execute(
                "INSERT INTO memory_fts(kind,ref_id,project,time_utc,text) VALUES(?,?,?,?,?)",
                ("message", message_id, "", when, text),
            )
        return message_id

    def recent_messages(self, user_id: int, limit: int = 8) -> list[dict[str, Any]]:
        self._ensure_open()
        bounded = _limit(limit, 100)
        if bounded == 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                """SELECT id,user_id,role,text,time_utc FROM messages WHERE user_id=?
                   ORDER BY time_utc DESC, rowid DESC LIMIT ?""",
                (int(user_id), bounded),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def record_attachment(self, metadata: dict[str, Any]) -> None:
        self._ensure_open()
        item = dict(metadata)
        attachment_id = str(item.get("id") or uuid4())
        item["id"] = attachment_id
        raw_time = item.get("time")
        if isinstance(raw_time, datetime):
            when = _dt_text(raw_time)
        elif isinstance(raw_time, str):
            try:
                when = _dt_text(datetime.fromisoformat(raw_time))
            except (ValueError, TypeError):
                when = _dt_text(datetime.now(timezone.utc))
        else:
            when = _dt_text(datetime.now(timezone.utc))
        metadata_json = _json_dumps(item)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO attachments(id,time_utc,metadata_json) VALUES(?,?,?)",
                (attachment_id, when, metadata_json),
            )
            self._conn.execute(
                "DELETE FROM memory_fts WHERE kind=? AND ref_id=?", ("attachment", attachment_id)
            )
            self._conn.execute(
                "INSERT INTO memory_fts(kind,ref_id,project,time_utc,text) VALUES(?,?,?,?,?)",
                ("attachment", attachment_id, "", when, metadata_json),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        self._ensure_open()
        with self._lock:
            row = self._conn.execute("SELECT value_json FROM state WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value_json"])

    def set_state(self, key: str, value: Any) -> None:
        self._ensure_open()
        now = _dt_text(datetime.now(timezone.utc))
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO state(key,value_json,updated_at) VALUES(?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,
                                                  updated_at=excluded.updated_at""",
                (key, _json_dumps(value), now),
            )

    def get_checkpoint(self, name: str, default: Any = None) -> Any:
        self._ensure_open()
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json FROM checkpoints WHERE name=?", (name,)
            ).fetchone()
        return default if row is None else json.loads(row["value_json"])

    def set_checkpoint(self, name: str, value: Any) -> None:
        self._ensure_open()
        now = _dt_text(datetime.now(timezone.utc))
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO checkpoints(name,value_json,updated_at) VALUES(?,?,?)
                   ON CONFLICT(name) DO UPDATE SET value_json=excluded.value_json,
                                                   updated_at=excluded.updated_at""",
                (name, _json_dumps(value), now),
            )

    def increment_metric(self, name: str, amount: int | float = 1) -> None:
        self._ensure_open()
        day = datetime.now(timezone.utc).date().isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO telemetry(day_utc,name,value) VALUES(?,?,?)
                   ON CONFLICT(day_utc,name) DO UPDATE SET value=value+excluded.value""",
                (day, name, float(amount)),
            )

    def metrics_today(self) -> dict[str, int | float]:
        self._ensure_open()
        day = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            rows = self._conn.execute(
                "SELECT name,value FROM telemetry WHERE day_utc=? ORDER BY name", (day,)
            ).fetchall()
        result: dict[str, int | float] = dict.fromkeys((
            "cloud_calls_today", "cloud_input_estimate", "cloud_output_estimate",
            "vision_calls_today", "background_reasoning_calls",
        ), 0)
        for row in rows:
            value = float(row["value"])
            result[row["name"]] = int(value) if value.is_integer() else value
        return result

    @staticmethod
    def _safe_project_filename(project: str) -> str:
        slug = re.sub(r"[^\w.-]+", "-", project, flags=re.UNICODE).strip("-._") or "project"
        digest = hashlib.sha1(project.encode("utf-8")).hexdigest()[:8]
        return f"{slug[:80]}-{digest}.md"

    def _atomic_write(self, path: Path, text: str) -> None:
        path = owned_path(self.root, *path.relative_to(self.root).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + "." + str(uuid4()) + ".tmp")
        try:
            with tmp.open("x", encoding="utf-8") as handle:
                handle.write(text)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    def export_day(self, day: date, tz: tzinfo) -> Path:
        self._ensure_open()
        local_start = datetime.combine(day, time.min, tzinfo=tz)
        local_end = datetime.combine(day.fromordinal(day.toordinal() + 1), time.min, tzinfo=tz)
        start_utc = local_start.astimezone(timezone.utc)
        end_utc = local_end.astimezone(timezone.utc)
        events = self.events_between(start_utc, end_utc, limit=5000)

        from .observers import summarize_activity
        lines = [f"# Activity {day.isoformat()}", "", f"Timezone: `{tz}`", "",
                 "## Deterministic summary", "", summarize_activity(events), "", "## Evidence (bounded view)", ""]
        if not events:
            lines.append("No recorded events.")
        else:
            for event in events:
                local_time = event.time.astimezone(tz).strftime("%H:%M:%S")
                project = f" project=`{event.project}`" if event.project else ""
                compact = _json_dumps(event.data)
                lines.append(
                    f"- {local_time} `{event.source}:{event.type}`{project} "
                    f"{compact} [event:{event.id}]"
                )

        day_path = self.memory_root / "days" / f"{day.isoformat()}.md"
        self._atomic_write(day_path, "\n".join(lines).rstrip() + "\n")

        projects = sorted({event.project for event in events if event.project})
        for project in projects:
            self._export_project(str(project), tz)
        return day_path

    def _export_project(self, project: str, tz: tzinfo) -> Path:
        with self._lock:
            rows = self._conn.execute(
                """SELECT id,time_utc,source,type,project,data_json FROM events
                   WHERE project=? ORDER BY time_utc DESC,id DESC LIMIT 500""",
                (project,),
            ).fetchall()
        from .observers import summarize_activity
        project_events = [self._row_to_event(row) for row in reversed(rows)]
        lines = [f"# Project: {project}", "", "Latest 500 events; older canonical evidence remains in SQLite.", "",
                 summarize_activity(project_events), "", "## Evidence", ""]
        for row in reversed(rows):
            event = self._row_to_event(row)
            when = event.time.astimezone(tz).isoformat(timespec="seconds")
            lines.append(
                f"- {when} `{event.source}:{event.type}` {_json_dumps(event.data)} "
                f"[event:{event.id}]"
            )
        path = self.memory_root / "projects" / self._safe_project_filename(project)
        self._atomic_write(path, "\n".join(lines).rstrip() + "\n")
        return path

    def update_context(self, text: str) -> None:
        self._ensure_open()
        path = self.memory_root / "current" / "context.md"
        body = "# Current context\n\n" + str(text).strip() + "\n"
        self._atomic_write(path, body)

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            try:
                self.set_state("runtime.status", "clean")
            finally:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
