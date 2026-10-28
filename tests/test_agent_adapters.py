from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from personal_agent.adapters.codex import CodexSessionObserver
from personal_agent.adapters.opencode import OpenCodeSessionObserver


class CodexSessionObserverTests(unittest.TestCase):
    def test_reads_bounded_task_metadata_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / "2026" / "10" / "26" / "thread.jsonl"
            session.parent.mkdir(parents=True)
            rows = [
                {"type": "session_meta", "payload": {
                    "session_id": "private-session-id", "cwd": r"C:\work\sample-project",
                }},
                {"type": "event_msg", "payload": {"type": "task_started", "started_at_ms": 1_798_472_400_000}},
                {"type": "event_msg", "payload": {"type": "thread_goal_updated", "goal": {
                    "objective": "Refactor the observer and inspect token=12345:abcdefghijklmnopqrstuvwxyzABCDEFG",
                    "status": "in_progress",
                }}},
                {"type": "event_msg", "payload": {"type": "item_completed", "item": {
                    "type": "CommandExecution", "command": ["pwsh", "-Command", "do-not-export"],
                    "aggregated_output": "private output", "status": "completed",
                }}},
                {"type": "event_msg", "payload": {"type": "item_completed", "item": {
                    "type": "FileChange", "changes": {"personal_agent/observers.py": {}, "tests/test_observers.py": {}},
                    "stdout": "private diff", "status": "completed",
                }}},
                {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                    "content": [{"text": "private assistant transcript"}]}},
            ]
            session.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            observer = CodexSessionObserver(root)

            events = observer.poll()
            self.assertEqual([event.type for event in events], [
                "codex.task_started", "codex.task_updated", "codex.command_completed", "codex.file_changed",
            ])
            self.assertTrue(all(event.source == "codex" for event in events))
            self.assertTrue(all(event.project == "sample-project" for event in events))
            self.assertEqual(events[1].data["status"], "in_progress")
            self.assertIn("[redacted]", events[1].data["task"])
            self.assertEqual(events[3].data["changed_files"], 2)
            exported = json.dumps([event.data for event in events])
            for forbidden in ("private-session-id", "do-not-export", "private output", "private diff", "private assistant transcript"):
                self.assertNotIn(forbidden, exported)
            self.assertEqual(observer.poll(), [])

            restarted = CodexSessionObserver(root).poll()
            self.assertEqual([event.id for event in restarted], [event.id for event in events])
            self.assertTrue(observer.health()["available"])

    def test_does_not_parse_an_incomplete_final_jsonl_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / "partial.jsonl"
            session.write_bytes(b'{"type":"event_msg","payload":{"type":"task_started"')
            observer = CodexSessionObserver(root)

            self.assertEqual(observer.poll(), [])
            with session.open("ab") as handle:
                handle.write(b'}}\n')
            events = observer.poll()
            self.assertEqual([event.type for event in events], ["codex.task_started"])


class OpenCodeSessionObserverTests(unittest.TestCase):
    def test_reads_session_metadata_only_and_emits_state_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "opencode.db"
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            connection = sqlite3.connect(db_path)
            try:
                connection.execute("""CREATE TABLE project (
                    id TEXT PRIMARY KEY, name TEXT, worktree TEXT
                )""")
                connection.execute("""CREATE TABLE session_v2 (
                    id TEXT PRIMARY KEY, project_id TEXT, directory TEXT, title TEXT,
                    time_created INTEGER, time_updated INTEGER, time_archived INTEGER,
                    time_suspended INTEGER, time_idle INTEGER, idle_outcome TEXT
                )""")
                connection.execute("INSERT INTO project VALUES(?,?,?)", (
                    "p1", "sample-project", r"C:\work\sample-project",
                ))
                connection.execute("INSERT INTO session_v2 VALUES(?,?,?,?,?,?,?,?,?,?)", (
                    "private-session-id", "p1", r"C:\work\sample-project",
                    "Inspect source token=12345:abcdefghijklmnopqrstuvwxyzABCDEFG",
                    now_ms - 1000, now_ms, None, None, None, None,
                ))
                connection.commit()
            finally:
                connection.close()
            observer = OpenCodeSessionObserver(db_path)

            events = observer.poll()
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event.type, "opencode.task_started")
            self.assertEqual(event.source, "opencode")
            self.assertEqual(event.project, "sample-project")
            self.assertIn("[redacted]", event.data["task"])
            self.assertNotEqual(event.data["session_id"], "private-session-id")
            self.assertEqual(observer.poll(), [])
            connection = sqlite3.connect(db_path)
            try:
                connection.execute("UPDATE session_v2 SET time_updated=? WHERE id=?", (
                    now_ms + 5000, "private-session-id",
                ))
                connection.commit()
            finally:
                connection.close()
            self.assertEqual(len(observer.poll()), 1)
            self.assertTrue(observer.health()["available"])

    def test_missing_database_reports_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            observer = OpenCodeSessionObserver(Path(temp) / "missing.db")

            self.assertEqual(observer.poll(), [])
            self.assertFalse(observer.health()["available"])


if __name__ == "__main__":
    unittest.main()
