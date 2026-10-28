from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from personal_agent.memory import MemoryStore
from personal_agent.models import TimelineEvent


class MemoryStoreTests(unittest.TestCase):
    def test_schema_migration_backfills_attachment_sender_and_preserves_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memory_root = root / "memory"
            memory_root.mkdir(parents=True)
            db_path = memory_root / "state.sqlite3"
            profile = {
                "user_id": 42,
                "text": "Сохранённый профиль",
                "characters": 19,
                "sha256": "existing-profile-hash",
                "source_attachment_id": "profile-source",
                "source_attachment_ids": ["profile-source"],
                "saved_at": "2026-10-25T12:00:00+00:00",
            }
            legacy_attachment = {"id": "legacy-attachment", "sender_id": 42, "name": "legacy.txt"}
            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.executescript(
                    """
                    CREATE TABLE attachments (
                        id TEXT PRIMARY KEY,
                        time_utc TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    );
                    CREATE TABLE state (
                        key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    "INSERT INTO attachments(id,time_utc,metadata_json) VALUES(?,?,?)",
                    (
                        "legacy-attachment",
                        "2026-10-25T12:00:00+00:00",
                        json.dumps(legacy_attachment),
                    ),
                )
                conn.execute(
                    "INSERT INTO state(key,value_json,updated_at) VALUES(?,?,?)",
                    (
                        "persistent_user_profile:42",
                        json.dumps(profile, ensure_ascii=False),
                        "2026-10-25T12:00:00+00:00",
                    ),
                )

            with MemoryStore(root) as store:
                self.assertEqual(store.get_user_profile(42), profile)
                self.assertEqual(
                    [item["id"] for item in store.recent_attachments(42)],
                    ["legacy-attachment"],
                )

            with closing(sqlite3.connect(db_path)) as conn, conn:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(attachments)")}
                indexes = {row[1] for row in conn.execute("PRAGMA index_list(attachments)")}
                sender_id = conn.execute(
                    "SELECT sender_id FROM attachments WHERE id='legacy-attachment'"
                ).fetchone()[0]
                version = conn.execute("PRAGMA user_version").fetchone()[0]
            self.assertIn("sender_id", columns)
            self.assertIn("idx_attachments_sender_time", indexes)
            self.assertEqual(sender_id, 42)
            self.assertEqual(version, 1)

    def test_recent_attachments_is_sender_scoped_and_new_rows_store_sender_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with MemoryStore(root) as store:
                store.record_attachment({"id": "target", "sender_id": 42, "name": "target.txt"})
                for index in range(5):
                    store.record_attachment(
                        {"id": f"other-{index}", "sender_id": 7, "name": f"other-{index}.txt"}
                    )
                recent = store.recent_attachments(42, limit=1)
                self.assertEqual([item["id"] for item in recent], ["target"])

            with closing(sqlite3.connect(root / "memory" / "state.sqlite3")) as conn, conn:
                sender_id = conn.execute(
                    "SELECT sender_id FROM attachments WHERE id='target'"
                ).fetchone()[0]
            self.assertEqual(sender_id, 42)

    def test_append_user_profile_preserves_existing_text_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with MemoryStore(Path(temp)) as store:
                first = store.save_user_profile(42, "Первый фрагмент", source_attachment_id="first")
                appended = store.save_user_profile(
                    42, "Второй фрагмент", source_attachment_id="second", append=True,
                )
                duplicate = store.save_user_profile(
                    42, "Второй фрагмент", source_attachment_id="second", append=True,
                )
                self.assertIn("Первый фрагмент\n\nВторой фрагмент", appended["text"])
                self.assertEqual(duplicate["text"], appended["text"])
                self.assertEqual(duplicate["source_attachment_ids"], ["first", "second"])
                self.assertNotEqual(first["sha256"], appended["sha256"])

    def test_retention_prunes_old_observations_but_keeps_profile_and_recent_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            now = datetime(2026, 10, 26, 18, 0, tzinfo=timezone.utc)
            store = MemoryStore(root)
            old_event = TimelineEvent(
                source="windows", type="window.foreground", data={"exe": "old-marker"},
                project="old-project", time=now - timedelta(days=31), id="old-observation",
            )
            recent_event = TimelineEvent(
                source="filesystem", type="file.modified", data={"path": "recent-marker"},
                project="recent-project", time=now - timedelta(days=10), id="recent-observation",
            )
            durable_event = TimelineEvent(
                source="memory", type="USER_PROFILE_SAVED", data={"user_id": 42},
                time=now - timedelta(days=31), id="durable-profile-event",
            )
            old_browser_event = TimelineEvent(
                source="browser", type="browser.context_updated", data={"title": "expired-page-marker"},
                time=now - timedelta(days=31), id="old-browser-observation",
            )
            store.record_events([old_event, recent_event, durable_event, old_browser_event])
            profile = store.save_user_profile(42, "Постоянный профиль", source_attachment_id="attachment-42")

            screenshots = root / "memory" / "screenshots"
            screenshots.mkdir(parents=True)
            expired_capture = screenshots / "screenshot-expired.png"
            recent_capture = screenshots / "screenshot-recent.png"
            expired_capture.write_bytes(b"old")
            recent_capture.write_bytes(b"new")
            old_time = (now - timedelta(days=8)).timestamp()
            recent_time = (now - timedelta(days=2)).timestamp()
            os.utime(expired_capture, (old_time, old_time))
            os.utime(recent_capture, (recent_time, recent_time))

            store.set_state("last_screenshot:42", {
                "kind": "screenshot", "time": (now - timedelta(days=8)).isoformat(),
                "path": str(expired_capture),
            })
            store.set_state("last_observation:42", {
                "kind": "list_directory", "time": (now - timedelta(days=31)).isoformat(),
            })

            removed = store.cleanup_retained_data(now=now, tz=timezone.utc)

            self.assertFalse(expired_capture.exists())
            self.assertTrue(recent_capture.exists())
            self.assertEqual(removed["screenshots_deleted"], 1)
            self.assertEqual(removed["observation_events_deleted"], 2)
            self.assertEqual(removed["observation_states_deleted"], 2)
            remaining_ids = {event.id for event in store.events_between(now - timedelta(days=40), now)}
            self.assertNotIn(old_event.id, remaining_ids)
            self.assertIn(recent_event.id, remaining_ids)
            self.assertIn(durable_event.id, remaining_ids)
            self.assertNotIn(old_browser_event.id, remaining_ids)
            self.assertNotIn("old-observation", {item["id"] for item in store.retrieve("old-marker")})
            self.assertEqual(store.get_user_profile(42)["sha256"], profile["sha256"])
            self.assertIsNone(store.get_state("last_screenshot:42"))
            self.assertIsNone(store.get_state("last_observation:42"))
            store.close()

    def test_restart_fts_timeline_and_summary_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            start = datetime(2026, 10, 26, 7, 0, tzinfo=timezone.utc)
            event = TimelineEvent(
                source="windows",
                type="file.modified",
                data={"path": "personal_agent/memory.py", "action": "debugged observer"},
                project="personal-agent",
                time=start + timedelta(minutes=5),
                id="event-memory-1",
            )

            store = MemoryStore(root)
            store.record_event(event)
            message_id = store.record_message(42, "user", "remember observer debugging")
            store.record_attachment({"id": "attachment-1", "name": "trace.txt"})
            store.set_state("test.value", {"ok": True})
            store.set_checkpoint("telegram.offset", 17)
            store.increment_metric("events", 2)
            store.update_context("Working on local observer memory.")
            self.assertEqual(store.metrics_today()["events"], 2)
            self.assertTrue((root / "memory" / "current" / "context.md").exists())
            store.close()

            reopened = MemoryStore(root)
            self.assertTrue(reopened.get_state("runtime.previous_shutdown_clean"))
            self.assertEqual(reopened.get_state("test.value"), {"ok": True})
            self.assertEqual(reopened.get_checkpoint("telegram.offset"), 17)

            timeline = reopened.events_between(start, start + timedelta(hours=1))
            self.assertEqual([item.id for item in timeline], [event.id])

            hits = reopened.retrieve("observer", limit=8)
            ids = {item["id"] for item in hits}
            self.assertIn(event.id, ids)
            self.assertIn(message_id, ids)
            event_hit = next(item for item in hits if item["id"] == event.id)
            self.assertEqual(event_hit["provenance"], {"table": "events", "id": event.id})
            self.assertEqual(reopened.retrieve('" OR *', limit=8), [])

            messages = reopened.recent_messages(42)
            self.assertEqual(messages[-1]["text"], "remember observer debugging")

            path = reopened.export_day(date(2026, 10, 26), timezone.utc)
            day_text = path.read_text(encoding="utf-8")
            self.assertIn(f"[event:{event.id}]", day_text)
            project_files = list((root / "memory" / "projects").glob("*.md"))
            self.assertEqual(len(project_files), 1)
            self.assertIn(f"[event:{event.id}]", project_files[0].read_text(encoding="utf-8"))
            reopened.close()

    def test_event_duplicate_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = MemoryStore(Path(temp))
            event = TimelineEvent(source="test", type="one", data={"value": 1}, id="stable-id")
            store.record_event(event)
            store.record_event(event)
            hits = store.retrieve("value")
            self.assertEqual(sum(item["id"] == "stable-id" for item in hits), 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
