from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from personal_agent.memory import MemoryStore
from personal_agent.models import TimelineEvent


class MemoryStoreTests(unittest.TestCase):
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
