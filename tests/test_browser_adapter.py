from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from personal_agent.adapters.browser import BrowserContextObserver


class BrowserContextObserverTests(unittest.TestCase):
    def test_current_context_reads_new_and_closed_snapshots_before_next_poll(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            now = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
            observer = BrowserContextObserver(root, clock=lambda: now)
            snapshot = {"schema_version": 1, "instance_id": "f27bc9ba01aa4f7d98d1374f7ef97b0c",
                        "browser": "Edge", "window_id": 1, "tab_id": 3, "captured_at": now.isoformat(),
                        "status": "active", "url": "https://example.test/old", "title": "Old", "dom_summary": {}}
            path = root / f"{snapshot['instance_id']}.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            observer.poll()
            snapshot.update(url="https://example.test/new", title="New")
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            self.assertEqual(observer.current_context()["title"], "New")
            snapshot.update(status="closed", tab_id=-1)
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            self.assertIsNone(observer.current_context())
            self.assertEqual([event.type for event in observer.poll()], ["browser.tab_closed"])

    def test_emits_metadata_but_keeps_page_text_only_in_current_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = [datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)]
            observer = BrowserContextObserver(root, clock=lambda: current[0])
            snapshot = {
                "schema_version": 1,
                "instance_id": "f27bc9ba01aa4f7d98d1374f7ef97b0c",
                "browser": "Chrome",
                "window_id": 2,
                "tab_id": 12,
                "captured_at": current[0].isoformat(),
                "status": "active",
                "url": "https://example.test/article",
                "title": "Example article",
                "page_text": "Private page body",
                "selected_text": "Selected passage",
                "dom_summary": {"headings": ["Heading"], "links": [], "link_count": 0,
                                "button_count": 0, "form_count": 0, "text_length": 18},
            }
            (root / f"{snapshot['instance_id']}.json").write_text(json.dumps(snapshot), encoding="utf-8")

            events = observer.poll()

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].source, "browser")
            self.assertEqual(events[0].type, "browser.context_updated")
            self.assertEqual(events[0].data["title"], "Example article")
            self.assertNotIn("page_text", events[0].data)
            self.assertNotIn("Private page body", json.dumps(events[0].data))
            current_context = observer.current_context()
            self.assertEqual(current_context["page_text"], "Private page body")
            self.assertEqual(current_context["selected_text"], "Selected passage")
            self.assertEqual(observer.poll(), [])

    def test_expires_stale_snapshot_and_removes_local_page_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = [datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)]
            observer = BrowserContextObserver(root, clock=lambda: current[0])
            snapshot = {
                "schema_version": 1,
                "instance_id": "f27bc9ba01aa4f7d98d1374f7ef97b0c",
                "browser": "Edge",
                "window_id": 1,
                "tab_id": 3,
                "captured_at": current[0].isoformat(),
                "status": "active",
                "url": "https://example.test/",
                "title": "Page",
                "page_text": "Transient page data",
                "selected_text": "",
                "dom_summary": {},
            }
            path = root / f"{snapshot['instance_id']}.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            observer.poll()
            self.assertIsNotNone(observer.current_context())

            current[0] += timedelta(minutes=31)
            expired = observer.poll()

            self.assertEqual([event.type for event in expired], ["browser.context_expired"])
            self.assertIsNone(observer.current_context())
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
