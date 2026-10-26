from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from personal_agent.config import Config
from personal_agent.context import ContextBuilder
from personal_agent.memory import MemoryStore
from personal_agent.models import TimelineEvent
from personal_agent.observers import sessionize, summarize_activity


class ContextTests(unittest.TestCase):
    def test_conversational_project_query_finds_technical_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            with MemoryStore(Path(temp)) as store:
                event = TimelineEvent("windows", "file.modified", {"path": "ModelLoader.java"}, "nagaport")
                store.record_event(event)
                hits = store.retrieve("Насколько сейчас готов nagaport?")
                self.assertTrue(any(hit["id"] == event.id for hit in hits))

    def test_context_includes_current_app_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            with MemoryStore(Path(temp)) as store:
                now = datetime.now(timezone.utc)
                store.record_event(TimelineEvent("windows", "window.foreground", {"exe": "Code.exe", "title": "Project"}, "project", now - timedelta(minutes=1)))
                store.set_state("current_foreground", {"exe": "Code.exe", "time": now.isoformat()})
                question = "Какая сейчас открыта программа? " + "x" * 11900
                packet = ContextBuilder(store, Config(Path(temp), 1, 2).tz).build(question, 1, now=now)
                self.assertLessEqual(len(packet), 24000)
                self.assertIn("Code.exe", packet)
                self.assertIn("UNTRUSTED", packet)
                self.assertIn("Какая сейчас открыта программа?", packet)
                self.assertIn("personal-agent-v2", packet)

    def test_summary_does_not_expand_all_event_ids_and_project_switch_splits(self):
        now = datetime.now(timezone.utc)
        events = [TimelineEvent("windows", "file.modified", {"path": "file.java"}, "one", now + timedelta(seconds=i)) for i in range(200)]
        events.append(TimelineEvent("windows", "window.foreground", {"exe": "Code.exe"}, "two", now + timedelta(seconds=210)))
        self.assertEqual(len(sessionize(events)), 2)
        summary = summarize_activity(events)
        self.assertLess(len(summary), 3000)
        self.assertIn(events[-1].id, summary)
        self.assertNotIn(events[100].id, summary)
