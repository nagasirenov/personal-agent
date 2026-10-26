from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone

from personal_agent.models import TimelineEvent
from personal_agent.observers import WindowsActivityObserver, sessionize, summarize_activity


class ObserverTests(unittest.TestCase):
    def test_deterministic_sampling_and_inference(self) -> None:
        base = datetime(2026, 10, 26, 8, 0, tzinfo=timezone.utc)
        snapshots = iter(
            [
                {
                    "foreground": {"hwnd": 1, "pid": 10, "exe": "C:/Tools/Code.exe", "title": "agent - Code"},
                    "idle_seconds": 3.0,
                    "processes": {10: "Code.exe", 20: "python.exe"},
                    "files": {"agent": {"personal_agent/memory.py": (1, 100)}},
                },
                {
                    "foreground": {"hwnd": 2, "pid": 30, "exe": "C:/Windows/System32/WindowsTerminal.exe", "title": "PowerShell"},
                    "idle_seconds": 2.0,
                    "processes": {10: "Code.exe", 20: "python.exe", 30: "WindowsTerminal.exe"},
                    "files": {"agent": {"personal_agent/memory.py": (2, 120)}},
                },
                {
                    "foreground": {"hwnd": 3, "pid": 40, "exe": "C:/Games/dota2.exe", "title": "game"},
                    "idle_seconds": 1.0,
                    "processes": {10: "Code.exe", 30: "WindowsTerminal.exe", 40: "dota2.exe"},
                    "files": {"agent": {"personal_agent/memory.py": (2, 120)}},
                },
            ]
        )
        times = iter([base, base + timedelta(minutes=2), base + timedelta(minutes=4)])
        observer = WindowsActivityObserver(
            {"agent": __import__("pathlib").Path("C:/work/agent")},
            snapshot_provider=lambda: next(snapshots),
            clock=lambda: next(times),
        )

        first = observer.sample()
        second = observer.sample()
        third = observer.sample()
        all_events = first + second + third

        self.assertTrue(any(event.type == "file.modified" for event in second))
        self.assertFalse(any(event.type.startswith("file.") for event in first))
        self.assertTrue(any(event.type == "process.started" and event.data["pid"] == 30 for event in second))
        self.assertTrue(any(event.type == "process.stopped" and event.data["pid"] == 20 for event in third))

        sessions = sessionize(all_events)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["activity"], "development/debugging")
        self.assertEqual(sessions[0]["confidence"], "high")
        self.assertIn("gaming", sessions[0]["transitions"])
        summary = summarize_activity(all_events)
        self.assertIn("Likely development/debugging", summary)
        self.assertIn("Later game foreground activity", summary)
        self.assertIn("Evidence:", summary)

        health = observer.health()
        self.assertEqual(health["samples"], 3)
        self.assertIsNone(health["last_error"])

    def test_uncertainty_is_explicit(self) -> None:
        event = TimelineEvent(
            source="windows",
            type="window.foreground",
            data={"exe": "unknown.exe"},
            time=datetime(2026, 10, 26, 8, 0, tzinfo=timezone.utc),
            id="unknown-1",
        )
        self.assertIn("uncertain", summarize_activity([event]).casefold())

    @unittest.skipUnless(sys.platform == "win32", "Windows-only live smoke check")
    def test_live_windows_snapshot_smoke(self) -> None:
        observer = WindowsActivityObserver({})
        snapshot = observer.snapshot()
        self.assertIn("foreground", snapshot)
        self.assertIn("idle_seconds", snapshot)
        self.assertIsInstance(snapshot["processes"], dict)
        events = observer.sample(snapshot=snapshot)
        self.assertIsInstance(events, list)
        self.assertTrue(observer.health()["available"])


if __name__ == "__main__":
    unittest.main()
