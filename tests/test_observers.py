from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from personal_agent.models import TimelineEvent
from personal_agent.observers import WindowsActivityObserver, sessionize, summarize_activity


class ObserverTests(unittest.TestCase):
    def test_event_queue_coalesces_paths_and_preserves_file_event_schema(self) -> None:
        class FakeWatchdog:
            instances: list["FakeWatchdog"] = []

            def __init__(self) -> None:
                self.scheduled = []
                self.started = False
                self.stopped = False
                FakeWatchdog.instances.append(self)

            def schedule(self, handler, path, recursive=False):
                self.scheduled.append((handler, path, recursive))

            def start(self) -> None:
                self.started = True

            def stop(self) -> None:
                self.stopped = True

            def join(self, timeout=None) -> None:
                pass

            def is_alive(self) -> bool:
                return self.started and not self.stopped

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "note.txt"
            target.write_text("one", encoding="utf-8")
            excluded = root / "node_modules" / "ignored.js"
            excluded.parent.mkdir()
            excluded.write_text("ignored", encoding="utf-8")
            runtime_state = root / ".state" / "memory" / "ignored.sqlite3"
            runtime_state.parent.mkdir(parents=True)
            runtime_state.write_text("runtime", encoding="utf-8")
            observer = WindowsActivityObserver(
                {"agent": root},
                event_queue_size=4,
                watchdog_loader=lambda: FakeWatchdog,
            )
            observer.start()
            fake = FakeWatchdog.instances[-1]
            handler, scheduled_root, recursive = fake.scheduled[0]
            self.assertEqual(Path(scheduled_root), root.resolve())
            self.assertTrue(recursive)

            handler.dispatch(SimpleNamespace(
                event_type="created", is_directory=False, src_path=str(target)
            ))
            handler.dispatch(SimpleNamespace(
                event_type="modified", is_directory=False, src_path=str(target)
            ))
            handler.dispatch(SimpleNamespace(
                event_type="created", is_directory=False, src_path=str(excluded)
            ))
            handler.dispatch(SimpleNamespace(
                event_type="modified", is_directory=False, src_path=str(runtime_state)
            ))
            events = observer.sample(snapshot={
                "foreground": None,
                "idle_seconds": 0.0,
                "processes": {},
            })
            file_events = [event for event in events if event.type.startswith("file.")]
            self.assertEqual(len(file_events), 1)
            self.assertEqual(file_events[0].type, "file.created")
            self.assertEqual(file_events[0].project, "agent")
            self.assertEqual(file_events[0].data["path"], "note.txt")
            self.assertIn("mtime_ns", file_events[0].data)
            self.assertEqual(file_events[0].data["size"], 3)

            target.unlink()
            handler.dispatch(SimpleNamespace(
                event_type="deleted", is_directory=False, src_path=str(target)
            ))
            deleted = [
                event for event in observer.sample(snapshot={
                    "foreground": None,
                    "idle_seconds": 0.0,
                    "processes": {},
                })
                if event.type.startswith("file.")
            ]
            self.assertEqual([(event.type, event.data) for event in deleted], [
                ("file.deleted", {"path": "note.txt"})
            ])
            health = observer.health()
            self.assertEqual(health["watched_projects"], ["agent"])
            self.assertGreaterEqual(health["coalesced_file_events"], 1)
            self.assertEqual(health["pending_file_events"], 0)
            observer.stop()
            self.assertTrue(fake.stopped)

    def test_event_queue_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("1", encoding="utf-8")
            second.write_text("2", encoding="utf-8")
            observer = WindowsActivityObserver(
                {"agent": root},
                event_queue_size=1,
                snapshot_provider=lambda: {},
            )
            observer._queue_path_change("agent", root, str(first), "created")
            observer._queue_path_change("agent", root, str(second), "created")
            health = observer.health()
            self.assertEqual(health["pending_file_events"], 1)
            self.assertEqual(health["dropped_file_events"], 1)
            self.assertEqual(health["file_event_queue_capacity"], 1)

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

    def test_worker_activity_is_summarized_without_raw_command_output(self) -> None:
        start = datetime(2026, 10, 26, 8, 0, tzinfo=timezone.utc)
        events = [
            TimelineEvent(
                source="codex",
                type="codex.task_started",
                data={"session_id": "opaque-1", "task": "Update observer tests", "status": "running"},
                project="personal-agent",
                time=start,
                id="worker-1",
            ),
            TimelineEvent(
                source="codex",
                type="codex.command_completed",
                data={"session_id": "opaque-1", "status": "running", "raw_output": "must not leak"},
                project="personal-agent",
                time=start + timedelta(minutes=2),
                id="worker-2",
            ),
            TimelineEvent(
                source="codex",
                type="codex.file_changed",
                data={"session_id": "opaque-1", "status": "completed"},
                project="personal-agent",
                time=start + timedelta(minutes=3),
                id="worker-3",
            ),
        ]

        summary = summarize_activity(events)

        self.assertIn("Codex activity", summary)
        self.assertIn("Update observer tests", summary)
        self.assertIn("1 command/tool events", summary)
        self.assertIn("1 file/change events", summary)
        self.assertIn("latest status completed", summary)
        self.assertNotIn("must not leak", summary)

    def test_worker_task_summary_redacts_common_credentials(self) -> None:
        event = TimelineEvent(
            source="opencode",
            type="opencode.task_started",
            data={"session_id": "opaque-2", "task": "Inspect bot token=12345:abcdefghijklmnopqrstuvwxyzABCDEFG", "status": "running"},
            time=datetime(2026, 10, 26, 8, 0, tzinfo=timezone.utc),
            id="worker-4",
        )

        summary = summarize_activity([event])

        self.assertIn("[redacted]", summary)
        self.assertNotIn("12345:abcdefghijklmnopqrstuvwxyzABCDEFG", summary)

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
