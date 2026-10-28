from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from personal_agent.adapters.vscode import VscodeCompanionObserver
from personal_agent.observers import summarize_activity


class VscodeCompanionObserverTests(unittest.TestCase):
    def test_reads_snapshot_metadata_and_deduplicates_unchanged_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            now = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
            observer = VscodeCompanionObserver(root, clock=lambda: now)
            snapshot_path = root / "window-a.json"
            snapshot = {
                "schema_version": 1,
                "captured_at": now.isoformat(),
                "workspace": "personal-agent",
                "workspace_folders": ["personal-agent"],
                "active_file": "personal_agent/daemon.py",
                "open_files": ["personal_agent/daemon.py", "tests/test_observers.py"],
                "selection": {"length": 24, "lines": 2},
                "diagnostics": {"error": 1, "warning": 2, "information": 0, "hint": 0},
                "git_branch": "codex/agent-observer",
                "active_editor_language": "python",
                "terminals": [{"name": "PowerShell", "process_id": 1234, "cwd": "C:/work/personal-agent"}],
            }
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

            events = observer.poll()

            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event.source, "vscode")
            self.assertEqual(event.type, "vscode.context_updated")
            self.assertEqual(event.project, "personal-agent")
            self.assertEqual(event.data["active_file"], "personal_agent/daemon.py")
            self.assertEqual(event.data["open_file_count"], 2)
            self.assertEqual(event.data["diagnostics"]["error"], 1)
            self.assertEqual(event.data["terminals"][0]["process_id"], 1234)
            self.assertEqual(observer.poll(), [])

            summary = summarize_activity(events)
            self.assertIn("VS Code activity", summary)
            self.assertIn("personal_agent/daemon.py", summary)
            self.assertIn("Git branch codex/agent-observer", summary)
            self.assertIn("diagnostics 1 errors, 2 warnings", summary)

            snapshot["captured_at"] = (now + timedelta(seconds=5)).isoformat()
            snapshot["active_file"] = "personal_agent/context.py"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            updated = observer.poll()
            self.assertEqual(len(updated), 1)
            self.assertEqual(updated[0].data["active_file"], "personal_agent/context.py")

    def test_expired_or_removed_snapshot_closes_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = [datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)]
            observer = VscodeCompanionObserver(root, clock=lambda: current[0])
            snapshot_path = root / "window-b.json"
            snapshot_path.write_text(json.dumps({
                "schema_version": 1,
                "captured_at": current[0].isoformat(),
                "workspace": "project-b",
            }), encoding="utf-8")
            self.assertEqual(observer.poll()[0].type, "vscode.context_updated")

            current[0] += timedelta(seconds=85)
            self.assertEqual(observer.poll()[0].type, "vscode.context_updated")
            snapshot_path.unlink()
            current[0] += timedelta(seconds=6)
            closed = observer.poll()

            self.assertEqual(len(closed), 1)
            self.assertEqual(closed[0].type, "vscode.workspace_closed")
            self.assertEqual(closed[0].project, "project-b")
            self.assertEqual(observer.health()["active_workspaces"], 0)

    def test_rejects_unknown_schema_and_large_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            observer = VscodeCompanionObserver(root)
            (root / "bad-schema.json").write_text(json.dumps({
                "schema_version": 2,
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }), encoding="utf-8")
            (root / "large.json").write_text(" " * (65 * 1024), encoding="utf-8")

            self.assertEqual(observer.poll(), [])
            self.assertEqual(observer.health()["active_workspaces"], 0)

    def test_preserves_paths_counts_full_lists_and_redacts_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            now = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
            observer = VscodeCompanionObserver(root, clock=lambda: now)
            workspace = r"C:\Users\user\Projects\a very long workspace path with  repeated spaces\child"
            (root / "window-c.json").write_text(json.dumps({
                "schema_version": 1,
                "captured_at": now.isoformat(),
                "workspace": workspace,
                "active_file": r"C:\work\file  name.py",
                "open_files": [r"C:\work\file  name.py"],
                "open_file_count": 83,
                "git_branch": "password=branch-secret-value",
                "terminals": [{"name": "PowerShell", "cwd": r"C:\work\folder  name"}],
                "terminal_count": 25,
            }), encoding="utf-8")

            event = observer.poll()[0]

            self.assertEqual(event.project, workspace)
            self.assertEqual(event.data["active_file"], r"C:\work\file  name.py")
            self.assertEqual(event.data["open_file_count"], 83)
            self.assertEqual(event.data["terminal_count"], 25)
            self.assertEqual(event.data["terminals"][0]["cwd"], r"C:\work\folder  name")
            self.assertIn("[redacted]", event.data["git_branch"])
            self.assertNotIn("branch-secret-value", json.dumps(event.data))
            self.assertIn(r"C:\work\file  name.py", summarize_activity([event]))


if __name__ == "__main__":
    unittest.main()
