from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from personal_agent.adapters.terminal import TerminalObserver, _read_process_details
from personal_agent.context import ContextBuilder
from personal_agent.memory import MemoryStore
from personal_agent.observers import summarize_activity


class TerminalObserverTests(unittest.TestCase):
    def test_reads_terminal_process_tree_and_deduplicates(self) -> None:
        current = [datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)]
        processes = [
            {"pid": 100, "parent_pid": 1, "name": "WindowsTerminal.exe"},
            {"pid": 101, "parent_pid": 100, "name": "OpenConsole.exe"},
            {"pid": 102, "parent_pid": 101, "name": "pwsh.exe"},
            {"pid": 103, "parent_pid": 102, "name": "python.exe"},
            {"pid": 104, "parent_pid": 102, "name": "node.exe"},
        ]
        observer = TerminalObserver(lambda: processes, clock=lambda: current[0])

        events = observer.poll()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source, "terminal")
        self.assertEqual(events[0].type, "terminal.context_updated")
        self.assertEqual(events[0].data["process"], "pwsh.exe")
        self.assertEqual(events[0].data["terminal_app"], "Windows Terminal")
        self.assertEqual(events[0].data["child_processes"], ["python.exe", "node.exe"])
        self.assertEqual(events[0].data["command_line_status"], "unavailable")
        self.assertEqual(events[0].data["process_cwd_status"], "unavailable")
        serialized = json.dumps(events[0].data)
        self.assertNotIn("command_line", events[0].data)
        for forbidden in ("terminal output", "shell history"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(observer.poll(), [])

        current[0] += timedelta(seconds=61)
        self.assertEqual(observer.poll()[0].type, "terminal.context_updated")
        self.assertEqual(observer.health()["active_terminals"], 1)

    def test_transient_enumeration_error_does_not_close_a_session(self) -> None:
        current = [datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)]
        available = [True]
        processes = [{"pid": 40, "parent_pid": 1, "name": "pwsh.exe"}]

        def enumerate_processes():
            if not available[0]:
                raise PermissionError("mocked access failure")
            return processes

        observer = TerminalObserver(enumerate_processes, clock=lambda: current[0])
        self.assertEqual(len(observer.poll()), 1)

        available[0] = False
        current[0] += timedelta(seconds=30)
        self.assertEqual(observer.poll(), [])
        self.assertEqual(observer.health()["error_type"], "PermissionError")
        self.assertEqual(observer.health()["active_terminals"], 1)

        available[0] = True
        processes.clear()
        self.assertEqual(observer.poll()[0].type, "terminal.session_closed")
        self.assertEqual(observer.health()["active_terminals"], 0)

    def test_command_lines_and_native_process_cwd_reach_model_context_safely(self) -> None:
        current = [datetime(2026, 10, 28, 11, 0, tzinfo=timezone.utc)]
        processes = [
            {"pid": 100, "parent_pid": 1, "name": "pwsh.exe"},
            {"pid": 101, "parent_pid": 100, "name": "python.exe"},
        ]
        details = {
            100: {"command_line": 'pwsh.exe -NoLogo --token "private-value"',
                  "process_cwd": r"C:\work\dir  with  spaces"},
            101: {"command_line": 'python.exe script.py --password secret-value',
                  "process_cwd": r"C:\work\child"},
        }
        observer = TerminalObserver(lambda: processes, details_provider=lambda pid: details[pid],
                                    clock=lambda: current[0])
        event = observer.poll()[0]
        self.assertEqual(event.data["process_cwd"], r"C:\work\dir  with  spaces")
        self.assertEqual(event.data["process_cwd_source"], "windows_process_parameters")
        self.assertEqual(event.data["command_line_status"], "known")
        self.assertEqual(event.data["child_details"][0]["command_line_status"], "known")
        self.assertNotIn("private-value", json.dumps(event.data))
        self.assertNotIn("secret-value", json.dumps(event.data))

        summary = summarize_activity([event])
        self.assertIn("native process cwd", summary)
        self.assertIn("shell location may differ", summary)
        self.assertIn("running child python.exe command line", summary)
        self.assertIn(r"child python.exe native process cwd C:\work\child", summary)
        with tempfile.TemporaryDirectory() as temp:
            with MemoryStore(Path(temp) / "state") as store:
                store.record_event(event)
                packet = ContextBuilder(store, timezone.utc).build("Что происходит в терминале?", 1,
                                                                      now=current[0] + timedelta(seconds=1))
                self.assertIn("native process cwd", packet)
                self.assertIn("running child", packet)
                self.assertIn("process_cwd может отличаться от пути оболочки", packet)
                self.assertNotIn("private-value", packet)
        details[100] = {"error": "access_denied_or_exited"}
        current[0] += timedelta(seconds=2)
        changed = observer.poll()[0]
        self.assertEqual(changed.data["command_line_status"], "unavailable")
        self.assertEqual(changed.data["process_cwd_status"], "unavailable")
        self.assertEqual(changed.data["unavailable_reason"], "access_denied_or_exited")
        self.assertNotIn("process_cwd", changed.data)
        self.assertIn("cwd unavailable", summarize_activity([changed]))

    @unittest.skipUnless(sys.platform == "win32", "Windows-only owned subprocess probe")
    def test_native_windows_reader_on_owned_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            marker = "agent-terminal-read-probe-20261028"
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(12)", marker],
                cwd=temp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                # A just-created process may expose its argv before its cwd is initialized.
                for _ in range(20):
                    details = _read_process_details(process.pid)
                    if details.get("process_cwd"):
                        break
                    time.sleep(0.05)
                self.assertNotIn("error", details, details)
                self.assertIn(marker, details.get("command_line", ""))
                self.assertEqual(os.path.normcase(os.path.normpath(details.get("process_cwd", ""))),
                                 os.path.normcase(os.path.normpath(temp)))
            finally:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
