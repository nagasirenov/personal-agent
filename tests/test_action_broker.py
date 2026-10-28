from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from personal_agent.action_broker import ActionBroker, ActionDenied, build_action_request
from personal_agent.security import ActionGate


class ActionRequestTests(unittest.TestCase):
    def assertDenied(self, code: str, fn) -> None:  # noqa: N802 - unittest style
        with self.assertRaises(ActionDenied) as caught:
            fn()
        self.assertEqual(caught.exception.code, code)
        self.assertTrue(caught.exception.public_message)

    def test_valid_native_tool_arguments_become_typed_requests(self) -> None:
        request = build_action_request("run_command", {"command": "Get-Location", "timeout_seconds": 5})
        self.assertEqual(request.kind, "run_command")
        self.assertEqual(request.arguments, {"command": "Get-Location", "timeout_seconds": 5})
        write = build_action_request("write_file", {"path": "C:\\x.txt", "text": "x", "append": True})
        self.assertEqual((write.kind, write.arguments["append"]), ("write_file", True))

    def test_rejects_malformed_oversized_and_unsafe_actions(self) -> None:
        invalid = [
            ("shell", {"command": "whoami"}, "unsupported_kind"),
            ("run_command", {"command": "x", "extra": True}, "malformed_plan"),
            ("run_command", {}, "invalid_arguments"),
            ("run_command", {"command": "x", "timeout_seconds": True}, "invalid_arguments"),
            ("run_command", {"command": "x", "timeout_seconds": 121}, "invalid_arguments"),
            ("run_command", {"command": "x\x00y"}, "invalid_arguments"),
            ("write_file", {"path": "C:\\x", "text": "x", "append": 1}, "invalid_arguments"),
            ("write_file", {"path": "C:\\x"}, "invalid_arguments"),
            ("write_file", {"path": "C:\\x", "text": "x" * (256 * 1024 + 1)}, "invalid_arguments"),
            ("write_file", "not an object", "malformed_plan"),
        ]
        for kind, arguments, code in invalid:
            with self.subTest(kind=kind, arguments=str(arguments)[:40]):
                self.assertDenied(code, lambda kind=kind, arguments=arguments: build_action_request(kind, arguments))


class ActionExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_secondary_is_denied_before_executor(self) -> None:
        with TemporaryDirectory() as folder:
            broker = ActionBroker(ActionGate(101, full_access=True), Path(folder))
            requests = [build_action_request("run_command", {"command": "whoami"})]
            executor = AsyncMock(side_effect=AssertionError("must not execute"))
            with patch("personal_agent.pc_tools.execute_pc_request", executor):
                with self.assertRaises(ActionDenied) as caught:
                    await broker.execute(202, requests)
            self.assertEqual(caught.exception.code, "action_denied")
            executor.assert_not_awaited()
            self.assertFalse(broker.available_for(202))
            self.assertTrue(broker.available_for(101))

    async def test_batch_is_fully_authorized_before_any_action_runs(self) -> None:
        class SelectiveGate:
            def authorize(self, sender_id, proposal):
                if proposal.get("kind") == "write_file":
                    raise PermissionError

        with TemporaryDirectory() as folder:
            broker = ActionBroker(SelectiveGate(), Path(folder))
            requests = [build_action_request("run_command", {"command": "whoami"}),
                        build_action_request("write_file", {"path": "C:\\x", "text": "x"})]
            executor = AsyncMock(side_effect=AssertionError("batch must not partially execute"))
            with patch("personal_agent.pc_tools.execute_pc_request", executor):
                with self.assertRaises(ActionDenied) as caught:
                    await broker.execute(101, requests)
            self.assertEqual(caught.exception.code, "action_denied")
            executor.assert_not_awaited()

    async def test_primary_actions_execute_sequentially(self) -> None:
        with TemporaryDirectory() as folder:
            broker = ActionBroker(ActionGate(101, full_access=True), Path(folder))
            requests = [build_action_request("run_command", {"command": "Get-Location"}),
                        build_action_request("write_file", {"path": "C:\\fixture.txt", "text": "ok"})]
            executor = AsyncMock(side_effect=[{"ok": True}, {"ok": True}])
            with patch("personal_agent.pc_tools.execute_pc_request", executor):
                result = await broker.execute(101, requests)
            self.assertEqual(result, [{"ok": True}, {"ok": True}])
            self.assertEqual(executor.await_count, 2)


if __name__ == "__main__":
    unittest.main()
