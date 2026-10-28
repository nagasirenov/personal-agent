"""Typing indicator, empty-final-answer retry and usage metrics around one model turn."""
from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace

from personal_agent.backends import BackendError
from personal_agent.models import Identity
from personal_agent.testing import ScriptedBrain, ScriptedTurn
from tests.test_security_integration import PRIMARY_ID, Harness, _message, _wait_supervisor_idle


class TypingTelegram:
    """Mixin target: wraps the harness FakeTelegram with send_chat_action."""

    def __init__(self, telegram, fail: bool = False) -> None:
        self.inner, self.fail, self.actions = telegram, fail, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.actions.append((chat_id, action))
        if self.fail:
            raise RuntimeError("telegram down")


class TypingIndicatorTests(unittest.IsolatedAsyncioTestCase):
    async def run_turn(self, text: str, *, fail=False, hold: float = 0.0):
        gate = asyncio.Event()
        brain = ScriptedBrain([ScriptedTurn(reply="готово", block=gate)])
        harness = Harness(brain)
        harness.supervisor.typing_interval = 0.02
        telegram = TypingTelegram(harness.telegram, fail=fail)
        harness.supervisor.telegram = telegram
        try:
            await harness.supervisor.submit(_message(PRIMARY_ID, text, update_id=1), Identity.PRIMARY)
            await asyncio.sleep(hold)
            gate.set()
            await _wait_supervisor_idle(harness.supervisor)
            return harness, telegram
        except BaseException:
            await harness.close()
            raise

    async def test_typing_is_sent_repeatedly_while_the_turn_runs_and_stops_after(self):
        harness, telegram = await self.run_turn("привет", hold=0.2)
        try:
            self.assertGreaterEqual(len(telegram.actions), 3)
            self.assertTrue(all(action == (PRIMARY_ID, "typing") for action in telegram.actions))
            count = len(telegram.actions)
            await asyncio.sleep(0.1)
            self.assertEqual(len(telegram.actions), count)  # the background task was cancelled
            self.assertEqual(harness.telegram.sent[-1], (PRIMARY_ID, "готово"))
            self.assertEqual([t for t in asyncio.all_tasks() if t.get_name().startswith("typing-")], [])
        finally:
            await harness.close()

    async def test_chat_action_errors_never_break_the_turn(self):
        harness, telegram = await self.run_turn("привет", fail=True, hold=0.1)
        try:
            self.assertGreaterEqual(len(telegram.actions), 2)
            self.assertEqual(harness.telegram.sent[-1], (PRIMARY_ID, "готово"))
            self.assertIsNone(harness.store.get_state("last_turn_error", None))
        finally:
            await harness.close()

    async def test_client_without_chat_action_support_is_tolerated(self):
        brain = ScriptedBrain(response="ок")
        harness = Harness(brain)
        harness.supervisor.typing_interval = 0.01
        try:
            await harness.supervisor.submit(_message(PRIMARY_ID, "привет", update_id=1), Identity.PRIMARY)
            await _wait_supervisor_idle(harness.supervisor)
            self.assertEqual(harness.telegram.sent[-1], (PRIMARY_ID, "ок"))
        finally:
            await harness.close()

    async def test_exact_service_commands_do_not_show_typing(self):
        for command in ("/status", " /Diagnostics "):
            harness, telegram = await self.run_turn(command, hold=0.1)
            try:
                with self.subTest(command=command):
                    self.assertEqual(telegram.actions, [])
            finally:
                await harness.close()


class EmptyAnswerBrain(ScriptedBrain):
    """First generate: runs tools, then reports empty_response like the real backend; second answers."""

    def __init__(self, calls, retry_reply="По результатам: 4721-ZETA", usage=None) -> None:
        super().__init__()
        self.calls, self.retry_reply, self.usage = calls, retry_reply, usage or {}
        self.consumed: list[str] = []

    async def generate(self, prompt, request_id=None, *, images=None, tools=None, tool_handler=None):
        self.generate_calls.append({"prompt": prompt, "request_id": request_id, "tool_names": [t["name"] for t in tools or []],
                                    "images": list(images or [])})
        if len(self.generate_calls) == 1:
            for position, (name, arguments) in enumerate(self.calls):
                await tool_handler(name, arguments, f"c{position}")
            raise BackendError("empty_response", "no final text")
        return self.retry_reply

    def consume_usage(self, request_id):
        self.consumed.append(request_id)
        return dict(self.usage)


class EmptyResponseRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_response_after_tools_is_retried_once_from_the_recorded_results(self):
        brain = EmptyAnswerBrain([("service_diagnostics", {})])
        harness = Harness(brain)
        try:
            await harness.supervisor.submit(_message(PRIMARY_ID, "ты работаешь?", update_id=1), Identity.PRIMARY)
            await _wait_supervisor_idle(harness.supervisor)
            self.assertEqual(len(brain.generate_calls), 2)
            first, second = brain.generate_calls
            self.assertTrue(first["tool_names"])
            self.assertEqual(second["tool_names"], [])  # no tool (and no action) can run twice
            self.assertEqual(second["request_id"], first["request_id"])
            self.assertTrue(second["prompt"].startswith(first["prompt"]))
            self.assertIn("SERVICE NOTE", second["prompt"])
            self.assertIn("do not invent", second["prompt"])
            self.assertIn("service_diagnostics", second["prompt"])
            self.assertEqual(harness.telegram.sent[-1], (PRIMARY_ID, "По результатам: 4721-ZETA"))
        finally:
            await harness.close()

    async def test_empty_response_without_tools_or_other_errors_are_not_retried(self):
        for error in (BackendError("empty_response", "x"), BackendError("rate_limited", "x")):
            brain = ScriptedBrain(error=error)
            harness = Harness(brain)
            try:
                await harness.supervisor.submit(_message(PRIMARY_ID, "привет", update_id=1), Identity.PRIMARY)
                await _wait_supervisor_idle(harness.supervisor)
                with self.subTest(code=error.code):
                    self.assertEqual(len(brain.generate_calls), 1)
                    self.assertIn(error.code, harness.telegram.sent[-1][1])
            finally:
                await harness.close()

    async def test_second_empty_response_is_reported_not_retried_again(self):
        class AlwaysEmpty(EmptyAnswerBrain):
            async def generate(self, prompt, request_id=None, **kwargs):
                if len(self.generate_calls) == 0:
                    return await super().generate(prompt, request_id, **kwargs)
                self.generate_calls.append({"prompt": prompt, "request_id": request_id, "tool_names": [], "images": []})
                raise BackendError("empty_response", "still nothing")

        brain = AlwaysEmpty([("service_diagnostics", {})])
        harness = Harness(brain)
        try:
            await harness.supervisor.submit(_message(PRIMARY_ID, "ты работаешь?", update_id=1), Identity.PRIMARY)
            await _wait_supervisor_idle(harness.supervisor)
            self.assertEqual(len(brain.generate_calls), 2)
            self.assertIn("empty_response", harness.telegram.sent[-1][1])
        finally:
            await harness.close()


class UsageMetricTests(unittest.IsolatedAsyncioTestCase):
    async def cloud_harness(self, brain) -> Harness:
        harness = Harness(brain)
        harness.supervisor.config = replace(harness.config, backend="codex_subscription")
        return harness

    async def test_cloud_calls_follow_the_upstream_requests_the_backend_reports(self):
        brain = EmptyAnswerBrain([("service_diagnostics", {})],
                                 usage={"upstream_requests": 4, "tool_calls": 3, "vision_requests": 2})
        harness = await self.cloud_harness(brain)
        try:
            await harness.supervisor.submit(_message(PRIMARY_ID, "статус?", update_id=1), Identity.PRIMARY)
            await _wait_supervisor_idle(harness.supervisor)
            metrics = harness.store.metrics_today()
            self.assertEqual(metrics["cloud_calls_today"], 4)
            self.assertEqual(metrics["vision_calls_today"], 2)
            self.assertEqual(metrics["tool_calls_today"], 3)
            self.assertGreater(metrics["cloud_input_estimate"], 0)
            self.assertEqual(len(brain.consumed), 1)
        finally:
            await harness.close()

    async def test_backend_without_usage_counts_one_call_per_generate_attempt(self):
        brain = ScriptedBrain(response="ок")
        harness = await self.cloud_harness(brain)
        try:
            await harness.supervisor.submit(_message(PRIMARY_ID, "привет", update_id=1), Identity.PRIMARY)
            await _wait_supervisor_idle(harness.supervisor)
            metrics = harness.store.metrics_today()
            self.assertEqual(metrics["cloud_calls_today"], 1)
            self.assertEqual(metrics["vision_calls_today"], 0)
        finally:
            await harness.close()


if __name__ == "__main__":
    unittest.main()
