import asyncio
import unittest

from personal_agent.models import Identity
from personal_agent.supervisor import TaskSupervisor
from personal_agent.telegram import TelegramError
from tests.test_security_integration import Harness, ImmediateBrain, PRIMARY_ID, _message, _wait_supervisor_idle


class SupervisorRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_delivery_is_not_retrieved_as_an_assistant_reply(self):
        harness = Harness(ImmediateBrain("undelivered draft"))
        try:
            async def failed_send(*args):
                raise TelegramError("offline send failure")
            harness.telegram.send_text = failed_send
            await harness.supervisor.submit(_message(PRIMARY_ID, "question", update_id=1), Identity.PRIMARY)
            await _wait_supervisor_idle(harness.supervisor)
            messages = harness.store.recent_messages(PRIMARY_ID)
            self.assertFalse(any(m["role"] == "assistant" for m in messages))
            self.assertTrue(harness.store.get_state("pending_replies"))
            await harness.supervisor.shutdown()
            restarted = TaskSupervisor(harness.config, harness.store, harness.brain, harness.telegram, asyncio.Event(), transcriber=harness.transcriber)
            try:
                self.assertTrue(harness.store.get_state("interrupted_replies"))
                self.assertFalse(restarted.tasks)
            finally:
                await restarted.shutdown()
        finally:
            await harness.close()

    async def test_message_burst_keeps_latest_steering(self):
        harness = Harness(ImmediateBrain())
        try:
            for i in range(40):
                await harness.supervisor.submit(_message(PRIMARY_ID, f"latest-message-{i}", update_id=i), Identity.PRIMARY)
            await _wait_supervisor_idle(harness.supervisor)
            self.assertEqual(len(harness.brain.calls), 1)
            self.assertIn("latest-message-39", harness.brain.calls[0][1])
            self.assertIsNone(harness.store.get_state("last_rejected_update"))
        finally:
            await harness.close()

    async def test_attachment_acknowledgment_needs_no_brain_call(self):
        harness = Harness(ImmediateBrain())
        try:
            await harness.supervisor.submit(_message(PRIMARY_ID, "", update_id=1, attachment={"kind": "document", "file_id": "local-fixture", "file_name": "notes.txt"}), Identity.PRIMARY)
            await _wait_supervisor_idle(harness.supervisor)
            self.assertEqual(harness.brain.calls, [])
            self.assertEqual(len(harness.telegram.downloads), 1)
            self.assertIn("сохранено локально", harness.telegram.sent[0][1])
        finally:
            await harness.close()
