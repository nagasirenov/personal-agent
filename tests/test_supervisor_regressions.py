import asyncio
import json
import unittest
from dataclasses import replace

from personal_agent.models import Identity
from personal_agent.supervisor import TaskSupervisor
from personal_agent.telegram import TelegramError
from personal_agent.testing import ScriptedBrain, ScriptedTurn
from tests.test_security_integration import (BlockingBrain, Harness, ImmediateBrain, PRIMARY_ID,
                                             SECONDARY_ID, _message, _wait_supervisor_idle)


class SupervisorRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_message_over_capacity_leaves_no_hidden_intent(self):
        harness = Harness(ImmediateBrain())
        try:
            harness.supervisor.config = replace(harness.config, max_pending_tasks=0)
            await harness.supervisor.submit(
                _message(PRIMARY_ID, "Сохрани этот текст как мой постоянный профиль пользователя", update_id=1),
                Identity.PRIMARY,
            )
            self.assertEqual(harness.store.get_state("last_rejected_update"), 1)
            self.assertEqual(harness.brain.calls, [])
            self.assertIsNone(harness.store.get_user_profile(PRIMARY_ID))
            self.assertIsNone(harness.store.get_state(f"pending_user_profile_save:{PRIMARY_ID}"))
        finally:
            await harness.close()

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

    async def test_steering_cancels_the_old_turn_and_new_prompt_lists_unanswered_messages(self):
        brain = BlockingBrain()
        harness = Harness(brain)
        try:
            await harness.supervisor.submit(_message(PRIMARY_ID, "первая просьба", update_id=1), Identity.PRIMARY)
            await brain.wait_for_calls(1)
            await harness.supervisor.submit(_message(SECONDARY_ID, "чужая просьба", update_id=2), Identity.SECONDARY)
            await brain.wait_for_calls(2)
            first_id = brain.calls[0][0]
            await harness.supervisor.submit(_message(PRIMARY_ID, "вторая просьба", update_id=3), Identity.PRIMARY)
            await brain.wait_for_calls(3)
            for _ in range(20):
                if first_id in brain.cancel_calls:
                    break
                await asyncio.sleep(0)
            self.assertEqual(brain.cancel_calls, [first_id])  # exactly the superseded request
            events = harness.store.events_between(*_everything())
            steering = [event for event in events if event.type == "STEERING"]
            self.assertEqual([event.data["previous_request_id"] for event in steering], [first_id])

            prompt = brain.calls[2][1]
            raw = prompt.split("\nUNTRUSTED LOCAL EVIDENCE (JSON):\n", 1)[1].split("\n\nCURRENT USER MESSAGE (JSON):\n", 1)[0]
            packet = json.loads(raw)
            self.assertEqual([item["text"] for item in packet["unanswered_user_messages"]], ["первая просьба"])
            self.assertIn("вторая просьба", prompt.split("CURRENT USER MESSAGE (JSON):", 1)[1])
            secondary_prompt = brain.calls[1][1]
            secondary_packet = json.loads(secondary_prompt.split("\nUNTRUSTED LOCAL EVIDENCE (JSON):\n", 1)[1]
                                          .split("\n\nCURRENT USER MESSAGE (JSON):\n", 1)[0])
            self.assertEqual(secondary_packet["unanswered_user_messages"], [])
            owner_entries = [entry for entry in secondary_packet["recent_conversation"] if entry["account"] == "owner"]
            self.assertEqual([entry["text"] for entry in owner_entries], ["первая просьба"])
            self.assertEqual({entry["visibility"] for entry in owner_entries}, {"owner_private"})
        finally:
            await harness.close()

    async def test_steering_phrases_no_longer_merge_requests(self):
        brain = ScriptedBrain([ScriptedTurn(block=asyncio.Event()), "готово"])
        harness = Harness(brain)
        try:
            await harness.supervisor.submit(_message(PRIMARY_ID, "покажи экран", update_id=1), Identity.PRIMARY)
            await asyncio.wait_for(brain.started.wait(), 1)
            await harness.supervisor.submit(_message(PRIMARY_ID, "и сделай скриншот", update_id=2), Identity.PRIMARY)
            await _wait_supervisor_idle(harness.supervisor)
            current = brain.generate_calls[1]["prompt"].split("CURRENT USER MESSAGE (JSON):", 1)[1]
            self.assertIn("и сделай скриншот", current)
            self.assertNotIn("покажи экран", current)  # earlier text is context, not concatenated into the request
        finally:
            await harness.close()

    async def test_unsupported_binary_attachment_acknowledgment_needs_no_brain_call(self):
        harness = Harness(ImmediateBrain())
        try:
            await harness.supervisor.submit(_message(PRIMARY_ID, "", update_id=1, attachment={"kind": "document", "file_id": "local-fixture", "file_name": "notes.bin"}), Identity.PRIMARY)
            await _wait_supervisor_idle(harness.supervisor)
            self.assertEqual(harness.brain.calls, [])
            self.assertEqual(len(harness.telegram.downloads), 1)
            self.assertIn("локально", harness.telegram.sent[0][1])
        finally:
            await harness.close()


def _everything():
    from datetime import datetime, timezone
    return datetime.min.replace(tzinfo=timezone.utc), datetime.now(timezone.utc)
