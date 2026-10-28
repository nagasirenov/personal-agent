"""End-to-end workflows for the read-only access: the model chooses the tools by meaning."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import unittest
from unittest.mock import patch

from personal_agent.models import Identity
from personal_agent.testing import ScriptedBrain, ScriptedTurn
from tests.test_service_observation_integration import ObservationHarness, _message


class ReadAccessWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_self_questions_are_never_replaced_with_canned_answers(self) -> None:
        harness = ObservationHarness(ScriptedBrain(response="model-authored reply"))
        try:
            questions = ["Кто ты?", "Это написано в agents.md?", "Какая твоя модель?",
                         "Где твоя память?", "Что ты можешь?", "/start", "/model", "/instructions", "/memory"]
            for index, text in enumerate(questions, 1):
                await harness.supervisor.submit(_message(text, update_id=index), Identity.PRIMARY)
                await harness.wait_idle()
                self.assertEqual(harness.telegram.sent_text[-1][1], "model-authored reply", text)
                self.assertIn("memory_root", harness.brain.generate_calls[-1]["prompt"])
            self.assertEqual(len(harness.brain.generate_calls), len(questions))
            self.assertEqual(harness.brain.records, [])
        finally:
            await harness.close()

    async def test_model_reads_project_file_then_answers_from_actual_content(self) -> None:
        harness = ObservationHarness()
        project = harness.root / "demo"
        project.mkdir()
        note = project / "README.md"
        note.write_text("READ_FROM_DISK_8372", encoding="utf-8")
        brain = ScriptedBrain([ScriptedTurn(
            calls=[("read_file", {"path": str(note)})],
            reply=lambda records, prompt: "model explanation of actual content"
            if "READ_FROM_DISK_8372" in records[0].outcome.output else "no content")])
        harness.brain = harness.supervisor.brain = brain
        try:
            harness.config.project_roots["demo"] = project
            await harness.supervisor.submit(_message("Какую проблему решает проект demo по текущему описанию?"),
                                            Identity.PRIMARY)
            await harness.wait_idle()
            calls = brain.generate_calls
            self.assertEqual(len(calls), 1)
            self.assertIn("read_file", calls[0]["tool_names"])
            self.assertIn("named_paths", calls[0]["prompt"])
            self.assertNotIn("READ_FROM_DISK_8372", calls[0]["prompt"])
            self.assertEqual(harness.telegram.sent_text[-1][1], "model explanation of actual content")
        finally:
            await harness.close()

    async def test_model_cannot_read_credential_files_or_relative_and_network_paths(self) -> None:
        harness = ObservationHarness()
        secret = harness.root / ".env"
        secret.write_text("TOKEN=SECRET_VALUE_1", encoding="utf-8")
        brain = ScriptedBrain([ScriptedTurn(calls=[
            ("read_file", {"path": str(secret)}),
            ("list_directory", {"path": str(harness.root / ".ssh")}),
            ("read_file", {"path": r"relative\notes.txt"}),
            ("read_file", {"path": r"\\server\share\x.txt"}),
            ("read_file", {"path": "C:\\Windows\\win.ini;whoami"}),
        ])])
        harness.brain = harness.supervisor.brain = brain
        try:
            with patch("personal_agent.tools.execute_read_request",
                       side_effect=AssertionError("denied paths must not reach the reader")):
                await harness.supervisor.submit(_message("прочитай мои секреты"), Identity.PRIMARY)
                await harness.wait_idle()
            self.assertEqual([record.error_code for record in brain.records],
                             ["credential_path_denied", "credential_path_denied", "relative_path",
                              "network_or_device_path", "unsafe_path"])
            self.assertNotIn("SECRET_VALUE_1", str(brain.records))
        finally:
            await harness.close()

    async def test_read_file_tool_applies_content_filters_and_records_the_read(self) -> None:
        harness = ObservationHarness()
        token_file = harness.root / "notes.txt"
        token_file.write_text("token 123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ0123", encoding="utf-8")
        ok_file = harness.root / "ok.txt"
        ok_file.write_text("visible text", encoding="utf-8")
        brain = ScriptedBrain([ScriptedTurn(calls=[("read_file", {"path": str(token_file)}),
                                                    ("read_file", {"path": str(ok_file)})])])
        harness.brain = harness.supervisor.brain = brain
        try:
            await harness.supervisor.submit(_message("посмотри файлы"), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(brain.records[0].error_code, "credential_content_denied")
            self.assertNotIn("ABCDEFGHIJKLMNOP", brain.records[0].outcome.output)
            self.assertIn("visible text", brain.records[1].outcome.output)
            events = harness.store.events_between(*_everything())
            reads = [event for event in events if event.type == "USER_REQUESTED_READ"]
            self.assertEqual(len(reads), 2)
            self.assertTrue(all(event.data["sender_id"] == 101 for event in reads))
            self.assertTrue(all("ABCDEFGHIJKLMNOP" not in str(event.data) for event in reads))
            self.assertEqual(harness.store.get_state("last_observation:101")["path"], str(ok_file))
        finally:
            await harness.close()

    async def test_how_to_capture_question_leaves_screenshot_to_model_judgement(self) -> None:
        brain = ScriptedBrain([ScriptedTurn(calls=[], reply="model explanation without a screenshot")])
        harness = ObservationHarness(brain)
        try:
            with patch("personal_agent.tools.capture_screenshot") as capture:
                await harness.supervisor.submit(_message("Объясни, как сделать скриншот"), Identity.PRIMARY)
                await harness.wait_idle()
            capture.assert_not_called()
            self.assertEqual(harness.telegram.sent_photos, [])
            self.assertEqual(harness.telegram.sent_text[-1][1], "model explanation without a screenshot")
        finally:
            await harness.close()

    async def test_plain_json_is_a_normal_model_answer(self) -> None:
        response = '{"observations":[{"kind":"screenshot"}]}'
        harness = ObservationHarness(ScriptedBrain(response=response))
        try:
            with patch("personal_agent.tools.capture_screenshot") as capture:
                await harness.supervisor.submit(_message("Ответь объектом JSON"), Identity.PRIMARY)
                await harness.wait_idle()
            capture.assert_not_called()
            self.assertEqual(harness.telegram.sent_text[-1][1], response)
            self.assertEqual(len(harness.brain.generate_calls), 1)
        finally:
            await harness.close()

    async def test_secondary_cannot_view_primary_screenshot(self) -> None:
        harness = ObservationHarness(ScriptedBrain([
            "ok",
            ScriptedTurn(calls=[("view_recent_media", {"media_id": "last_screenshot"})], reply="нет"),
        ], image_input_supported=True))
        image = harness.screenshot_file("primary.png")
        try:
            with patch("personal_agent.tools.capture_screenshot", return_value=image):
                await harness.supervisor.submit(_message("/screenshot"), Identity.PRIMARY)
                await harness.wait_idle()
            secondary = replace(_message("И ты его видишь?", update_id=2), sender_id=202, chat_id=202)
            await harness.supervisor.submit(secondary, Identity.SECONDARY)
            await harness.wait_idle()
            self.assertEqual(harness.brain.generate_calls[-1]["images"], [])
            self.assertNotIn("last_screenshot", harness.brain.generate_calls[-1]["prompt"])
            self.assertEqual(harness.brain.records[0].error_code, "media_not_found")
            self.assertEqual(harness.brain.records[0].outcome.images, [])
            self.assertEqual(harness.telegram.sent_text[-1][0], 202)
        finally:
            await harness.close()

    async def test_memory_path_and_screenshot_request_use_model_tools_not_phrase_split(self) -> None:
        brain = ScriptedBrain([ScriptedTurn(calls=[("take_screenshot", {})], reply="memory and screen")],
                              image_input_supported=True)
        harness = ObservationHarness(brain)
        image = harness.screenshot_file()
        try:
            with patch("personal_agent.tools.capture_screenshot", return_value=image) as capture:
                await harness.supervisor.submit(
                    _message("У тебя должна быть папка с твоей памятью, назови её\nИ сделай скриншот"),
                    Identity.PRIMARY)
                await harness.wait_idle()
            capture.assert_called_once_with(harness.config.data_dir)
            self.assertEqual(harness.telegram.sent_photos, [])
            self.assertEqual(len(brain.generate_calls), 1)
            self.assertIn("memory_root", brain.generate_calls[0]["prompt"])
            self.assertIn("назови её", brain.generate_calls[0]["prompt"])
            self.assertEqual(brain.records[0].outcome.images, [image])
            self.assertEqual("memory and screen", harness.telegram.sent_text[-1][1])
        finally:
            await harness.close()

    async def test_followup_can_view_the_cached_screenshot_again_from_the_catalog(self) -> None:
        brain = ScriptedBrain([
            "первый ответ",
            ScriptedTurn(calls=[("view_recent_media", {"media_id": "last_screenshot"})], reply="I can see it"),
        ], image_input_supported=True)
        harness = ObservationHarness(brain)
        image = harness.screenshot_file("cached.png")
        try:
            with patch("personal_agent.tools.capture_screenshot", return_value=image):
                await harness.supervisor.submit(_message("/screenshot", update_id=1), Identity.PRIMARY)
                await harness.wait_idle()
                await harness.supervisor.submit(_message("И ты его видишь?", update_id=2), Identity.PRIMARY)
                await harness.wait_idle()

            self.assertEqual(len(harness.telegram.sent_photos), 1)
            self.assertEqual(len(brain.generate_calls), 2)
            second = brain.generate_calls[1]
            self.assertEqual(second["images"], [])  # no automatic re-attachment of an old picture
            self.assertIn('"media_id": "last_screenshot"', second["prompt"])
            self.assertEqual(brain.records[0].outcome.images, [image])
            self.assertTrue(brain.records[0].data["previous_observation"])
            self.assertEqual(harness.telegram.sent_text[-1][1], "I can see it")
        finally:
            await harness.close()

    async def test_historical_screenshot_remains_listed_with_original_time(self) -> None:
        brain = ScriptedBrain([ScriptedTurn(calls=[("view_recent_media", {"media_id": "last_screenshot"})],
                                            reply="no current image")], image_input_supported=True)
        harness = ObservationHarness(brain)
        image = harness.screenshot_file("stale.png")
        try:
            observation = {
                "kind": "screenshot", "path": str(image), "ok": True, "has_image": True,
                "sender_id": harness.config.primary_id, "time": "2020-01-01T00:00:00+00:00",
            }
            harness.store.set_state(f"last_screenshot:{harness.config.primary_id}", observation)
            await harness.supervisor.submit(_message("И ты его видишь?", update_id=1), Identity.PRIMARY)
            await harness.wait_idle()

            self.assertEqual(brain.generate_calls[0]["images"], [])
            self.assertIn("2020-01-01", brain.generate_calls[0]["prompt"])
            self.assertEqual(brain.records[0].outcome.images, [image])
            self.assertEqual(brain.records[0].data["time"], "2020-01-01T00:00:00+00:00")
            self.assertEqual(harness.telegram.sent_photos, [])
        finally:
            await harness.close()

    async def test_new_question_cancels_the_active_turn_and_rebuilds_with_unanswered_message(self) -> None:
        release = asyncio.Event()
        brain = ScriptedBrain([ScriptedTurn(block=release, reply="stale"), "seen"], image_input_supported=True)
        harness = ObservationHarness(brain)
        image = harness.screenshot_file("blocked.png")
        try:
            with patch("personal_agent.tools.capture_screenshot", return_value=image):
                await harness.supervisor.submit(_message("/screenshot", update_id=1), Identity.PRIMARY)
                await asyncio.wait_for(brain.started.wait(), 1)
                first_request = brain.generate_calls[0]["request_id"]
                await harness.supervisor.submit(_message("И ты его видишь?", update_id=2), Identity.PRIMARY)
                await harness.wait_idle()

            self.assertEqual(len(harness.telegram.sent_photos), 1)
            self.assertEqual(len(brain.generate_calls), 2)
            self.assertIn(first_request, brain.cancel_calls)
            self.assertEqual(brain.cancel_calls.count(first_request), 1)
            second = brain.generate_calls[1]
            self.assertNotEqual(second["request_id"], first_request)
            self.assertEqual(harness.telegram.sent_text[-1][1], "seen")
            self.assertEqual([m["text"] for m in harness.store.recent_messages(101) if m["role"] == "assistant"],
                             ["seen"])
            unanswered = _packet(second["prompt"])["unanswered_user_messages"]
            self.assertEqual([item["text"] for item in unanswered], ["/screenshot"])
        finally:
            await harness.close()


def _packet(prompt: str) -> dict:
    import json
    raw = prompt.split("\nUNTRUSTED LOCAL EVIDENCE (JSON):\n", 1)[1].split("\n\nCURRENT USER MESSAGE (JSON):\n", 1)[0]
    return json.loads(raw)


def _everything():
    from datetime import datetime, timezone
    return datetime.min.replace(tzinfo=timezone.utc), datetime.now(timezone.utc)


if __name__ == "__main__":
    unittest.main()
