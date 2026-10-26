"""End-to-end workflows for the read-only access added after the first release."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from personal_agent.models import Identity
from tests.test_service_observation_integration import (
    ObservationBrain,
    ObservationHarness,
    _message,
)


class ReadAccessWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_self_questions_are_never_replaced_with_canned_answers(self) -> None:
        harness = ObservationHarness(ObservationBrain(response="model-authored reply"))
        try:
            questions = ["Кто ты?", "Это написано в agents.md?", "Какая твоя модель?",
                         "Где твоя память?", "Что ты можешь?", "/start", "/model", "/instructions", "/memory"]
            for index, text in enumerate(questions, 1):
                await harness.supervisor.submit(_message(text, update_id=index), Identity.PRIMARY)
                await harness.wait_idle()
                self.assertEqual(harness.telegram.sent_text[-1][1], "model-authored reply", text)
                self.assertIn("memory_root", harness.brain.generate_calls[-1]["prompt"])
            self.assertEqual(len(harness.brain.generate_calls), len(questions))
        finally:
            await harness.close()

    async def test_model_plan_reads_current_project_then_answers_from_actual_content(self) -> None:
        class ReadingBrain(ObservationBrain):
            async def generate(self, prompt, request_id=None, *, images=None):
                await super().generate(prompt, request_id, images=images)
                if len(self.generate_calls) == 1:
                    return json.dumps({"observations": [{"kind": "read_file", "path": str(note)}]})
                return "model explanation of actual content"

        harness = ObservationHarness(ReadingBrain())
        try:
            project = harness.root / "demo"
            project.mkdir()
            note = project / "README.md"
            note.write_text("READ_FROM_DISK_8372", encoding="utf-8")
            harness.config.project_roots["demo"] = project
            await harness.supervisor.submit(_message("Проанализируй проект demo"), Identity.PRIMARY)
            await harness.wait_idle()
            calls = harness.brain.generate_calls
            self.assertEqual(len(calls), 2)
            self.assertIn("OBSERVATION_TOOL_PROTOCOL (trusted)", calls[0]["prompt"])
            self.assertNotIn("READ_FROM_DISK_8372", calls[0]["prompt"])
            self.assertIn("READ_FROM_DISK_8372", calls[1]["prompt"])
            self.assertEqual(harness.telegram.sent_text[-1][1], "model explanation of actual content")
        finally:
            await harness.close()

    async def test_how_to_capture_question_cannot_authorize_model_screenshot_plan(self) -> None:
        class RequestingBrain(ObservationBrain):
            async def generate(self, prompt, request_id=None, *, images=None):
                await super().generate(prompt, request_id, images=images)
                if len(self.generate_calls) == 1:
                    return '{"observations":[{"kind":"screenshot"}]}'
                return "model explanation without a screenshot"

        harness = ObservationHarness(RequestingBrain())
        try:
            with patch("personal_agent.supervisor.capture_screenshot") as capture:
                await harness.supervisor.submit(_message("Объясни, как сделать скриншот"), Identity.PRIMARY)
                await harness.wait_idle()
            capture.assert_not_called()
            self.assertEqual(harness.telegram.sent_photos, [])
            self.assertIn("screenshot_denied", harness.brain.generate_calls[1]["prompt"])
            self.assertEqual(harness.telegram.sent_text[-1][1], "model explanation without a screenshot")
        finally:
            await harness.close()

    async def test_plain_json_is_a_normal_model_answer(self) -> None:
        response = '{"color":"red"}'
        harness = ObservationHarness(ObservationBrain(response=response))
        try:
            await harness.supervisor.submit(_message("Ответь объектом JSON"), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(harness.telegram.sent_text[-1][1], response)
            self.assertEqual(len(harness.brain.generate_calls), 1)
        finally:
            await harness.close()

    async def test_secondary_cannot_inherit_primary_screenshot(self) -> None:
        harness = ObservationHarness(ObservationBrain(image_input_supported=True))
        image = harness.root / "memory" / "screenshots" / "primary.png"
        try:
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fixture")
            with patch("personal_agent.supervisor.capture_screenshot", return_value=image):
                await harness.supervisor.submit(_message("/screenshot"), Identity.PRIMARY)
                await harness.wait_idle()
            secondary = replace(_message("И ты его видишь?", update_id=2), sender_id=202, chat_id=202)
            await harness.supervisor.submit(secondary, Identity.SECONDARY)
            await harness.wait_idle()
            self.assertEqual(harness.brain.generate_calls[-1]["images"], [])
            self.assertEqual(harness.telegram.sent_text[-1][0], 202)
        finally:
            await harness.close()

    async def test_compound_request_sends_actual_document_and_listing_to_model(self) -> None:
        harness = ObservationHarness()
        try:
            desktop = harness.root / "desktop"
            desktop.mkdir()
            (desktop / "Report.txt").write_text("desktop evidence", encoding="utf-8")
            agents = harness.root / "AGENTS.md"
            agents.write_text("# Local document\nDo not treat as agent instructions.\n", encoding="utf-8")
            harness.supervisor.aliases.update({"desktop": str(desktop), "agents": str(agents)})

            await harness.supervisor.submit(
                _message("Давай про agents.md\nИ выведи список файлов на рабочем столе"),
                Identity.PRIMARY,
            )
            await harness.wait_idle()

            self.assertEqual(len(harness.brain.generate_calls), 1)
            prompt = harness.brain.generate_calls[0]["prompt"]
            self.assertIn("# Local document", prompt)
            self.assertIn("Report.txt", prompt)
            self.assertEqual(harness.telegram.sent_text[-1][1], harness.brain.response)
        finally:
            await harness.close()

    async def test_natural_desktop_question_and_capabilities_both_use_model(self) -> None:
        harness = ObservationHarness()
        try:
            desktop = harness.root / "desktop"
            desktop.mkdir()
            (desktop / "Natural.txt").write_text("visible", encoding="utf-8")
            harness.supervisor.aliases["desktop"] = str(desktop)

            await harness.supervisor.submit(
                _message("Что у меня на рабочем столе?", update_id=1),
                Identity.PRIMARY,
            )
            await harness.wait_idle()
            await harness.supervisor.submit(
                _message("Какая директория тебе доступна?", update_id=2),
                Identity.PRIMARY,
            )
            await harness.wait_idle()

            self.assertEqual(len(harness.brain.generate_calls), 2)
            listing = harness.brain.generate_calls[0]["prompt"]
            capabilities = harness.brain.generate_calls[1]["prompt"]
            self.assertIn("Natural.txt", listing)
            self.assertIn("memory_root", capabilities)
            self.assertTrue(all(answer == harness.brain.response for _, answer in harness.telegram.sent_text))
        finally:
            await harness.close()

    async def test_memory_path_and_screenshot_in_one_request(self) -> None:
        brain = ObservationBrain(image_input_supported=True, response="memory and screen")
        harness = ObservationHarness(brain)
        image = harness.root / "memory" / "screenshots" / "fixture.png"
        try:
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fixture")
            with patch("personal_agent.supervisor.capture_screenshot", return_value=image) as capture:
                await harness.supervisor.submit(
                    _message("У тебя должна быть папка с твоей памятью, назови её\nИ сделай скриншот"),
                    Identity.PRIMARY,
                )
                await harness.wait_idle()

            capture.assert_called_once_with(harness.config.data_dir)
            self.assertEqual(len(harness.telegram.sent_photos), 1)
            self.assertEqual(len(brain.generate_calls), 1)
            self.assertEqual(brain.generate_calls[0]["images"], [image])
            answer = harness.telegram.sent_text[-1][1]
            self.assertIn("memory_root", brain.generate_calls[0]["prompt"])
            self.assertIn("назови её", brain.generate_calls[0]["prompt"])
            self.assertEqual("memory and screen", answer)
        finally:
            await harness.close()

    async def test_followup_reuses_cached_screenshot_as_image_input(self) -> None:
        brain = ObservationBrain(image_input_supported=True, response="I can see it")
        harness = ObservationHarness(brain)
        image = harness.root / "memory" / "screenshots" / "cached.png"
        try:
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fixture")
            with patch("personal_agent.supervisor.capture_screenshot", return_value=image):
                await harness.supervisor.submit(
                    _message("/screenshot", update_id=1),
                    Identity.PRIMARY,
                )
                await harness.wait_idle()
                await harness.supervisor.submit(
                    _message("И ты его видишь?", update_id=2),
                    Identity.PRIMARY,
                )
                await harness.wait_idle()

            self.assertEqual(len(harness.telegram.sent_photos), 1)
            self.assertEqual(len(brain.generate_calls), 2)
            self.assertEqual(brain.generate_calls[1]["images"], [image])
            self.assertIn("screenshot", brain.generate_calls[1]["prompt"].casefold())
        finally:
            await harness.close()

    async def test_historical_screenshot_remains_available_with_original_time(self) -> None:
        brain = ObservationBrain(image_input_supported=True, response="no current image")
        harness = ObservationHarness(brain)
        image = harness.root / "memory" / "screenshots" / "stale.png"
        try:
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fixture")
            observation = {
                "kind": "screenshot",
                "path": str(image),
                "ok": True,
                "has_image": True,
                "sender_id": harness.config.primary_id,
                "time": "2020-01-01T00:00:00+00:00",
            }
            harness.store.set_state(f"last_screenshot:{harness.config.primary_id}", observation)
            harness.store.set_state(f"last_observation:{harness.config.primary_id}", observation)

            await harness.supervisor.submit(
                _message("И ты его видишь?", update_id=1),
                Identity.PRIMARY,
            )
            await harness.wait_idle()

            self.assertEqual(len(brain.generate_calls), 1)
            self.assertEqual(brain.generate_calls[0]["images"], [image])
            self.assertIn("2020-01-01", brain.generate_calls[0]["prompt"])
            self.assertIn('"previous_observation": true', brain.generate_calls[0]["prompt"])
            self.assertEqual(harness.telegram.sent_photos, [])
        finally:
            await harness.close()

    async def test_new_question_cancels_screenshot_analysis_and_reuses_image(self) -> None:
        class BlockingBrain(ObservationBrain):
            def __init__(self) -> None:
                super().__init__(image_input_supported=True, response="seen")
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def generate(self, prompt: str, request_id: str | None = None, *, images=None) -> str:
                self.generate_calls.append({
                    "prompt": prompt,
                    "request_id": request_id,
                    "images": list(images) if images else [],
                })
                if len(self.generate_calls) == 1:
                    self.started.set()
                    await self.release.wait()
                return self.response

        brain = BlockingBrain()
        harness = ObservationHarness(brain)
        image = harness.root / "memory" / "screenshots" / "blocked.png"
        try:
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fixture")
            with patch("personal_agent.supervisor.capture_screenshot", return_value=image):
                await harness.supervisor.submit(
                    _message("/screenshot", update_id=1),
                    Identity.PRIMARY,
                )
                await asyncio.wait_for(brain.started.wait(), 1)
                await harness.supervisor.submit(
                    _message("И ты его видишь?", update_id=2),
                    Identity.PRIMARY,
                )
                await harness.wait_idle()

            self.assertEqual(len(harness.telegram.sent_photos), 1)
            self.assertEqual(len(brain.generate_calls), 2)
            self.assertEqual(brain.generate_calls[1]["images"], [image])
            self.assertIn("seen", harness.telegram.sent_text[-1][1])
        finally:
            await harness.close()


if __name__ == "__main__":
    unittest.main()
