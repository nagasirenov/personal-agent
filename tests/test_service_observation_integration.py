from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from personal_agent.backends import BackendError
from personal_agent.attachments import AttachmentResult
from personal_agent.config import Config
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage, TimelineEvent
from personal_agent.security import ActionGate
from personal_agent.supervisor import TaskSupervisor
from personal_agent.testing import ScriptedBrain, ScriptedTurn


PRIMARY_ID = 101
SECONDARY_ID = 202

# Existing workflow tests build on this name; the double now speaks the native tool contract.
ObservationBrain = ScriptedBrain


def _message(text: str, *, update_id: int = 1) -> IncomingMessage:
    return IncomingMessage(
        update_id=update_id,
        sender_id=PRIMARY_ID,
        chat_id=PRIMARY_ID,
        text=text,
        message_id=update_id,
    )


class FakeTelegram:
    def __init__(self) -> None:
        self.sent_text: list[tuple[int, str]] = []
        self.sent_photos: list[tuple[int, Path, str]] = []

    async def send_text(self, chat_id: int, text: str) -> None:
        self.sent_text.append((chat_id, text))

    async def send_photo(self, chat_id: int, path: Path, caption: str = "") -> None:
        self.sent_photos.append((chat_id, path, caption))

    async def download(self, file_id: str, destination: Path, max_bytes: int) -> int:
        raise AssertionError("observation integration tests do not download attachments")


class FakeTranscriber:
    async def close(self) -> None:
        return None


class ObservationHarness:
    def __init__(self, brain: ScriptedBrain | None = None) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.store = MemoryStore(self.root)
        self.telegram = FakeTelegram()
        self.brain = brain or ScriptedBrain()
        self.stop = asyncio.Event()
        self.config = Config(
            data_dir=self.root,
            primary_id=PRIMARY_ID,
            secondary_id=SECONDARY_ID,
            backend="mock",
            timezone_name="UTC",
            brain_timeout=2.0,
        )
        self.supervisor = TaskSupervisor(
            self.config,
            self.store,
            self.brain,
            self.telegram,
            self.stop,
            transcriber=FakeTranscriber(),
        )

    def enable_full_access(self) -> None:
        """Local-configuration grant (pc_access=full); PRIMARY gets write_file/run_command."""
        self.config = replace(self.config, pc_access="full")
        self.supervisor.config = self.config
        self.supervisor.action_gate = ActionGate(self.config.primary_id, full_access=True)

    def screenshot_file(self, name: str = "fixture.png") -> Path:
        image = self.root / "memory" / "screenshots" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"synthetic image")
        return image

    async def wait_idle(self) -> None:
        tasks = list(self.supervisor.tasks)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1.0)

    async def close(self) -> None:
        await self.supervisor.shutdown()
        self.store.close()
        self._temp.cleanup()


class ServiceObservationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_screenshot_delivery_reaches_only_the_current_chat_once(self) -> None:
        brain = ScriptedBrain([ScriptedTurn(
            calls=[("take_screenshot", {"send_to_user": True}), ("take_screenshot", {"send_to_user": True})],
            reply="Основная панель красная.")], image_input_supported=True)
        harness = ObservationHarness(brain)
        harness.enable_full_access()
        image = harness.screenshot_file("synthetic-red.png")
        request = "Оцени оформление открытого приложения по цвету главной панели. Хочу увидеть точную копию его текущего вида в этом чате."
        try:
            with patch("personal_agent.tools.capture_screenshot", return_value=image) as capture:
                await harness.supervisor.submit(_message(request), Identity.PRIMARY)
                await harness.wait_idle()
            self.assertEqual(capture.call_count, 2)
            capture.assert_called_with(harness.config.data_dir)
            self.assertEqual(harness.telegram.sent_photos, [(PRIMARY_ID, image, "Скриншот по твоему запросу.")])
            first, second = brain.records
            self.assertEqual(first.outcome.images, [image])
            self.assertTrue(first.data["delivered_to_user"])
            self.assertTrue(second.data["delivered_to_user"])
            self.assertEqual(harness.telegram.sent_text[-1][1], "Основная панель красная.")
        finally:
            await harness.close()

    async def test_internal_capture_is_private_and_failed_delivery_is_not_claimed(self) -> None:
        brain = ScriptedBrain([ScriptedTurn(calls=[("take_screenshot", {}),
                                                   ("take_screenshot", {"send_to_user": True})])],
                              image_input_supported=True)
        harness = ObservationHarness(brain)
        image = harness.screenshot_file()

        async def failing_send(chat_id, path, caption=""):
            raise RuntimeError("telegram offline")

        harness.telegram.send_photo = failing_send
        try:
            with patch("personal_agent.tools.capture_screenshot", return_value=image):
                await harness.supervisor.submit(_message("Что за окно открыто?"), Identity.PRIMARY)
                await harness.wait_idle()
            private, requested = brain.records
            self.assertFalse(private.data["delivered_to_user"])
            self.assertEqual(private.outcome.images, [image])
            self.assertFalse(requested.data["delivered_to_user"])
            self.assertEqual(requested.data["delivery_error"]["code"], "delivery_failed")
            self.assertIsNone(harness.store.get_state("last_screenshot_sent"))
        finally:
            await harness.close()

    async def test_screenshot_without_image_support_reports_no_pixels(self) -> None:
        brain = ScriptedBrain([ScriptedTurn(calls=[("take_screenshot", {})])], image_input_supported=False)
        harness = ObservationHarness(brain)
        image = harness.screenshot_file()
        try:
            with patch("personal_agent.tools.capture_screenshot", return_value=image):
                await harness.supervisor.submit(_message("Опиши экран"), Identity.PRIMARY)
                await harness.wait_idle()
            record = brain.records[0]
            self.assertEqual(record.outcome.images, [])
            self.assertFalse(record.data["image_attached"])
            self.assertIn("note", record.data)
        finally:
            await harness.close()

    async def test_missing_initial_browser_snapshot_lets_model_read_the_page(self) -> None:
        page = {
            "browser": "Edge", "page_text": "BROWSER_TOOL_EVIDENCE", "title": "Test",
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        provider_calls = 0

        def browser_contexts():
            nonlocal provider_calls
            provider_calls += 1
            return [] if provider_calls == 1 else [page]

        brain = ScriptedBrain([ScriptedTurn(
            calls=[("read_browser_page", {})],
            reply=lambda records, prompt: "На странице есть нужные данные." if "BROWSER_TOOL_EVIDENCE" in records[0].outcome.output else "нет")])
        harness = ObservationHarness(brain)
        try:
            harness.supervisor.browser_context_provider = browser_contexts
            await harness.supervisor.submit(_message("Что сейчас открыто в браузере?"), Identity.PRIMARY)
            await harness.wait_idle()

            prompt = brain.generate_calls[0]["prompt"]
            self.assertIn("current_browser_snapshot говорит лишь о том, что вкладка доступна", prompt)
            self.assertIn("read_browser_page", prompt)
            self.assertNotIn("BROWSER_TOOL_EVIDENCE", prompt)
            self.assertEqual(len(brain.generate_calls), 1)
            self.assertTrue(brain.records[0].ok)
            self.assertEqual(harness.telegram.sent_text[-1][1], "На странице есть нужные данные.")
        finally:
            await harness.close()

    async def test_unavailable_browser_page_is_reported_not_invented(self) -> None:
        brain = ScriptedBrain([ScriptedTurn(calls=[("read_browser_page", {})])])
        harness = ObservationHarness(brain)
        try:
            harness.supervisor.browser_context_provider = lambda: []
            await harness.supervisor.submit(_message("Можешь проверить?"), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(brain.records[0].error_code, "browser_page_unavailable")
        finally:
            await harness.close()

    async def test_voice_caption_and_transcript_reach_model_together(self) -> None:
        harness = ObservationHarness()
        try:
            attachment = AttachmentResult(transcript="содержимое голосового")
            with patch.object(harness.supervisor, "_attachment", new=AsyncMock(return_value=attachment)):
                await harness.supervisor.submit(_message("Переведи голосовое"), Identity.PRIMARY)
                await harness.wait_idle()
            prompt = harness.brain.generate_calls[-1]["prompt"]
            self.assertIn("Переведи голосовое", prompt)
            self.assertIn("содержимое голосового", prompt)
        finally:
            await harness.close()

    async def test_exact_slash_list_and_read_run_before_model_and_give_results_to_it(self) -> None:
        harness = ObservationHarness()
        try:
            list_result = {
                "ok": True, "kind": "list_directory", "path": r"C:\demo",
                "data": {"entries": [{"name": "folder", "type": "directory"},
                                     {"name": "note.txt", "type": "file"}],
                         "count": 2, "truncated": False, "bytes": 64},
                "provenance": {"source": "local_filesystem", "read_only": True},
            }
            read_result = {
                "ok": True, "kind": "read_file", "path": r"C:\demo\note.txt",
                "data": {"text": "bounded evidence", "bytes": 16, "truncated": False},
                "provenance": {"source": "local_filesystem", "read_only": True},
            }
            with patch("personal_agent.tools.execute_read_request", side_effect=[list_result, read_result]) as execute:
                await harness.supervisor.submit(_message(r"/ls C:\demo", update_id=1), Identity.PRIMARY)
                await harness.wait_idle()
                await harness.supervisor.submit(_message(r"/read C:\demo\note.txt", update_id=2), Identity.PRIMARY)
                await harness.wait_idle()

            self.assertEqual(execute.call_count, 2)
            recorded = harness.store.events_between(datetime.min.replace(tzinfo=timezone.utc), datetime.now(timezone.utc))
            reads = [event for event in recorded if event.type == "USER_REQUESTED_READ"]
            self.assertEqual(len(reads), 2)
            self.assertTrue(all(event.data.get("sender_id") == PRIMARY_ID for event in reads))
            self.assertEqual(len(harness.brain.generate_calls), 2)
            self.assertIn("note.txt", harness.brain.generate_calls[0]["prompt"])
            self.assertIn("bounded evidence", harness.brain.generate_calls[1]["prompt"])
            self.assertEqual(harness.brain.records, [])  # the model made no tool call of its own
            self.assertTrue(all(answer == harness.brain.response for _, answer in harness.telegram.sent_text))
            with self.assertRaises(PermissionError):
                harness.supervisor.action_gate.authorize(PRIMARY_ID, {"action": "shell", "command": "whoami"})
        finally:
            await harness.close()

    async def test_slash_ls_without_a_path_passes_needs_path_instead_of_guessing(self) -> None:
        harness = ObservationHarness()
        try:
            await harness.supervisor.submit(_message("/ls"), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(len(harness.brain.generate_calls), 1)
            self.assertIn('"needs_path": true', harness.brain.generate_calls[0]["prompt"])
        finally:
            await harness.close()

    async def test_slash_read_of_credential_file_is_denied_before_any_io(self) -> None:
        harness = ObservationHarness()
        try:
            with patch("personal_agent.tools.execute_read_request",
                       side_effect=AssertionError("credential path must not reach the reader")):
                await harness.supervisor.submit(_message(r"/read C:\Users\user\.env"), Identity.PRIMARY)
                await harness.wait_idle()
            prompt = harness.brain.generate_calls[0]["prompt"]
            self.assertIn("credential_path_denied", prompt)
            self.assertEqual(harness.telegram.sent_text[0][1], harness.brain.response)
        finally:
            await harness.close()

    async def test_model_read_error_renders_safe_code_without_executor_detail(self) -> None:
        brain = ScriptedBrain([ScriptedTurn(calls=[("read_file", {"path": r"C:\Users\user\notes.txt"})])])
        harness = ObservationHarness(brain)
        failure = {
            "ok": False, "kind": "read_file", "path": r"C:\Users\user\notes.txt", "denied": True,
            "error": {"code": "private_key_denied", "message": "SECRET=must-not-reach-the-model"},
            "data": {"text": "SECRET_BODY"},
            "provenance": {"source": "local_filesystem", "read_only": True},
        }
        try:
            with patch("personal_agent.tools.execute_read_request", return_value=failure):
                await harness.supervisor.submit(_message("Что в моих заметках?"), Identity.PRIMARY)
                await harness.wait_idle()
            record = brain.records[0]
            self.assertEqual(record.error_code, "private_key_denied")
            self.assertNotIn("SECRET", record.outcome.output)
        finally:
            await harness.close()

    async def test_slash_screenshot_delivers_photo_and_gives_the_model_the_image(self) -> None:
        brain = ScriptedBrain(image_input_supported=True, response="вижу рабочий стол")
        harness = ObservationHarness(brain)
        image = harness.screenshot_file()
        try:
            with patch("personal_agent.tools.capture_screenshot", return_value=image) as capture:
                await harness.supervisor.submit(_message("/screenshot"), Identity.PRIMARY)
                await harness.wait_idle()
            capture.assert_called_once_with(harness.config.data_dir)
            self.assertEqual(harness.telegram.sent_photos, [(PRIMARY_ID, image, "Скриншот по твоему запросу.")])
            self.assertEqual(brain.generate_calls[0]["images"], [image])
            self.assertIn('"attached_image_count": 1', brain.generate_calls[0]["prompt"])
            self.assertEqual(brain.records, [])
            self.assertIn("вижу рабочий стол", harness.telegram.sent_text[-1][1])
        finally:
            await harness.close()

    async def test_slash_screenshot_without_image_support_gives_backend_facts_to_brain(self) -> None:
        brain = ScriptedBrain(image_input_supported=False)
        harness = ObservationHarness(brain)
        image = harness.screenshot_file()
        try:
            with patch("personal_agent.tools.capture_screenshot", return_value=image) as capture:
                await harness.supervisor.submit(_message("/screenshot"), Identity.PRIMARY)
                await harness.wait_idle()
            capture.assert_called_once_with(harness.config.data_dir)
            self.assertEqual(len(harness.telegram.sent_photos), 1)
            self.assertEqual(brain.generate_calls[0]["images"], [])
            self.assertIn('"image_input_supported": false', brain.generate_calls[0]["prompt"])
            self.assertEqual(harness.telegram.sent_text[-1][1], brain.response)
        finally:
            await harness.close()

    async def test_diagnostics_command_routes_to_health_without_generate(self) -> None:
        brain = ScriptedBrain()
        harness = ObservationHarness(brain)
        try:
            harness.store.set_state("observer_health", {"available": True, "last_error": None})
            await harness.supervisor.submit(_message("/diagnostics"), Identity.PRIMARY)
            await harness.wait_idle()

            self.assertEqual(brain.health_calls, 1)
            self.assertEqual(brain.generate_calls, [])
            self.assertIn("самопроверка не вызывает llm", harness.telegram.sent_text[-1][1].casefold())
        finally:
            await harness.close()

    async def test_status_command_needs_no_model(self) -> None:
        harness = ObservationHarness()
        try:
            await harness.supervisor.submit(_message("/status"), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(harness.brain.generate_calls, [])
            self.assertIn("Модель:", harness.telegram.sent_text[-1][1])
        finally:
            await harness.close()

    async def test_model_selects_fresh_diagnostics_for_indirect_status_question(self) -> None:
        brain = ScriptedBrain([ScriptedTurn(
            calls=[("service_diagnostics", {})],
            reply=lambda records, prompt: "Последняя запись polling содержит ошибку; наблюдатель недоступен.")])
        harness = ObservationHarness(brain)
        try:
            harness.store.set_state("observer_health", {"available": False,
                                     "last_sample_time": "2026-10-27T12:00:00+00:00"})
            harness.store.set_state("telegram_health", {"ok": False})
            await harness.supervisor.submit(_message("Слушай, у тебя там всё в порядке со связью?"), Identity.PRIMARY)
            await harness.wait_idle()

            self.assertEqual(brain.health_calls, 1)
            self.assertEqual(len(brain.generate_calls), 1)
            self.assertNotIn('"kind": "service_diagnostics"', brain.generate_calls[0]["prompt"])
            data = brain.records[0].data
            self.assertEqual(data["kind"], "service_diagnostics")
            self.assertIs(data["telegram_polling"]["ok_last_recorded"], False)
            self.assertIs(data["observer"]["available"], False)
            self.assertIn("polling содержит ошибку", harness.telegram.sent_text[-1][1])
        finally:
            await harness.close()

    async def test_diagnostics_probe_not_triggered_without_model_selection(self) -> None:
        harness = ObservationHarness()
        try:
            await harness.supervisor.submit(_message("Как диагностировать автомобиль?"), Identity.PRIMARY)
            await harness.wait_idle()
            self.assertEqual(harness.brain.health_calls, 0)
            self.assertEqual(len(harness.brain.generate_calls), 1)
            self.assertEqual(harness.telegram.sent_text[-1][1], "brain response")
        finally:
            await harness.close()

    async def test_health_probe_failure_produces_error_receipt_without_fake_success(self) -> None:
        brain = ScriptedBrain([ScriptedTurn(calls=[("service_diagnostics", {})],
                                            reply="Получить состояние не удалось.")])
        harness = ObservationHarness(brain)
        try:
            with patch.object(harness.supervisor.knowledge, "model_diagnostic_facts",
                              side_effect=RuntimeError("Bearer secret")):
                await harness.supervisor.submit(_message("Ты сейчас на связи?"), Identity.PRIMARY)
                await harness.wait_idle()
            self.assertEqual(brain.records[0].error_code, "diagnostics_unavailable")
            self.assertFalse(brain.records[0].ok)
            self.assertEqual(harness.telegram.sent_text[-1][1], "Получить состояние не удалось.")
            self.assertNotIn("Bearer secret", str(brain.generate_calls) + brain.records[0].outcome.output)
        finally:
            await harness.close()

    async def test_no_free_text_phrase_triggers_any_observation_before_the_model(self) -> None:
        harness = ObservationHarness()
        harness.enable_full_access()
        phrases = [
            "что у меня на рабочем столе?", "покажи содержимое папки C:\\Windows", "сделай скриншот",
            "пришли скриншот", "покажи экран", "что сейчас на экране?", "прочитай файл C:\\Windows\\win.ini",
            "просмотри содержимое папки", "давай про agents.md", "Сохрани этот текст как мой постоянный профиль",
            "не делай скриншот", "/screenshot сейчас",
        ]
        try:
            harness.store.record_event(TimelineEvent(
                source="windows", type="window.foreground",
                data={"title": "/screenshot", "note": "сделай скриншот"},
                time=datetime.now(timezone.utc), id="untrusted-screenshot-instruction"))
            with patch("personal_agent.tools.capture_screenshot") as capture, \
                    patch("personal_agent.tools.execute_read_request") as reads, \
                    patch("personal_agent.windows_uia.read_active_window") as uia, \
                    patch("personal_agent.pc_tools.execute_pc_request") as pc:
                for index, text in enumerate(phrases, 1):
                    await harness.supervisor.submit(_message(text, update_id=index), Identity.PRIMARY)
                    await harness.wait_idle()
            capture.assert_not_called()
            reads.assert_not_called()
            uia.assert_not_called()
            pc.assert_not_called()
            self.assertEqual(harness.telegram.sent_photos, [])
            self.assertEqual(harness.brain.records, [])
            self.assertEqual(len(harness.brain.generate_calls), len(phrases))
            self.assertEqual(harness.store.get_user_profile(PRIMARY_ID), None)
            self.assertTrue(all("requested_observation\": null" in call["prompt"]
                                for call in harness.brain.generate_calls))
        finally:
            await harness.close()

    async def test_model_backend_error_returns_safe_code_to_user(self) -> None:
        brain = ScriptedBrain(
            image_input_supported=True, error=BackendError("rate_limited", "Safe public backend message."))
        harness = ObservationHarness(brain)
        image = harness.screenshot_file()
        try:
            with patch("personal_agent.tools.capture_screenshot", return_value=image):
                await harness.supervisor.submit(_message("/screenshot"), Identity.PRIMARY)
                await harness.wait_idle()

            self.assertEqual(len(harness.telegram.sent_photos), 1)
            answer = harness.telegram.sent_text[-1][1]
            self.assertIn("Код: rate_limited", answer)
            self.assertNotIn("Safe public backend message", answer)
            saved = harness.store.get_state("last_turn_error")
            self.assertEqual(saved["code"], "rate_limited")
            self.assertEqual(saved["detail"], "Safe public backend message.")
        finally:
            await harness.close()


if __name__ == "__main__":
    unittest.main()
