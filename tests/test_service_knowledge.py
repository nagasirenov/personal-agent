from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from personal_agent.config import Config
from personal_agent.memory import MemoryStore
from personal_agent.self_knowledge import (
    PUBLIC_RULES, ServiceKnowledge, remember_preferences, wants_diagnostics,
)

PRIMARY_ID = 101
SECONDARY_ID = 202


class KnowledgeBrain:
    def __init__(self) -> None:
        self.health_calls = 0
        self.generate_calls = 0

    async def health(self) -> dict:
        self.health_calls += 1
        return {"ok": True, "backend": "fixture"}

    async def generate(self, *args, **kwargs) -> str:
        self.generate_calls += 1
        raise AssertionError("service knowledge must not generate")


class ServiceKnowledgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.store = MemoryStore(self.root)
        self.brain = KnowledgeBrain()
        self.config = Config(
            data_dir=self.root, primary_id=PRIMARY_ID, secondary_id=SECONDARY_ID,
            backend="codex_subscription", model="gpt-6.1-sol", reasoning="medium",
            timezone_name="UTC",
        )
        self.knowledge = ServiceKnowledge(self.config, self.store, self.brain)

    def tearDown(self) -> None:
        self.store.close()
        self._temp.cleanup()

    def test_facts_describe_identity_model_memory_and_public_rules(self) -> None:
        facts = self.knowledge.facts(PRIMARY_ID)
        self.assertIn("персональный агент", facts["identity"].casefold())
        self.assertIn("Telegram", facts["identity"])
        self.assertEqual(facts["model_requested"], "gpt-6.1-sol")
        self.assertEqual(facts["reasoning"], "medium")
        self.assertEqual(facts["brain_transport"], "codex_subscription")
        memory = (self.root / "memory").resolve()
        self.assertEqual(facts["agent_data_dir"], str(self.root.resolve()))
        self.assertEqual(facts["memory_root"], str(memory))
        self.assertEqual(facts["memory_database"], str(memory / "state.sqlite3"))
        self.assertTrue(Path(facts["memory_database"]).is_file())
        self.assertEqual(facts["named_paths"]["memory"], str(memory))
        self.assertEqual(facts["public_rules"], list(PUBLIC_RULES))
        self.assertEqual(facts["instruction_scope"], "personal_agent_only")
        self.assertFalse(facts["host_coding_instructions"])
        self.assertFalse(facts["subagent_orchestration"])
        self.assertFalse(facts["environment_actions"])
        self.assertEqual(self.brain.health_calls, 0)
        self.assertEqual(self.brain.generate_calls, 0)

    def test_historical_preferences_use_only_account_user_messages(self) -> None:
        self.store.record_message(PRIMARY_ID, "assistant", "Обращайся на ты.")
        self.store.record_message(PRIMARY_ID, "user", "Отвечай попроще и дружелюбно.")
        self.assertEqual(
            self.knowledge.facts(PRIMARY_ID)["conversation_preferences"],
            {"tone": "friendly"},
        )
        self.assertEqual(self.knowledge.facts(SECONDARY_ID)["conversation_preferences"], {})
        self.assertEqual(self.store.get_state(f"conversation_preferences:{SECONDARY_ID}"), {})

    def test_updated_preferences_remain_account_scoped(self) -> None:
        self.knowledge.facts(PRIMARY_ID)
        self.knowledge.facts(SECONDARY_ID)
        remember_preferences(self.store, PRIMARY_ID, "Пиши попроще.")
        remember_preferences(self.store, SECONDARY_ID, "Обращайся на ты.")
        self.assertEqual(
            self.knowledge.facts(PRIMARY_ID)["conversation_preferences"],
            {"tone": "friendly"},
        )
        self.assertEqual(
            self.knowledge.facts(SECONDARY_ID)["conversation_preferences"],
            {"address": "informal"},
        )

    async def test_conversation_and_descriptive_commands_are_never_intercepted(self) -> None:
        texts = (
            "/start", "/model", "/instructions", "/memory", "/capabilities",
            "Кто ты?", "Какая у тебя модель?", "Какие твои инструкции?",
            "Где твоя память?", "Что ты можешь?", "Это написано в AGENTS.md?",
            "Какая директория тебе доступна?", "Дай ему уже доступы",
            "Объясни, кто ты в этой истории", "Проведи самодиагностику",
            "Проверь свое состояние", "/status и объясни расход",
            "/diagnostics подробно", "/help me",
        )
        for text in texts:
            with self.subTest(text=text):
                self.assertIsNone(await self.knowledge.command_report(text))
        self.assertEqual(self.brain.health_calls, 0)
        self.assertEqual(self.brain.generate_calls, 0)

    async def test_status_reports_configured_model_and_actual_usage(self) -> None:
        self.store.increment_metric("cloud_calls_today", 7)
        self.store.increment_metric("vision_calls_today", 2)
        self.store.increment_metric("background_reasoning_calls", 3)
        report = await self.knowledge.command_report("/status")
        self.assertIsNotNone(report)
        self.assertIn("gpt-6.1-sol (medium)", report)
        self.assertIn("запросов к облачной модели — 7", report)
        self.assertIn("vision — 2", report)
        self.assertIn("фоновых рассуждений — 3", report)
        self.assertEqual(self.brain.health_calls, 0)
        self.assertEqual(self.brain.generate_calls, 0)

    async def test_diagnostics_reports_health_once_without_generation(self) -> None:
        observer = {"available": True, "last_error": None}
        self.store.set_state("observer_health", observer)
        self.store.set_state("telegram_health", {"ok": True})
        report = await self.knowledge.command_report("/diagnostics")
        self.assertIsNotNone(report)
        self.assertIn("gpt-6.1-sol, medium", report)
        self.assertIn("Backend: доступен", report)
        self.assertIn("Windows observer: работает", report)
        self.assertIn("Telegram polling: соединение работает", report)
        self.assertEqual(self.brain.health_calls, 1)
        self.assertEqual(self.brain.generate_calls, 0)
        saved = self.store.get_state("last_diagnostics")
        self.assertEqual(saved["brain"], {"ok": True, "backend": "fixture"})
        self.assertEqual(saved["observer"], observer)
        self.assertEqual(saved["model"], "gpt-6.1-sol")
        self.assertEqual(saved["backend"], "codex_subscription")

    async def test_help_reports_command_reference_without_health_or_generation(self) -> None:
        report = await self.knowledge.command_report("/help")
        self.assertIsNotNone(report)
        for command in ("/model", "/instructions", "/capabilities", "/memory",
                        "/status", "/diagnostics", "/ls", "/read", "/screenshot"):
            with self.subTest(command=command):
                self.assertIn(command, report)
        self.assertEqual(self.brain.health_calls, 0)
        self.assertEqual(self.brain.generate_calls, 0)

    async def test_natural_diagnostics_select_data_without_intercepting_reply(self) -> None:
        for text in ("Проведи самодиагностику", "Проведи диагностику",
                     "Проверь себя", "Проверь своё состояние"):
            with self.subTest(text=text):
                self.assertTrue(wants_diagnostics(text))
                self.assertIsNone(await self.knowledge.command_report(text))
        for text in ("Кто ты?", "Где твоя память?", "Расскажи о диагностике автомобиля"):
            with self.subTest(text=text):
                self.assertFalse(wants_diagnostics(text))
        self.assertEqual(self.brain.health_calls, 0)
        self.assertEqual(self.brain.generate_calls, 0)


if __name__ == "__main__":
    unittest.main()
