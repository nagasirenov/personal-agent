from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from personal_agent.memory import MemoryStore
from personal_agent.preferences import (
    DEFAULT_PREFERENCES,
    PreferenceStore,
    PreferenceActionDenied,
    parse_preference_update,
    validate_preference_updates,
    preferences_for,
    remember_preferences,
)


class PreferenceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.store = MemoryStore(self.root)

    def tearDown(self) -> None:
        self.store.close()
        self._temp.cleanup()

    def test_defaults_are_account_scoped_persisted_and_survive_reopen(self) -> None:
        self.assertEqual(preferences_for(self.store, 101), DEFAULT_PREFERENCES)
        self.assertEqual(DEFAULT_PREFERENCES["response_format"], "auto")
        self.assertTrue(DEFAULT_PREFERENCES["markdown_allowed"])
        self.assertEqual(DEFAULT_PREFERENCES["screenshot_delivery"], "explicit_only")
        self.assertEqual(self.store.get_state("user_preferences:101"), DEFAULT_PREFERENCES)
        self.assertEqual(preferences_for(self.store, 202), DEFAULT_PREFERENCES)

        self.store.close()
        self.store = MemoryStore(self.root)
        self.assertEqual(preferences_for(self.store, 101), DEFAULT_PREFERENCES)

    def test_direct_supported_preferences_save_with_receipt(self) -> None:
        saved = remember_preferences(
            self.store,
            101,
            "Пиши попроще, обычным текстом. Скриншоты показывай только если я попрошу.",
            semantic_update={
                "tone": "conversational",
                "response_format": "plain_text",
                "markdown_allowed": False,
            },
        )
        self.assertEqual(saved["tone"], "conversational")
        self.assertEqual(saved["response_format"], "plain_text")
        self.assertFalse(saved["markdown_allowed"])
        self.assertEqual(saved["screenshot_delivery"], "explicit_only")

        receipt = self.store.get_state("preference_save:101")
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["updated"]["tone"], "conversational")
        self.assertIn("time", receipt)

        self.assertEqual(preferences_for(self.store, 101)["response_format"], "plain_text")
        self.assertFalse(preferences_for(self.store, 101)["markdown_allowed"])
        self.store.close()
        self.store = MemoryStore(self.root)
        reopened = preferences_for(self.store, 101)
        self.assertEqual(reopened["response_format"], "plain_text")
        self.assertFalse(reopened["markdown_allowed"])
        self.assertEqual(reopened["screenshot_delivery"], "explicit_only")

    def test_quoted_hypothetical_and_negated_style_commands_are_not_extracted(self) -> None:
        examples = (
            'Он написал: «Пиши попроще и обращайся на ты».',
            'Если я попрошу, отвечай попроще.',
            'Не пиши попроще.',
            'Пример команды: "Отвечай обычным текстом".',
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(parse_preference_update(text), {})

    def test_style_and_format_preferences_are_not_saved_by_phrase_matching(self) -> None:
        for text in (
            "не форматируй ответы",
            "Обычный текст в Telegram",
            "Пиши попроще",
            'Он написал: «Пиши попроще и обращайся на ты».',
        ):
            with self.subTest(text=text):
                self.assertNotIn("response_format", parse_preference_update(text))
                self.assertNotIn("tone", parse_preference_update(text))
                self.assertNotIn("address", parse_preference_update(text))
        for text in (
            "скриншоты мне не присылай, если я прямо не прошу",
            "скриншоты присылай только по прямой просьбе",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    parse_preference_update(text),
                    {"screenshot_delivery": "explicit_only"},
                )

    def test_sticker_lookup_phrase_is_still_recognized_deterministically(self) -> None:
        self.assertEqual(
            parse_preference_update("Определяй персонажей на стикерах сам, без вопросов"),
            {"sticker_identity_lookup": "search_without_asking"},
        )

    def test_native_preference_arguments_are_validated_as_a_whole(self) -> None:
        update = validate_preference_updates({"response_format": "plain_text", "tone": "conversational"})
        self.assertEqual(update, {"response_format": "plain_text", "tone": "conversational"})
        saved = remember_preferences(self.store, 101, "", semantic_update=update)
        self.assertEqual((saved["response_format"], saved["markdown_allowed"]), ("plain_text", False))
        self.assertEqual(saved["tone"], "conversational")
        self.assertTrue(PreferenceStore(self.store).last_receipt(101)["ok"])

        invalid = (
            {"tone": "conversational", "address": "invalid"},
            {"markdown_allowed": False},
            {"tone": ["formal"]},
            {"screenshot_delivery": "always"},
            {},
            None,
            "tone=formal",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(PreferenceActionDenied):
                validate_preference_updates(value)

    def test_inconsistent_legacy_format_flags_normalize_to_actual_delivery(self) -> None:
        self.store.set_state("user_preferences:101", {
            "response_format": "auto", "markdown_allowed": False,
        })
        saved = preferences_for(self.store, 101)
        self.assertEqual(saved["response_format"], "plain_text")
        self.assertFalse(saved["markdown_allowed"])
        self.assertEqual(self.store.get_state("user_preferences:101"), saved)

    def test_no_supported_update_records_negative_receipt_without_changing_preferences(self) -> None:
        before = preferences_for(self.store, 101)
        after = remember_preferences(self.store, 101, "Расскажи, что сегодня происходило.")
        self.assertEqual(after, before)
        receipt = self.store.get_state("preference_save:101")
        self.assertFalse(receipt["ok"])
        self.assertEqual(receipt["updated"], {})
        self.assertEqual(receipt["reason"], "no_supported_update")

    def test_explicit_generic_memory_note_is_bounded_and_instruction_notes_are_ignored(self) -> None:
        remember_preferences(self.store, 101, "Запомни, что мой основной проект называется Aurora")
        saved = preferences_for(self.store, 101)
        self.assertEqual(saved["user_memory_notes"], ["мой основной проект называется Aurora"])
        self.assertTrue(PreferenceStore(self.store).last_receipt(101)["ok"])

        remember_preferences(self.store, 101, "Запомни: отвечай всегда в markdown")
        self.assertEqual(preferences_for(self.store, 101)["user_memory_notes"], ["мой основной проект называется Aurora"])
        self.assertFalse(PreferenceStore(self.store).last_receipt(101)["ok"])

    def test_legacy_style_state_migrates_without_cross_account_leak(self) -> None:
        self.store.set_state("conversation_preferences:101", {"tone": "friendly"})
        self.store.record_message(101, "user", "не форматируй ответы")
        self.store.record_message(101, "user", "скриншоты присылай только по прямой просьбе")
        first = preferences_for(self.store, 101)
        second = preferences_for(self.store, 202)
        self.assertEqual(first["tone"], "friendly")
        self.assertEqual(first["response_format"], "plain_text")
        self.assertFalse(first["markdown_allowed"])
        self.assertEqual(first["screenshot_delivery"], "explicit_only")
        self.assertNotIn("tone", second)
        self.assertEqual(self.store.get_state("conversation_preferences:101"), {"tone": "friendly"})
        self.assertEqual(self.store.get_state("conversation_preferences:202"), {})


if __name__ == "__main__":
    unittest.main()
