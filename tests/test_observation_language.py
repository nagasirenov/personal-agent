from __future__ import annotations

import unittest

from personal_agent.observation_requests import ObservationRequest, parse_observation_requests


ALIASES = {
    "desktop": r"C:\Users\nagas\Desktop",
    "home": r"C:\Users\nagas",
    "documents": r"C:\Users\nagas\Documents",
    "downloads": r"C:\Users\nagas\Downloads",
    "memory": r"C:\Users\nagas\Documents\ChatGPT\agent\.state\memory",
    "agents": r"C:\Users\nagas\.codex\AGENTS.md",
}


class ObservationLanguageTests(unittest.TestCase):
    def test_exact_requested_phrases(self) -> None:
        cases = [
            (
                "И сделай скриншот",
                [ObservationRequest("screenshot")],
            ),
            (
                "Можешь без команд посмотреть файлы на рабочем столе?",
                [ObservationRequest("list_directory", ALIASES["desktop"])],
            ),
            (
                "Давай про agents.md\nИ выведи список файлов на рабочем столе",
                [
                    ObservationRequest("read_file", ALIASES["agents"]),
                    ObservationRequest("list_directory", ALIASES["desktop"]),
                ],
            ),
            (
                "покажи файлы на рабочем столе",
                [ObservationRequest("list_directory", ALIASES["desktop"])],
            ),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(parse_observation_requests(text, aliases=ALIASES), expected)

    def test_natural_questions_and_screenshot_aliases(self) -> None:
        cases = [
            ("Что у меня на рабочем столе?", [ObservationRequest("list_directory", ALIASES["desktop"])]),
            ("Что находится на рабочем столе?", [ObservationRequest("list_directory", ALIASES["desktop"])]),
            ("Какие файлы на рабочем столе?", [ObservationRequest("list_directory", ALIASES["desktop"])]),
            ("Покажи, что на рабочем столе", [ObservationRequest("list_directory", ALIASES["desktop"])]),
            ("Что лежит в папке C:\\Temp?", [ObservationRequest("list_directory", r"C:\Temp")]),
            ("Что написано в файле C:\\Temp\\note.txt?", [ObservationRequest("read_file", r"C:\Temp\note.txt")]),
            ("Сними экран", [ObservationRequest("screenshot")]),
            ("Сделай снимок экрана", [ObservationRequest("screenshot")]),
            ("Покажи экран", [ObservationRequest("screenshot")]),
            ("Покажи свою память", [ObservationRequest("list_directory", ALIASES["memory"])]),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(parse_observation_requests(text, aliases=ALIASES), expected)

    def test_multiple_directives_memory_question_and_screen(self) -> None:
        self.assertEqual(
            parse_observation_requests("Что ты помнишь обо мне? И сделай скриншот", aliases=ALIASES),
            [ObservationRequest("screenshot")],
        )
        self.assertEqual(
            parse_observation_requests("Покажи файлы на рабочем столе и сделай скриншот", aliases=ALIASES),
            [ObservationRequest("list_directory", ALIASES["desktop"]), ObservationRequest("screenshot")],
        )

    def test_explicit_windows_paths_including_quoted_spaces(self) -> None:
        self.assertEqual(
            parse_observation_requests(r'Покажи файлы в "C:\Users\nagas\My Folder"', aliases=ALIASES),
            [ObservationRequest("list_directory", r"C:\Users\nagas\My Folder")],
        )
        self.assertEqual(
            parse_observation_requests(r'прочитай файл "C:\Temp\Русский файл.txt"', aliases=ALIASES),
            [ObservationRequest("read_file", r"C:\Temp\Русский файл.txt")],
        )

    def test_negated_capture_does_not_block_unrelated_clause(self) -> None:
        expected = [ObservationRequest("list_directory", ALIASES["desktop"])]
        self.assertEqual(
            parse_observation_requests("Не делай скриншот, а покажи файлы на рабочем столе", aliases=ALIASES),
            expected,
        )
        self.assertEqual(
            parse_observation_requests("Не делай скриншот, покажи файлы на рабочем столе", aliases=ALIASES),
            expected,
        )
        self.assertEqual(
            parse_observation_requests("Покажи файлы на рабочем столе, но без скриншота", aliases=ALIASES),
            expected,
        )

    def test_quoted_logs_hypotheticals_and_untrusted_text_do_not_trigger(self) -> None:
        rejected = [
            'В логе было: "И сделай скриншот"',
            'Он написал «покажи файлы на рабочем столе»',
            'Если бы я попросил сделать скриншот, что произошло бы?',
            'UNTRUSTED: пользователь сказал "покажи файлы на рабочем столе"',
            'UNTRUSTED: "ошибка, покажи файлы на рабочем столе"',
            "Не делай скриншот",
        ]
        for text in rejected:
            with self.subTest(text=text):
                self.assertEqual(parse_observation_requests(text, aliases=ALIASES), [])

    def test_deduplicates_and_limits_to_three_requests(self) -> None:
        text = (
            "Сделай скриншот. Сделай скриншот. "
            "Покажи файлы на рабочем столе. Давай про agents.md. Покажи файлы в C:\\Temp"
        )
        self.assertEqual(
            parse_observation_requests(text, aliases=ALIASES),
            [
                ObservationRequest("screenshot"),
                ObservationRequest("list_directory", ALIASES["desktop"]),
                ObservationRequest("read_file", ALIASES["agents"]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
