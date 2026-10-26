from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import personal_agent.observation_requests as observation_requests
from personal_agent.observation_requests import (
    MAX_DIRECTORY_ENTRIES,
    MAX_FILE_BYTES,
    ObservationRequest,
    execute_read_request,
    known_paths,
    parse_observation_request,
    parse_observation_requests,
)


class ObservationRequestTests(unittest.TestCase):
    def test_explicit_parser_contract(self) -> None:
        self.assertEqual(parse_observation_request(r"/ls C:\Users\nagas"), ObservationRequest("list_directory", r"C:\Users\nagas"))
        self.assertEqual(parse_observation_request(r'/read "C:\Temp\a file.txt"'), ObservationRequest("read_file", r"C:\Temp\a file.txt"))
        self.assertEqual(parse_observation_request("/screenshot"), ObservationRequest("screenshot"))
        self.assertEqual(parse_observation_request(r"покажи содержимое папки C:\Temp"), ObservationRequest("list_directory", r"C:\Temp"))
        self.assertEqual(parse_observation_request("просмотри содержимое папки"), ObservationRequest("list_directory", None))
        self.assertEqual(parse_observation_request("сделай скриншот"), ObservationRequest("screenshot"))
        self.assertEqual(parse_observation_request("можешь сделать скриншот?"), ObservationRequest("screenshot"))

    def test_parser_rejects_ordinary_negated_hypothetical_and_quoted_text(self) -> None:
        rejected = [
            "я сейчас читаю про наблюдение экрана",
            "не делай скриншот",
            "если бы я попросил сделать скриншот, что произошло бы?",
            'команда "/screenshot" должна работать',
            "`/screenshot`",
            "покажи как написать /ls C:\\Temp",
        ]
        for text in rejected:
            with self.subTest(text=text):
                self.assertIsNone(parse_observation_request(text))

    def test_missing_path_is_explicit(self) -> None:
        result = execute_read_request(ObservationRequest("list_directory"))
        self.assertFalse(result["ok"])
        self.assertTrue(result["needs_path"])
        self.assertEqual(result["error"]["code"], "needs_path")

    def test_batch_slash_ls_preserves_missing_path(self) -> None:
        aliases = {"desktop": r"C:\Users\nagas\Desktop"}
        self.assertEqual(
            parse_observation_requests("/ls", aliases=aliases, last_path=r"C:\Temp"),
            [ObservationRequest("list_directory")],
        )
        self.assertEqual(
            parse_observation_requests("/ls", aliases=aliases),
            [ObservationRequest("list_directory")],
        )
        self.assertEqual(parse_observation_request("/ls"), ObservationRequest("list_directory", None))

    def test_known_paths_includes_memory_projects_and_existing_global_agents(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home = root / "home"
            agents = home / ".codex" / "AGENTS.md"
            agents.parent.mkdir(parents=True)
            agents.write_text("rules", encoding="utf-8")
            with patch.object(Path, "home", return_value=home), patch.object(
                observation_requests, "_desktop_path", return_value=home / "OneDrive" / "Desktop"
            ):
                aliases = known_paths(root / "state", {"nagaport": root / "repo"})
            self.assertEqual(aliases["desktop"], str(home / "OneDrive" / "Desktop"))
            self.assertEqual(aliases["memory"], str(root / "state" / "memory"))
            self.assertEqual(aliases["agents"], str(agents))
            self.assertEqual(aliases["nagaport"], str(root / "repo"))

    def test_rejects_unc_device_and_relative_paths_without_io(self) -> None:
        for raw in (r"\\server\share", r"\\?\C:\Temp", r"\\.\PhysicalDrive0", r"relative\path"):
            with self.subTest(raw=raw):
                result = execute_read_request(ObservationRequest("list_directory", raw))
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"]["code"], "invalid_path")

    def test_bounded_directory_listing_and_no_writes_to_observed_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(MAX_DIRECTORY_ENTRIES + 7):
                (root / f"item-{index:03d}.txt").write_text("x", encoding="utf-8")
            before = {p.name: p.stat().st_mtime_ns for p in root.iterdir()}
            result = execute_read_request(ObservationRequest("list_directory", str(root)))
            after = {p.name: p.stat().st_mtime_ns for p in root.iterdir()}
            self.assertTrue(result["ok"])
            self.assertLessEqual(result["data"]["count"], MAX_DIRECTORY_ENTRIES)
            self.assertTrue(result["data"]["truncated"])
            self.assertEqual(before, after)

    def test_utf8_read_is_capped_and_sensitive_or_binary_files_are_denied(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text_path = root / "large.txt"
            text_path.write_text("я" * MAX_FILE_BYTES, encoding="utf-8")
            result = execute_read_request(ObservationRequest("read_file", str(text_path)))
            self.assertTrue(result["ok"])
            self.assertLessEqual(result["data"]["bytes"], MAX_FILE_BYTES)
            self.assertTrue(result["data"]["truncated"])

            env_path = root / ".env"
            env_path.write_text("TOKEN=secret", encoding="utf-8")
            denied = execute_read_request(ObservationRequest("read_file", str(env_path)))
            self.assertTrue(denied["denied"])
            self.assertNotIn("data", denied)

            binary = root / "blob.bin"
            binary.write_bytes(b"abc\x00def")
            denied_binary = execute_read_request(ObservationRequest("read_file", str(binary)))
            self.assertTrue(denied_binary["denied"])
            self.assertEqual(denied_binary["error"]["code"], "binary_file")


if __name__ == "__main__":
    unittest.main()
