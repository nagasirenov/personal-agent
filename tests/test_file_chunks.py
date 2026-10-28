"""Chunked reading of large text files and bounded recursive file search."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from personal_agent.observation_requests import (
    FIND_MAX_DEPTH, MAX_CHUNK_BYTES, MAX_FILE_BYTES, ObservationRequest, execute_read_request, read_text_chunk,
)
from personal_agent.tools import DEFAULT_REGISTRY
from tests.test_service_observation_integration import PRIMARY_ID, ObservationHarness, _message


def read_all(path: Path, max_bytes: int = MAX_FILE_BYTES) -> tuple[str, int, list[dict]]:
    text, offset, chunks = "", 0, []
    while True:
        chunk = read_text_chunk(path, offset, max_bytes)
        chunks.append(chunk)
        text += chunk["text"]
        if chunk["eof"]:
            return text, chunk["total_bytes"], chunks
        assert chunk["next_offset"] > offset
        offset = chunk["next_offset"]


class ChunkReadingTests(unittest.TestCase):
    def test_large_utf8_file_is_reassembled_exactly_without_splitting_characters(self):
        body = "".join(f"строка {i}: Привет, мир — ёжик ✓\n" for i in range(6000))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "big.txt"
            path.write_bytes(body.encode("utf-8"))
            text, total, chunks = read_all(path)
            self.assertEqual(text, body)
            self.assertEqual(total, path.stat().st_size)
            self.assertGreater(len(chunks), 3)
            self.assertTrue(all(chunk["bytes"] <= MAX_FILE_BYTES for chunk in chunks))
            self.assertEqual(chunks[0]["encoding"], "utf-8")
            self.assertFalse(chunks[0]["eof"])
            self.assertTrue(chunks[0]["truncated"])
            self.assertEqual(chunks[1]["offset"], chunks[0]["next_offset"])

    def test_chunk_size_is_capped_and_offset_past_end_is_a_clean_eof(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.txt"
            path.write_text("x" * (MAX_CHUNK_BYTES * 2 + 5), encoding="utf-8")
            chunk = read_text_chunk(path, 0, 10**9)
            self.assertEqual(chunk["bytes"], MAX_CHUNK_BYTES)
            tail = read_text_chunk(path, 10**9)
            self.assertEqual((tail["text"], tail["eof"], tail["next_offset"]),
                             ("", True, path.stat().st_size))

    def test_misaligned_offset_inside_a_character_moves_to_a_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ru.txt"
            path.write_text("я" * 3000, encoding="utf-8")  # every character is two bytes
            chunk = read_text_chunk(path, 1001, 1024)
            self.assertEqual(chunk["offset"], 1002)
            self.assertTrue(set(chunk["text"]) <= {"я"})

    def test_bom_utf8_utf16_and_cp1251_are_decoded_and_reported(self):
        body = "Привет, это тест кодировок.\n" * 2000
        cases = {
            "utf-8-bom": (codecs_bom := b"\xef\xbb\xbf") + body.encode("utf-8"),
            "utf-16le": b"\xff\xfe" + body.encode("utf-16-le"),
            "utf-16be": b"\xfe\xff" + body.encode("utf-16-be"),
            "cp1251": body.encode("cp1251"),
        }
        with tempfile.TemporaryDirectory() as temp:
            for label, raw in cases.items():
                with self.subTest(encoding=label):
                    path = Path(temp) / f"{label}.txt"
                    path.write_bytes(raw)
                    text, _, chunks = read_all(path, 4096)
                    self.assertEqual(text, body)
                    self.assertEqual({chunk["encoding"] for chunk in chunks}, {label})

    def test_denials_survive_chunking(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "blob.bin").write_bytes(b"abc\x00def" * 100)
            (root / "key.txt").write_text("a" * 70000 + "-----BEGIN RSA PRIVATE KEY-----\nxxx\n", encoding="utf-8")
            (root / ".env").write_text("A=1", encoding="utf-8")
            denied = execute_read_request(ObservationRequest("read_file", str(root / "blob.bin")))
            self.assertEqual(denied["error"]["code"], "binary_file")
            late_key = execute_read_request(ObservationRequest("read_file", str(root / "key.txt"), offset=MAX_CHUNK_BYTES))
            self.assertEqual(late_key["error"]["code"], "private_key_denied")
            self.assertNotIn("data", late_key)
            self.assertTrue(execute_read_request(ObservationRequest("read_file", str(root / ".env")))["denied"])

    def test_private_key_marker_split_by_a_chunk_boundary_is_still_denied(self):
        marker = b"-----BEGIN OPENSSH PRIVATE KEY-----"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "notes.txt"
            path.write_bytes(b"a" * (1024 - 10) + marker + b"\nzzz\n" + b"b" * 3000)
            second = execute_read_request(ObservationRequest("read_file", str(path), offset=1024, max_bytes=1024))
            self.assertFalse(second["ok"])
            self.assertEqual(second["error"]["code"], "private_key_denied")


class FindFilesTests(unittest.TestCase):
    def build_tree(self, root: Path) -> None:
        (root / "docs" / "deep" / "deeper").mkdir(parents=True)
        (root / ".ssh").mkdir()
        (root / "node_modules" / "pkg").mkdir(parents=True)
        for name in ("docs/Report-2026.pdf", "docs/deep/report-notes.txt", "docs/deep/deeper/REPORT.md",
                     ".ssh/report_key.txt", "node_modules/pkg/report.js", "top.txt", "credentials.json"):
            (root / name).write_text("x", encoding="utf-8")

    def test_bare_word_is_a_case_insensitive_name_contains_search_shallow_first(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.build_tree(root)
            result = execute_read_request(ObservationRequest("find_files", str(root), pattern="report"))
            self.assertTrue(result["ok"])
            found = [Path(item["path"]).name for item in result["data"]["matches"]]
            self.assertEqual(found, ["Report-2026.pdf", "report-notes.txt", "REPORT.md"])
            self.assertFalse(result["data"]["truncated"])

    def test_glob_limits_and_sensitive_or_noisy_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.build_tree(root)
            txt = execute_read_request(ObservationRequest("find_files", str(root), pattern="*.txt"))
            self.assertEqual(sorted(Path(i["path"]).name for i in txt["data"]["matches"]),
                             ["report-notes.txt", "top.txt"])  # .ssh content skipped
            self.assertEqual(execute_read_request(
                ObservationRequest("find_files", str(root), pattern="credentials*"))["data"]["count"], 0)
            limited = execute_read_request(ObservationRequest("find_files", str(root), pattern="*", max_results=2))
            self.assertEqual(limited["data"]["count"], 2)
            self.assertEqual(limited["data"]["truncated_reason"], "result_limit")
            bad = execute_read_request(ObservationRequest("find_files", str(root), pattern="a/b"))
            self.assertEqual(bad["error"]["code"], "invalid_pattern")
            missing = execute_read_request(ObservationRequest("find_files", str(root / "nope"), pattern="x"))
            self.assertEqual(missing["error"]["code"], "not_a_directory")

    def test_depth_limit_and_entry_limit(self):
        import personal_agent.observation_requests as module
        with tempfile.TemporaryDirectory() as temp:
            deep = Path(temp)
            for index in range(FIND_MAX_DEPTH + 2):
                deep /= f"d{index}"
            deep.mkdir(parents=True)
            (deep / "needle.txt").write_text("x", encoding="utf-8")
            result = execute_read_request(ObservationRequest("find_files", temp, pattern="needle"))
            self.assertEqual(result["data"]["count"], 0)  # deeper than the depth limit
            old = module.FIND_MAX_ENTRIES
            module.FIND_MAX_ENTRIES = 3
            try:
                for index in range(10):
                    (Path(temp) / f"f{index}.txt").write_text("x", encoding="utf-8")
                limited = execute_read_request(ObservationRequest("find_files", temp, pattern="zzz"))
            finally:
                module.FIND_MAX_ENTRIES = old
            self.assertEqual(limited["data"]["truncated_reason"], "entry_limit")

    @unittest.skipUnless(os.name == "nt", "directory junctions are a Windows feature")
    def test_junctions_are_not_followed(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
            (Path(outside) / "secret-report.txt").write_text("x", encoding="utf-8")
            link = Path(temp) / "link"
            made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), outside], capture_output=True)
            if made.returncode != 0:
                self.skipTest("cannot create a junction here")
            result = execute_read_request(ObservationRequest("find_files", temp, pattern="report"))
            self.assertEqual(result["data"]["count"], 0)


class ChunkToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_can_page_through_a_large_file_and_find_it_first(self):
        harness = ObservationHarness()
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "журнал-проекта.log"
                body = "".join(f"запись {i} событие\n" for i in range(9000))
                path.write_bytes(body.encode("utf-8"))
                message = replace(_message("x"), sender_id=PRIMARY_ID, chat_id=PRIMARY_ID)
                turn = harness.supervisor._turn_tools(message, "rid-chunks", [])
                _, handler = DEFAULT_REGISTRY.for_turn(turn)

                found = json.loads((await handler("find_files", {"root": temp, "pattern": "журнал"}, "c0")).output)
                self.assertEqual([Path(m["path"]).name for m in found["data"]["matches"]], [path.name])

                text, offset, calls = "", 0, 0
                while True:
                    args = {"path": str(path)} | ({"offset": offset} if offset else {})
                    result = json.loads((await handler("read_file", args, f"c{calls}")).output)
                    self.assertTrue(result["ok"], result)
                    text += result["data"]["text"]
                    calls += 1
                    if result["data"]["eof"]:
                        break
                    offset = result["data"]["next_offset"]
                self.assertEqual(text, body)
                self.assertGreater(calls, 3)

                for bad in ({"path": str(path), "offset": -1}, {"path": str(path), "offset": "0"},
                            {"path": str(path), "offset": True}, {"path": str(path), "extra": 1}):
                    refused = json.loads((await handler("read_file", bad, "cx")).output)
                    self.assertEqual(refused["error"]["code"], "invalid_arguments")
                clamped = json.loads((await handler("read_file", {"path": str(path), "max_bytes": 10**9}, "cy")).output)
                self.assertLessEqual(clamped["data"]["bytes"], MAX_CHUNK_BYTES)
        finally:
            await harness.close()


if __name__ == "__main__":
    unittest.main()
