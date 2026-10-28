"""Offline tests for the send_file tool: method selection, path denials, per-turn limits, the Telegram
upload cap (video re-encode, Google Drive link through a fake rclone, split into parts) and receipts."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from personal_agent.config import Config
from personal_agent.file_delivery import choose_method, find_rclone
from personal_agent.memory import MemoryStore
from personal_agent.telegram import CLOUD_UPLOAD_CAP, LOCAL_UPLOAD_CAP, TelegramClient, TelegramError
from personal_agent.tools import DEFAULT_REGISTRY, ToolArgumentError, TurnTools
from personal_agent.video import VideoAnalyzer

PRIMARY_ID, SECONDARY_ID = 101, 202


class FakeTelegram:
    def __init__(self, cap: int = 1000, fail: bool = False) -> None:
        self.upload_cap_bytes, self.fail = cap, fail
        self.sent: list[tuple[str, int, str, bytes, str]] = []
        self.texts: list[tuple[int, str]] = []

    async def _record(self, method: str, chat_id: int, path: Path, caption: str) -> str:
        if self.fail:
            raise TelegramError("Telegram upload failed")
        self.sent.append((method, chat_id, path.name, path.read_bytes(), caption))
        return method

    async def send_photo(self, chat_id, path, caption=""):
        return await self._record("sendPhoto", chat_id, path, caption)

    async def send_video(self, chat_id, path, caption=""):
        return await self._record("sendVideo", chat_id, path, caption)

    async def send_audio(self, chat_id, path, caption=""):
        return await self._record("sendAudio", chat_id, path, caption)

    async def send_document(self, chat_id, path, caption=""):
        return await self._record("sendDocument", chat_id, path, caption)

    async def send_text(self, chat_id, text):
        self.texts.append((chat_id, text))


class SendFileTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._state, self._files = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
        self.data, self.files = Path(self._state.name), Path(self._files.name)
        self.store = MemoryStore(self.data)
        self.telegram = FakeTelegram()
        self.config = Config(self.data, PRIMARY_ID, SECONDARY_ID, backend="mock", timezone_name="UTC",
                             ffmpeg_executable="fake-ffmpeg")

    async def asyncTearDown(self) -> None:
        self.store.close()
        self._state.cleanup()
        self._files.cleanup()

    def turn(self, sender: int = PRIMARY_ID, config: Config | None = None) -> TurnTools:
        return TurnTools(config=config or self.config, store=self.store, context=None, knowledge=None,
                         action_broker=type("B", (), {"available_for": lambda self, s: False})(), profile_store=None, telegram=self.telegram, sender_id=sender,
                         chat_id=sender, request_id="req-1", video_analyzer=VideoAnalyzer(self.data, ffmpeg="x"))

    def make(self, name: str, content: bytes = b"data") -> Path:
        path = self.files / name
        path.write_bytes(content)
        return path

    def test_method_selection_and_cap_constants(self) -> None:
        picks = {name: choose_method(Path(name)) for name in ("a.JPG", "a.png", "a.mp4", "a.mkv", "a.mp3", "a.m4a",
                                                                "a.pdf", "a.zip", "a")}
        self.assertEqual(picks, {"a.JPG": "sendPhoto", "a.png": "sendPhoto", "a.mp4": "sendVideo", "a.mkv": "sendVideo",
                                 "a.mp3": "sendAudio", "a.m4a": "sendAudio", "a.pdf": "sendDocument",
                                 "a.zip": "sendDocument", "a": "sendDocument"})
        self.assertEqual(choose_method(Path("a.png"), "document"), "sendDocument")
        self.assertEqual(choose_method(Path("a.bin"), "video"), "sendVideo")
        self.assertEqual(TelegramClient("1:t").upload_cap_bytes, CLOUD_UPLOAD_CAP)
        self.assertEqual(TelegramClient("1:t", api_base="http://127.0.0.1:8081").upload_cap_bytes, LOCAL_UPLOAD_CAP)

    async def test_auto_selection_receipts_and_registry(self) -> None:
        self.telegram.upload_cap_bytes = CLOUD_UPLOAD_CAP
        turn = self.turn()
        receipts = [await turn.send_file({"path": str(self.make(name)), "caption": "hi"})
                    for name in ("p.jpg", "v.mp4", "s.mp3", "d.txt")]
        self.assertEqual([r["method"] for r in receipts], ["sendPhoto", "sendVideo", "sendAudio", "sendDocument"])
        self.assertTrue(all(r["ok"] and r["parts"] == 1 and r["reencoded"] is False and r["size"] == 4 for r in receipts))
        self.assertEqual({entry[1] for entry in self.telegram.sent}, {PRIMARY_ID})
        self.assertEqual(self.telegram.sent[0][4], "hi")
        forced = await turn.send_file({"path": str(self.make("big.png")), "as": "document"})
        self.assertEqual(forced["method"], "sendDocument")
        self.assertIn("send_file", [spec.name for spec in DEFAULT_REGISTRY.specs if spec.availability(turn)])

    async def test_denials_limits_and_current_chat_only(self) -> None:
        turn = self.turn(PRIMARY_ID)
        env = self.files / ".env"
        env.write_text("X=1")
        secret = self.data / "bot_state.db"
        secret.write_bytes(b"x")
        for bad in (str(env), str(secret), "relative.txt", r"\\server\share\a.txt", str(self.files / "missing.txt"),
                    str(self.files)):
            self.assertFalse((await turn.send_file({"path": bad}))["ok"], bad)
        self.assertEqual((await turn.send_file({"path": str(env)}))["error"]["code"], "credential_path_denied")
        self.assertEqual((await turn.send_file({"path": str(secret)}))["error"]["code"], "credential_path_denied")
        with self.assertRaises(ToolArgumentError):  # there is no way to name another chat
            await turn.send_file({"path": str(self.make("a.txt")), "chat_id": PRIMARY_ID})
        with self.assertRaises(ToolArgumentError):
            await turn.send_file({})
        self.assertEqual(self.telegram.sent, [])
        self.assertTrue((await turn.send_file({"path": str(self.make("a.txt"))}))["ok"])
        self.assertEqual((await turn.send_file({"path": str(self.files / "a.txt")}))["error"]["code"], "duplicate_in_turn")
        self.assertEqual([entry[1] for entry in self.telegram.sent], [PRIMARY_ID])
        for index in range(9):
            self.assertTrue((await turn.send_file({"path": str(self.make(f"n{index}.txt"))}))["ok"])
        self.assertEqual((await turn.send_file({"path": str(self.make("eleven.txt"))}))["error"]["code"],
                         "too_many_files_in_turn")

    async def test_media_id_sends_cached_video_and_screenshot(self) -> None:
        analyzer = VideoAnalyzer(self.data, ffmpeg="x")
        directory = analyzer.entry_dir(PRIMARY_ID, "v-cached")
        directory.mkdir(parents=True)
        (directory / "video.mp4").write_bytes(b"clip")
        (directory / "meta.json").write_text(
            json.dumps({"sender_id": PRIMARY_ID, "video_id": "v-cached", "file": "video.mp4"}), encoding="utf-8")
        shot = self.data / "memory" / "screenshots" / "s.png"
        shot.parent.mkdir(parents=True)
        shot.write_bytes(b"png")
        self.store.set_state(f"last_screenshot:{PRIMARY_ID}", {"sender_id": PRIMARY_ID, "path": str(shot), "time": "t"})
        turn = self.turn()
        video = await turn.send_file({"media_id": "v-cached"})
        picture = await turn.send_file({"media_id": "last_screenshot"})
        self.assertEqual((video["method"], picture["method"]), ("sendVideo", "sendPhoto"))
        self.assertFalse((await self.turn().send_file({"media_id": "nope"}))["ok"])
        self.assertFalse((await self.turn(SECONDARY_ID).send_file({"media_id": "v-cached"}))["ok"])

    async def test_big_file_is_split_when_drive_is_not_configured(self) -> None:
        content = bytes(range(256)) * 10  # 2560 bytes, cap 1000 -> 980-byte parts
        receipt = await self.turn().send_file({"path": str(self.make("archive.zip", content))})
        self.assertTrue(receipt["ok"])
        self.assertEqual((receipt["parts"], receipt["parts_sent"], receipt["drive_unavailable"]), (3, 3, True))
        parts = self.telegram.sent
        self.assertEqual([p[2] for p in parts], ["archive.zip.part01", "archive.zip.part02", "archive.zip.part03"])
        self.assertTrue(all(p[0] == "sendDocument" and len(p[3]) <= 980 for p in parts))
        self.assertEqual(b"".join(p[3] for p in parts), content)
        self.assertIn("copy /b archive.zip.part01+archive.zip.part02+archive.zip.part03 archive.zip", parts[0][4])

    async def test_video_over_cap_is_reencoded_not_split(self) -> None:
        calls: list[list[str]] = []

        async def fake_process(argv, *, timeout):
            calls.append(list(argv))
            if "-y" in argv:
                Path(argv[-1]).write_bytes(b"small")
                return 0, b"", b""
            return 1, b"", (b"Duration: 00:01:40.00, start: 0\n  Stream #0:0: Video: h264, yuv420p, 1920x1080\n"
                            b"  Stream #0:1: Audio: aac\n")

        original = self.make("movie.mp4", b"x" * 5000)
        with patch("personal_agent.file_delivery.run_process", new=fake_process):
            receipt = await self.turn().send_file({"path": str(original)})
        self.assertTrue(receipt["ok"])
        self.assertTrue(receipt["reencoded"])
        self.assertEqual((receipt["method"], receipt["parts"]), ("sendVideo", 1))
        self.assertEqual(self.telegram.sent[0][3], b"small")
        encode = calls[-1]
        self.assertIn("-b:v", encode)
        self.assertIn("scale=-2:240", encode[encode.index("-vf") + 1])  # tiny budget forces a downscale

    async def test_big_file_goes_to_drive_through_rclone(self) -> None:
        rclone = self.make("rclone.exe")
        config = replace(self.config, drive_remote="gdrive:PersonalAgent", rclone_executable=str(rclone))
        calls: list[list[str]] = []

        async def fake_process(argv, *, timeout):
            calls.append(list(argv))
            if argv[1] == "link":
                return 0, b"https://drive.google.com/file/d/abc/view?usp=sharing\n", b""
            return 0, b"", b"ya29.SECRET-TOKEN"

        with patch("personal_agent.file_delivery.run_process", new=fake_process):
            receipt = await self.turn(config=config).send_file({"path": str(self.make("dump.bin", b"y" * 3000))})
        self.assertEqual((receipt["ok"], receipt["method"]), (True, "drive_link"))
        self.assertNotIn("drive_unavailable", receipt)
        self.assertEqual(calls[0][1], "copyto")
        self.assertRegex(calls[0][3], r"^gdrive:PersonalAgent/\d{4}-\d\d-\d\d/dump\.bin$")
        self.assertEqual(calls[1][1:], ["link", calls[0][3]])
        self.assertEqual(self.telegram.sent, [])
        self.assertIn("https://drive.google.com/file/d/abc/view", self.telegram.texts[0][1])
        self.assertIn("dump.bin", self.telegram.texts[0][1])

        async def failing(argv, *, timeout):
            return 1, b"", b"token ya29.SECRET-TOKEN rejected"

        with patch("personal_agent.file_delivery.run_process", new=failing):
            fallback = await self.turn(config=config).send_file({"path": str(self.make("dump2.bin", b"y" * 3000))})
        self.assertTrue(fallback["ok"] and fallback["drive_unavailable"])
        self.assertEqual(fallback["drive_error"], "drive_upload_failed")
        self.assertNotIn("SECRET", json.dumps(fallback))

    async def test_upload_failure_is_not_reported_as_sent(self) -> None:
        self.telegram.fail = True
        receipt = await self.turn().send_file({"path": str(self.make("a.txt"))})
        self.assertFalse(receipt["ok"])
        self.assertEqual(receipt["error"]["code"], "telegram_upload_failed")

    def test_find_rclone_prefers_configured_file(self) -> None:
        exe = self.make("rclone.exe")
        self.assertEqual(find_rclone(str(exe)), str(exe))
        self.assertEqual(find_rclone(str(self.files / "missing.exe")), "")


if __name__ == "__main__":
    unittest.main()
