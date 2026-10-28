from __future__ import annotations

from dataclasses import dataclass
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from personal_agent.attachments import AttachmentPipeline
from personal_agent.models import IncomingMessage
from personal_agent.animated_stickers import RenderedSticker, StickerRenderError


PRIMARY_ID = 101
SECONDARY_ID = 202


@dataclass
class FakeConfig:
    data_dir: Path
    max_attachment_bytes: int = 20 * 1024 * 1024


class FakeStore:
    def __init__(self) -> None:
        self.attachments: list[dict] = []
        self.events: list[object] = []
        self.messages: list[tuple[int, str, str]] = []
        self.state: dict[str, object] = {}

    def record_attachment(self, metadata: dict) -> None:
        self.attachments.append(dict(metadata))

    def record_event(self, event) -> None:
        self.events.append(event)

    def record_message(self, sender_id: int, role: str, text: str) -> None:
        self.messages.append((sender_id, role, text))

    def set_state(self, key: str, value) -> None:
        self.state[key] = value

    def get_state(self, key: str, default=None):
        return self.state.get(key, default)


class FakeTelegram:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.downloads: list[tuple[str, Path, int]] = []

    async def download(self, file_id: str, destination: Path, max_bytes: int) -> int:
        self.downloads.append((file_id, destination, max_bytes))
        payload = self.payloads[file_id]
        if len(payload) > max_bytes:
            raise AssertionError("fixture exceeds requested download bound")
        destination.write_bytes(payload)
        return len(payload)


class FakeTranscriber:
    def __init__(self, transcript: str = "offline transcript") -> None:
        self.transcript = transcript
        self.paths: list[Path] = []

    async def transcribe(self, path: Path) -> str:
        self.paths.append(path)
        return self.transcript


def png(width: int = 2, height: int = 3) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"


def webp(width: int = 512, height: int = 512) -> bytes:
    payload = bytes([0, 0, 0, 0]) + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    chunk = b"VP8X" + len(payload).to_bytes(4, "little") + payload
    return b"RIFF" + (4 + len(chunk)).to_bytes(4, "little") + b"WEBP" + chunk


def message(sender_id: int, attachment: dict, *, update_id: int = 1) -> IncomingMessage:
    return IncomingMessage(
        update_id=update_id,
        sender_id=sender_id,
        chat_id=sender_id,
        text="",
        message_id=update_id,
        attachment=attachment,
        is_text=False,
    )


class AttachmentPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = FakeStore()
        self.transcriber = FakeTranscriber()

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    def pipeline(self, payloads: dict[str, bytes]) -> tuple[AttachmentPipeline, FakeTelegram]:
        telegram = FakeTelegram(payloads)
        return AttachmentPipeline(FakeConfig(self.root), self.store, telegram, self.transcriber), telegram

    async def test_image_document_is_detected_from_bytes_and_cached_per_sender(self) -> None:
        pipeline, telegram = self.pipeline({"img": png()})
        result = await pipeline.prepare(message(PRIMARY_ID, {
            "kind": "document",
            "file_id": "img",
            "file_name": "looks-like-text.txt",
            "mime_type": "text/plain",
            "file_size": len(png()),
        }))

        self.assertEqual(len(result.images), 1)
        self.assertTrue(result.images[0].is_file())
        self.assertEqual(result.observations[0]["kind"], "attachment_image")
        self.assertEqual(result.observations[0]["image_format"], "png")
        self.assertEqual(result.observations[0]["sender_id"], PRIMARY_ID)
        self.assertTrue(result.observations[0]["has_image"])
        self.assertEqual(self.store.get_state(f"last_attachment_image:{PRIMARY_ID}"), result.observations[0])
        self.assertIsNone(self.store.get_state(f"last_attachment_image:{SECONDARY_ID}"))
        self.assertEqual(telegram.downloads[0][2], 20 * 1024 * 1024)
        self.assertFalse(any(event.type == "ATTACHMENT_TEXT" for event in self.store.events))

    async def test_static_webp_sticker_goes_to_vision_with_ten_mib_bound(self) -> None:
        payload = webp()
        pipeline, telegram = self.pipeline({"sticker": payload})
        result = await pipeline.prepare(message(PRIMARY_ID, {
            "kind": "sticker",
            "file_id": "sticker",
            "file_size": len(payload),
            "sticker_format": "webp",
            "is_animated": False,
            "is_video": False,
            "width": 512,
            "height": 512,
        }))

        self.assertEqual(len(result.images), 1)
        self.assertEqual(result.observations[0]["image_format"], "webp")
        self.assertEqual(telegram.downloads[0][2], 10 * 1024 * 1024)
        self.assertEqual(self.store.attachments[0]["sticker"]["format"], "webp")

    async def test_failed_animated_sticker_reports_error_and_invalidates_old_image(self) -> None:
        pipeline, _ = self.pipeline({"sticker": b"\x1f\x8bTGS-fixture"})
        self.store.set_state(f"last_attachment_image:{PRIMARY_ID}", {"kind": "attachment_image", "has_image": True})
        with patch('personal_agent.attachments.render_sticker', side_effect=StickerRenderError('invalid_tgs')):
            result = await pipeline.prepare(message(PRIMARY_ID, {
            "kind": "sticker",
            "file_id": "sticker",
            "file_size": 13,
            "sticker_format": "tgs",
            "is_animated": True,
            "is_video": False,
            }))

        self.assertEqual(result.images, [])
        self.assertIn("TGS", result.notice)
        self.assertIn("Не удалось", result.notice)
        self.assertEqual(result.observations[0]["status"], "invalid_tgs")
        self.assertFalse(result.observations[0]["has_image"])
        self.assertFalse(self.store.get_state(f"last_attachment_image:{PRIMARY_ID}")["has_image"])

    async def test_animation_frames_and_timestamps_are_cached_as_a_sequence(self):
        pipeline, _ = self.pipeline({"sticker": b"animation"})
        directory = self.root / 'memory' / 'attachments' / 'fixture-frames'
        directory.mkdir(parents=True)
        frames = [directory / f'{index}.png' for index in range(4)]
        for frame in frames:
            frame.write_bytes(png())
        rendered = RenderedSticker(frames, [0., .5, 1., 1.5], 2.)
        for is_video, format in ((False, 'tgs'), (True, 'webm')):
            with patch('personal_agent.attachments.render_sticker', new=AsyncMock(return_value=rendered)) as render:
                result = await pipeline.prepare(message(PRIMARY_ID, {
                    'kind': 'sticker', 'file_id': 'sticker', 'is_animated': not is_video,
                    'is_video': is_video, 'sticker_format': format}))
            self.assertEqual(result.images, frames)
            self.assertEqual([o['frame_time'] for o in result.observations], rendered.timestamps)
            self.assertEqual(render.call_args.kwargs['format'], format)
            cache = self.store.get_state(f'last_attachment_image:{PRIMARY_ID}')
            self.assertEqual(cache['paths'], [str(p) for p in frames])
            self.assertEqual(cache['timestamps'], rendered.timestamps)

    async def test_fake_png_filename_does_not_enter_vision(self) -> None:
        pipeline, _ = self.pipeline({"fake": b"not an image"})
        result = await pipeline.prepare(message(PRIMARY_ID, {
            "kind": "document",
            "file_id": "fake",
            "file_name": "photo.png",
            "mime_type": "image/png",
            "file_size": 12,
        }))

        self.assertEqual(result.images, [])
        self.assertEqual(result.observations, [])
        self.assertIn("сохранено локально", result.notice)

    async def test_oversized_dimensions_are_saved_but_not_sent_to_vision(self) -> None:
        payload = png(20_000, 20_000)
        pipeline, _ = self.pipeline({"huge": payload})
        result = await pipeline.prepare(message(PRIMARY_ID, {
            "kind": "photo",
            "file_id": "huge",
            "file_size": len(payload),
        }))

        self.assertEqual(result.images, [])
        self.assertIn("слишком велики", result.notice)
        self.assertIsNone(self.store.get_state(f"last_attachment_image:{PRIMARY_ID}"))

    async def test_voice_and_text_document_keep_existing_semantics(self) -> None:
        pipeline, telegram = self.pipeline({"voice": b"voice", "text": b"hello\nworld"})
        voice = await pipeline.prepare(message(PRIMARY_ID, {
            "kind": "voice",
            "file_id": "voice",
            "file_size": 5,
            "duration": 10,
        }))
        text = await pipeline.prepare(message(PRIMARY_ID, {
            "kind": "document",
            "file_id": "text",
            "file_name": "notes.md",
            "mime_type": "text/markdown",
            "file_size": 11,
        }, update_id=2))

        self.assertEqual(voice.transcript, "offline transcript")
        self.assertIn("расшифровано локально", voice.notice)
        self.assertEqual(self.store.messages, [(PRIMARY_ID, "user", "offline transcript")])
        text_event = next(event for event in self.store.events if event.type == "ATTACHMENT_TEXT")
        self.assertEqual(text_event.data["sender_id"], PRIMARY_ID)
        self.assertIn("локально", text.notice)
        self.assertEqual([call[2] for call in telegram.downloads], [20 * 1024 * 1024, 20 * 1024 * 1024])

    async def test_oversized_attachment_is_not_downloaded(self) -> None:
        pipeline, telegram = self.pipeline({"big": b"x"})
        result = await pipeline.prepare(message(PRIMARY_ID, {
            "kind": "video",
            "file_id": "big",
            "file_size": 25 * 1024 * 1024,
        }))
        self.assertIn("превышает", result.notice)
        self.assertEqual(len(telegram.downloads), 0)


if __name__ == "__main__":
    unittest.main()
