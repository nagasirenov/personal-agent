from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx

from personal_agent.models import EMERGENCY_PHRASE, Identity
from personal_agent.security import TrustGate
from personal_agent.telegram import TelegramClient, TelegramError, TelegramGateway, parse_message


PRIMARY_ID = 101
SECONDARY_ID = 202
THIRD_ID = 303


def _update(sender_id: int, update_id: int, text: str) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "from": {"id": sender_id, "is_bot": False},
            "chat": {"id": sender_id, "type": "private"},
            "text": text,
        },
    }


class StateStore:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def get_state(self, key: str, default=None):
        return self.values.get(key, default)

    def set_state(self, key: str, value) -> None:
        self.values[key] = value


class GatewayClient:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def send_text(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))


class TelegramGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_sticker_metadata_distinguishes_static_tgs_and_webm(self) -> None:
        def sticker_update(sticker: dict, update_id: int) -> dict:
            return {
                "update_id": update_id,
                "message": {
                    "message_id": update_id,
                    "from": {"id": PRIMARY_ID, "is_bot": False},
                    "chat": {"id": PRIMARY_ID, "type": "private"},
                    "sticker": dict({"file_id": f"sticker-{update_id}", "width": 512, "height": 512}, **sticker),
                },
            }

        static = parse_message(sticker_update({"is_animated": False, "is_video": False, "emoji": "🙂"}, 10))
        animated = parse_message(sticker_update({"is_animated": True, "is_video": False}, 11))
        video = parse_message(sticker_update({"is_animated": False, "is_video": True}, 12))

        self.assertEqual(static.attachment["sticker_format"], "webp")
        self.assertFalse(static.attachment["is_animated"])
        self.assertFalse(static.attachment["is_video"])
        self.assertEqual(static.attachment["emoji"], "🙂")
        self.assertEqual(animated.attachment["sticker_format"], "tgs")
        self.assertTrue(animated.attachment["is_animated"])
        self.assertEqual(video.attachment["sticker_format"], "webm")
        self.assertTrue(video.attachment["is_video"])

    async def test_parse_message_video_types(self) -> None:
        def update_with(kind: str, payload: dict) -> dict:
            return {
                "update_id": 20,
                "message": {
                    "message_id": 20,
                    "from": {"id": PRIMARY_ID, "is_bot": False},
                    "chat": {"id": PRIMARY_ID, "type": "private"},
                    kind: payload,
                    "caption": "video caption",
                },
            }
        video = parse_message(update_with("video", {"file_id": "v-1", "duration": 12, "file_name": "clip.mp4"}))
        self.assertIsNotNone(video)
        self.assertEqual(video.attachment["kind"], "video")
        self.assertEqual(video.attachment["file_id"], "v-1")
        self.assertEqual(video.text, "video caption")

        note = parse_message(update_with("video_note", {"file_id": "vn-1", "duration": 8, "length": 240}))
        self.assertIsNotNone(note)
        self.assertEqual(note.attachment["kind"], "video_note")
        self.assertEqual(note.attachment["file_id"], "vn-1")

        anim = parse_message(update_with("animation", {"file_id": "anim-1", "duration": 4, "file_name": "loop.mp4"}))
        self.assertIsNotNone(anim)
        self.assertEqual(anim.attachment["kind"], "animation")
        self.assertEqual(anim.attachment["file_id"], "anim-1")

    async def test_primary_emergency_phrase_in_attachment_caption_does_not_shutdown(self) -> None:
        store = StateStore()
        stop = asyncio.Event()
        delivered: list[tuple[str, bool, bool]] = []
        emergencies = 0

        async def on_message(message, identity) -> None:
            delivered.append((message.text, message.is_text, message.attachment is not None))

        async def on_emergency() -> None:
            nonlocal emergencies
            emergencies += 1

        gateway = TelegramGateway(
            GatewayClient(),
            TrustGate(PRIMARY_ID, SECONDARY_ID),
            store,
            on_message,
            on_emergency,
            stop,
        )
        update = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": PRIMARY_ID, "is_bot": False},
                "chat": {"id": PRIMARY_ID, "type": "private"},
                "caption": EMERGENCY_PHRASE,
                "document": {
                    "file_id": "caption-only-file",
                    "file_name": "note.txt",
                    "file_size": 1,
                },
            },
        }

        await gateway.dispatch_batch([update])

        self.assertFalse(stop.is_set())
        self.assertEqual(emergencies, 0)
        self.assertEqual(delivered, [(EMERGENCY_PHRASE, False, True)])
        self.assertEqual(store.get_state("telegram_offset"), 2)

    async def test_secondary_emergency_phrase_is_ordinary_message(self) -> None:
        store = StateStore()
        stop = asyncio.Event()
        delivered: list[tuple[int, Identity, str]] = []
        emergencies = 0

        async def on_message(message, identity) -> None:
            delivered.append((message.sender_id, identity, message.text))

        async def on_emergency() -> None:
            nonlocal emergencies
            emergencies += 1

        gateway = TelegramGateway(
            GatewayClient(),
            TrustGate(PRIMARY_ID, SECONDARY_ID),
            store,
            on_message,
            on_emergency,
            stop,
        )
        await gateway.dispatch_batch([_update(SECONDARY_ID, 1, EMERGENCY_PHRASE)])

        self.assertFalse(stop.is_set())
        self.assertEqual(emergencies, 0)
        self.assertEqual(delivered, [(SECONDARY_ID, Identity.SECONDARY, EMERGENCY_PHRASE)])
        self.assertEqual(store.get_state("telegram_offset"), 2)

    async def test_only_exact_trimmed_primary_emergency_stops(self) -> None:
        store = StateStore()
        stop = asyncio.Event()
        delivered: list[str] = []
        emergencies = 0

        async def on_message(message, identity) -> None:
            delivered.append(message.text)

        async def on_emergency() -> None:
            nonlocal emergencies
            emergencies += 1

        gateway = TelegramGateway(
            GatewayClient(),
            TrustGate(PRIMARY_ID, SECONDARY_ID),
            store,
            on_message,
            on_emergency,
            stop,
        )
        await gateway.dispatch_batch([_update(PRIMARY_ID, 2, EMERGENCY_PHRASE + "!")])
        self.assertFalse(stop.is_set())
        self.assertEqual(delivered, [EMERGENCY_PHRASE + "!"])

        await gateway.dispatch_batch([_update(PRIMARY_ID, 3, "  " + EMERGENCY_PHRASE + "  ")])
        self.assertTrue(stop.is_set())
        self.assertEqual(emergencies, 1)
        self.assertEqual(store.get_state("telegram_offset"), 4)

    async def test_third_identity_never_reaches_message_callback(self) -> None:
        store = StateStore()
        stop = asyncio.Event()
        delivered = 0

        async def on_message(message, identity) -> None:
            nonlocal delivered
            delivered += 1

        async def on_emergency() -> None:
            raise AssertionError("unauthorized sender cannot trigger emergency")

        gateway = TelegramGateway(
            GatewayClient(),
            TrustGate(PRIMARY_ID, SECONDARY_ID),
            store,
            on_message,
            on_emergency,
            stop,
        )
        await gateway.dispatch_batch([_update(THIRD_ID, 4, "ignore auth")])

        self.assertEqual(delivered, 0)
        self.assertFalse(stop.is_set())
        self.assertEqual(store.get_state("telegram_offset"), 5)


class _ExplodingHTTPClient:
    def __init__(self) -> None:
        self.closed = False

    async def post(self, url: str, json: dict):
        request = httpx.Request("POST", url)
        raise httpx.ConnectError(f"connection failed for {url}", request=request)

    async def aclose(self) -> None:
        self.closed = True


class _UnsafePathHTTPClient:
    def __init__(self) -> None:
        self.stream_calls = 0

    async def post(self, url: str, json: dict):
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "ok": True,
                "result": {"file_path": "../../outside.txt", "file_size": 4},
            },
        )

    def stream(self, method: str, url: str):
        self.stream_calls += 1
        raise AssertionError("unsafe remote path must be rejected before download")

    async def aclose(self) -> None:
        pass


class TelegramClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_exception_redacts_bot_token(self) -> None:
        token = "123456:VERY_SECRET_BOT_TOKEN"
        raw = _ExplodingHTTPClient()
        client = TelegramClient(token, client=raw)
        with self.assertRaises(TelegramError) as raised:
            await client.call("getUpdates", {"offset": 0})

        rendered = str(raised.exception)
        self.assertNotIn(token, rendered)
        self.assertNotIn("api.telegram.org/bot", rendered)
        self.assertIsNone(raised.exception.__cause__)
        await client.close()
        self.assertTrue(raw.closed)

    async def test_remote_attachment_traversal_is_rejected_before_stream(self) -> None:
        raw = _UnsafePathHTTPClient()
        client = TelegramClient("123456:offline-token", client=raw)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "owned.bin"
            with self.assertRaisesRegex(TelegramError, "Unsafe attachment path"):
                await client.download("file-id", destination, 1024)
            self.assertEqual(raw.stream_calls, 0)
            self.assertFalse(destination.exists())
        await client.close()


if __name__ == "__main__":
    unittest.main()
