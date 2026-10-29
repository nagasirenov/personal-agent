"""Built-in image generation: gateway payload/extraction/replay, backend hand-off and delivery to the chat."""
from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from personal_agent.backends import BrainBackend, CodexSubscriptionBackend, MockBackend
from personal_agent.brain_gateway import BrainRequestGateway, GeneratedImage, ToolOutcome, _replayable
from personal_agent.config import Config
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage
from personal_agent.supervisor import TaskSupervisor
from personal_agent.telegram import TelegramError
from personal_agent.testing import ScriptedBrain, ScriptedTurn
from personal_agent.tools import TurnTools
from tests.test_gateway_tool_loop import TOOLS, _CLIENT, _HOST, call, message, reasoning, sse
from tests.test_group_chats import GROUP, MEMBER, OWNER, GroupHarness, group_update, mention, BOT_NAME

PNG = b"\x89PNG\r\n\x1a\n" + b"IMAGE-ONE" * 8
PNG2 = b"\x89PNG\r\n\x1a\n" + b"IMAGE-TWO" * 8
JPEG = b"\xff\xd8\xff\xe0" + b"JPEGDATA" * 8
BUILTIN = [{"type": "image_generation"}]
SECONDARY = 5002


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def image_item(data: bytes = PNG, item_id: str = "ig_1", prompt: str = "a blue square") -> dict:
    return {"type": "image_generation_call", "id": item_id, "status": "completed", "result": b64(data),
            "revised_prompt": prompt, "size": "1254x1254"}


class GatewayImageTests(unittest.IsolatedAsyncioTestCase):
    def gateway(self, **kwargs) -> BrainRequestGateway:
        kwargs.setdefault("upstream_url", "http://127.0.0.1:1/responses")
        kwargs.setdefault("require_auth", False)
        return BrainRequestGateway(instructions="PERSONA", prompt="PROMPT", model="gpt-6-luna",
                                   reasoning_effort="medium", **kwargs)

    def upstream(self, responses):
        self.bodies: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.bodies.append(body)
            result = responses[len(self.bodies) - 1]
            return httpx.Response(200, content=result, headers={"Content-Type": "text/event-stream"})

        return patch("personal_agent.brain_gateway.httpx.AsyncClient",
                     side_effect=lambda **kw: _CLIENT(transport=httpx.MockTransport(handler), **kw))

    async def post(self, gateway) -> httpx.Response:
        async with _CLIENT(trust_env=False, timeout=10) as caller:
            return await caller.post(gateway.base_url + "/responses", json=_HOST)

    async def test_builtin_tool_only_when_requested(self):
        final = sse([message("ok")])
        with self.upstream([final]):
            async with self.gateway() as gateway:
                await self.post(gateway)
        self.assertEqual(self.bodies[0]["tools"], [])
        self.assertNotIn("tool_choice", self.bodies[0])
        with self.upstream([final]):
            async with self.gateway(builtin_tools=BUILTIN) as gateway:
                response = await self.post(gateway)
                self.assertEqual(gateway.upstream_requests, 1)
        self.assertEqual(response.content, final)
        self.assertEqual(self.bodies[0]["tools"], BUILTIN)
        self.assertEqual(self.bodies[0]["include"], ["reasoning.encrypted_content"])
        with self.upstream([final]):
            async with self.gateway(tools=TOOLS, tool_handler=lambda *a: None, builtin_tools=BUILTIN) as gateway:
                await self.post(gateway)
        self.assertEqual(self.bodies[0]["tools"], [*TOOLS, *BUILTIN])

    def test_invalid_builtin_tools_are_rejected(self):
        for bad in ([{"type": "web_search"}], [{"type": "image_generation"}, {"type": "image_generation"}],
                    ["image_generation"], [{"name": "x"}]):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "gateway_invalid_builtin_tools"):
                self.gateway(builtin_tools=bad)

    async def test_image_taken_from_output_item_done_when_completed_output_is_empty(self):
        final = sse([image_item(), message("Готово")], with_completed_output=False)
        with self.upstream([final]):
            async with self.gateway(builtin_tools=BUILTIN) as gateway:
                await self.post(gateway)
                images = list(gateway.generated_images)
        self.assertEqual([(i.data, i.revised_prompt, i.item_id, i.suffix) for i in images],
                         [(PNG, "a blue square", "ig_1", ".png")])

    async def test_image_taken_from_completed_output_and_jpeg_is_accepted(self):
        with self.upstream([sse([image_item(JPEG, "ig_j")], with_completed_output=True)]):
            async with self.gateway(builtin_tools=BUILTIN) as gateway:
                await self.post(gateway)
        self.assertEqual([(i.data, i.suffix) for i in gateway.generated_images], [(JPEG, ".jpg")])

    async def test_multi_round_replays_image_with_id_and_result_and_dedupes(self):
        first = sse([reasoning(), image_item(), call("call_1")])
        second = sse([image_item(), image_item(PNG2, "ig_2", "second"), message("Две картинки")])

        async def handler(name, arguments, call_id):
            return ToolOutcome('{"ok":true}')

        with self.upstream([first, second]):
            async with self.gateway(tools=TOOLS, tool_handler=handler, builtin_tools=BUILTIN) as gateway:
                await self.post(gateway)
                self.assertEqual(gateway.upstream_requests, 2)
        replayed = [item for item in self.bodies[1]["input"] if item.get("type") == "image_generation_call"]
        self.assertEqual(replayed, [{"type": "image_generation_call", "id": "ig_1", "status": "completed",
                                     "result": b64(PNG), "revised_prompt": "a blue square"}])
        self.assertNotIn("size", replayed[0])
        # ig_1 was reported by both rounds but kept once; ig_2 only by the last.
        self.assertEqual([(i.item_id, i.data) for i in gateway.generated_images], [("ig_1", PNG), ("ig_2", PNG2)])

    def test_replay_requires_id_and_result(self):
        self.assertIsNotNone(_replayable(image_item()))
        self.assertIsNone(_replayable({"type": "image_generation_call", "id": "ig_1", "status": "generating"}))
        self.assertIsNone(_replayable({**image_item(), "result": ""}))
        self.assertIsNone(_replayable({**image_item(), "id": None}))

    async def test_invalid_base64_magic_and_oversize_are_ignored(self):
        items = [
            {**image_item(item_id="ig_a"), "result": "!!!not base64!!!"},
            image_item(b"GIF89a" + b"x" * 40, "ig_b"),                 # not PNG/JPEG
            image_item(b"plain text, not a picture at all", "ig_c"),
            {**image_item(item_id="ig_d"), "result": ""},
            {"type": "image_generation_call", "id": "ig_e", "status": "in_progress"},
            {**image_item(item_id="ig_f"), "result": 12345},
            image_item(PNG2, "ig_ok", "fine"),
        ]
        with self.upstream([sse(items, with_completed_output=False)]):
            async with self.gateway(builtin_tools=BUILTIN) as gateway:
                gateway.MAX_GENERATED_IMAGE = 64
                await self.post(gateway)
                self.assertEqual([i.item_id for i in gateway.generated_images], [])
        small = PNG2[:20]
        with self.upstream([sse(items[:-1] + [image_item(small, "ig_small")], with_completed_output=False)]):
            async with self.gateway(builtin_tools=BUILTIN) as gateway:
                gateway.MAX_GENERATED_IMAGE = 64
                await self.post(gateway)
        self.assertEqual([i.item_id for i in gateway.generated_images], ["ig_small"])

    async def test_count_cap(self):
        items = [image_item(PNG + bytes([n]), f"ig_{n}") for n in range(7)]
        with self.upstream([sse(items, with_completed_output=False)]):
            async with self.gateway(builtin_tools=BUILTIN) as gateway:
                await self.post(gateway)
        self.assertEqual(len(gateway.generated_images), BrainRequestGateway.MAX_GENERATED_IMAGES)


class BackendImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_passes_builtin_tools_and_reports_support(self):
        backend = CodexSubscriptionBackend(model="gpt-6-luna")
        self.assertEqual(backend._request_gateway("p", (), builtin_tools=BUILTIN)._builtin_tools, tuple(BUILTIN))
        self.assertEqual(backend._request_gateway("p", ())._builtin_tools, ())
        self.assertIs(backend.model_info()["image_generation_supported"], True)
        self.assertIn("image_generation", CodexSubscriptionBackend.SAFE_DISABLE_FEATURES)  # the CLI stays without it

    async def test_mock_backend_and_default_hand_off(self):
        backend = MockBackend()
        self.assertIn("offline mock", await backend.generate("hi", "r", builtin_tools=BUILTIN))
        self.assertEqual(backend.consume_generated_images("r"), [])
        self.assertEqual(BrainBackend.consume_generated_images(backend, None), [])

    async def test_generated_images_are_consumed_once_and_counted(self):
        backend = CodexSubscriptionBackend(model="gpt-6-luna")
        picture = GeneratedImage(PNG, "p", "ig_1")
        backend._generated["r1"] = [picture]
        self.assertEqual(backend.consume_generated_images("r1"), [picture])
        self.assertEqual(backend.consume_generated_images("r1"), [])
        self.assertEqual(backend.consume_generated_images(None), [])


class SupervisorImageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.store = MemoryStore(self.root)
        self.telegram = FakeTelegram()
        self.config = Config(self.root, OWNER, SECONDARY, backend="mock", timezone_name="UTC", brain_timeout=2.0)

    async def asyncTearDown(self) -> None:
        self.store.close()
        self._temp.cleanup()

    def supervisor(self, brain, config=None) -> TaskSupervisor:
        return TaskSupervisor(config or self.config, self.store, brain, self.telegram, asyncio.Event())

    async def ask(self, supervisor, sender: int = OWNER, text: str = "нарисуй квадрат", message_id: int = 7) -> None:
        identity = Identity.PRIMARY if sender == OWNER else Identity.SECONDARY
        await supervisor.submit(IncomingMessage(1, sender, sender, text, message_id=message_id), identity)
        await asyncio.wait_for(asyncio.gather(*list(supervisor.tasks)), 5)

    def drawing_brain(self, reply="Вот квадрат", **kwargs) -> ScriptedBrain:
        return ScriptedBrain([ScriptedTurn(reply=reply, generated=[GeneratedImage(PNG, "a blue square", "ig_1")])],
                             image_generation_supported=True, **kwargs)

    async def test_photo_is_sent_before_text_and_saved_per_sender(self):
        brain = self.drawing_brain()
        sup = self.supervisor(brain)
        await self.ask(sup)
        self.assertEqual(brain.generate_calls[0]["builtin_tools"], BUILTIN)
        self.assertEqual([e[0] for e in self.telegram.log], ["photo", "text"])
        kind, chat, name, data, extra = self.telegram.log[0]
        self.assertEqual((chat, data, extra), (OWNER, PNG, None))
        self.assertEqual(self.telegram.log[1][:3], ("text", OWNER, "Вот квадрат"))
        saved = self.root / "memory" / "generated" / str(OWNER) / name
        self.assertTrue(saved.is_file())
        self.assertEqual(saved.read_bytes(), PNG)
        events = [e for e in self.events() if e.type == "IMAGE_GENERATED"]
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].data["ok"])
        self.assertEqual(events[0].data["revised_prompt"], "a blue square")

    async def test_failed_send_adds_note_and_keeps_model_text(self):
        brain = self.drawing_brain(reply="Нарисовал, держи")
        sup = self.supervisor(brain)
        self.telegram.fail_photo = True
        await self.ask(sup)
        texts = [e[2] for e in self.telegram.log if e[0] == "text"]
        self.assertEqual(len(texts), 1)
        self.assertTrue(texts[0].startswith("Нарисовал, держи"))
        self.assertIn("не удалось отправить", texts[0])
        events = [e for e in self.events() if e.type == "IMAGE_GENERATED"]
        self.assertFalse(events[0].data["ok"])
        self.assertEqual(events[0].data["error"], "TelegramError")

    async def test_picture_without_text_sends_only_the_photo(self):
        sup = self.supervisor(self.drawing_brain(reply=""))
        await self.ask(sup)
        self.assertEqual([e[0] for e in self.telegram.log], ["photo"])

    async def test_no_image_without_generation_or_with_flag_off(self):
        plain = ScriptedBrain(response="просто ответ")
        await self.ask(self.supervisor(plain))
        self.assertEqual(plain.generate_calls[0]["builtin_tools"], [])
        self.assertEqual([e[0] for e in self.telegram.log], ["text"])
        off = self.drawing_brain()
        await self.ask(self.supervisor(off, Config(self.root, OWNER, SECONDARY, backend="mock", timezone_name="UTC",
                                                    brain_timeout=2.0, image_generation=False)))
        self.assertEqual(off.generate_calls[0]["builtin_tools"], [])
        self.assertEqual([e[0] for e in self.telegram.log], ["text", "text"])

    async def test_secondary_account_can_draw_too_and_folders_are_separate(self):
        owner_brain, second_brain = self.drawing_brain(), ScriptedBrain(
            [ScriptedTurn(reply="и тебе", generated=[GeneratedImage(PNG2, "second", "ig_9")])],
            image_generation_supported=True)
        await self.ask(self.supervisor(owner_brain))
        await self.ask(self.supervisor(second_brain), SECONDARY)
        folders = {p.name: sorted(f.name for f in p.iterdir() if f.suffix == ".png")
                   for p in (self.root / "memory" / "generated").iterdir()}
        self.assertEqual(set(folders), {str(OWNER), str(SECONDARY)})
        self.assertTrue(all(len(v) == 1 for v in folders.values()))
        self.assertEqual({e[1] for e in self.telegram.log if e[0] == "photo"}, {OWNER, SECONDARY})

    def turn(self, sender: int) -> TurnTools:
        return TurnTools(config=self.config, store=self.store, context=None, knowledge=None,
                         action_broker=type("B", (), {"available_for": lambda self, s: False})(),
                         profile_store=None, telegram=self.telegram, sender_id=sender, chat_id=sender,
                         request_id="later", video_analyzer=__import__("personal_agent.video", fromlist=["x"]).VideoAnalyzer(self.root, ffmpeg="x"))

    async def test_generated_image_is_in_recent_media_and_isolated(self):
        await self.ask(self.supervisor(self.drawing_brain()))
        own, other = self.turn(OWNER), self.turn(SECONDARY)
        catalog = own.public_media_catalog()
        self.assertEqual([item["kind"] for item in catalog], ["generated_image"])
        media_id = catalog[0]["media_id"]
        self.assertEqual(catalog[0]["prompt"], "a blue square")
        self.assertEqual(other.public_media_catalog(), [])
        result, attached = await own.view_recent_media({"media_id": media_id})
        self.assertTrue(result["ok"])
        self.assertEqual(result["media_kind"], "generated_image")
        self.assertEqual((await other.view_recent_media({"media_id": media_id}))["error"]["code"], "media_not_found")
        self.telegram.log.clear()
        receipt = await own.send_file({"media_id": media_id})
        self.assertTrue(receipt["ok"])
        self.assertEqual([(e[0], e[1], e[3]) for e in self.telegram.log], [("photo", OWNER, PNG)])
        self.assertEqual((await other.send_file({"media_id": media_id}))["error"]["code"], "media_not_found")

    async def test_group_photo_replies_to_asker_and_is_stored_per_chat(self):
        harness = GroupHarness(ScriptedBrain(
            [ScriptedTurn(reply="группе", generated=[GeneratedImage(PNG, "group art", "ig_g")])],
            image_generation_supported=True))
        self.addAsyncCleanup(harness.close)
        photos: list[dict] = []

        async def send_photo(chat_id, path, caption="", *, reply_to=None):
            photos.append({"chat_id": chat_id, "reply_to": reply_to, "data": path.read_bytes()})
            return "sendPhoto"
        harness.bot.send_photo = send_photo
        text = f"@{BOT_NAME} нарисуй кота"
        await harness.send(group_update(11, text, entities=mention(text), sender=MEMBER))
        self.assertEqual(photos, [{"chat_id": GROUP, "reply_to": 11, "data": PNG}])
        self.assertEqual(harness.bot.sent[-1]["reply_to"], 11)
        self.assertEqual(harness.brain.generate_calls[0]["builtin_tools"], BUILTIN)
        self.assertTrue((harness.root / "memory" / "generated" / str(GROUP)).is_dir())
        self.assertFalse((harness.root / "memory" / "generated" / str(MEMBER)).exists())
        self.assertFalse((harness.root / "memory" / "generated" / str(OWNER)).exists())

    def events(self):
        from datetime import datetime, timezone
        return self.store.events_between(datetime(2000, 1, 1, tzinfo=timezone.utc),
                                         datetime(2100, 1, 1, tzinfo=timezone.utc))


class FakeTelegram:
    def __init__(self) -> None:
        self.log: list[tuple] = []
        self.fail_photo = False
        self.upload_cap_bytes = 50 * 1024 * 1024

    async def send_text(self, chat_id, text, *, reply_to=None):
        self.log.append(("text", chat_id, text))

    async def send_chat_action(self, chat_id, action="typing"):
        pass

    async def send_photo(self, chat_id, path, caption="", *, reply_to=None):
        if self.fail_photo:
            raise TelegramError("Telegram upload failed")
        self.log.append(("photo", chat_id, path.name, path.read_bytes(), reply_to))
        return "sendPhoto"


if __name__ == "__main__":
    unittest.main()
