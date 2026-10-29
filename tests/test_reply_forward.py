"""Replies and forwards: the bot sees which message was answered and who wrote a forwarded original."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from personal_agent.config import Config
from personal_agent.context import ContextBuilder
from personal_agent.groups import BotIdentity
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage
from personal_agent.preferences import PreferenceStore
from personal_agent.reply_context import memory_prefix, strip_memory_prefix
from personal_agent.telegram import parse_message
from personal_agent.testing import ScriptedBrain
from scripts.verify_conversation import red_png
from tests.test_context_isolation import packet_from
from tests.test_group_chats import BOT_ID, BOT_NAME, GROUP, MEMBER, OWNER, SECONDARY, GroupHarness, group_update, mention
from tests.test_service_observation_integration import ObservationHarness, _message

BOT = BotIdentity(BOT_ID, BOT_NAME, "Personal Agent")
HINT = "current_message_reply_to — сообщение, на которое пользователь ответил"
LEAK = "СЕКРЕТ-ВЛАДЕЛЬЦА"


def private(message: dict, sender: int = OWNER) -> dict:
    return {"update_id": 1, "message": {"message_id": 50, "from": {"id": sender, "is_bot": False, "first_name": "Я"},
                                        "chat": {"id": sender, "type": "private"}, "text": "а почему?", **message}}


def group(message: dict) -> dict:
    return {"update_id": 1, "message": {"message_id": 50, "from": {"id": MEMBER, "is_bot": False, "first_name": "Маша"},
                                        "chat": {"id": GROUP, "type": "supergroup", "title": "Друзья"},
                                        "text": "а почему?", **message}}


class ParseReplyTests(unittest.TestCase):
    def test_reply_to_the_bot(self) -> None:
        reply = {"message_id": 7, "from": {"id": BOT_ID, "is_bot": True, "first_name": "Personal Agent"},
                 "text": "Курс 92 рубля"}
        parsed = parse_message(private({"reply_to_message": reply}), BOT)
        self.assertEqual(parsed.reply_to, {"message_id": 7, "text": "Курс 92 рубля",
                                           "author": {"name": "Personal Agent", "is_bot_self": True}})
        self.assertIsNone(parsed.forwarded_from)

    def test_reply_to_own_message_in_private_chat(self) -> None:
        parsed = parse_message(private({"reply_to_message": {"message_id": 3, "from": {"id": OWNER, "first_name": "Я"},
                                                             "text": "заметка"}}), BOT)
        self.assertEqual(parsed.reply_to["author"], {"name": "Я", "is_self": True})

    def test_reply_to_someone_else_in_a_group(self) -> None:
        reply = {"message_id": 9, "from": {"id": 7002, "is_bot": False, "first_name": "Петя", "last_name": "Иванов"},
                 "caption": "смотрите", "photo": [{"file_id": "small", "file_size": 10},
                                                   {"file_id": "big", "file_size": 99}]}
        parsed = parse_message(group({"reply_to_message": reply}), BOT)
        self.assertEqual(parsed.reply_to["author"], {"name": "Петя Иванов"})
        self.assertEqual(parsed.reply_to["text"], "смотрите")
        self.assertEqual(parsed.reply_to["media"], {"kind": "photo", "file_id": "big", "file_size": 99})
        self.assertFalse(parsed.addressed)  # replying to a person is not addressing the bot

    def test_sticker_reply_keeps_the_format_fields_of_a_main_attachment(self) -> None:
        sticker = {"file_id": "st", "file_size": 5, "is_animated": False, "is_video": True, "emoji": "x"}
        parsed = parse_message(private({"reply_to_message": {"message_id": 4, "from": {"id": 1}, "sticker": sticker}}),
                               BOT)
        own = parse_message(private({"sticker": sticker}), BOT).attachment
        media = parsed.reply_to["media"]
        for key in ("kind", "file_id", "file_size", "is_animated", "is_video", "sticker_format"):
            self.assertEqual(media[key], own[key])
        self.assertEqual(media["sticker_format"], "webm")

    def test_every_media_kind_is_recognised(self) -> None:
        for kind in ("photo", "sticker", "document", "video", "voice", "audio", "animation", "video_note"):
            value = [{"file_id": "f"}] if kind == "photo" else {"file_id": "f", "file_size": 1}
            parsed = parse_message(private({"reply_to_message": {"message_id": 4, "from": {"id": 1}, kind: value}}), BOT)
            self.assertEqual(parsed.reply_to["media"]["kind"], kind)

    def test_external_reply_and_quote(self) -> None:
        external = {"message_id": 12, "origin": {"type": "user", "sender_user": {"id": 77, "first_name": "Олег"}},
                    "text": "из другого чата"}
        parsed = parse_message(private({"external_reply": external, "quote": {"text": "другого"}}), BOT)
        self.assertEqual(parsed.reply_to, {"message_id": 12, "author": {"name": "Олег"}, "text": "из другого чата",
                                           "quote": "другого"})

    def test_external_reply_to_the_bot_is_recognised(self) -> None:
        external = {"origin": {"type": "user", "sender_user": {"id": BOT_ID, "first_name": "Personal Agent"}}}
        self.assertTrue(parse_message(private({"external_reply": external}), BOT).reply_to["author"]["is_bot_self"])

    def test_text_is_bounded_and_cleaned(self) -> None:
        reply = {"message_id": 1, "from": {"id": 5, "first_name": "A\x00" + "я" * 200}, "text": "x\x07" + "я" * 5000}
        parsed = parse_message(private({"reply_to_message": reply, "quote": {"text": "q" * 3000}}), BOT)
        self.assertEqual(len(parsed.reply_to["text"]), 2000)
        self.assertEqual(len(parsed.reply_to["quote"]), 1000)
        self.assertLessEqual(len(parsed.reply_to["author"]["name"]), 64)
        self.assertNotIn("\\u0000", json.dumps(parsed.reply_to))
        self.assertNotIn("\x07", parsed.reply_to["text"])

    def test_a_reply_that_is_itself_forwarded_carries_the_original_author(self) -> None:
        reply = {"message_id": 2, "from": {"id": OWNER, "first_name": "Я"}, "text": "чужое",
                 "forward_origin": {"type": "hidden_user", "sender_user_name": "Скрытый Автор", "date": 1_700_000_000}}
        info = parse_message(private({"reply_to_message": reply}), BOT).reply_to["forwarded_from"]
        self.assertEqual(info, {"type": "hidden_user", "name": "Скрытый Автор", "date": "2023-11-14T22:13:20+00:00"})

    def test_forum_topic_service_message_is_not_a_reply(self) -> None:
        reply = {"message_id": 2, "from": {"id": 5, "first_name": "A"}, "forum_topic_created": {"name": "t"}}
        self.assertIsNone(parse_message(group({"reply_to_message": reply}), BOT).reply_to)

    def test_plain_message_has_neither_field(self) -> None:
        parsed = parse_message(private({}), BOT)
        self.assertIsNone(parsed.reply_to)
        self.assertIsNone(parsed.forwarded_from)


class ParseForwardTests(unittest.TestCase):
    def forward(self, **fields) -> dict | None:
        return parse_message(private(fields), BOT).forwarded_from

    def test_forward_origin_user(self) -> None:
        origin = {"type": "user", "date": 1_700_000_000,
                  "sender_user": {"id": 5, "first_name": "Иван", "last_name": "Петров", "username": "ivan_p"}}
        self.assertEqual(self.forward(forward_origin=origin), {
            "type": "user", "name": "Иван Петров", "username": "ivan_p", "date": "2023-11-14T22:13:20+00:00"})

    def test_forward_origin_hidden_user(self) -> None:
        self.assertEqual(self.forward(forward_origin={"type": "hidden_user", "sender_user_name": "Иван Петров"}),
                         {"type": "hidden_user", "name": "Иван Петров"})

    def test_forward_origin_chat_and_channel(self) -> None:
        chat = self.forward(forward_origin={"type": "chat", "sender_chat": {"id": -5, "title": "Клуб", "type": "supergroup"},
                                            "author_signature": "админ"})
        self.assertEqual(chat, {"type": "chat", "name": "Клуб (админ)"})
        channel = self.forward(forward_origin={"type": "channel", "chat": {"id": -9, "title": "Новости", "username": "news"},
                                               "message_id": 3})
        self.assertEqual(channel, {"type": "channel", "name": "Новости", "username": "news"})

    def test_legacy_fields(self) -> None:
        self.assertEqual(self.forward(forward_from={"id": 5, "first_name": "Иван", "username": "iv"},
                                      forward_date=1_700_000_000),
                         {"type": "user", "name": "Иван", "username": "iv", "date": "2023-11-14T22:13:20+00:00"})
        self.assertEqual(self.forward(forward_sender_name="Скрытый"), {"type": "hidden_user", "name": "Скрытый"})
        self.assertEqual(self.forward(forward_from_chat={"id": -9, "type": "channel", "title": "Новости"},
                                      forward_signature="Автор"), {"type": "channel", "name": "Новости (Автор)"})

    def test_origin_wins_over_legacy_and_garbage_is_ignored(self) -> None:
        both = self.forward(forward_origin={"type": "hidden_user", "sender_user_name": "Новый"},
                            forward_sender_name="Старый")
        self.assertEqual(both["name"], "Новый")
        self.assertIsNone(self.forward(forward_origin={"type": "mystery"}))
        self.assertIsNone(self.forward(forward_origin="x"))


class MemoryPrefixTests(unittest.TestCase):
    def message(self, **fields) -> IncomingMessage:
        return IncomingMessage(update_id=1, sender_id=OWNER, chat_id=OWNER, text="а почему?", **fields)

    def test_markers(self) -> None:
        self.assertEqual(memory_prefix(self.message()), "")
        self.assertEqual(memory_prefix(self.message(forwarded_from={"type": "user", "name": "Иван Петров"})),
                         "[Переслано от: Иван Петров]\n")
        bot = memory_prefix(self.message(reply_to={"text": "я так " * 100, "author": {"name": "Бот", "is_bot_self": True}}))
        self.assertTrue(bot.startswith("[В ответ на сообщение бота: «я так я так"))
        self.assertTrue(bot.endswith("…»]\n"))
        self.assertLess(len(bot), 260)
        self.assertEqual(memory_prefix(self.message(reply_to={"text": "привет", "author": {"name": "Петя"}})),
                         "[В ответ на сообщение Петя: «привет»]\n")
        self.assertEqual(memory_prefix(self.message(reply_to={"media": {"kind": "photo"}})), "[В ответ на фото]\n")
        self.assertEqual(memory_prefix(self.message(reply_to={"media": {"kind": "sticker"},
                                                              "author": {"is_bot_self": True}})),
                         "[В ответ на стикер бота]\n")

    def test_strip(self) -> None:
        self.assertEqual(strip_memory_prefix("[Переслано от: Иван]\n[В ответ на фото]\nсамо сообщение"), "само сообщение")
        self.assertEqual(strip_memory_prefix("[Переслано]\nх"), "[Переслано]\nх")


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.store = MemoryStore(Path(self._temp.name))
        self.addCleanup(self.store.close)
        self.config = Config(Path(self._temp.name), OWNER, SECONDARY, backend="mock", timezone_name="UTC")
        self.builder = ContextBuilder(self.store, self.config.tz, config=self.config)
        self.context = {
            "reply_to": {"message_id": 7, "text": "Курс 92 рубля", "author": {"name": "Бот", "is_bot_self": True},
                         "media": {"kind": "photo", "file_id": "SECRET-FILE-ID", "file_size": 5}, "media_id": "m-1"},
            "forwarded_from": {"type": "user", "name": "Иван Петров", "username": "ivan"}}

    def test_every_account_sees_the_fields_and_the_hint(self) -> None:
        for user in (OWNER, SECONDARY):
            prompt = self.builder.build("а почему?", user, reply_context=self.context)
            packet = packet_from(prompt)
            self.assertIn(HINT, prompt.split("CURRENT INPUTS")[0])
            self.assertEqual(packet["current_message_reply_to"], {
                "message_id": 7, "text": "Курс 92 рубля", "author": {"name": "Бот", "is_bot_self": True},
                "media_kind": "photo", "media_id": "m-1"})
            self.assertEqual(packet["current_message_forwarded_from"],
                             {"type": "user", "name": "Иван Петров", "username": "ivan"})
            self.assertNotIn("SECRET-FILE-ID", prompt)

    def test_absent_context_adds_no_keys(self) -> None:
        packet = packet_from(self.builder.build("привет", OWNER))
        self.assertNotIn("current_message_reply_to", packet)
        self.assertNotIn("current_message_forwarded_from", packet)

    def test_fields_survive_the_budget_cut(self) -> None:
        self.context["reply_to"]["text"] = "я" * 2000
        self.context["reply_to"]["quote"] = "ц" * 1000
        big = {"results": [{"kind": "x", "text": "z" * 5000} for _ in range(20)]}
        seen = set()
        for max_chars in (9000, 12000, 16000):
            small = ContextBuilder(self.store, self.config.tz, config=self.config, max_chars=max_chars)
            for user in (OWNER, SECONDARY):
                packet = packet_from(small.build("а почему?", user, reply_context=self.context, observation=big))
                if "notice" not in packet:
                    continue
                seen.add(packet["notice"])
                self.assertEqual(packet["current_message_forwarded_from"]["name"], "Иван Петров")
                reply = packet["current_message_reply_to"]
                self.assertEqual(reply["author"], {"name": "Бот", "is_bot_self": True})
                self.assertEqual(len(reply["text"]), 500)
                self.assertEqual(len(reply["quote"]), 300)
        self.assertEqual(seen, {"Context limited", "Observation exceeds context budget; results truncated"})


def png_fixture(root: Path) -> Path:
    fixture = root / "fixture.png"
    red_png(fixture)
    return fixture


def install_download(telegram, fixture: Path) -> list[str]:
    calls: list[str] = []

    async def download(file_id, destination, max_bytes):
        calls.append(file_id)
        shutil.copyfile(fixture, destination)
        return destination.stat().st_size

    telegram.download = download
    return calls


def reply_message(**fields) -> IncomingMessage:
    return replace(_message("а что на этой картинке?", update_id=60), **fields)


class SupervisorReplyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.h = ObservationHarness(ScriptedBrain(image_input_supported=True, response="ок"))
        self.addAsyncCleanup(self.h.close)
        self.fixture = png_fixture(self.h.root)

    async def ask(self, message: IncomingMessage) -> dict:
        await self.h.supervisor.submit(message, Identity.PRIMARY)
        await self.h.wait_idle()
        return self.h.brain.generate_calls[-1]

    async def test_reply_and_forward_reach_the_prompt(self) -> None:
        call = await self.ask(reply_message(
            reply_to={"message_id": 7, "text": "Курс 92", "author": {"name": "Бот", "is_bot_self": True}},
            forwarded_from={"type": "user", "name": "Иван Петров"}))
        packet = packet_from(call["prompt"])
        self.assertEqual(packet["current_message_reply_to"]["text"], "Курс 92")
        self.assertTrue(packet["current_message_reply_to"]["author"]["is_bot_self"])
        self.assertEqual(packet["current_message_forwarded_from"]["name"], "Иван Петров")
        self.assertIn('CURRENT USER MESSAGE (JSON):\n"а что на этой картинке?"', call["prompt"])

    async def test_photo_of_the_reply_is_downloaded_and_attached(self) -> None:
        calls = install_download(self.h.telegram, self.fixture)
        call = await self.ask(reply_message(reply_to={
            "message_id": 7, "author": {"name": "Бот", "is_bot_self": True},
            "media": {"kind": "photo", "file_id": "bot-photo", "file_size": 100}}))
        self.assertEqual(calls, ["bot-photo"])
        self.assertEqual(len(call["images"]), 1)
        reply = packet_from(call["prompt"])["current_message_reply_to"]
        self.assertEqual(reply["media_kind"], "photo")
        self.assertTrue(reply["media_id"])
        self.assertNotIn("media_unavailable", reply)
        self.assertNotIn("bot-photo", call["prompt"])

    async def test_saved_attachment_is_reused_without_downloading(self) -> None:
        calls = install_download(self.h.telegram, self.fixture)
        await self.h.supervisor.submit(replace(_message("", update_id=21), attachment={
            "kind": "photo", "file_id": "orig", "file_size": 100}), Identity.PRIMARY)
        await self.h.wait_idle()
        saved_id = self.h.store.recent_attachments(101)[0]["id"]
        self.assertEqual(calls, ["orig"])
        call = await self.ask(reply_message(reply_to={
            "message_id": 21, "media": {"kind": "photo", "file_id": "orig", "file_size": 100}}))
        self.assertEqual(calls, ["orig"])  # no second download
        self.assertEqual(len(call["images"]), 1)
        packet = packet_from(call["prompt"])
        self.assertEqual(packet["current_message_reply_to"]["media_id"], saved_id)
        self.assertNotIn(saved_id, [item["media_id"] for item in packet["recent_media"]])

    async def test_download_failure_does_not_fail_the_turn(self) -> None:
        async def broken(file_id, destination, max_bytes):
            raise RuntimeError("network down")

        self.h.telegram.download = broken
        call = await self.ask(reply_message(reply_to={
            "message_id": 7, "text": "вот", "media": {"kind": "photo", "file_id": "x", "file_size": 1}}))
        reply = packet_from(call["prompt"])["current_message_reply_to"]
        self.assertTrue(reply["media_unavailable"])
        self.assertEqual(call["images"], [])
        self.assertEqual(self.h.telegram.sent_text[-1][1], "ок")

    async def test_oversized_reply_picture_is_not_downloaded(self) -> None:
        calls = install_download(self.h.telegram, self.fixture)
        call = await self.ask(reply_message(reply_to={
            "message_id": 7, "media": {"kind": "photo", "file_id": "huge", "file_size": 50 * 1024 * 1024}}))
        self.assertEqual(calls, [])
        self.assertTrue(packet_from(call["prompt"])["current_message_reply_to"]["media_unavailable"])

    async def test_own_attachment_means_no_extra_download(self) -> None:
        calls = install_download(self.h.telegram, self.fixture)
        call = await self.ask(reply_message(
            attachment={"kind": "photo", "file_id": "mine", "file_size": 10},
            reply_to={"message_id": 7, "media": {"kind": "photo", "file_id": "theirs", "file_size": 10}}))
        self.assertEqual(calls, ["mine"])
        self.assertEqual(len(call["images"]), 1)

    async def test_non_picture_reply_media_is_only_described(self) -> None:
        calls = install_download(self.h.telegram, self.fixture)
        call = await self.ask(reply_message(reply_to={
            "message_id": 7, "media": {"kind": "document", "file_id": "doc", "file_size": 10}}))
        self.assertEqual(calls, [])
        reply = packet_from(call["prompt"])["current_message_reply_to"]
        self.assertEqual(reply["media_kind"], "document")
        self.assertNotIn("media_unavailable", reply)

    async def test_history_gets_the_prefix_but_the_model_gets_the_plain_text(self) -> None:
        call = await self.ask(reply_message(
            reply_to={"message_id": 7, "text": "Курс 92", "author": {"name": "Бот", "is_bot_self": True}},
            forwarded_from={"type": "user", "name": "Иван Петров"}))
        stored = [m for m in self.h.store.recent_messages(101) if m["role"] == "user"][-1]["text"]
        self.assertEqual(stored, "[Переслано от: Иван Петров]\n[В ответ на сообщение бота: «Курс 92»]\n"
                                 "а что на этой картинке?")
        self.assertIn('CURRENT USER MESSAGE (JSON):\n"а что на этой картинке?"', call["prompt"])
        # the next turn sees the context in the conversation feed
        await self.ask(_message("и что дальше?", update_id=61))
        feed = packet_from(self.h.brain.generate_calls[-1]["prompt"])["recent_conversation"]
        self.assertTrue(any("[Переслано от: Иван Петров]" in item["text"] for item in feed))

    async def test_unanswered_messages_keep_their_markers(self) -> None:
        self.h.store.record_message(101, "user", memory_prefix(IncomingMessage(
            update_id=1, sender_id=101, chat_id=101, text="", forwarded_from={"type": "user", "name": "Иван"})) + "раз")
        call = await self.ask(_message("и два", update_id=2))
        pending = packet_from(call["prompt"])["unanswered_user_messages"]
        self.assertEqual([item["text"] for item in pending], ["[Переслано от: Иван]\nраз"])

    async def test_prefix_does_not_change_preference_parsing_of_history(self) -> None:
        question = "если бы я попросил отвечать на ты, что бы ты сделал?"
        self.h.store.record_message(101, "user", f"[В ответ на сообщение бота: «ок»]\n{question}")
        with_prefix = PreferenceStore(self.h.store)._historical_update(101)
        self.h.store.record_message(101, "assistant", "x")
        self.assertEqual(with_prefix, PreferenceStore(self.h.store)._historical_update(101))
        self.assertNotIn("address", with_prefix)


class GroupReplyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.h = GroupHarness(ScriptedBrain(image_input_supported=True, response="ответ группе"))
        self.addAsyncCleanup(self.h.close)
        self.fixture = png_fixture(self.h.root)
        self.calls = install_download(self.h.bot, self.fixture)

    def addressed(self, update_id: int, **extra) -> dict:
        text = f"@{BOT_NAME} что это?"
        return group_update(update_id, text, entities=mention(text), **extra)

    async def test_group_reply_to_a_member_photo_reaches_the_prompt_and_history(self) -> None:
        reply = {"message_id": 5, "from": {"id": 7002, "is_bot": False, "first_name": "Петя"},
                 "caption": "мой кот", "photo": [{"file_id": "cat", "file_size": 100}]}
        await self.h.send(self.addressed(10, reply_to_message=reply))
        call = self.h.brain.generate_calls[-1]
        packet = packet_from(call["prompt"])
        self.assertEqual(packet["current_message_reply_to"]["author"], {"name": "Петя"})
        self.assertEqual(packet["current_message_reply_to"]["text"], "мой кот")
        self.assertEqual(self.calls, ["cat"])
        self.assertEqual(len(call["images"]), 1)
        self.assertEqual(packet["context_policy"]["current_account"], "group")
        saved = [m for m in self.h.store.recent_chat_messages(GROUP) if m["role"] == "user"][-1]["text"]
        self.assertTrue(saved.startswith("[В ответ на сообщение Петя: «мой кот»]\n"))
        stored = self.h.store.recent_attachments(GROUP)
        self.assertEqual(len(stored), 1)  # saved under the chat's scope and keyed by the replied-to message id
        self.assertEqual(stored[0]["message_id"], 5)

    async def test_saved_group_attachment_is_found_by_message_id(self) -> None:
        text = f"@{BOT_NAME} запомни"
        await self.h.send(group_update(30, text, entities=mention(text), photo=[{"file_id": "first", "file_size": 100}]))
        reply = {"message_id": 30, "from": {"id": MEMBER, "first_name": "Маша"},
                 "photo": [{"file_id": "first", "file_size": 100}]}
        await self.h.send(self.addressed(31, reply_to_message=reply))
        self.assertEqual(self.calls, ["first"])
        self.assertEqual(len(self.h.brain.generate_calls[-1]["images"]), 1)

    async def test_nothing_of_the_owner_leaks_through_the_new_fields(self) -> None:
        self.h.store.record_message(OWNER, "user", LEAK)
        reply = {"message_id": 5, "from": {"id": BOT_ID, "is_bot": True, "first_name": "Personal Agent"},
                 "text": "привет"}
        await self.h.send(self.addressed(10, reply_to_message=reply,
                                         forward_origin={"type": "hidden_user", "sender_user_name": "Иван"}))
        prompt = self.h.brain.generate_calls[-1]["prompt"]
        packet = packet_from(prompt)
        self.assertNotIn(LEAK, prompt)
        self.assertNotIn("activity_summary", packet)
        self.assertEqual({key for key in packet if key.startswith("current_message")},
                         {"current_message_author", "current_message_reply_to", "current_message_forwarded_from"})
        self.assertIn(HINT, prompt)
        self.assertIn("ничего не раскрываешь о владельце", prompt)


class SecondaryReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_secondary_gets_only_what_it_sent_and_no_owner_data(self) -> None:
        h = ObservationHarness(ScriptedBrain(response="ок"))
        self.addAsyncCleanup(h.close)
        h.store.record_message(101, "user", LEAK)
        message = IncomingMessage(update_id=5, sender_id=202, chat_id=202, text="а почему?", message_id=5,
                                  reply_to={"message_id": 4, "text": "мой факт", "author": {"name": "Я", "is_self": True}},
                                  forwarded_from={"type": "user", "name": "Иван"})
        await h.supervisor.submit(message, Identity.SECONDARY)
        await h.wait_idle()
        packet = packet_from(h.brain.generate_calls[-1]["prompt"])
        self.assertNotIn(LEAK, json.dumps(packet.get("recent_conversation")))
        self.assertNotIn("activity_summary", packet)
        self.assertEqual(packet["current_message_reply_to"]["author"], {"name": "Я", "is_self": True})
        self.assertEqual(packet["current_message_forwarded_from"]["name"], "Иван")
        stored = [m for m in h.store.recent_messages(202) if m["role"] == "user"][-1]["text"]
        self.assertTrue(stored.startswith("[Переслано от: Иван]\n[В ответ на своё сообщение: «мой факт»]\n"))


if __name__ == "__main__":
    unittest.main()
