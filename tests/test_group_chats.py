"""Group chats: served only while the owner is a member, with the shared account's rights and per-chat memory."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import httpx

from personal_agent.config import Config
from personal_agent.context import ContextBuilder
from personal_agent.groups import GroupAdmission, admitted_group, owner_is_member
from personal_agent.memory import MemoryStore
from personal_agent.models import EMERGENCY_PHRASE, Identity, IncomingMessage
from personal_agent.preferences import PreferenceStore
from personal_agent.security import TrustGate
from personal_agent.self_knowledge import account_role
from personal_agent.supervisor import SECONDARY_COMMAND_REFUSAL, TaskSupervisor
from personal_agent.telegram import TelegramClient, TelegramGateway, parse_message
from personal_agent.testing import ScriptedBrain, ScriptedTurn
from personal_agent.tools import DEFAULT_REGISTRY
from tests.test_context_isolation import packet_from, runtime_facts_from

OWNER, SECONDARY, MEMBER, OTHER = 5001, 5002, 7001, 7002
GROUP, GROUP2, UNKNOWN_GROUP = -1001, -1002, -1003
BOT_ID, BOT_NAME = 9000, "PersonalAgentBot"
OWNER_FACT = "ПОЛЯРИС-731 ТВЕРЬ"
SECONDARY_FACT = "СЕКОНДАРИ-ФАКТ-55"
GROUP_TOOLS = {"web_search", "web_open", "watch_video", "send_file", "view_recent_media", "read_attachment",
               "save_preferences", "save_profile", "search_memory", "ask_owner_permission"}
PC_TOOLS = ("list_directory", "read_file", "find_files", "read_active_window", "take_screenshot",
            "read_browser_page", "activity_timeline", "service_diagnostics", "write_file", "run_command",
            "share_with_secondary", "deny_secondary_request", "list_shared_context", "revoke_shared_note")


class FakeBot:
    """Stands in for the Bot API client: getMe, getChatMember, leaveChat, sendMessage, typing."""

    def __init__(self) -> None:
        self.members: dict[tuple[int, int], dict] = {}
        self.member_error: set[int] = set()
        self.member_calls: list[tuple[int, int]] = []
        self.left: list[int] = []
        self.sent: list[dict] = []
        self.typing: list[int] = []

    def set_member(self, chat_id: int, user_id: int, status: str = "member", **extra) -> None:
        self.members[(chat_id, user_id)] = {"status": status, "user": {"id": user_id}, **extra}

    async def get_me(self) -> dict:
        return {"id": BOT_ID, "username": BOT_NAME, "first_name": "Personal Agent", "is_bot": True}

    async def get_chat_member(self, chat_id: int, user_id: int) -> dict:
        self.member_calls.append((chat_id, user_id))
        if chat_id in self.member_error:
            raise RuntimeError("network down")
        return self.members.get((chat_id, user_id), {"status": "left", "user": {"id": user_id}})

    async def leave_chat(self, chat_id: int) -> None:
        self.left.append(chat_id)

    async def send_text(self, chat_id: int, text: str, *, reply_to: int | None = None) -> None:
        self.sent.append({"chat_id": chat_id, "text": text, "reply_to": reply_to})

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.typing.append(chat_id)

    def texts_to(self, chat_id: int) -> list[str]:
        return [item["text"] for item in self.sent if item["chat_id"] == chat_id]


def group_update(update_id: int, text: str, *, chat_id: int = GROUP, sender: int = MEMBER, name: str = "Маша",
                 title: str = "Друзья", entities=None, reply_to_bot: bool = False, chat_type: str = "supergroup",
                 **extra) -> dict:
    message = {"message_id": update_id, "from": {"id": sender, "is_bot": False, "first_name": name},
               "chat": {"id": chat_id, "type": chat_type, "title": title}, "text": text, **extra}
    if entities:
        message["entities"] = entities
    if reply_to_bot:
        message["reply_to_message"] = {"message_id": 1, "from": {"id": BOT_ID, "is_bot": True}, "text": "ответ"}
    return {"update_id": update_id, "message": message}


def mention(text: str, handle: str = "@" + BOT_NAME) -> list[dict]:
    return [{"type": "mention", "offset": text.index(handle), "length": len(handle)}]


def private_update(update_id: int, sender: int, text: str) -> dict:
    return {"update_id": update_id, "message": {"message_id": update_id, "from": {"id": sender, "is_bot": False},
                                                "chat": {"id": sender, "type": "private"}, "text": text}}


class GroupHarness:
    def __init__(self, brain: ScriptedBrain | None = None) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.store = MemoryStore(self.root)
        self.bot = FakeBot()
        self.brain = brain or ScriptedBrain()
        self.stop = asyncio.Event()
        self.config = Config(self.root, OWNER, SECONDARY, backend="mock", timezone_name="UTC", brain_timeout=2.0)
        self.supervisor = TaskSupervisor(self.config, self.store, self.brain, self.bot, self.stop)
        self.clock = [0.0]
        self.admission = GroupAdmission(self.bot, self.store, OWNER, clock=lambda: self.clock[0])
        self.gateway = TelegramGateway(self.bot, TrustGate(OWNER, SECONDARY), self.store, self.supervisor.submit,
                                       self.supervisor.shutdown, self.stop, groups=self.admission)
        self.bot.set_member(GROUP, OWNER, "member")
        self.bot.set_member(GROUP2, OWNER, "creator")

    async def send(self, *updates: dict) -> None:
        await self.gateway.dispatch_batch(list(updates))
        await self.idle()

    async def idle(self) -> None:
        tasks = list(self.supervisor.tasks)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3.0)

    def events(self, kind: str) -> list:
        return [event for event in self.store.events_between(
            __import__("datetime").datetime(2000, 1, 1, tzinfo=__import__("datetime").timezone.utc),
            __import__("datetime").datetime(2100, 1, 1, tzinfo=__import__("datetime").timezone.utc))
            if event.type == kind]

    async def close(self) -> None:
        await self.supervisor.shutdown()
        self.store.close()
        self._temp.cleanup()


class GroupTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.h = GroupHarness(ScriptedBrain(response="ответ группе"))
        self.addAsyncCleanup(self.h.close)

    def addressed(self, update_id: int, text: str = "привет", **kwargs) -> dict:
        text = f"@{BOT_NAME} {text}"
        return group_update(update_id, text, entities=mention(text), **kwargs)


class AdmissionTests(GroupTestCase):
    async def test_owner_member_group_is_served_with_a_reply_to_the_question(self) -> None:
        release = asyncio.Event()
        self.h.brain.turns = [ScriptedTurn(reply="ответ группе", block=release)]
        await self.h.gateway.dispatch_batch([self.addressed(10, "расскажи шутку")])
        await asyncio.wait_for(self.h.brain.started.wait(), 1)
        await asyncio.sleep(0)
        self.assertEqual(self.h.bot.typing, [GROUP])  # typing shows in the group while the answer is prepared
        release.set()
        await self.h.idle()
        self.assertEqual(len(self.h.brain.generate_calls), 1)
        self.assertEqual(self.h.bot.sent, [{"chat_id": GROUP, "text": "ответ группе", "reply_to": 10}])
        self.assertIsNotNone(admitted_group(self.h.store, GROUP))
        self.assertEqual(self.h.bot.left, [])

    async def test_owner_absent_leaves_notifies_owner_and_never_calls_the_model(self) -> None:
        await self.h.send(self.addressed(10, chat_id=UNKNOWN_GROUP, title="Чужой чат"))
        self.assertEqual(self.h.brain.generate_calls, [])
        self.assertEqual(self.h.bot.left, [UNKNOWN_GROUP])
        self.assertEqual(self.h.bot.sent, [{"chat_id": OWNER, "reply_to": None,
                                            "text": "Меня добавили в чат «Чужой чат», но тебя там нет — я вышел."}])
        rejected = self.h.events("GROUP_REJECTED")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].data["reason"], "owner_not_member")
        self.assertIsNone(admitted_group(self.h.store, UNKNOWN_GROUP))
        self.assertEqual(self.h.store.recent_chat_messages(UNKNOWN_GROUP), [])  # nothing from it is stored

    async def test_membership_check_error_fails_closed(self) -> None:
        self.h.bot.member_error.add(GROUP)
        await self.h.send(self.addressed(10))
        self.assertEqual(self.h.brain.generate_calls, [])
        self.assertEqual(self.h.bot.left, [GROUP])
        self.assertEqual(self.h.events("GROUP_REJECTED")[0].data["reason"], "membership_check_failed")
        self.assertEqual(len(self.h.bot.texts_to(OWNER)), 1)
        self.assertEqual(self.h.bot.texts_to(GROUP), [])

    async def test_restricted_owner_counts_only_while_still_a_member(self) -> None:
        self.assertTrue(owner_is_member({"status": "restricted", "is_member": True}))
        self.assertFalse(owner_is_member({"status": "restricted", "is_member": False}))
        for status in ("left", "kicked"):
            self.assertFalse(owner_is_member({"status": status}))
        self.assertFalse(owner_is_member(None))
        self.h.bot.set_member(GROUP, OWNER, "restricted", is_member=False)
        await self.h.send(self.addressed(10))
        self.assertEqual(self.h.brain.generate_calls, [])
        self.assertEqual(self.h.bot.left, [GROUP])

    async def test_bot_added_to_a_group_checks_at_once(self) -> None:
        def added(update_id: int, chat_id: int, status: str = "member", chat_type: str = "group") -> dict:
            return {"update_id": update_id, "my_chat_member": {
                "chat": {"id": chat_id, "type": chat_type, "title": "Новый"}, "from": {"id": MEMBER},
                "new_chat_member": {"status": status, "user": {"id": BOT_ID}}}}
        await self.h.send(added(1, GROUP), added(2, UNKNOWN_GROUP), added(3, -5000, chat_type="channel"))
        self.assertIsNotNone(admitted_group(self.h.store, GROUP))
        self.assertEqual(self.h.bot.left, [UNKNOWN_GROUP])  # the channel is simply ignored
        self.assertEqual(self.h.bot.member_calls, [(GROUP, OWNER), (UNKNOWN_GROUP, OWNER)])
        await self.h.send(added(4, GROUP, status="kicked"))
        self.assertIsNone(admitted_group(self.h.store, GROUP))

    async def test_owner_leaving_makes_the_bot_leave(self) -> None:
        await self.h.send(self.addressed(10))
        self.assertIsNotNone(admitted_group(self.h.store, GROUP))
        leaving = {"update_id": 11, "chat_member": {
            "chat": {"id": GROUP, "type": "supergroup", "title": "Друзья"}, "from": {"id": OWNER},
            "new_chat_member": {"status": "left", "user": {"id": OWNER}}}}
        await self.h.send(leaving)
        self.assertEqual(self.h.bot.left, [GROUP])
        self.assertIsNone(admitted_group(self.h.store, GROUP))
        self.assertEqual(len(self.h.bot.texts_to(OWNER)), 1)
        # A second signal about the same departure (the service message) does not repeat anything.
        gone = group_update(12, "", left_chat_member={"id": OWNER, "first_name": "Хозяин"})
        await self.h.send(gone)
        self.assertEqual(self.h.bot.left, [GROUP])
        self.assertEqual(len(self.h.bot.texts_to(OWNER)), 1)
        # Another member leaving changes nothing.
        await self.h.send(group_update(13, "", chat_id=GROUP2, left_chat_member={"id": MEMBER}))
        self.assertEqual(self.h.bot.left, [GROUP])

    async def test_left_chat_member_message_alone_removes_a_served_group(self) -> None:
        await self.h.send(self.addressed(10))
        await self.h.send(group_update(11, "", left_chat_member={"id": OWNER}))
        self.assertEqual(self.h.bot.left, [GROUP])
        self.h.brain.generate_calls.clear()
        await self.h.send(self.addressed(12))  # cached as rejected: still no answer
        self.assertEqual(self.h.brain.generate_calls, [])

    async def test_membership_is_cached_for_five_minutes_then_rechecked(self) -> None:
        await self.h.send(self.addressed(10), self.addressed(11))
        self.assertEqual(self.h.bot.member_calls, [(GROUP, OWNER)])
        self.h.clock[0] = 301.0
        self.h.bot.members[(GROUP, OWNER)] = {"status": "left", "user": {"id": OWNER}}  # left silently
        await self.h.send(self.addressed(12))
        self.assertEqual(len(self.h.bot.member_calls), 2)
        self.assertEqual(self.h.bot.left, [GROUP])
        self.assertEqual(len(self.h.brain.generate_calls), 1)  # the third message is refused

    async def test_channels_and_private_strangers_are_ignored(self) -> None:
        channel = group_update(10, "hello", chat_id=-1009, chat_type="channel")
        await self.h.send(channel, private_update(11, 99999, "привет"))
        self.assertIsNone(parse_message(channel))
        self.assertEqual(self.h.brain.generate_calls, [])
        self.assertEqual(self.h.bot.sent, [])
        self.assertEqual(self.h.bot.member_calls, [])

    async def test_a_forged_group_identity_for_an_unadmitted_chat_is_dropped_by_the_supervisor(self) -> None:
        message = parse_message(group_update(10, "x"), None)
        await self.h.supervisor.submit(message, Identity.GROUP)  # never admitted: no feed, no turn
        self.assertEqual(self.h.store.recent_chat_messages(GROUP), [])
        self.assertEqual(self.h.supervisor.tasks, set())
        private = IncomingMessage(1, OWNER, OWNER, "x", 1)
        await self.h.supervisor.submit(private, Identity.GROUP)
        self.assertEqual(self.h.supervisor.tasks, set())


class AddressingTests(GroupTestCase):
    async def test_only_addressed_messages_start_a_model_call_but_all_reach_the_feed(self) -> None:
        plain = group_update(10, "ребята, кто идёт в кино?", sender=OTHER, name="Витя")
        await self.h.send(plain)
        self.assertEqual(self.h.brain.generate_calls, [])
        self.assertEqual(self.h.bot.sent, [])
        # Even the unaddressed message was admission-checked and is now context for the next answer.
        await self.h.send(self.addressed(11, "а ты что думаешь?"))
        self.assertEqual(len(self.h.brain.generate_calls), 1)
        feed = packet_from(self.h.brain.generate_calls[0]["prompt"])["recent_conversation"]
        self.assertEqual([(item["from"], item["text"]) for item in feed][0], ("Витя", "ребята, кто идёт в кино?"))

    async def test_addressing_forms(self) -> None:
        bot = await self.h.admission.ensure_identity()
        text = f"@{BOT_NAME.lower()} привет"
        cases = {
            "mention": group_update(1, f"эй @{BOT_NAME} ответь", entities=mention(f"эй @{BOT_NAME} ответь")),
            "mention-case": group_update(2, text, entities=mention(text, "@" + BOT_NAME.lower())),
            "plain-at": group_update(3, f"привет, @{BOT_NAME}!"),
            "reply": group_update(4, "а почему?", reply_to_bot=True),
            "name-prefix": group_update(5, "Personal Agent, привет"),
            "username-prefix": group_update(6, f"{BOT_NAME} расскажи"),
            "command": group_update(7, f"/help@{BOT_NAME}", entities=[{"type": "bot_command", "offset": 0,
                                                                     "length": len(f"/help@{BOT_NAME}")}]),
            "emoji-offset": group_update(8, f"😀 @{BOT_NAME} hi", entities=[{"type": "mention", "offset": 3,
                                                                          "length": len(BOT_NAME) + 1}]),
        }
        for name, update in cases.items():
            self.assertTrue(parse_message(update, bot).addressed, name)
        not_addressed = {
            "chatter": group_update(20, "всем привет"),
            "other-bot": group_update(21, "@OtherBot привет", entities=mention("@OtherBot привет", "@OtherBot")),
            "similar-name": group_update(22, f"@{BOT_NAME}Fan привет"),
            "plain-command": group_update(23, "/help", entities=[{"type": "bot_command", "offset": 0, "length": 5}]),
            "other-bot-command": group_update(24, "/help@OtherBot"),
            "reply-to-human": group_update(25, "ага", reply_to_message={"message_id": 2, "from": {"id": OTHER}}),
        }
        for name, update in not_addressed.items():
            self.assertFalse(parse_message(update, bot).addressed, name)
        self.assertFalse(parse_message(cases["mention"], None).addressed)  # unknown identity: never addressed
        self.assertTrue(parse_message(private_update(30, OWNER, "привет")).addressed)

    async def test_two_members_asking_at_once_both_get_answers(self) -> None:
        release = asyncio.Event()
        self.h.brain.turns = [ScriptedTurn(reply="первому", block=release), ScriptedTurn(reply="второму")]
        await self.h.gateway.dispatch_batch([self.addressed(10, "вопрос А", sender=MEMBER, name="Маша")])
        await asyncio.wait_for(self.h.brain.started.wait(), 1)
        await self.h.gateway.dispatch_batch([self.addressed(11, "вопрос Б", sender=OTHER, name="Витя")])
        release.set()
        await self.h.idle()
        self.assertEqual(sorted((item["text"], item["reply_to"]) for item in self.h.bot.sent),
                         [("второму", 11), ("первому", 10)])

    async def test_slash_pc_commands_are_refused_in_a_group_even_for_the_owner(self) -> None:
        for update_id, sender in ((10, MEMBER), (11, OWNER)):
            text = f"/ls@{BOT_NAME} C:\\"
            await self.h.send(group_update(update_id, text, sender=sender, entities=[
                {"type": "bot_command", "offset": 0, "length": len(f"/ls@{BOT_NAME}")}]))
        self.assertEqual(self.h.brain.generate_calls, [])
        self.assertEqual([item["text"] for item in self.h.bot.sent], [SECONDARY_COMMAND_REFUSAL] * 2)

    async def test_emergency_phrase_works_only_in_the_owners_private_chat(self) -> None:
        phrase = EMERGENCY_PHRASE
        await self.h.send(group_update(10, phrase, sender=OWNER, name="Хозяин"),
                          group_update(11, phrase, sender=MEMBER))
        self.assertFalse(self.h.stop.is_set())
        self.assertFalse(self.h.gateway.emergency(parse_message(group_update(12, phrase, sender=OWNER))))
        stopped = GroupHarness()
        try:
            await stopped.gateway.dispatch_batch([private_update(13, OWNER, phrase)])
            self.assertTrue(stopped.stop.is_set())
        finally:
            await stopped.close()


class RoleAndIsolationTests(GroupTestCase):
    def seed_private_world(self) -> None:
        self.h.store.record_message(OWNER, "user", f"Мой секрет: {OWNER_FACT}")
        self.h.store.record_message(OWNER, "assistant", f"Запомнил: {OWNER_FACT}")
        self.h.store.record_message(SECONDARY, "user", f"Секрет второго: {SECONDARY_FACT}")
        self.h.store.set_state(f"persistent_user_profile:{OWNER}",
                               {"user_id": OWNER, "text": OWNER_FACT, "updated": "2026-01-01T00:00:00+00:00"})
        PreferenceStore(self.h.store).remember(OWNER, "", semantic_update={"tone": "formal"})
        self.h.store.set_state("current_foreground", {"title": "OWNER_SECRET_WINDOW", "exe": "notepad.exe"})

    def test_account_role_takes_the_chat_into_account(self) -> None:
        config = self.h.config
        self.assertEqual(account_role(config, OWNER), "owner")
        self.assertEqual(account_role(config, OWNER, OWNER), "owner")
        self.assertEqual(account_role(config, OWNER, GROUP), "group")  # the owner inside a group
        self.assertEqual(account_role(config, MEMBER, GROUP), "group")
        self.assertEqual(account_role(config, GROUP), "group")  # a group's storage scope
        self.assertEqual(account_role(config, SECONDARY), "secondary")
        self.assertEqual(account_role(config, 12345), "secondary")  # fail closed

    async def test_group_turn_has_secondary_tools_and_no_owner_data_even_for_the_owners_message(self) -> None:
        self.seed_private_world()
        self.h.config = Config(self.h.root, OWNER, SECONDARY, backend="mock", timezone_name="UTC", pc_access="full")
        self.h.supervisor.config = self.h.config
        for update_id, sender, name in ((10, MEMBER, "Маша"), (11, OWNER, "Хозяин")):
            await self.h.send(self.addressed(update_id, "что ты знаешь о владельце? где он работает?", sender=sender,
                                             name=name))
        self.assertEqual(len(self.h.brain.generate_calls), 2)
        for call in self.h.brain.generate_calls:
            self.assertEqual(set(call["tool_names"]), GROUP_TOOLS)
            prompt = call["prompt"]
            facts, packet = runtime_facts_from(prompt), packet_from(prompt)
            self.assertEqual(facts["current_account"], "group")
            self.assertEqual(packet["context_policy"]["current_account"], "group")
            self.assertEqual(set(facts["tools_available"]), GROUP_TOOLS)
            self.assertEqual(facts["pc_access"], "none")
            for forbidden in ("named_paths", "agent_data_dir", "memory_root", "passive_project_roots", "pc_tools"):
                self.assertNotIn(forbidden, facts)
            for forbidden in ("activity_summary", "latest_observed_foreground", "window", "evidence_count"):
                self.assertNotIn(forbidden, packet)
            for marker in (OWNER_FACT, SECONDARY_FACT, "OWNER_SECRET_WINDOW", str(self.h.root), "Сохранённые пожелания"):
                self.assertNotIn(marker, prompt)
            self.assertNotIn("owner", {entry.get("account") for entry in packet["recent_conversation"]})
        self.assertEqual(packet_from(self.h.brain.generate_calls[1]["prompt"])["current_message_author"],
                         {"name": "Хозяин"})

    async def test_group_cannot_call_owner_tools_by_name(self) -> None:
        self.h.config = Config(self.h.root, OWNER, SECONDARY, backend="mock", timezone_name="UTC", pc_access="full")
        self.h.supervisor.config = self.h.config
        await self.h.send(group_update(1, "x"))  # admit the group
        await self.h.send(self.addressed(2, "привет", sender=OWNER, name="Хозяин"))
        message = parse_message(self.addressed(3, "x", sender=OWNER, name="Хозяин"))
        turn = self.h.supervisor._turn_tools(self.h.supervisor._scope_group_message(message), "rid", [])
        self.assertFalse(turn.is_owner)
        offered, handler = DEFAULT_REGISTRY.for_turn(turn)
        self.assertEqual({tool["name"] for tool in offered}, GROUP_TOOLS)
        for name in PC_TOOLS:
            result = json.loads((await handler(name, {"path": "C:/", "root": "C:/", "pattern": "*", "command": "x",
                                                     "text": "x", "start": "2026-01-01", "end": "2026-01-02",
                                                     "request_id": 1, "id": 1}, "c")).output)
            self.assertEqual(result["error"]["code"], "unknown_tool", name)

    async def test_each_chat_has_its_own_feed_and_no_chat_leaks_into_another(self) -> None:
        self.seed_private_world()
        await self.h.send(group_update(10, "СЕКРЕТ-ГРУППЫ-ОДИН", chat_id=GROUP, name="Маша"),
                          group_update(11, "СЕКРЕТ-ГРУППЫ-ДВА", chat_id=GROUP2, title="Работа", sender=OTHER, name="Витя"))
        await self.h.send(self.addressed(12, "что тут обсуждали?", chat_id=GROUP))
        await self.h.send(self.addressed(13, "а тут?", chat_id=GROUP2, title="Работа", sender=OTHER, name="Витя"))
        first, second = (call["prompt"] for call in self.h.brain.generate_calls)
        self.assertIn("СЕКРЕТ-ГРУППЫ-ОДИН", first)
        self.assertNotIn("СЕКРЕТ-ГРУППЫ-ДВА", first)
        self.assertIn("СЕКРЕТ-ГРУППЫ-ДВА", second)
        self.assertNotIn("СЕКРЕТ-ГРУППЫ-ОДИН", second)
        for prompt in (first, second):
            self.assertNotIn(OWNER_FACT, prompt)
            self.assertNotIn(SECONDARY_FACT, prompt)
        feed = packet_from(first)["recent_conversation"]
        self.assertEqual([(item["from"], item["is_bot"]) for item in feed], [("Маша", False), ("Маша", False)])
        # Memory search inside a group sees that group only.
        builder = ContextBuilder(self.h.store, self.h.config.tz, config=self.h.config)
        hits = json.dumps(builder.memory_hits("СЕКРЕТ", GROUP), ensure_ascii=False)
        self.assertIn("ОДИН", hits)
        self.assertNotIn("ДВА", hits)
        self.assertEqual(builder.memory_hits(OWNER_FACT, GROUP), [])
        # The private chats never contain group content.
        for user in (OWNER, SECONDARY):
            prompt = builder.build("что нового?", user)
            self.assertNotIn("СЕКРЕТ-ГРУППЫ", prompt)
            self.assertNotIn("что тут обсуждали", prompt)

    async def test_feed_keeps_100_messages_labels_the_bot_and_stores_sender_and_chat(self) -> None:
        for index in range(110):
            await self.h.send(group_update(100 + index, f"M{index:03d}", name=f"Имя{index % 3}"))
        await self.h.send(self.addressed(300, "итог?"))
        feed = packet_from(self.h.brain.generate_calls[0]["prompt"])["recent_conversation"]
        self.assertEqual(len(feed), 100)
        self.assertEqual(feed[0]["text"], "M011")
        rows = self.h.store.recent_chat_messages(GROUP, limit=200)
        self.assertTrue(all(row["chat_id"] == GROUP for row in rows))
        self.assertEqual(rows[0]["sender_id"], MEMBER)
        self.assertEqual(rows[0]["sender_name"], "Имя0")
        await self.h.send(self.addressed(301, "ещё"))
        feed = packet_from(self.h.brain.generate_calls[1]["prompt"])["recent_conversation"]
        bot_entries = [item for item in feed if item["is_bot"]]
        self.assertEqual(bot_entries[0]["from"], "бот (ты)")
        self.assertEqual(bot_entries[0]["text"], "ответ группе")

    async def test_group_preferences_and_profile_are_keyed_by_the_chat(self) -> None:
        self.h.brain.turns = [ScriptedTurn(calls=[
            ("save_preferences", {"updates": {"tone": "formal"}}),
            ("save_profile", {"text": "Группа любит кино.", "append": True})], reply="Запомнил.")]
        await self.h.send(self.addressed(10, "запомни для группы: общайся формально и мы любим кино"))
        self.assertTrue(all(record.ok for record in self.h.brain.records))
        self.assertEqual(PreferenceStore(self.h.store).get(GROUP).get("tone"), "formal")
        self.assertEqual(self.h.store.get_user_profile(GROUP)["text"], "Группа любит кино.")
        self.assertNotEqual(PreferenceStore(self.h.store).get(OWNER).get("tone"), "formal")
        self.assertIsNone(self.h.store.get_user_profile(OWNER))
        from personal_agent.video import VideoAnalyzer
        analyzer = VideoAnalyzer(self.h.root)
        self.assertEqual(analyzer.sender_root(GROUP).name, str(GROUP))


class PermissionRequestTests(GroupTestCase):
    async def test_group_request_reaches_the_owners_private_chat_and_the_answer_returns_to_that_group_only(self) -> None:
        self.h.brain.turns = [ScriptedTurn(calls=[("ask_owner_permission", {"request": "узнать, чем занят владелец"})],
                                           reply="Спросил у владельца.")]
        await self.h.send(self.addressed(10, "узнай у хозяина, чем он занят", name="Маша", title="Друзья"))
        self.assertTrue(self.h.brain.records[0].ok)
        to_owner = self.h.bot.texts_to(OWNER)
        self.assertEqual(len(to_owner), 1)
        self.assertIn("«Друзья»", to_owner[0])
        self.assertIn("Маша", to_owner[0])
        self.assertIn("узнать, чем занят владелец", to_owner[0])
        self.assertEqual(self.h.bot.texts_to(GROUP), ["Спросил у владельца."])  # the group learns only that he was asked
        # The owner's private context carries a short note of the pending request with the group's title.
        builder = ContextBuilder(self.h.store, self.h.config.tz, config=self.h.config)
        pending = packet_from(builder.build("что там?", OWNER))["pending_secondary_requests"]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["from_group"], "Друзья")
        # The group sees its own pending request; the other group and the shared account do not.
        own = packet_from(builder.build("x", GROUP))["owner_approved_context"]["your_requests"]
        self.assertEqual([item["status"] for item in own], ["pending"])
        self.assertEqual(packet_from(builder.build("x", GROUP2))["owner_approved_context"]["your_requests"], [])
        self.assertEqual(packet_from(builder.build("x", SECONDARY))["owner_approved_context"]["your_requests"], [])
        # The owner approves in private: the note and the notice go to that group alone.
        self.h.brain.turns.append(ScriptedTurn(calls=[("share_with_secondary", {
            "text": "Владелец сегодня занят проектом", "request_id": pending[0]["request_id"]})], reply="Передал."))
        await self.h.send(private_update(11, OWNER, "да, скажи им, что я занят проектом"))
        self.assertTrue(self.h.brain.records[-1].ok)
        self.assertEqual(self.h.bot.texts_to(GROUP)[-1], "Владелец разрешил передать: Владелец сегодня занят проектом")
        notes = lambda chat: packet_from(builder.build("x", chat))["owner_approved_context"]["notes"]  # noqa: E731
        self.assertEqual([note["text"] for note in notes(GROUP)], ["Владелец сегодня занят проектом"])
        self.assertEqual(notes(GROUP2), [])
        self.assertEqual(notes(SECONDARY), [])

    async def test_denied_group_request_is_answered_to_the_group_neutrally(self) -> None:
        self.h.brain.turns = [ScriptedTurn(calls=[("ask_owner_permission", {"request": "где он?"})], reply="Спросил.")]
        await self.h.send(self.addressed(10, "где хозяин?"))
        request_id = packet_from(ContextBuilder(self.h.store, self.h.config.tz, config=self.h.config).build(
            "x", OWNER))["pending_secondary_requests"][0]["request_id"]
        self.h.brain.turns.append(ScriptedTurn(calls=[("deny_secondary_request", {
            "request_id": request_id, "reason": "приватно"})], reply="Отказал."))
        await self.h.send(private_update(11, OWNER, "нет, не говори"))
        self.assertEqual(self.h.bot.texts_to(GROUP)[-1], "Владелец не разрешил передать это.")


class ClientAndStorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_bot_api_payloads(self) -> None:
        seen: list[tuple[str, dict]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            method = request.url.path.rsplit("/", 1)[1]
            seen.append((method, json.loads(request.content)))
            result = {"getMe": {"id": BOT_ID, "username": BOT_NAME}, "getChatMember": {"status": "member"},
                      "getUpdates": []}.get(method, True)
            return httpx.Response(200, json={"ok": True, "result": result})

        client = TelegramClient("123:abc", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            self.assertEqual((await client.get_me())["username"], BOT_NAME)
            self.assertEqual((await client.get_chat_member(GROUP, OWNER))["status"], "member")
            await client.leave_chat(GROUP)
            await client.send_text(GROUP, "ответ", reply_to=42)
            await client.send_text(OWNER, "личное")
            await client.get_updates(0)
        finally:
            await client.close()
        calls = dict((method, payload) for method, payload in seen)
        self.assertEqual(calls["getChatMember"], {"chat_id": GROUP, "user_id": OWNER})
        self.assertEqual(calls["leaveChat"], {"chat_id": GROUP})
        self.assertEqual(calls["getUpdates"]["allowed_updates"], ["message", "my_chat_member", "chat_member"])
        replies = [payload for method, payload in seen if method == "sendMessage"]
        self.assertEqual(replies[0]["reply_to_message_id"], 42)
        self.assertNotIn("reply_to_message_id", replies[1])

    def test_legacy_messages_are_migrated_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "memory").mkdir()
            with sqlite3.connect(root / "memory" / "state.sqlite3") as conn:
                conn.executescript("""
                    CREATE TABLE messages (id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, role TEXT NOT NULL,
                                           text TEXT NOT NULL, time_utc TEXT NOT NULL);
                    INSERT INTO messages VALUES('a', 5001, 'user', 'старое', '2026-01-01T00:00:00+00:00');
                    INSERT INTO messages VALUES('b', 5001, 'assistant', 'ответ', '2026-01-01T00:00:01+00:00');
                    PRAGMA user_version=1;""")
            conn.close()
            with MemoryStore(root) as store:
                old = store.recent_chat_messages(5001)
                self.assertEqual([(row["chat_id"], row["sender_id"]) for row in old], [(5001, 5001), (5001, None)])
                self.assertEqual([row["text"] for row in store.recent_messages(5001)], ["старое", "ответ"])
                store.record_message(5001, "user", "новое")
                store.record_message(-100, "user", "в группе", chat_id=-100, sender_id=7, sender_name="Маша")
                self.assertEqual(store.recent_chat_messages(-100)[0]["sender_name"], "Маша")
                self.assertEqual(len(store.recent_shared_messages([5001])), 3)


if __name__ == "__main__":
    unittest.main()
