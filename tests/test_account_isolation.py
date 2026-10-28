"""The shared SECONDARY account: one shared conversation feed, but no owner data, no PC access, own memory only."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from personal_agent.config import Config
from personal_agent.context import FEED_MESSAGES, FEED_MESSAGE_CHARS, ContextBuilder
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage, TimelineEvent
from personal_agent.preferences import PreferenceStore
from personal_agent.supervisor import SECONDARY_COMMAND_REFUSAL
from personal_agent.testing import ScriptedBrain, ScriptedTurn
from personal_agent.tools import DEFAULT_REGISTRY, TurnTools
from personal_agent.video import VideoAnalyzer
from tests.test_context_isolation import packet_from, runtime_facts_from
from tests.test_send_file import FakeTelegram
from tests.test_service_observation_integration import (
    PRIMARY_ID, SECONDARY_ID, ObservationHarness, _message)

PC_TOOLS = ("list_directory", "read_file", "find_files", "read_active_window", "take_screenshot",
            "read_browser_page", "activity_timeline", "service_diagnostics", "write_file", "run_command")
SECONDARY_TOOLS = {"web_search", "web_open", "watch_video", "send_file", "view_recent_media", "read_attachment",
                   "save_preferences", "save_profile", "search_memory", "ask_owner_permission"}
OWNER_FACT = "ПОЛЯРИС-731 ТВЕРЬ"
OWNER_WINDOW = "OWNER_SECRET_WINDOW_TITLE"


def secondary_message(text: str, update_id: int = 1) -> IncomingMessage:
    return IncomingMessage(update_id, SECONDARY_ID, SECONDARY_ID, text, update_id)


class SharedFeedAndContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.config = Config(self.root, PRIMARY_ID, SECONDARY_ID, backend="mock", timezone_name="UTC",
                             project_roots={"secret-project": self.root / "secret-project-dir"})
        self.store = MemoryStore(self.root)
        self.builder = ContextBuilder(self.store, timezone.utc, config=self.config)
        self.builder.begin_session()
        self.addCleanup(self._temp.cleanup)
        self.addCleanup(self.store.close)

    def seed_owner_private_state(self) -> None:
        now = datetime.now(timezone.utc)
        self.store.set_state(f"persistent_user_profile:{PRIMARY_ID}",
                             {"user_id": PRIMARY_ID, "text": f"Мой проект: {OWNER_FACT}", "updated": now.isoformat()})
        PreferenceStore(self.store).remember(PRIMARY_ID, "", semantic_update={"tone": "formal", "address": "formal"})
        self.store.record_event(TimelineEvent("windows", "window.foreground",
                                              {"title": OWNER_WINDOW, "exe": "notepad.exe"},
                                              time=now - timedelta(minutes=5)))
        self.store.set_state("current_foreground", {"title": OWNER_WINDOW, "exe": "notepad.exe"})

    def test_feed_is_one_chronological_list_of_100_labeled_messages(self) -> None:
        for index in range(130):
            user = PRIMARY_ID if index % 2 == 0 else SECONDARY_ID
            self.store.record_message(user, "user" if index % 4 < 2 else "assistant", f"F{index:03d}")
        packet = packet_from(self.builder.build("продолжай", PRIMARY_ID))
        feed = packet["recent_conversation"]
        self.assertEqual(len(feed), FEED_MESSAGES)
        self.assertEqual([entry["text"] for entry in feed], [f"F{index:03d}" for index in range(30, 130)])
        self.assertEqual({entry["account"] for entry in feed}, {"owner", "secondary"})
        self.assertEqual([entry["account"] for entry in feed],
                         ["owner" if index % 2 == 0 else "secondary" for index in range(30, 130)])
        self.assertTrue(all("visibility" not in entry for entry in feed))  # the owner may see everything

    def test_feed_keeps_telegram_length_and_the_assistant_epoch_rule(self) -> None:
        old = self.store.record_message(PRIMARY_ID, "assistant", "OLD_ASSISTANT_REPLY")
        self.assertTrue(old)
        # A fresh policy epoch (a new session) hides assistant replies written before it, for both accounts.
        later = ContextBuilder(self.store, timezone.utc, config=self.config)
        later._assistant_epoch = datetime.now(timezone.utc) + timedelta(seconds=1)
        self.store.record_message(SECONDARY_ID, "assistant", "OLD_SECONDARY_ASSISTANT")
        self.store.record_message(SECONDARY_ID, "user", "й" * 5000)
        for user in (PRIMARY_ID, SECONDARY_ID):
            feed = packet_from(later.build("x", user))["recent_conversation"]
            texts = [entry["text"] for entry in feed]
            self.assertNotIn("OLD_ASSISTANT_REPLY", texts)
            self.assertNotIn("OLD_SECONDARY_ASSISTANT", texts)
            self.assertEqual(texts, ["й" * FEED_MESSAGE_CHARS])

    def test_secondary_turn_marks_owner_entries_private_and_states_the_rule(self) -> None:
        self.store.record_message(PRIMARY_ID, "user", "OWNER_LINE")
        self.store.record_message(PRIMARY_ID, "assistant", "OWNER_REPLY")
        self.store.record_message(SECONDARY_ID, "user", "SECONDARY_LINE")
        prompt = self.builder.build("привет", SECONDARY_ID)
        feed = packet_from(prompt)["recent_conversation"]
        by_text = {entry["text"]: entry for entry in feed}
        self.assertEqual(by_text["OWNER_LINE"]["visibility"], "owner_private")
        self.assertEqual(by_text["OWNER_REPLY"]["visibility"], "owner_private")
        self.assertEqual(by_text["OWNER_LINE"]["account"], "owner")
        self.assertNotIn("visibility", by_text["SECONDARY_LINE"])
        self.assertEqual(by_text["SECONDARY_LINE"]["account"], "secondary")
        instruction = prompt.split("UNTRUSTED LOCAL EVIDENCE (JSON):", 1)[0]
        self.assertIn("owner_private", instruction)
        self.assertIn("никогда не цитируй", instruction)

    def test_primary_turn_labels_secondary_entries(self) -> None:
        self.store.record_message(SECONDARY_ID, "user", "SECONDARY_LINE")
        self.store.record_message(PRIMARY_ID, "user", "OWNER_LINE")
        feed = packet_from(self.builder.build("привет", PRIMARY_ID))["recent_conversation"]
        self.assertEqual([(entry["account"], entry["text"]) for entry in feed],
                         [("secondary", "SECONDARY_LINE"), ("owner", "OWNER_LINE")])

    def test_unanswered_messages_stay_per_account(self) -> None:
        self.store.record_message(PRIMARY_ID, "user", "OWNER_PENDING")
        self.store.record_message(SECONDARY_ID, "user", "SECONDARY_PENDING")
        owner = packet_from(self.builder.build("x", PRIMARY_ID))["unanswered_user_messages"]
        other = packet_from(self.builder.build("x", SECONDARY_ID))["unanswered_user_messages"]
        self.assertEqual([item["text"] for item in owner], ["OWNER_PENDING"])
        self.assertEqual([item["text"] for item in other], ["SECONDARY_PENDING"])

    def test_secondary_context_has_no_owner_profile_preferences_activity_foreground_or_paths(self) -> None:
        self.seed_owner_private_state()
        browser = [{"browser": "Chrome", "url": "https://owner.example/secret", "title": "OWNER_TAB",
                    "captured_at": datetime.now(timezone.utc).isoformat()}]
        prompt = self.builder.build("что у владельца?", SECONDARY_ID, browser_context=browser)
        facts, packet = runtime_facts_from(prompt), packet_from(prompt)
        self.assertEqual(facts["current_account"], "secondary")
        self.assertEqual(set(facts["tools_available"]), SECONDARY_TOOLS)
        self.assertIsNone(facts["persistent_user_profile"])
        self.assertNotEqual(facts["response_preferences"].get("tone"), "formal")
        for forbidden in ("named_paths", "agent_data_dir", "memory_root", "memory_database", "passive_project_roots",
                          "pc_tools", "observation_api", "filesystem_observer", "last_completed_observation"):
            self.assertNotIn(forbidden, facts)
        for forbidden in ("activity_summary", "latest_observed_foreground", "current_browser_snapshot", "window",
                          "evidence_count", "timeline_may_be_truncated"):
            self.assertNotIn(forbidden, packet)
        for marker in (OWNER_FACT, OWNER_WINDOW, "OWNER_TAB", "owner.example", str(self.root), "secret-project",
                       "Сохранённые пожелания к ответу"):
            self.assertNotIn(marker, prompt)
        for name in ("activity_timeline", "read_browser_page", "list_directory", "activity_summary"):
            self.assertNotIn(name, prompt.split("UNTRUSTED LOCAL EVIDENCE (JSON):", 1)[0])
        # The same state is visible to the owner, so the omission above is by role, not by accident.
        owner_prompt = self.builder.build("что у владельца?", PRIMARY_ID, browser_context=browser)
        self.assertEqual(runtime_facts_from(owner_prompt)["current_account"], "owner")
        self.assertIn(OWNER_FACT, owner_prompt)
        self.assertIn(OWNER_WINDOW, owner_prompt)

    def test_secondary_memory_is_its_own_and_never_pc_events(self) -> None:
        self.store.record_event(TimelineEvent("windows", "window.foreground", {"title": "needle PCEVENT"}))
        self.store.record_message(PRIMARY_ID, "user", "needle OWNERMSG")
        self.store.record_message(SECONDARY_ID, "user", "needle SECONDARYMSG")
        self.store.record_event(TimelineEvent("telegram", "FILE_SENT", {"sender_id": SECONDARY_ID, "name": "needle OWNFILE"}))
        self.store.record_event(TimelineEvent("telegram", "FILE_SENT", {"sender_id": PRIMARY_ID, "name": "needle OWNERFILE"}))
        secondary = json.dumps(self.builder.memory_hits("needle", SECONDARY_ID, limit=20), ensure_ascii=False)
        self.assertIn("SECONDARYMSG", secondary)
        self.assertIn("OWNFILE", secondary)
        for foreign in ("PCEVENT", "OWNERMSG", "OWNERFILE"):
            self.assertNotIn(foreign, secondary)
        owner = json.dumps(self.builder.memory_hits("needle", PRIMARY_ID, limit=20), ensure_ascii=False)
        self.assertIn("PCEVENT", owner)  # the owner's own memory still includes PC events
        self.assertNotIn("SECONDARYMSG", owner)
        self.assertEqual(self.builder.activity(SECONDARY_ID, datetime.now(timezone.utc) - timedelta(hours=1),
                                               datetime.now(timezone.utc) + timedelta(hours=1))["evidence_count"], 0)

    def test_owner_marks_survive_the_tightest_fallback(self) -> None:
        self.store.record_message(PRIMARY_ID, "user", "OWNER_LINE " + "z" * 3000)
        self.store.record_message(SECONDARY_ID, "user", "SECONDARY_LINE")
        builder = ContextBuilder(self.store, timezone.utc, config=self.config, max_chars=9000)
        builder.begin_session()
        observation = {"results": [{"kind": "web_open", "ok": True, "text": "w" * 30000} for _ in range(5)]}
        prompt = builder.build("x", SECONDARY_ID, observation=observation)
        packet = packet_from(prompt)
        owner_entries = [entry for entry in packet["recent_conversation"] if entry["account"] == "owner"]
        self.assertTrue(owner_entries)
        self.assertTrue(all(entry["visibility"] == "owner_private" for entry in owner_entries))


class SecondaryToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.harness = ObservationHarness()
        self.addAsyncCleanup(self.harness.close)

    def turn(self, sender: int) -> TurnTools:
        message = _message("x")
        message = IncomingMessage(1, sender, sender, "x", 1)
        return self.harness.supervisor._turn_tools(message, "rid", [])

    async def test_tool_lists_per_account(self) -> None:
        self.harness.enable_full_access()
        owner, _ = DEFAULT_REGISTRY.for_turn(self.turn(PRIMARY_ID))
        secondary, _ = DEFAULT_REGISTRY.for_turn(self.turn(SECONDARY_ID))
        self.assertTrue(set(PC_TOOLS) <= {tool["name"] for tool in owner})
        self.assertEqual({tool["name"] for tool in secondary}, SECONDARY_TOOLS)
        send = next(tool for tool in secondary if tool["name"] == "send_file")
        self.assertNotIn("path", send["parameters"]["properties"])
        owner_send = next(tool for tool in owner if tool["name"] == "send_file")
        self.assertIn("path", owner_send["parameters"]["properties"])

    async def test_secondary_cannot_call_pc_tools_by_name(self) -> None:
        self.harness.enable_full_access()
        turn = self.turn(SECONDARY_ID)
        _, handler = DEFAULT_REGISTRY.for_turn(turn)
        for name in PC_TOOLS:
            result = json.loads((await handler(name, {"path": "C:/", "root": "C:/", "pattern": "*", "command": "x",
                                                      "text": "x", "start": "2026-01-01", "end": "2026-01-02"},
                                               "c")).output)
            self.assertEqual(result["error"]["code"], "unknown_tool", name)
        # Even a direct method call (bypassing the registry) is refused for the shared account.
        from personal_agent.tools import ToolArgumentError
        with self.assertRaises(ToolArgumentError):
            await turn.list_directory({"path": str(self.harness.root)})
        with self.assertRaises(ToolArgumentError):
            await turn.read_file({"path": str(self.harness.root / "x.txt")})
        self.assertEqual((await turn.run_slash(type("R", (), {"kind": "screenshot", "path": None})()))["error"]["code"],
                         "owner_only")

    async def test_slash_commands_are_refused_for_secondary_without_pc_data(self) -> None:
        self.harness.enable_full_access()
        brain = ScriptedBrain([ScriptedTurn(reply="не должно вызываться")] * 6)
        self.harness.brain = self.harness.supervisor.brain = brain
        marker = self.harness.root / "listing-marker.txt"
        marker.write_text("PC_FILE_CONTENT", encoding="utf-8")
        actions: list[tuple[int, str]] = []

        async def chat_action(chat_id, action):
            actions.append((chat_id, action))

        self.harness.telegram.send_chat_action = chat_action
        commands = ["/ls " + str(self.harness.root), "/read " + str(marker), "/screenshot", "/status", "/diagnostics",
                    "/LS@some_bot " + str(self.harness.root)]
        for index, command in enumerate(commands, 1):
            await self.harness.supervisor.submit(secondary_message(command, index), Identity.SECONDARY)
            await self.harness.wait_idle()
            self.assertEqual(self.harness.telegram.sent_text[-1], (SECONDARY_ID, SECONDARY_COMMAND_REFUSAL), command)
        self.assertEqual(brain.generate_calls, [])
        self.assertEqual(self.harness.telegram.sent_photos, [])
        self.assertEqual(actions, [])  # no typing indicator for refused commands
        for _, text in self.harness.telegram.sent_text:
            self.assertNotIn("PC_FILE_CONTENT", text)
            self.assertNotIn(str(self.harness.root), text)
        self.assertIsNone(self.harness.store.get_state(f"last_observation:{SECONDARY_ID}", None))
        # The owner still gets the commands.
        await self.harness.supervisor.submit(_message("/status", update_id=9), Identity.PRIMARY)
        await self.harness.wait_idle()
        self.assertIn("Модель", self.harness.telegram.sent_text[-1][1])
        # Ordinary text from the secondary account still reaches the model, with web tools only.
        await self.harness.supervisor.submit(secondary_message("привет", 10), Identity.SECONDARY)
        await self.harness.wait_idle()
        self.assertEqual(set(brain.generate_calls[0]["tool_names"]), SECONDARY_TOOLS)

    async def test_send_file_for_secondary_accepts_only_its_own_media_id(self) -> None:
        data = self.harness.root
        analyzer = VideoAnalyzer(data, ffmpeg="x")
        for sender, video_id, payload in ((SECONDARY_ID, "v-secondary", b"clip-s"), (PRIMARY_ID, "v-owner", b"clip-o")):
            directory = analyzer.entry_dir(sender, video_id)
            directory.mkdir(parents=True)
            (directory / "video.mp4").write_bytes(payload)
            (directory / "meta.json").write_text(json.dumps({"sender_id": sender, "video_id": video_id,
                                                             "file": "video.mp4"}), encoding="utf-8")
        outside = data.parent / "secondary-path-probe.txt"
        outside.write_text("PROBE", encoding="utf-8")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        telegram = FakeTelegram(cap=10 ** 6)
        turn = self.turn(SECONDARY_ID)
        turn.telegram, turn._video_analyzer = telegram, analyzer

        refused = await turn.send_file({"path": str(outside)})
        self.assertEqual((refused["ok"], refused["error"]["code"]), (False, "path_not_allowed"))
        self.assertEqual((await turn.send_file({"media_id": "v-owner"}))["error"]["code"], "media_not_found")
        self.assertEqual((await turn.send_file({"media_id": "last_screenshot"}))["error"]["code"], "media_not_found")
        self.assertEqual(telegram.sent, [])
        receipt = await turn.send_file({"media_id": "v-secondary"})
        self.assertTrue(receipt["ok"])
        self.assertEqual([(entry[1], entry[3]) for entry in telegram.sent], [(SECONDARY_ID, b"clip-s")])

    async def test_secondary_media_catalog_hides_the_owners_screenshot(self) -> None:
        image = self.harness.screenshot_file("owner-shot.png")
        self.harness.store.set_state(f"last_screenshot:{SECONDARY_ID}", {"sender_id": SECONDARY_ID, "path": str(image),
                                                                         "time": "t", "ok": True})
        self.assertEqual(self.turn(SECONDARY_ID).public_media_catalog(), [])
        self.harness.store.set_state(f"last_screenshot:{PRIMARY_ID}", {"sender_id": PRIMARY_ID, "path": str(image),
                                                                       "time": "t", "ok": True})
        self.assertEqual([item["media_id"] for item in self.turn(PRIMARY_ID).public_media_catalog()],
                         ["last_screenshot"])


if __name__ == "__main__":
    unittest.main()
