"""Native tool registry: definitions per identity, argument validation, bounded results, dispatch."""
from __future__ import annotations

import asyncio
import json
import re
import unittest
from datetime import datetime, timedelta, timezone
from dataclasses import replace

from personal_agent.models import Identity, TimelineEvent
from personal_agent.testing import ScriptedBrain, ScriptedTurn
from personal_agent.tools import DEFAULT_REGISTRY, MAX_TOOL_RESULT_CHARS, ToolRegistry, ToolSpec, encode_result
from tests.test_service_observation_integration import (
    PRIMARY_ID, SECONDARY_ID, ObservationHarness, _message)


READ_TOOLS = {"list_directory", "find_files", "read_file", "read_active_window", "take_screenshot", "read_browser_page",
              "view_recent_media", "read_attachment", "web_search", "web_open", "search_memory",
              "activity_timeline", "save_preferences", "save_profile", "service_diagnostics", "watch_video", "send_file",
              "share_with_secondary", "deny_secondary_request", "list_shared_context", "revoke_shared_note"}
ACTION_TOOLS = {"write_file", "run_command"}
SECONDARY_TOOLS = {"web_search", "web_open", "watch_video", "send_file", "view_recent_media", "read_attachment",
                   "save_preferences", "save_profile", "search_memory", "ask_owner_permission"}


def turn_for(harness, sender_id=PRIMARY_ID):
    message = replace(_message("x"), sender_id=sender_id, chat_id=sender_id)
    return harness.supervisor._turn_tools(message, "rid-1", [])


class ToolDefinitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_definitions_are_valid_responses_function_tools(self):
        harness = ObservationHarness()
        harness.enable_full_access()
        try:
            definitions, _ = DEFAULT_REGISTRY.for_turn(turn_for(harness))
            names = [definition["name"] for definition in definitions]
            self.assertEqual(set(names), READ_TOOLS | ACTION_TOOLS)
            self.assertEqual(len(names), len(set(names)))
            for definition in definitions:
                self.assertEqual(definition["type"], "function")
                self.assertTrue(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", definition["name"]))
                self.assertIs(definition["strict"], False)
                self.assertGreater(len(definition["description"]), 60)
                schema = definition["parameters"]
                self.assertEqual(schema["type"], "object")
                self.assertIs(schema["additionalProperties"], False)
                self.assertTrue(set(schema.get("required", [])) <= set(schema["properties"]))
                json.dumps(definition)  # serializable for the gateway
        finally:
            await harness.close()

    async def test_availability_depends_only_on_identity_and_configuration(self):
        harness = ObservationHarness()
        try:
            def offered(sender):
                return {d["name"] for d in DEFAULT_REGISTRY.for_turn(turn_for(harness, sender))[0]}

            self.assertEqual(offered(PRIMARY_ID), READ_TOOLS)  # pc_access=read: nobody gets actions
            self.assertEqual(offered(SECONDARY_ID), SECONDARY_TOOLS)  # never the owner's PC
            harness.enable_full_access()
            self.assertEqual(offered(PRIMARY_ID), READ_TOOLS | ACTION_TOOLS)
            self.assertEqual(offered(SECONDARY_ID), SECONDARY_TOOLS)
            self.assertFalse({"take_screenshot", "read_active_window", "list_directory"} & offered(SECONDARY_ID))
        finally:
            await harness.close()

    def test_duplicate_tool_names_are_refused(self):
        async def noop(turn, args):
            return {}

        spec = ToolSpec("same", "d" * 70, {"type": "object", "properties": {}}, noop)
        with self.assertRaises(ValueError):
            ToolRegistry([spec, spec])


class ToolDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_malformed_and_failing_calls_return_safe_error_codes(self):
        harness = ObservationHarness()
        try:
            turn = turn_for(harness)

            async def boom(turn_, args):
                raise RuntimeError("Bearer SECRET_TOKEN_9 leaked in message")

            registry = ToolRegistry([ToolSpec("boom", "d" * 70, {"type": "object", "properties": {}}, boom)])
            _, handler = registry.for_turn(turn)
            unknown = json.loads((await handler("nope", {}, "c1")).output)
            self.assertEqual(unknown["error"]["code"], "unknown_tool")
            bad = json.loads((await handler("boom", ["not", "an", "object"], "c2")).output)
            self.assertEqual(bad["error"]["code"], "invalid_arguments")
            failed = (await handler("boom", {}, "c3")).output
            self.assertEqual(json.loads(failed)["error"], {"code": "tool_failed", "type": "RuntimeError"})
            self.assertNotIn("SECRET_TOKEN_9", failed)
        finally:
            await harness.close()

    async def test_cancellation_inside_a_tool_propagates(self):
        harness = ObservationHarness()
        try:
            started = asyncio.Event()

            async def wait_forever(turn_, args):
                started.set()
                await asyncio.Event().wait()

            registry = ToolRegistry([ToolSpec("wait", "d" * 70, {"type": "object", "properties": {}}, wait_forever)])
            _, handler = registry.for_turn(turn_for(harness))
            task = asyncio.create_task(handler("wait", {}, "c1"))
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            await harness.close()

    def test_results_are_bounded_valid_json_with_truncation_marker(self):
        small = encode_result({"ok": True, "text": "x"})
        self.assertEqual(json.loads(small), {"ok": True, "text": "x"})
        huge = {"ok": True, "kind": "read_file", "data": {"text": "я" * 400_000}, "items": ["y" * 5000] * 500}
        encoded = encode_result(huge)
        self.assertLessEqual(len(encoded), MAX_TOOL_RESULT_CHARS)
        decoded = json.loads(encoded)
        self.assertTrue(decoded["output_truncated"])
        self.assertEqual(decoded["kind"], "read_file")

    async def test_list_directory_happy_path_and_arguments(self):
        harness = ObservationHarness()
        folder = harness.root / "listing"
        folder.mkdir()
        (folder / "alpha.txt").write_text("a", encoding="utf-8")
        (folder / "sub").mkdir()
        brain = ScriptedBrain([ScriptedTurn(calls=[
            ("list_directory", {"path": str(folder)}),
            ("list_directory", {}),
            ("list_directory", {"path": str(folder), "recursive": True}),
            ("list_directory", {"path": ""}),
            ("read_file", {"path": str(folder / "alpha.txt")}),
            ("read_file", {"path": str(folder)}),
        ])])
        harness.brain = harness.supervisor.brain = brain
        try:
            await harness.supervisor.submit(_message("что в папке?"), Identity.PRIMARY)
            await harness.wait_idle()
            listing = brain.records[0].data
            self.assertTrue(listing["ok"])
            self.assertEqual({entry["name"]: entry["type"] for entry in listing["data"]["entries"]},
                             {"alpha.txt": "file", "sub": "directory"})
            self.assertEqual([record.error_code for record in brain.records[1:4]], ["invalid_arguments"] * 3)
            self.assertEqual(brain.records[4].data["data"]["text"], "a")
            self.assertFalse(brain.records[5].ok)  # a directory is not a readable file
        finally:
            await harness.close()


class MemoryToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_memory_returns_only_this_senders_hits(self):
        harness = ObservationHarness()
        harness.store.record_message(PRIMARY_ID, "user", "ZEBRAMARKER my private plan")
        harness.store.record_message(SECONDARY_ID, "user", "ZEBRAMARKER secondary secret plan")
        brain = ScriptedBrain([ScriptedTurn(calls=[("search_memory", {"query": "ZEBRAMARKER"})]),
                               ScriptedTurn(calls=[("search_memory", {"query": "ZEBRAMARKER"}), ("search_memory", {})])])
        harness.brain = harness.supervisor.brain = brain
        try:
            await harness.supervisor.submit(_message("найди в памяти"), Identity.PRIMARY)
            await harness.wait_idle()
            primary = brain.records[0].outcome.output
            self.assertIn("my private plan", primary)
            self.assertNotIn("secondary secret", primary)
            secondary_message = replace(_message("найди", update_id=2), sender_id=SECONDARY_ID, chat_id=SECONDARY_ID)
            await harness.supervisor.submit(secondary_message, Identity.SECONDARY)
            await harness.wait_idle()
            secondary = brain.records[1].outcome.output
            self.assertIn("secondary secret", secondary)
            self.assertNotIn("my private plan", secondary)
            self.assertEqual(brain.records[2].error_code, "invalid_arguments")
        finally:
            await harness.close()

    async def test_activity_timeline_is_bounded_owner_filtered_and_validated(self):
        harness = ObservationHarness()
        now = datetime.now(timezone.utc)
        harness.store.record_event(TimelineEvent("windows", "window.foreground",
                                                 {"exe": "Code.exe", "title": "ZEBRA_PROJECT_WINDOW"},
                                                 "proj", now - timedelta(hours=30)))
        harness.store.record_event(TimelineEvent("filesystem", "USER_REQUESTED_READ",
                                                 {"sender_id": SECONDARY_ID, "text": "OTHERS_PRIVATE_READ"},
                                                 None, now - timedelta(hours=30)))
        start = (now - timedelta(hours=40)).isoformat()
        end = (now - timedelta(hours=20)).isoformat()
        brain = ScriptedBrain([ScriptedTurn(calls=[
            ("activity_timeline", {"start": start, "end": end}),
            ("activity_timeline", {"start": (now - timedelta(days=30)).isoformat(), "end": end}),
            ("activity_timeline", {"start": end, "end": start}),
            ("activity_timeline", {"start": "вчера", "end": end}),
            ("activity_timeline", {"start": start}),
            ("activity_timeline", {"start": "2026-10-27T09:00:00", "end": "2026-10-27T10:00:00"}),
        ])])
        harness.brain = harness.supervisor.brain = brain
        try:
            await harness.supervisor.submit(_message("что я делал вчера?"), Identity.PRIMARY)
            await harness.wait_idle()
            first = brain.records[0].data
            self.assertTrue(first["ok"])
            self.assertEqual(first["evidence_count"], 1)
            self.assertIn("Code.exe", first["summary"] + brain.records[0].outcome.output)
            self.assertNotIn("OTHERS_PRIVATE_READ", brain.records[0].outcome.output)
            clamped = brain.records[1].data
            self.assertTrue(clamped["clamped_to_7_days"])
            span = datetime.fromisoformat(clamped["end"]) - datetime.fromisoformat(clamped["start"])
            self.assertEqual(span, timedelta(days=7))
            self.assertEqual(brain.records[2].error_code, "invalid_window")
            self.assertEqual(brain.records[3].error_code, "invalid_datetime")
            self.assertEqual(brain.records[4].error_code, "invalid_arguments")
            self.assertTrue(brain.records[5].ok)  # naive datetimes are interpreted in the configured timezone
            # The prompt itself only carries the fixed last-two-hours summary.
            packet_window = brain.generate_calls[0]["prompt"].split('"window": ', 1)[1].split("}", 1)[0]
            self.assertIn(now.strftime("%Y-%m-%d"), packet_window)
        finally:
            await harness.close()


if __name__ == "__main__":
    unittest.main()
