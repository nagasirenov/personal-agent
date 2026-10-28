from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from personal_agent.backends import (CodexChatGPTWebBackend, CodexSubscriptionBackend, MockBackend,
                                     resolve_codex_command)
from personal_agent.brain_gateway import BrainRequestGateway, ToolOutcome

_CLIENT = httpx.AsyncClient
_PNG = b"\x89PNG\r\n\x1a\nTOOL_IMAGE_FIXTURE"
_HOST = {"instructions": "HOST_AGENTS_AND_COORDINATOR_PRIVATE", "tools": [{"type": "function", "name": "shell"}],
         "input": [{"role": "developer", "content": "HOST_CONTEXT"}], "model": "host-model"}
TOOLS = [
    {"type": "function", "name": "lookup", "description": "Look up a value.",
     "parameters": {"type": "object", "properties": {"key": {"type": "string"}}}, "strict": False},
    {"type": "function", "name": "snap", "description": "Take an image.",
     "parameters": {"type": "object", "properties": {}}, "strict": False},
]


def sse(items: list[dict], *, with_completed_output: bool = True) -> bytes:
    events = [{"type": "response.created", "response": {"id": "resp_1"}}]
    events += [{"type": "response.output_item.done", "item": item} for item in items]
    events.append({"type": "response.completed", "response": {
        "id": "resp_1", "output": items if with_completed_output else []}})
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def reasoning(n: int = 1) -> dict:
    return {"type": "reasoning", "id": f"rs_{n}", "summary": [], "encrypted_content": f"ENC_{n}"}


def call(call_id: str, name: str = "lookup", arguments: str = '{"key": "alpha"}') -> dict:
    return {"type": "function_call", "id": f"fc_{call_id}", "status": "completed", "call_id": call_id,
            "name": name, "arguments": arguments}


def message(text: str) -> dict:
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


class ToolLoopTests(unittest.IsolatedAsyncioTestCase):
    def gateway(self, **kwargs) -> BrainRequestGateway:
        kwargs.setdefault("upstream_url", "http://127.0.0.1:1/responses")
        kwargs.setdefault("require_auth", False)
        return BrainRequestGateway(instructions="PERSONAL_AGENT_PERSONA", prompt="PERSONAL_AGENT_PROMPT",
                                   model="gpt-6-luna", reasoning_effort="medium", **kwargs)

    def upstream(self, responses):
        """responses: list of bytes/Response or a callable(body)->bytes/Response; records request bodies."""
        self.bodies: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.bodies.append(body)
            result = responses(body) if callable(responses) else responses[len(self.bodies) - 1]
            if isinstance(result, httpx.Response):
                return result
            return httpx.Response(200, content=result, headers={"Content-Type": "text/event-stream"})

        def factory(**kwargs):
            return _CLIENT(transport=httpx.MockTransport(handler), **kwargs)
        return patch("personal_agent.brain_gateway.httpx.AsyncClient", side_effect=factory)

    async def post(self, gateway) -> httpx.Response:
        async with _CLIENT(trust_env=False, timeout=10) as caller:
            return await caller.post(gateway.base_url + "/responses", json=_HOST)

    async def test_one_round_then_final_replays_items_without_ids(self):
        final = sse([message("The value is 4721-ZETA.")])
        seen = []

        async def handler(name, arguments, call_id):
            seen.append((name, arguments, call_id))
            return ToolOutcome('{"ok":true,"value":"4721-ZETA"}')

        with self.upstream([sse([reasoning(), call("call_1")]), final]):
            async with self.gateway(tools=TOOLS, tool_handler=handler) as gateway:
                response = await self.post(gateway)
                self.assertEqual(gateway.upstream_requests, 2)
                self.assertEqual(gateway.request_count, 1)
                self.assertIsNone(gateway.last_error_code)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream")
        self.assertEqual(response.content, final)
        self.assertEqual(seen, [("lookup", {"key": "alpha"}, "call_1")])
        first, second = self.bodies
        for body in (first, second):
            self.assertEqual(body["tools"], TOOLS)
            self.assertEqual(body["tool_choice"], "auto")
            self.assertIs(body["parallel_tool_calls"], True)
            self.assertEqual(body["include"], ["reasoning.encrypted_content"])
            self.assertIs(body["stream"], True)
            self.assertIs(body["store"], False)
            self.assertEqual(body["instructions"], "PERSONAL_AGENT_PERSONA")
            self.assertEqual(body["model"], "gpt-6-luna")
            self.assertNotIn("HOST_", json.dumps(body))
            self.assertNotIn("AGENTS", json.dumps(body).replace("PERSONAL_AGENT", ""))
        user = {"role": "user", "content": [{"type": "input_text", "text": "PERSONAL_AGENT_PROMPT"}]}
        self.assertEqual(first["input"], [user])
        self.assertEqual(second["input"], [
            user,
            {"type": "reasoning", "summary": [], "encrypted_content": "ENC_1"},
            {"type": "function_call", "call_id": "call_1", "name": "lookup",
             "arguments": '{"key": "alpha"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": '{"ok":true,"value":"4721-ZETA"}'},
        ])

    async def test_parallel_calls_all_answered_in_order(self):
        order = []

        async def handler(name, arguments, call_id):
            order.append(call_id)
            return ToolOutcome(json.dumps({"id": call_id, "key": arguments["key"]}))

        first = sse([reasoning(), call("c1", arguments='{"key":"a"}'), call("c2", arguments='{"key":"b"}')])
        with self.upstream([first, sse([message("done")])]):
            async with self.gateway(tools=TOOLS, tool_handler=handler) as gateway:
                response = await self.post(gateway)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(order, ["c1", "c2"])
        tail = self.bodies[1]["input"][1:]
        self.assertEqual([i["type"] for i in tail], ["reasoning", "function_call", "function_call",
                                                     "function_call_output", "function_call_output"])
        self.assertEqual([i["call_id"] for i in tail[-2:]], ["c1", "c2"])
        self.assertEqual(json.loads(tail[-1]["output"]), {"id": "c2", "key": "b"})

    async def test_output_items_fall_back_to_item_done_events(self):
        async def handler(name, arguments, call_id):
            return ToolOutcome("{}")
        first = sse([reasoning(), call("c1")], with_completed_output=False)
        with self.upstream([first, sse([message("ok")])]):
            async with self.gateway(tools=TOOLS, tool_handler=handler) as gateway:
                await self.post(gateway)
        self.assertEqual([i["type"] for i in self.bodies[1]["input"][1:]],
                         ["reasoning", "function_call", "function_call_output"])

    async def test_tool_image_follows_outputs_as_user_message(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "shot.png"
            image.write_bytes(_PNG)
            bad = Path(temp) / "bad.png"
            bad.write_bytes(b"not an image")

            async def handler(name, arguments, call_id):
                return ToolOutcome('{"ok":true}', [image, bad])

            with self.upstream([sse([call("c9", name="snap", arguments="{}")]), sse([message("red")])]):
                async with self.gateway(tools=TOOLS, tool_handler=handler) as gateway:
                    response = await self.post(gateway)
        self.assertEqual(response.status_code, 200)
        tail = self.bodies[1]["input"][1:]
        self.assertEqual([i.get("type") or i.get("role") for i in tail],
                         ["function_call", "function_call_output", "user"])
        content = tail[-1]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text":
                                      "Images returned by tool snap (call c9). 1 image(s) could not be attached."})
        self.assertEqual(content[1], {"type": "input_image", "image_url":
                                      "data:image/png;base64," + base64.b64encode(_PNG).decode()})
        self.assertEqual(len(content), 2)

    async def test_handler_failures_become_json_errors_without_leaking(self):
        async def handler(name, arguments, call_id):
            raise RuntimeError("SECRET_TRACEBACK_DETAIL")

        calls = [call("c1"), call("c2", arguments="not json"), call("c3", name="shell"),
                 call("c4", arguments="[1]")]
        with self.upstream([sse(calls), sse([message("fine")])]):
            async with self.gateway(tools=TOOLS, tool_handler=handler) as gateway:
                response = await self.post(gateway)
                self.assertIsNone(gateway.last_error_code)
        self.assertEqual(response.status_code, 200)
        outputs = {i["call_id"]: json.loads(i["output"]) for i in self.bodies[1]["input"]
                   if i.get("type") == "function_call_output"}
        self.assertEqual(outputs, {
            "c1": {"ok": False, "error": {"code": "tool_failed"}},
            "c2": {"ok": False, "error": {"code": "invalid_arguments"}},
            "c3": {"ok": False, "error": {"code": "unknown_tool"}},
            "c4": {"ok": False, "error": {"code": "invalid_arguments"}},
        })
        self.assertNotIn("SECRET_TRACEBACK_DETAIL", json.dumps(self.bodies))
        self.assertNotIn("Traceback", json.dumps(self.bodies))

    async def test_handler_timeout_and_bad_return_value(self):
        async def slow(name, arguments, call_id):
            if call_id == "c2":
                return object()
            await asyncio.sleep(10)

        with self.upstream([sse([call("c1"), call("c2")]), sse([message("x")])]):
            gateway = self.gateway(tools=TOOLS, tool_handler=slow)
            gateway.TOOL_TIMEOUT = 0.05
            async with gateway:
                await self.post(gateway)
        outputs = {i["call_id"]: json.loads(i["output"])["error"]["code"] for i in self.bodies[1]["input"]
                   if i.get("type") == "function_call_output"}
        self.assertEqual(outputs, {"c1": "tool_timeout", "c2": "tool_output_invalid"})

    async def test_round_limit_forces_final_answer_with_tool_choice_none(self):
        async def handler(name, arguments, call_id):
            return ToolOutcome('{"ok":true}')

        counter = iter(range(100))

        def responses(body):
            if body.get("tool_choice") == "none":
                return sse([message("answer from what I have")])
            return sse([reasoning(), call(f"c{next(counter)}")])

        with self.upstream(responses):
            gateway = self.gateway(tools=TOOLS, tool_handler=handler)
            gateway.MAX_TOOL_ROUNDS = 2
            async with gateway:
                response = await self.post(gateway)
                self.assertEqual(gateway.upstream_requests, 3)
                self.assertEqual(gateway.tool_call_count, 2)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"answer from what I have", response.content)
        self.assertEqual([b["tool_choice"] for b in self.bodies], ["auto", "auto", "none"])
        self.assertEqual(self.bodies[2]["tools"], TOOLS)

    async def test_call_budget_answers_excess_calls_and_forces_final(self):
        seen = []

        async def handler(name, arguments, call_id):
            seen.append(call_id)
            return ToolOutcome('{"ok":true}')

        responses = [sse([call("c1"), call("c2"), call("c3")]), sse([message("done")])]
        with self.upstream(responses):
            gateway = self.gateway(tools=TOOLS, tool_handler=handler)
            gateway.MAX_TOOL_CALLS = 2
            async with gateway:
                await self.post(gateway)
        self.assertEqual(seen, ["c1", "c2"])
        outputs = {i["call_id"]: json.loads(i["output"]) for i in self.bodies[1]["input"]
                   if i.get("type") == "function_call_output"}
        self.assertEqual(outputs["c3"], {"ok": False, "error": {"code": "tool_budget_exhausted"}})
        self.assertEqual(self.bodies[1]["tool_choice"], "none")

    async def test_model_ignoring_tool_choice_none_is_a_static_error(self):
        async def handler(name, arguments, call_id):
            return ToolOutcome("{}")
        with self.upstream(lambda body: sse([call("again")])):
            gateway = self.gateway(tools=TOOLS, tool_handler=handler)
            gateway.MAX_TOOL_ROUNDS = 1
            async with gateway:
                response = await self.post(gateway)
                self.assertEqual(gateway.last_error_code, "gateway_tool_loop_exhausted")
        self.assertEqual(response.status_code, 502)

    async def test_upstream_errors_in_later_rounds_stay_static(self):
        async def handler(name, arguments, call_id):
            return ToolOutcome("{}")
        failing = httpx.Response(429, text="PRIVATE_UPSTREAM_SECRET")
        with self.upstream([sse([call("c1")]), failing]):
            async with self.gateway(tools=TOOLS, tool_handler=handler) as gateway:
                response = await self.post(gateway)
                self.assertEqual(gateway.last_error_code, "gateway_upstream_http_429")
        self.assertEqual(response.status_code, 429)
        self.assertNotIn("PRIVATE", response.text)

    async def test_sse_error_event_without_calls_is_relayed_to_the_cli(self):
        raw = b'event: error\ndata: {"type":"error","error":{"code":"x"}}\n\n'
        async def handler(name, arguments, call_id):
            return ToolOutcome("{}")
        with self.upstream([raw]):
            async with self.gateway(tools=TOOLS, tool_handler=handler) as gateway:
                response = await self.post(gateway)
        self.assertEqual((response.status_code, response.content), (200, raw))

    async def test_disconnect_cancels_running_handler_and_stops_the_loop(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def handler(name, arguments, call_id):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with self.upstream([sse([call("c1")])]):
            async with self.gateway(tools=TOOLS, tool_handler=handler) as gateway:
                from urllib.parse import urlsplit
                url = urlsplit(gateway.base_url)
                reader, writer = await asyncio.open_connection(url.hostname, url.port)
                body = b"{}"
                writer.write((f"POST {url.path}/responses HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                              f"Content-Length: {len(body)}\r\n\r\n").encode() + body)
                await writer.drain()
                await asyncio.wait_for(started.wait(), 2)
                writer.close()
                await writer.wait_closed()
                await asyncio.wait_for(cancelled.wait(), 2)
                await asyncio.wait_for(asyncio.gather(*gateway._tasks), 2)
                self.assertEqual(gateway.upstream_requests, 1)

    async def test_gateway_exit_cancels_running_handler(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def handler(name, arguments, call_id):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with self.upstream([sse([call("c1")])]):
            async with self.gateway(tools=TOOLS, tool_handler=handler) as gateway:
                caller = asyncio.create_task(self.post(gateway))
                await asyncio.wait_for(started.wait(), 2)
            self.assertTrue(cancelled.is_set())
            caller.cancel()
            await asyncio.gather(caller, return_exceptions=True)

    async def test_no_tools_payload_is_unchanged(self):
        with self.upstream([sse([message("plain")])]):
            async with self.gateway() as gateway:
                response = await self.post(gateway)
                self.assertEqual(gateway.upstream_requests, 1)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.bodies[0], {
            "instructions": "PERSONAL_AGENT_PERSONA", "model": "gpt-6-luna",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "PERSONAL_AGENT_PROMPT"}]}],
            "reasoning": {"effort": "medium"}, "stream": True, "store": False, "tools": [],
        })

    def test_invalid_tool_configuration_is_rejected(self):
        handler = lambda *a: None  # noqa: E731
        for kwargs in (
            {"tools": TOOLS},  # no handler
            {"tools": [{"type": "web_search"}], "tool_handler": handler},
            {"tools": [{"type": "function", "name": "bad name!"}], "tool_handler": handler},
            {"tools": [TOOLS[0], TOOLS[0]], "tool_handler": handler},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "gateway_invalid_tools"):
                self.gateway(**kwargs)


class BackendNativeToolsTests(unittest.TestCase):
    def test_backend_passes_tools_to_gateway_and_reports_support(self):
        backend = CodexSubscriptionBackend(model="gpt-6-luna")
        handler = lambda *a: None  # noqa: E731
        gateway = backend._request_gateway("p", (), TOOLS, handler)
        self.assertEqual(gateway._tools, tuple(TOOLS))
        self.assertIs(gateway._tool_handler, handler)
        self.assertEqual(backend._request_gateway("p", ())._tools, ())
        self.assertIs(backend.model_info()["native_tools_supported"], True)
        self.assertIs(CodexChatGPTWebBackend(enabled=True).model_info()["native_tools_supported"], True)
        self.assertEqual(backend.model_info()["tools"], [])

    def test_idle_timeout_is_configured_for_the_isolated_provider(self):
        args = CodexSubscriptionBackend()._gateway_provider_args("http://127.0.0.1:1/v1")
        self.assertIn("model_providers.personal_agent_isolated.stream_idle_timeout_ms=600000", args)
        args = CodexSubscriptionBackend(stream_idle_timeout_ms=900000)._gateway_provider_args("http://x/v1")
        self.assertIn("model_providers.personal_agent_isolated.stream_idle_timeout_ms=900000", args)
        with self.assertRaises(ValueError):
            CodexSubscriptionBackend(stream_idle_timeout_ms=5)


class BackendNativeToolsAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_mock_backend_accepts_and_ignores_tools(self):
        text = await MockBackend().generate("hi", "r", images=None, tools=TOOLS, tool_handler=lambda *a: None)
        self.assertIn("offline mock", text)


class ResolveCodexCommandTests(unittest.TestCase):
    def test_existing_path_then_path_lookup_then_newest_desktop_install_then_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            explicit = root / "codex-explicit.exe"
            explicit.write_bytes(b"")
            self.assertEqual(resolve_codex_command(str(explicit)), str(explicit))
            with patch("personal_agent.backends.shutil.which", return_value="C:/bin/codex.exe"):
                self.assertEqual(resolve_codex_command("codex"), "C:/bin/codex.exe")
            old, new = (root / "OpenAI" / "Codex" / "bin" / name / "codex.exe" for name in ("a", "b"))
            for path, mtime in ((old, 1_000_000), (new, 2_000_000)):
                path.parent.mkdir(parents=True)
                path.write_bytes(b"")
                os.utime(path, (mtime, mtime))
            with patch("personal_agent.backends.shutil.which", return_value=None), \
                    patch.dict(os.environ, {"LOCALAPPDATA": str(root)}):
                self.assertEqual(resolve_codex_command("codex"), str(new))
            with patch("personal_agent.backends.shutil.which", return_value=None), \
                    patch.dict(os.environ, {"LOCALAPPDATA": str(root / "missing")}):
                self.assertEqual(resolve_codex_command("codex"), "codex")


if __name__ == "__main__":
    unittest.main()
