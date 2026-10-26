from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import httpx

from personal_agent.brain_gateway import BrainRequestGateway


_CLIENT = httpx.AsyncClient
_SSE = b'event: response.completed\ndata: {"type":"response.completed"}\n\n'
_INJECTED = {
    "instructions": "HOST_AGENTS_AND_COORDINATOR_PRIVATE",
    "input": [{"role": "developer", "content": "HOST_CONTEXT"}],
    "metadata": {"thread": "HOST_THREAD"},
    "tools": [{"type": "function", "name": "shell"}],
    "model": "host-model", "reasoning": {"effort": "low"},
    "previous_response_id": "HOST_RESPONSE", "store": True,
}


class _BlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self):
        self.started.set()
        yield b"event: response.created\ndata: {}\n\n"
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed.set()


class BrainGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_sse_without_mime_header_is_accepted_but_untyped_json_is_not(self):
        for payload, accepted in ((_SSE, True), (b'{"unexpected":"json"}', False)):
            async def upstream(request):
                return httpx.Response(200, content=payload)
            with self.mocked_upstream(upstream):
                async with self.gateway(upstream_url="http://127.0.0.1:9/responses", require_auth=False) as gateway:
                    reader, writer = await self.send_raw(gateway)
                    response = await asyncio.wait_for(reader.read(), 2)
                    writer.close()
                    await writer.wait_closed()
                    self.assertTrue(response.startswith(b"HTTP/1.1 200" if accepted else b"HTTP/1.1 502"))
                    if accepted:
                        self.assertIn(_SSE, response)

    def gateway(self, **kwargs) -> BrainRequestGateway:
        return BrainRequestGateway(
            instructions="PERSONAL_AGENT_PERSONA", prompt="PERSONAL_AGENT_PROMPT",
            model="gpt-6-sol", reasoning_effort="high", **kwargs,
        )

    def mocked_upstream(self, handler):
        def factory(**kwargs):
            self.assertIs(kwargs["trust_env"], False)
            self.assertIs(kwargs["follow_redirects"], False)
            return _CLIENT(transport=httpx.MockTransport(handler), **kwargs)
        return patch("personal_agent.brain_gateway.httpx.AsyncClient", side_effect=factory)

    async def send_raw(self, gateway, *, route=None, extra=b"", body=b"{}", length=None):
        url = urlsplit(gateway.base_url)
        reader, writer = await asyncio.open_connection(url.hostname, url.port)
        path = route if route is not None else url.path + "/responses"
        request = (f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                   "Content-Type: application/json\r\n"
                   f"Content-Length: {len(body) if length is None else length}\r\n").encode()
        writer.write(request + extra + b"\r\n" + body)
        await writer.drain()
        return reader, writer

    async def test_rebuilds_body_and_forwards_only_allowed_auth(self):
        captured = []
        async def upstream(request):
            captured.append(request)
            return httpx.Response(200, content=_SSE, headers={"Content-Type": "text/event-stream"})

        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "fixture.png"
            image_bytes = b"\x89PNG\r\n\x1a\nEXPLICIT_IMAGE_FIXTURE"
            image.write_bytes(image_bytes)
            with self.mocked_upstream(upstream):
                async with self.gateway(images=[image]) as gateway:
                    async with _CLIENT(trust_env=False) as caller:
                        response = await caller.post(
                            gateway.base_url + "/responses", json=_INJECTED,
                            headers={"Authorization": "Bearer fixture-token",
                                     "ChatGPT-Account-ID": "fixture-account",
                                     "Cookie": "private=cookie", "X-Host-Context": "HOST_CONTEXT",
                                     "session_id": "HOST_THREAD", "originator": "injected"},
                        )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.content, _SSE)
                    self.assertEqual(gateway.request_count, 1)
                    self.assertIsNone(gateway.last_error_code)

        request = captured[0]
        self.assertEqual(str(request.url), "https://chatgpt.com/backend-api/codex/responses")
        body = json.loads(request.content)
        self.assertEqual(body, {
            "instructions": "PERSONAL_AGENT_PERSONA", "model": "gpt-6-sol",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "PERSONAL_AGENT_PROMPT"},
                {"type": "input_image", "image_url": "data:image/png;base64," +
                 base64.b64encode(image_bytes).decode("ascii")},
            ]}], "reasoning": {"effort": "high"},
            "stream": True, "store": False, "tools": [],
        })
        self.assertNotIn("HOST_", request.content.decode())
        self.assertEqual(request.headers["Authorization"], "Bearer fixture-token")
        self.assertEqual(request.headers["ChatGPT-Account-ID"], "fixture-account")
        for name in ("Cookie", "X-Host-Context", "session_id"):
            self.assertNotIn(name, request.headers)
        self.assertEqual(request.headers["originator"], "codex_cli_rs")

    async def test_fixture_mode_strips_credentials_even_when_caller_supplies_them(self):
        captured = []
        async def upstream(request):
            captured.append(request)
            return httpx.Response(200, content=_SSE, headers={"Content-Type": "text/event-stream"})

        with self.mocked_upstream(upstream):
            async with self.gateway(upstream_url="http://127.0.0.1:1/responses", require_auth=False) as gateway:
                async with _CLIENT(trust_env=False) as caller:
                    response = await caller.post(gateway.base_url + "/responses", json=_INJECTED,
                        headers={"Authorization": "Bearer never-forward", "ChatGPT-Account-ID": "never-forward"})
                self.assertEqual(response.status_code, 200)
        self.assertNotIn("authorization", captured[0].headers)
        self.assertNotIn("chatgpt-account-id", captured[0].headers)

    async def test_rejects_missing_auth_without_upstream_call(self):
        async def upstream(request):
            self.fail("unauthenticated request reached upstream")
        with self.mocked_upstream(upstream):
            async with self.gateway() as gateway:
                async with _CLIENT(trust_env=False) as caller:
                    response = await caller.post(gateway.base_url + "/responses", json={})
                self.assertEqual(response.status_code, 401)
                self.assertEqual(gateway.last_error_code, "gateway_auth_required")
                self.assertEqual(gateway.request_count, 0)

    async def test_upstream_cookie_is_never_replayed_on_retry(self):
        captured = []
        async def upstream(request):
            captured.append(request)
            return httpx.Response(200, content=_SSE, headers={
                "Content-Type": "text/event-stream", "Set-Cookie": "private=never-replay; Path=/",
            })
        with self.mocked_upstream(upstream):
            async with self.gateway(upstream_url="http://127.0.0.1:1/responses", require_auth=False) as gateway:
                async with _CLIENT(trust_env=False) as caller:
                    for _ in range(2):
                        response = await caller.post(gateway.base_url + "/responses", json={})
                        self.assertEqual(response.status_code, 200)
                self.assertEqual(gateway.request_count, 2)
        self.assertTrue(all("cookie" not in request.headers for request in captured))

    async def test_rejects_bad_route_chunking_compression_and_oversized_or_bad_body(self):
        async def upstream(request):
            self.fail("malformed request reached upstream")
        cases = [
            ({"route": "/responses"}, 404),
            ({"extra": b"Transfer-Encoding: chunked\r\n"}, 400),
            ({"extra": b"Content-Encoding: zstd\r\n"}, 415),
            ({"length": BrainRequestGateway.MAX_BODY + 1}, 413),
            ({"body": b"not-json"}, 400),
            ({"extra": b"Content-Length: 2\r\n"}, 400),
            ({"extra": b"X-Large: " + b"a" * BrainRequestGateway.MAX_HEADERS + b"\r\n"}, 431),
        ]
        with self.mocked_upstream(upstream):
            async with self.gateway(upstream_url="http://127.0.0.1:1/responses", require_auth=False) as gateway:
                for kwargs, status in cases:
                    with self.subTest(status=status, keys=tuple(kwargs)):
                        reader, writer = await self.send_raw(gateway, **kwargs)
                        response = await asyncio.wait_for(reader.read(), 2)
                        self.assertTrue(response.startswith(f"HTTP/1.1 {status} ".encode()))
                        writer.close()
                        await writer.wait_closed()
                self.assertEqual(gateway.request_count, 0)

    async def test_upstream_errors_are_static_and_redirects_are_not_followed(self):
        for status in (302, 400, 401, 403, 429, 500):
            async def upstream(request):
                return httpx.Response(status, text="PRIVATE_UPSTREAM_SECRET", headers={"Location": "https://invalid.example"})
            with self.subTest(status=status), self.mocked_upstream(upstream):
                async with self.gateway(upstream_url="http://127.0.0.1:1/responses", require_auth=False) as gateway:
                    async with _CLIENT(trust_env=False) as caller:
                        response = await caller.post(gateway.base_url + "/responses", json={})
                    self.assertEqual(response.status_code, status if status >= 400 else 502)
                    self.assertNotIn("PRIVATE", response.text)
                    self.assertEqual(gateway.last_error_code, f"gateway_upstream_http_{status}")

    async def test_exit_closes_active_upstream_and_listening_socket(self):
        stream = _BlockingStream()
        async def upstream(request):
            return httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        with self.mocked_upstream(upstream):
            async with self.gateway(upstream_url="http://127.0.0.1:1/responses", require_auth=False) as gateway:
                reader, writer = await self.send_raw(gateway)
                await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2)
                await asyncio.wait_for(stream.started.wait(), 2)
                port = urlsplit(gateway.base_url).port
                client = gateway._client
            self.assertTrue(stream.closed.is_set())
            self.assertTrue(client.is_closed)
            self.assertFalse(gateway._tasks)
            await asyncio.wait_for(reader.read(), 2)
            writer.close()
            await writer.wait_closed()
            with self.assertRaises(OSError):
                await asyncio.open_connection("127.0.0.1", port)

    async def test_caller_disconnect_cancels_upstream_stream(self):
        stream = _BlockingStream()
        async def upstream(request):
            return httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})
        with self.mocked_upstream(upstream):
            async with self.gateway(upstream_url="http://127.0.0.1:1/responses", require_auth=False) as gateway:
                reader, writer = await self.send_raw(gateway)
                await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2)
                writer.close()
                await writer.wait_closed()
                await asyncio.wait_for(stream.closed.wait(), 2)
                await asyncio.wait_for(asyncio.gather(*gateway._tasks), 2)
                self.assertEqual(gateway.last_error_code, "gateway_client_disconnected")
                self.assertFalse(gateway._tasks)

    def test_upstream_allowlist_and_image_limits(self):
        for url in ("https://example.com/responses", "http://chatgpt.com/backend-api/codex/responses",
                    "https://chatgpt.com/other", "https://chatgpt.com/backend-api/codex/responses?private=x",
                    "http://token@127.0.0.1/responses"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "gateway_upstream_not_allowed"):
                self.gateway(upstream_url=url)
        with self.assertRaisesRegex(ValueError, "gateway_upstream_not_allowed"):
            self.gateway(require_auth=False)
        with self.assertRaisesRegex(ValueError, "gateway_invalid_images"):
            self.gateway(images=[Path("fixture.png")] * 9)

    async def test_image_byte_limit_and_unsupported_type_fail_before_listening(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "fixture.png"
            image.write_bytes(b"unsupported")
            gateway = self.gateway(images=[image])
            with self.assertRaisesRegex(ValueError, "gateway_image_unsupported"):
                await gateway.__aenter__()
            self.assertEqual(gateway.base_url, "")
            image.write_bytes(b"\x89PNG\r\n\x1a\nTOO_LARGE")
            gateway = self.gateway(images=[image])
            gateway.MAX_IMAGE = 8
            with self.assertRaisesRegex(ValueError, "gateway_image_too_large_or_invalid"):
                await gateway.__aenter__()
            self.assertEqual(gateway.base_url, "")


if __name__ == "__main__":
    unittest.main()
