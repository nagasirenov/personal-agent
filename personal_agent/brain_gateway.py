"""Request-local subscription proxy; caller context never crosses this boundary."""
from __future__ import annotations

import asyncio
import base64
import json
import re
import secrets
from collections.abc import Awaitable, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx


class _Rejected(Exception):
    def __init__(self, status: int, code: str) -> None:
        self.status, self.code = status, code


@dataclass
class ToolOutcome:
    """Result of one native tool call: JSON text for the model plus optional images to show it."""

    output: str
    images: list[Path] = field(default_factory=list)


ToolHandler = Callable[[str, dict, str], Awaitable[ToolOutcome]]

# Output items replayed to the endpoint on the next tool round (store=false).  Server-assigned
# ids ("rs_...", "fc_...", "msg_...") refer to stored items and are never replayed; only the
# self-contained content (including reasoning encrypted_content) is.
_REPLAY_FIELDS = {
    "reasoning": ("type", "summary", "encrypted_content"),
    "function_call": ("type", "call_id", "name", "arguments"),
    "message": ("type", "role", "content", "phase"),
}


def _tool_error(code: str) -> str:
    return json.dumps({"ok": False, "error": {"code": code}}, separators=(",", ":"))


def _parse_sse(raw: bytes) -> list[dict]:
    events: list[dict] = []
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    for block in text.split("\n\n"):
        data = "\n".join(line[5:].lstrip(" ") for line in block.split("\n")
                         if line.startswith("data:"))
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _output_items(events: list[dict]) -> list[dict]:
    done: list[dict] = []
    for event in events:
        kind = event.get("type")
        if kind == "response.completed":
            output = (event.get("response") or {}).get("output")
            if isinstance(output, list) and output:
                return [item for item in output if isinstance(item, dict)]
        elif kind == "response.output_item.done" and isinstance(event.get("item"), dict):
            done.append(event["item"])
    return done


def _replayable(item: dict) -> dict | None:
    fields = _REPLAY_FIELDS.get(item.get("type"))
    if fields is None:
        return None
    return {key: item[key] for key in fields if key in item}


_SSE_HEAD = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
             b"Cache-Control: no-store\r\nConnection: close\r\n\r\n")
_SSE_STARTS = (b"event:", b"data:", b":")


class BrainRequestGateway:
    MAX_HEADERS = 16 * 1024
    MAX_BODY = 32 * 1024 * 1024
    MAX_IMAGE = 10 * 1024 * 1024
    MAX_IMAGES_TOTAL = 20 * 1024 * 1024
    MAX_IMAGES = 8
    REQUEST_TIMEOUT = 15.0
    MAX_TOOL_ROUNDS = 8
    MAX_TOOL_CALLS = 24
    MAX_TOOL_IMAGES = 8
    MAX_TOOL_OUTPUT = 256 * 1024
    MAX_BUFFERED_RESPONSE = 16 * 1024 * 1024
    TOOL_TIMEOUT = 300.0

    def __init__(
        self, *, instructions: str, prompt: str, model: str, reasoning_effort: str,
        images: Sequence[Path] = (),
        upstream_url: str = "https://chatgpt.com/backend-api/codex/responses",
        require_auth: bool = True,
        tools: Sequence[dict] = (),
        tool_handler: ToolHandler | None = None,
    ) -> None:
        try:
            url = urlsplit(upstream_url)
            loopback = url.hostname in {"127.0.0.1", "::1", "localhost"}
            official = (url.scheme == "https" and url.hostname == "chatgpt.com"
                        and url.port in {None, 443}
                        and url.path == "/backend-api/codex/responses")
            allowed = (official or (loopback and url.scheme in {"http", "https"}))
            allowed = allowed and not (url.username or url.password or url.query or url.fragment)
        except ValueError:
            allowed = loopback = False
        if not allowed or (not require_auth and not loopback):
            raise ValueError("gateway_upstream_not_allowed")
        if (not isinstance(instructions, str) or not isinstance(prompt, str)
                or len(instructions.encode("utf-8")) > 256 * 1024
                or len(prompt.encode("utf-8")) > 1024 * 1024
                or not re.fullmatch(r"[A-Za-z0-9_./-]{1,128}", model)
                or reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}):
            raise ValueError("gateway_invalid_configuration")
        self._instructions, self._prompt = instructions, prompt
        self._model, self._reasoning = model, reasoning_effort
        self._images = tuple(images)
        if len(self._images) > self.MAX_IMAGES or any(not isinstance(p, Path) for p in self._images):
            raise ValueError("gateway_invalid_images")
        try:
            self._tools = tuple(json.loads(json.dumps(list(tools))))
        except (TypeError, ValueError):
            raise ValueError("gateway_invalid_tools") from None
        names = [tool.get("name") if isinstance(tool, dict) else None for tool in self._tools]
        if (any(not isinstance(t, dict) or t.get("type") != "function" for t in self._tools)
                or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", n)
                       for n in names)
                or len(set(names)) != len(names)
                or (self._tools and not callable(tool_handler))):
            raise ValueError("gateway_invalid_tools")
        self._tool_names = frozenset(names)
        self._tool_handler = tool_handler
        self._tool_image_count = self._tool_image_bytes = 0
        self.tool_call_count = 0
        self.upstream_requests = 0
        self.vision_requests = 0  # upstream requests whose input carried at least one image
        self._upstream_url, self._require_auth = upstream_url, require_auth
        self._path = "/" + secrets.token_urlsafe(32) + "/v1/responses"
        self._tasks: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()
        self._server: asyncio.Server | None = None
        self._client: httpx.AsyncClient | None = None
        self._payload: dict = {}
        self._closing = False
        self.base_url = ""
        self.request_count = 0
        self.last_error_code: str | None = None

    def _count_upstream_request(self) -> None:
        self.upstream_requests += 1
        if self._images or self._tool_image_count:
            self.vision_requests += 1

    def _note_error(self, code: str) -> None:
        """Keep the FIRST failure: later errors are usually the CLI's follow-up retry, not the cause.

        The CLI also probes sibling routes (for example a model list) that the gateway rejects by design;
        such a probe is only a placeholder and never hides the real failure of the actual request.
        """
        if self.last_error_code is None or (
                self.last_error_code == "gateway_route_rejected" and code != "gateway_route_rejected"):
            self.last_error_code = code

    def _encode_image(self, path: Path, total: int) -> tuple[dict, int]:
        try:
            if not path.is_file() or path.stat().st_size > self.MAX_IMAGE:
                raise ValueError("gateway_image_too_large_or_invalid")
            with path.open("rb") as handle:
                data = handle.read(self.MAX_IMAGE + 1)
        except OSError:
            raise ValueError("gateway_image_unreadable") from None
        total += len(data)
        if len(data) > self.MAX_IMAGE or total > self.MAX_IMAGES_TOTAL:
            raise ValueError("gateway_images_too_large")
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif data[:6] in {b"GIF87a", b"GIF89a"}:
            mime = "image/gif"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            mime = "image/webp"
        else:
            raise ValueError("gateway_image_unsupported")
        return ({"type": "input_image", "image_url":
                 f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}, total)

    def _make_payload(self) -> dict:
        content = [{"type": "input_text", "text": self._prompt}]
        total = 0
        for path in self._images:
            part, total = self._encode_image(path, total)
            content.append(part)
        payload = {"instructions": self._instructions, "model": self._model,
                   "input": [{"role": "user", "content": content}],
                   "reasoning": {"effort": self._reasoning},
                   "stream": True, "store": False, "tools": list(self._tools)}
        if self._tools:
            payload.update({"tool_choice": "auto", "parallel_tool_calls": True,
                            "include": ["reasoning.encrypted_content"]})
        return payload

    async def __aenter__(self) -> BrainRequestGateway:
        if self._server is not None or self._closing:
            raise ValueError("gateway_already_used")
        self._payload = self._make_payload()
        self._client = httpx.AsyncClient(
            trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(120.0, connect=15.0),
        )
        try:
            self._server = await asyncio.start_server(
                self._accept, "127.0.0.1", 0, limit=self.MAX_HEADERS,
            )
        except BaseException:
            await self._client.aclose()
            raise
        port = self._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}" + self._path.removesuffix("/responses")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._closing = True
        if self._server is not None:
            self._server.close()
        for writer in self._writers:
            writer.close()
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()
        if self._client is not None:
            await self._client.aclose()
        self._writers.clear()
        self._payload = {}

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._closing or len(self._tasks) >= 4:
            writer.close()
            return
        task = asyncio.create_task(self._handle(reader, writer))
        self._writers.add(writer)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _read_request(self, reader: asyncio.StreamReader) -> dict[str, str]:
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
        except asyncio.LimitOverrunError:
            raise _Rejected(431, "gateway_headers_too_large") from None
        if len(raw) > self.MAX_HEADERS:
            raise _Rejected(431, "gateway_headers_too_large")
        lines = raw[:-4].decode("ascii").split("\r\n")
        if lines[0] != f"POST {self._path} HTTP/1.1":
            raise _Rejected(404, "gateway_route_rejected")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            key, separator, value = line.partition(":")
            if (not separator or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key)
                    or key.lower() in headers or any(ord(c) < 32 or ord(c) == 127 for c in value)):
                raise _Rejected(400, "gateway_invalid_headers")
            headers[key.lower()] = value.strip()
        if "transfer-encoding" in headers:
            raise _Rejected(400, "gateway_chunked_unsupported")
        if headers.get("content-encoding", "identity").lower() != "identity":
            raise _Rejected(415, "gateway_compression_unsupported")
        length = headers.get("content-length", "")
        if not re.fullmatch(r"[0-9]{1,9}", length):
            raise _Rejected(411, "gateway_content_length_required")
        if not 0 < int(length) <= self.MAX_BODY:
            raise _Rejected(413, "gateway_body_too_large")
        if headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise _Rejected(415, "gateway_json_required")
        auth = headers.get("authorization", "")
        account = headers.get("chatgpt-account-id", "")
        forwarded = {}
        if self._require_auth:
            if not re.fullmatch(r"Bearer [\x21-\x7e]{1,8192}", auth):
                raise _Rejected(401, "gateway_auth_required")
            if account and not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", account):
                raise _Rejected(400, "gateway_invalid_account")
            forwarded["Authorization"] = auth
            if account:
                forwarded["ChatGPT-Account-ID"] = account
        body = await reader.readexactly(int(length))
        if not isinstance(json.loads(body), dict):
            raise _Rejected(400, "gateway_invalid_json")
        return forwarded

    async def _reply(self, writer: asyncio.StreamWriter, status: int, code: str) -> None:
        body = json.dumps({"error": {"code": code, "message": code}}).encode("ascii")
        try:
            writer.write(f"HTTP/1.1 {status} Error\r\nContent-Type: application/json\r\n"
                         f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii") + body)
            await writer.drain()
        except OSError:
            pass

    @staticmethod
    def _upstream_headers(headers: dict[str, str]) -> dict[str, str]:
        headers = dict(headers)
        headers.update({"Content-Type": "application/json", "Accept": "text/event-stream",
                        "Accept-Encoding": "identity", "User-Agent": "PersonalAgent/1.0",
                        "OpenAI-Beta": "responses=experimental", "originator": "codex_cli_rs"})
        return headers

    async def _forward(self, writer: asyncio.StreamWriter, headers: dict[str, str]) -> None:
        assert self._client is not None
        if self._tools:
            await self._forward_with_tools(writer, headers)
            return
        headers = self._upstream_headers(headers)
        started = False
        try:
            self._count_upstream_request()
            request = self._client.build_request("POST", self._upstream_url,
                                                 headers=headers, json=self._payload)
            request.headers.pop("cookie", None)
            response = await self._client.send(request, stream=True)
            async with aclosing(response):
                if response.status_code != 200:
                    status = response.status_code
                    raise _Rejected(status if 400 <= status <= 599 else 502,
                                    f"gateway_upstream_http_{status}")
                mime = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if mime not in {"", "text/event-stream"}:
                    raise _Rejected(502, "gateway_upstream_not_sse")
                prefix = bytearray()
                async for chunk in response.aiter_bytes():
                    if not started:
                        prefix.extend(chunk)
                        if b"\n" not in prefix:
                            if len(prefix) > 8192:
                                raise _Rejected(502, "gateway_upstream_not_sse")
                            continue
                        first_line = bytes(prefix).lstrip(b"\xef\xbb\xbf\r\n \t").split(b"\n", 1)[0]
                        if not first_line.startswith(_SSE_STARTS):
                            raise _Rejected(502, "gateway_upstream_not_sse")
                        writer.write(_SSE_HEAD)
                        started = True
                        chunk = bytes(prefix)
                        prefix.clear()
                    writer.write(chunk)
                    await writer.drain()
                if not started:
                    raise _Rejected(502, "gateway_upstream_not_sse")
        except (httpx.HTTPError, _Rejected) as exc:
            code = exc.code if isinstance(exc, _Rejected) else "gateway_upstream_unavailable"
            self._note_error(code)
            if not started:
                await self._reply(writer, exc.status if isinstance(exc, _Rejected) else 502, code)

    async def _post_buffered(self, headers: dict[str, str], body: dict) -> bytes:
        """One upstream request; the complete SSE response is buffered locally."""
        assert self._client is not None
        self._count_upstream_request()
        request = self._client.build_request("POST", self._upstream_url, headers=headers, json=body)
        request.headers.pop("cookie", None)
        response = await self._client.send(request, stream=True)
        async with aclosing(response):
            if response.status_code != 200:
                status = response.status_code
                raise _Rejected(status if 400 <= status <= 599 else 502,
                                f"gateway_upstream_http_{status}")
            mime = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if mime not in {"", "text/event-stream"}:
                raise _Rejected(502, "gateway_upstream_not_sse")
            raw = bytearray()
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > self.MAX_BUFFERED_RESPONSE:
                    raise _Rejected(502, "gateway_upstream_response_too_large")
        if not bytes(raw[:8192]).lstrip(b"\xef\xbb\xbf\r\n \t").startswith(_SSE_STARTS):
            raise _Rejected(502, "gateway_upstream_not_sse")
        return bytes(raw)

    async def _run_tool(self, name: str, raw_arguments: object, call_id: str) -> ToolOutcome:
        """Never raises (except cancellation); every failure becomes a JSON error for the model."""
        if name not in self._tool_names:
            return ToolOutcome(_tool_error("unknown_tool"))
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else None
        except json.JSONDecodeError:
            arguments = None
        if not isinstance(arguments, dict):
            return ToolOutcome(_tool_error("invalid_arguments"))
        assert self._tool_handler is not None
        try:
            outcome = await asyncio.wait_for(self._tool_handler(name, arguments, call_id),
                                             self.TOOL_TIMEOUT)
        except TimeoutError:
            return ToolOutcome(_tool_error("tool_timeout"))
        except Exception:
            return ToolOutcome(_tool_error("tool_failed"))
        if isinstance(outcome, str):
            outcome = ToolOutcome(outcome)
        if (not isinstance(outcome, ToolOutcome) or not isinstance(outcome.output, str)
                or len(outcome.output.encode("utf-8")) > self.MAX_TOOL_OUTPUT):
            return ToolOutcome(_tool_error("tool_output_invalid"))
        return outcome

    def _tool_images_message(self, name: str, call_id: str, paths: Sequence[Path]) -> dict | None:
        parts: list[dict] = []
        omitted = 0
        for path in paths:
            try:
                if (not isinstance(path, Path)
                        or self._tool_image_count >= self.MAX_TOOL_IMAGES):
                    raise ValueError("gateway_tool_image_limit")
                part, total = self._encode_image(path, self._tool_image_bytes)
            except ValueError:
                omitted += 1
                continue
            self._tool_image_count += 1
            self._tool_image_bytes = total
            parts.append(part)
        if not parts and not omitted:
            return None
        text = f"Images returned by tool {name} (call {call_id})."
        if omitted:
            text += f" {omitted} image(s) could not be attached."
        return {"role": "user", "content": [{"type": "input_text", "text": text}, *parts]}

    async def _forward_with_tools(self, writer: asyncio.StreamWriter, headers: dict[str, str]) -> None:
        headers = self._upstream_headers(headers)
        conversation = list(self._payload["input"])
        rounds = 0
        try:
            while True:
                exhausted = (rounds >= self.MAX_TOOL_ROUNDS
                             or self.tool_call_count >= self.MAX_TOOL_CALLS)
                body = {**self._payload, "input": conversation}
                if exhausted:
                    body["tool_choice"] = "none"
                raw = await self._post_buffered(headers, body)
                items = _output_items(_parse_sse(raw))
                calls = [item for item in items if item.get("type") == "function_call"]
                if not calls:
                    writer.write(_SSE_HEAD + raw)
                    await writer.drain()
                    return
                if exhausted:
                    raise _Rejected(502, "gateway_tool_loop_exhausted")
                rounds += 1
                conversation.extend(r for r in map(_replayable, items) if r is not None)
                outputs: list[dict] = []
                image_messages: list[dict] = []
                for call in calls:
                    call_id, name = call.get("call_id"), call.get("name")
                    if not isinstance(call_id, str) or not isinstance(name, str):
                        raise _Rejected(502, "gateway_upstream_bad_tool_call")
                    if self.tool_call_count >= self.MAX_TOOL_CALLS:
                        outcome = ToolOutcome(_tool_error("tool_budget_exhausted"))
                    else:
                        self.tool_call_count += 1
                        outcome = await self._run_tool(name, call.get("arguments"), call_id)
                    outputs.append({"type": "function_call_output", "call_id": call_id,
                                    "output": outcome.output})
                    if outcome.images:
                        message = self._tool_images_message(name, call_id, outcome.images)
                        if message:
                            image_messages.append(message)
                conversation.extend(outputs)
                conversation.extend(image_messages)
        except (httpx.HTTPError, _Rejected) as exc:
            code = exc.code if isinstance(exc, _Rejected) else "gateway_upstream_unavailable"
            self._note_error(code)
            await self._reply(writer, exc.status if isinstance(exc, _Rejected) else 502, code)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        children: list[asyncio.Task] = []
        try:
            headers = await asyncio.wait_for(self._read_request(reader), self.REQUEST_TIMEOUT)
            self.request_count += 1
            relay = asyncio.create_task(self._forward(writer, headers))
            disconnected = asyncio.create_task(reader.read(1))
            children = [relay, disconnected]
            done, _ = await asyncio.wait(children, return_when=asyncio.FIRST_COMPLETED)
            if relay in done:
                await relay
            else:
                self._note_error("gateway_client_disconnected")
        except _Rejected as exc:
            self._note_error(exc.code)
            await self._reply(writer, exc.status, exc.code)
        except (ValueError, UnicodeError, asyncio.IncompleteReadError):
            self._note_error("gateway_malformed_request")
            await self._reply(writer, 400, "gateway_malformed_request")
        except TimeoutError:
            self._note_error("gateway_request_timeout")
            await self._reply(writer, 408, "gateway_request_timeout")
        except OSError:
            self._note_error("gateway_client_disconnected")
        except Exception:
            self._note_error("gateway_internal_error")
        finally:
            writer.close()
            self._writers.discard(writer)
            for child in children:
                child.cancel()
            await asyncio.gather(*children, return_exceptions=True)
            try:
                await writer.wait_closed()
            except OSError:
                pass
