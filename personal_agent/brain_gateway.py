"""Request-local subscription proxy; caller context never crosses this boundary."""
from __future__ import annotations

import asyncio
import base64
import json
import re
import secrets
from collections.abc import Sequence
from contextlib import aclosing
from pathlib import Path
from urllib.parse import urlsplit

import httpx


class _Rejected(Exception):
    def __init__(self, status: int, code: str) -> None:
        self.status, self.code = status, code


class BrainRequestGateway:
    MAX_HEADERS = 16 * 1024
    MAX_BODY = 32 * 1024 * 1024
    MAX_IMAGE = 10 * 1024 * 1024
    MAX_IMAGES_TOTAL = 20 * 1024 * 1024
    MAX_IMAGES = 8
    REQUEST_TIMEOUT = 15.0

    def __init__(
        self, *, instructions: str, prompt: str, model: str, reasoning_effort: str,
        images: Sequence[Path] = (),
        upstream_url: str = "https://chatgpt.com/backend-api/codex/responses",
        require_auth: bool = True,
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

    def _make_payload(self) -> dict:
        content = [{"type": "input_text", "text": self._prompt}]
        total = 0
        for path in self._images:
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
            content.append({"type": "input_image", "image_url":
                            f"data:{mime};base64," + base64.b64encode(data).decode("ascii")})
        return {"instructions": self._instructions, "model": self._model,
                "input": [{"role": "user", "content": content}],
                "reasoning": {"effort": self._reasoning},
                "stream": True, "store": False, "tools": []}

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

    async def _forward(self, writer: asyncio.StreamWriter, headers: dict[str, str]) -> None:
        assert self._client is not None
        headers.update({"Content-Type": "application/json", "Accept": "text/event-stream",
                        "Accept-Encoding": "identity", "User-Agent": "PersonalAgent/1.0",
                        "OpenAI-Beta": "responses=experimental", "originator": "codex_cli_rs"})
        started = False
        try:
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
                        if not first_line.startswith((b"event:", b"data:", b":")):
                            raise _Rejected(502, "gateway_upstream_not_sse")
                        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                                     b"Cache-Control: no-store\r\nConnection: close\r\n\r\n")
                        started = True
                        chunk = bytes(prefix)
                        prefix.clear()
                    writer.write(chunk)
                    await writer.drain()
                if not started:
                    raise _Rejected(502, "gateway_upstream_not_sse")
        except (httpx.HTTPError, _Rejected) as exc:
            code = exc.code if isinstance(exc, _Rejected) else "gateway_upstream_unavailable"
            self.last_error_code = code
            if not started:
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
                self.last_error_code = "gateway_client_disconnected"
        except _Rejected as exc:
            self.last_error_code = exc.code
            await self._reply(writer, exc.status, exc.code)
        except (ValueError, UnicodeError, asyncio.IncompleteReadError):
            self.last_error_code = "gateway_malformed_request"
            await self._reply(writer, 400, self.last_error_code)
        except TimeoutError:
            self.last_error_code = "gateway_request_timeout"
            await self._reply(writer, 408, self.last_error_code)
        except OSError:
            self.last_error_code = "gateway_client_disconnected"
        except Exception:
            self.last_error_code = "gateway_internal_error"
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
