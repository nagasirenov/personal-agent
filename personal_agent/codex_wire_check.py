"""Local wire self-check of an unknown Codex CLI version (no model call, no network beyond loopback).

The real CLI is pointed at our request gateway, whose upstream is a local stub that answers with a
synthetic final SSE.  The check passes only if the CLI behaves as the isolation design expects:

* it reaches the gateway route with Bearer authentication and without websocket or other fallbacks,
* the one request that arrives at the stub is the gateway-rebuilt Responses request (our persona, our
  prompt, exactly the offered tools, no extra messages, no credentials of any kind in the body),
* the CLI turns the synthetic SSE into the expected final text,
* the CLI opens no connection to a host outside the OpenAI/ChatGPT domains.  The installed CLI is known to
  open housekeeping connections to chatgpt.com (feature flags, model lists) on its own; all proxy-aware
  traffic is routed to a local sink, so these cannot succeed and only the destination is recorded.  The
  inference request itself must still arrive through the gateway at the stub.

Credentials the CLI sends stay in memory: only their presence is recorded.
"""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4


def synthetic_sse(text: str) -> bytes:
    """A minimal, well-formed Responses stream that ends in one assistant message."""
    message = {"type": "message", "id": "msg_selfcheck", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": text, "annotations": []}]}
    events = [
        {"type": "response.created", "response": {"id": "resp_selfcheck", "status": "in_progress"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {**message, "status": "in_progress", "content": []}},
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        {"type": "response.completed", "response": {
            "id": "resp_selfcheck", "status": "completed", "output": [message],
            "usage": {"input_tokens": 1, "input_tokens_details": None, "output_tokens": 1,
                      "output_tokens_details": None, "total_tokens": 2}}},
    ]
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


class _StubHandler(BaseHTTPRequestHandler):
    server: "CaptureStub"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        authorization = self.headers.get("Authorization", "")
        self.server.requests.append({
            "method": "POST", "path": self.path, "raw": raw,
            # Only the shape of the credential is kept, never its value.
            "bearer_auth": authorization.startswith("Bearer ") and len(authorization) > 10,
            "header_names": sorted(key.lower() for key in self.headers.keys())})
        payload = self.server.sse
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self.server.requests.append({"method": "GET", "path": self.path, "raw": b"", "bearer_auth": False,
                                     "header_names": []})
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class CaptureStub(ThreadingHTTPServer):
    """Loopback upstream that records requests and answers every POST with a synthetic final SSE."""

    daemon_threads = True

    def __init__(self, text: str) -> None:
        super().__init__(("127.0.0.1", 0), _StubHandler)
        self.requests: list[dict[str, Any]] = []
        self.sse = synthetic_sse(text)
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/v1"

    def start(self) -> "CaptureStub":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
        self._thread.join(timeout=2.0)


class ProxySink:
    """Records connection attempts of proxy-aware clients and drops them; real traffic never leaves."""

    def __init__(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(16)
        self._socket.settimeout(0.2)
        self.attempts: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def port(self) -> int:
        return self._socket.getsockname()[1]

    @property
    def env(self) -> dict[str, str]:
        url = f"http://127.0.0.1:{self.port}"
        return {"HTTP_PROXY": url, "HTTPS_PROXY": url, "ALL_PROXY": url,
                "http_proxy": url, "https_proxy": url, "all_proxy": url,
                "NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1"}

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                return
            with connection:
                connection.settimeout(0.5)
                try:
                    first = connection.recv(256).split(b"\r\n", 1)[0]
                except OSError:
                    first = b""
                # Method and host only; paths and bodies are not interesting and may carry data.
                parts = first.decode("ascii", errors="replace").split(" ")
                self.attempts.append(" ".join(parts[:2])[:120] or "connect")

    def start(self) -> "ProxySink":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._socket.close()
        self._thread.join(timeout=2.0)


EXPECTED_HOUSEKEEPING_DOMAINS = ("chatgpt.com", "openai.com")


def unexpected_hosts(attempts: list[str]) -> list[str]:
    """Destinations of recorded proxy attempts that are outside the expected OpenAI/ChatGPT domains."""
    unexpected = []
    for attempt in attempts:
        target = attempt.split(" ")[-1]
        host = target.rsplit(":", 1)[0].strip("[]").lower() if target else ""
        if not any(host == domain or host.endswith("." + domain) for domain in EXPECTED_HOUSEKEEPING_DOMAINS):
            unexpected.append(attempt)
    return unexpected


def new_marker() -> str:
    return f"WIRE_SELFCHECK_OK_{uuid4().hex[:12]}"


def evaluate(*, stub_requests: list[dict], proxy_attempts: list[str], gateway_requests: int,
             gateway_error: str | None, result_text: str | None, marker: str, instructions: str, prompt: str,
             tool_names: list[str], generate_error: str | None) -> dict[str, bool]:
    """Pure judgement of one self-check run; every key must be True for a PASS."""
    body: dict = {}
    if len(stub_requests) == 1:
        try:
            decoded = json.loads(stub_requests[0]["raw"])
            body = decoded if isinstance(decoded, dict) else {}
        except (ValueError, TypeError):
            body = {}
    messages = body.get("input") if isinstance(body.get("input"), list) else []
    user_text = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            user_text.append("".join(part.get("text", "") for part in content if isinstance(part, dict)))
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    return {
        "cli_reached_gateway_route": gateway_requests >= 1,
        # The CLI also probes sibling routes (for example a model list), which the gateway rejects by design.
        "gateway_served_exactly_one_request": gateway_error in (None, "gateway_route_rejected")
                                              and gateway_requests == 1,
        "exactly_one_upstream_request": len(stub_requests) == 1,
        "upstream_route_and_method": bool(stub_requests) and stub_requests[0]["method"] == "POST"
                                     and stub_requests[0]["path"] == "/v1/responses",
        "cli_sent_bearer_auth": bool(stub_requests) and all(item["bearer_auth"] for item in stub_requests),
        "persona_instructions_exact": body.get("instructions") == instructions,
        "only_our_prompt_as_input": len(messages) == 1 and user_text == [prompt]
                                    and all(isinstance(m, dict) and m.get("role") == "user" for m in messages),
        "tools_exactly_those_offered": [t.get("name") for t in tools if isinstance(t, dict)] == tool_names,
        "request_is_stateless_stream": body.get("stream") is True and body.get("store") is False,
        "cli_parsed_final_text": generate_error is None and result_text == marker,
        "no_connection_to_unexpected_hosts": not unexpected_hosts(proxy_attempts),
    }


def describe_failures(checks: dict[str, bool], generate_error: str | None, proxy_attempts: list[str]) -> str:
    failed = [name for name, ok in checks.items() if not ok]
    detail = "failed checks: " + ", ".join(failed)
    if generate_error:
        detail += f"; backend error code: {generate_error}"
    if unexpected_hosts(proxy_attempts):
        detail += f"; unexpected connection attempt(s), first: {unexpected_hosts(proxy_attempts)[0]}"
    return detail
