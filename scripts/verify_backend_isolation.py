from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import sys
import tempfile
import threading
import tomllib
import zlib
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personal_agent.backends import BackendError, CodexSubscriptionBackend  # noqa: E402


class _CaptureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _CaptureHandler)
        self.captured: dict[str, Any] | None = None
        self.captured_event = threading.Event()


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    transfer_encoding = handler.headers.get("Transfer-Encoding", "").lower()
    if "chunked" not in transfer_encoding:
        length = int(handler.headers.get("Content-Length", "0"))
        return handler.rfile.read(length)

    chunks: list[bytes] = []
    while True:
        line = handler.rfile.readline().strip()
        size = int(line.split(b";", 1)[0], 16)
        if size == 0:
            handler.rfile.readline()
            break
        chunks.append(handler.rfile.read(size))
        handler.rfile.read(2)
    return b"".join(chunks)


class _CaptureHandler(BaseHTTPRequestHandler):
    server: _CaptureServer

    def do_POST(self) -> None:  # noqa: N802
        raw = _read_body(self)
        if self.server.captured is None:
            self.server.captured = {
                "path": self.path,
                "headers": {key.lower(): value for key, value in self.headers.items()},
                "raw": raw,
            }
            self.server.captured_event.set()

        payload = b'{"error":{"message":"intentional local isolation capture stop"}}'
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


class _LoopbackCaptureBackend(CodexSubscriptionBackend):
    def __init__(self, base_url: str, **kwargs: Any) -> None:
        self._capture_base_url = base_url.rstrip("/")
        super().__init__(**kwargs)

    def _upstream_url(self) -> str:
        # Capture gateway egress: that is the payload the real model would see.
        return self._capture_base_url + "/responses"

    def _gateway_auth_required(self) -> bool:
        # The fake upstream must never receive a subscription credential.
        return False


def _decode_body(raw: bytes, content_encoding: str) -> bytes:
    encoding = content_encoding.strip().lower()
    if encoding in {"", "identity"}:
        return raw
    if encoding == "gzip":
        return gzip.decompress(raw)
    if encoding == "deflate":
        return zlib.decompress(raw)
    if encoding == "zstd":
        try:
            import zstandard  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "wire payload used zstd Content-Encoding, but Python zstandard is unavailable"
            ) from exc
        return zstandard.ZstdDecompressor().decompress(raw)
    raise RuntimeError(f"unsupported wire Content-Encoding: {encoding!r}")


def _collect_user_instruction_values(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "user_instructions" and isinstance(child, str) and child.strip():
                found.append(child)
            found.extend(_collect_user_instruction_values(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_user_instruction_values(child))
    return found


def _host_private_markers() -> tuple[list[str], list[str]]:
    agents_markers: list[str] = []
    agents_path = Path.home() / ".codex" / "AGENTS.md"
    if agents_path.is_file():
        text = agents_path.read_text(encoding="utf-8", errors="replace")
        agents_markers = [
            line.strip()
            for line in text.splitlines()
            if len(line.strip()) >= 24 and not line.lstrip().startswith("#")
        ]

    instruction_values: list[str] = []
    config_path = Path.home() / ".codex" / "config.toml"
    if config_path.is_file():
        try:
            with config_path.open("rb") as handle:
                parsed = tomllib.load(handle)
            instruction_values = _collect_user_instruction_values(parsed)
        except (OSError, tomllib.TOMLDecodeError):
            pass
    return agents_markers, instruction_values


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _input_messages(value: object) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [("user", value)]
    if not isinstance(value, list):
        return []
    messages: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if isinstance(role, str):
            messages.append((role, _message_text(item.get("content"))))
    return messages


def _tool_labels(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    labels: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            labels.append(type(item).__name__)
            continue
        name = item.get("name")
        tool_type = item.get("type")
        function = item.get("function")
        if isinstance(name, str):
            labels.append(name)
        elif isinstance(function, dict) and isinstance(function.get("name"), str):
            labels.append(function["name"])
        elif isinstance(tool_type, str):
            labels.append(tool_type)
        else:
            labels.append("unknown")
    return labels


async def _invoke(base_url: str, work_root: Path, prompt: str) -> str | None:
    backend = _LoopbackCaptureBackend(
        base_url,
        model="gpt-6-sol",
        reasoning_effort="high",
        work_root=work_root,
    )
    try:
        await backend.generate(prompt, "isolation-wire-capture")
    except BackendError as exc:
        return exc.code
    finally:
        await backend.close()
    return None


def main() -> int:
    run_id = uuid4().hex
    synthetic_marker = f"PERSONAL_AGENT_SYNTHETIC_AGENTS_{run_id}"
    prompt = f"PERSONAL_AGENT_ISOLATION_PROMPT_{run_id}"
    host_agents_markers, host_user_instructions = _host_private_markers()

    server = _CaptureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backend_error: str | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="personal-agent-isolation-") as temp:
            controlled_parent = Path(temp)
            (controlled_parent / "AGENTS.md").write_text(
                "# AGENTS.md\n" + synthetic_marker + "\n",
                encoding="utf-8",
            )
            base_url = f"http://127.0.0.1:{server.server_port}/v1"
            backend_error = asyncio.run(
                _invoke(base_url, controlled_parent, prompt)
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    captured = server.captured
    if captured is None:
        print("FAIL: Codex did not reach the local Responses endpoint.")
        print(f"backend_error={backend_error or 'none'}")
        return 1

    headers = captured["headers"]
    try:
        decoded = _decode_body(captured["raw"], headers.get("content-encoding", ""))
        body = json.loads(decoded)
    except (RuntimeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"FAIL: could not decode captured request body: {exc}")
        return 1

    serialized = json.dumps(body, ensure_ascii=False, sort_keys=True)
    instructions = body.get("instructions")
    messages = _input_messages(body.get("input"))
    user_messages = [text for role, text in messages if role == "user"]
    non_user_messages = [(role, text) for role, text in messages if role != "user" and text]

    host_agents_matches = sum(1 for marker in host_agents_markers if marker in serialized)
    host_user_instruction_matches = sum(
        1 for marker in host_user_instructions if marker in serialized
    )
    coordinator_markers = (
        "Act as the model backend for the Codex task encoded below.",
        "CODEX_LATEST_USER_PROMPT_JSON",
        "codex_transport_resume",
    )
    coordinator_matches = sum(marker in serialized for marker in coordinator_markers)
    host_subagent_rule_matches = sum(
        marker in serialized
        for marker in host_agents_markers
        if "субагент" in marker.lower() and "разреш" in marker.lower()
    )
    tools = body.get("tools")
    tool_labels = _tool_labels(tools)
    role_counts = Counter(role for role, _ in messages)
    persona_locations = int(
        isinstance(instructions, str)
        and instructions == CodexSubscriptionBackend.APPLICATION_INSTRUCTIONS
    ) + sum(
        text == CodexSubscriptionBackend.APPLICATION_INSTRUCTIONS
        for _, text in messages
    )
    prompt_match_count = sum(text == prompt for role, text in messages if role == "user")
    expected_message_matches = prompt_match_count + sum(
        text == CodexSubscriptionBackend.APPLICATION_INSTRUCTIONS
        for _, text in messages
    )
    unexpected_message_count = len(
        [(role, text) for role, text in messages if text]
    ) - expected_message_matches

    checks = {
        "authorization_header_absent": "authorization" not in headers,
        "tools_empty": tools is None or tools == [],
        "synthetic_agents_marker_absent": synthetic_marker not in serialized,
        "host_agents_content_absent": host_agents_matches == 0,
        "agents_instruction_frame_absent": "# AGENTS.md" not in serialized,
        "coordinator_context_absent": coordinator_matches == 0,
        "host_user_instructions_absent": host_user_instruction_matches == 0,
        "persona_exact_once": persona_locations == 1,
        "provided_prompt_exact_once": prompt_match_count == 1,
        "only_persona_and_prompt_messages": unexpected_message_count == 0,
    }

    print(f"request_path={captured['path']}")
    print(f"content_encoding={headers.get('content-encoding', 'identity') or 'identity'}")
    print(f"body_sha256={hashlib.sha256(decoded).hexdigest()}")
    print(f"host_agents_marker_count={len(host_agents_markers)}")
    print(f"host_agents_match_count={host_agents_matches}")
    print(f"host_subagent_rule_match_count={host_subagent_rule_matches}")
    print(f"agents_instruction_frame_count={serialized.count('# AGENTS.md')}")
    print(f"host_user_instructions_value_count={len(host_user_instructions)}")
    print(f"host_user_instructions_match_count={host_user_instruction_matches}")
    print(f"coordinator_marker_match_count={coordinator_matches}")
    print(f"tools_count={len(tools) if isinstance(tools, list) else -1}")
    print(f"tool_labels={json.dumps(tool_labels, ensure_ascii=True)}")
    print(f"input_message_count={len(messages)}")
    print(f"input_role_counts={json.dumps(dict(role_counts), sort_keys=True)}")
    print(f"user_message_count={len(user_messages)}")
    print(f"provided_prompt_match_count={prompt_match_count}")
    print(f"non_user_message_count={len(non_user_messages)}")
    print(f"unexpected_message_count={unexpected_message_count}")
    print(f"instructions_chars={len(instructions) if isinstance(instructions, str) else -1}")
    print(f"persona_chars={len(CodexSubscriptionBackend.APPLICATION_INSTRUCTIONS)}")
    print(f"persona_exact_location_count={persona_locations}")
    print(f"backend_error_after_capture={backend_error or 'none'}")
    for name, ok in checks.items():
        print(f"{name}={'PASS' if ok else 'FAIL'}")

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print("FAIL: " + ", ".join(failed))
        return 1
    print("PASS: captured Codex Responses payload is isolated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
