"""Authenticated loopback receiver for the read-only browser companion."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import struct
import sys
import threading
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID

from .security import owned_path

_MAX_BODY_BYTES = 128 * 1024
_SNAPSHOT_TTL = timedelta(minutes=30)
_WS_PATH = "/v1/ws"
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_WS_SUBPROTOCOL_PREFIX = "pa."
_WS_MAX_MESSAGE = 512 * 1024
_WS_IDLE_SECONDS = 45.0  # the extension pings every ~20 s; silence for longer means the worker is gone
_WS_MAX_CLIENTS = 8
CAPTURE_TIMEOUT_SECONDS = 3.0
_TOKEN_FILENAME = "browser_context_token"
_EXTENSION_ORIGIN = re.compile(r"^[a-z][a-z0-9-]*-extension://[a-z0-9]{16,64}$", re.IGNORECASE)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b\d{5,}:[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b"),
    re.compile(r"(?i)\b(?:token|password|secret|api[_ -]?key)\s*[:=]\s*\S+"),
)


class _LocalHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def browser_token_path(data_dir: Path) -> Path:
    return owned_path(Path(data_dir), _TOKEN_FILENAME)


def get_or_create_browser_token(data_dir: Path) -> str:
    path = browser_token_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            value = secrets.token_urlsafe(32)
            handle.write(value + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return value
    except FileExistsError:
        value = path.read_text(encoding="utf-8").strip()
        if len(value) >= 32 and re.fullmatch(r"[A-Za-z0-9_-]+", value):
            return value
        raise ValueError("Browser companion token file is invalid") from None


def _bounded_text(value: object, limit: int, *, preserve_lines: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    text = value.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(char if char == "\n" or ord(char) >= 32 else " " for char in text)
    if not preserve_lines:
        text = " ".join(text.split())
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text[:limit]


def _safe_url(value: object) -> str:
    if not isinstance(value, str) or len(value) > 8192:
        return ""
    try:
        parsed = urlsplit(value)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port:
            host += f":{parsed.port}"
        query = []
        for key, val in parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=128):
            if re.search(r"token|secret|key|password|code|auth|session|credential", key, re.IGNORECASE):
                val = "[redacted]"
            query.append((key, val))
        result = urlunsplit((parsed.scheme.casefold(), host, parsed.path, urlencode(query), ""))
    except (ValueError, UnicodeError):
        return ""
    return _bounded_text(result, 2048)


def _bounded_count(value: object, maximum: int = 1_000_000) -> int:
    if type(value) is not int:
        return 0
    return min(max(value, 0), maximum)


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def normalize_browser_snapshot(raw: object, *, captured_at: datetime | None = None) -> dict | None:
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        return None
    try:
        instance_id = UUID(str(raw.get("instance_id"))).hex
    except (ValueError, TypeError, AttributeError):
        return None
    status = raw.get("status", "active")
    if status not in {"active", "closed"}:
        return None
    browser = raw.get("browser")
    browser = browser if browser in {"Chrome", "Edge", "Chromium"} else "Chromium"
    window_id = raw.get("window_id")
    tab_id = raw.get("tab_id")
    if type(window_id) is not int or window_id < 0:
        return None
    if type(tab_id) is not int or (tab_id < 0 and status != "closed"):
        return None

    dom_raw = raw.get("dom_summary") if isinstance(raw.get("dom_summary"), dict) else {}
    headings_raw = dom_raw.get("headings")
    headings = [
        _bounded_text(item, 160)
        for item in headings_raw[:24]
        if isinstance(item, str) and _bounded_text(item, 160)
    ] if isinstance(headings_raw, list) else []
    links_raw = dom_raw.get("links")
    links: list[dict[str, str]] = []
    if isinstance(links_raw, list):
        for item in links_raw[:20]:
            if not isinstance(item, dict):
                continue
            label = _bounded_text(item.get("text"), 120)
            href = _safe_url(item.get("url"))
            if label or href:
                links.append({"text": label, "url": href})

    closed = status == "closed"
    content_available = raw.get("content_available", True) is not False
    stamp = captured_at or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    stamp = stamp.astimezone(timezone.utc)
    interaction = _parse_time(raw.get("interaction_at"))
    if interaction is None or interaction > stamp + timedelta(minutes=2):
        interaction = stamp
    result = {
        "schema_version": 1,
        "instance_id": instance_id,
        "browser": browser,
        "window_id": window_id,
        "tab_id": tab_id,
        "captured_at": stamp.isoformat(),
        "last_interaction_at": interaction.isoformat(),
        "status": status,
        "content_available": content_available and not closed,
        "url": "" if closed else _safe_url(raw.get("url")),
        "title": "" if closed else _bounded_text(raw.get("title"), 400),
        "page_text": "" if closed or not content_available
        else _bounded_text(raw.get("page_text"), 20_000, preserve_lines=True),
        "selected_text": "" if closed or not content_available
        else _bounded_text(raw.get("selected_text"), 4_000, preserve_lines=True),
        "dom_summary": {
            "headings": [] if closed else headings,
            "links": [] if closed else links,
            "link_count": 0 if closed else _bounded_count(dom_raw.get("link_count")),
            "button_count": 0 if closed else _bounded_count(dom_raw.get("button_count")),
            "form_count": 0 if closed else _bounded_count(dom_raw.get("form_count")),
            "text_length": 0 if closed else _bounded_count(dom_raw.get("text_length")),
        },
    }
    if not closed and not result["url"]:
        return None
    return result


class _WSClient:
    """One authenticated extension WebSocket; sends are serialized, reads happen in the handler thread."""

    def __init__(self, connection) -> None:
        self.connection = connection
        self.send_lock = threading.Lock()
        self.closed = False

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        length = len(payload)
        if length < 126:
            header = bytes([0x80 | opcode, length])
        elif length < 65536:
            header = bytes([0x80 | opcode, 126]) + struct.pack(">H", length)
        else:
            header = bytes([0x80 | opcode, 127]) + struct.pack(">Q", length)
        with self.send_lock:
            if self.closed:
                raise OSError("websocket closed")
            self.connection.sendall(header + payload)

    def send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode("utf-8"))

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._send_frame(0x8, b"")
        except OSError:
            pass
        self.closed = True
        try:
            self.connection.shutdown(2)
        except OSError:
            pass
        try:
            self.connection.close()
        except OSError:
            pass


def _read_exact(stream, count: int) -> bytes:
    data = stream.read(count)
    if data is None or len(data) != count:
        raise ConnectionError("websocket closed")
    return data


def _read_ws_message(stream, client: _WSClient) -> str | None:
    """Next complete text message from the client, or None when it closed.  Answers ping frames."""
    message = bytearray()
    started = False
    while True:
        first, second = _read_exact(stream, 2)
        fin, opcode = bool(first & 0x80), first & 0x0F
        if first & 0x70 or not second & 0x80:  # reserved bits, or an unmasked client frame
            raise ConnectionError("websocket protocol error")
        length = second & 0x7F
        if length == 126:
            length = struct.unpack(">H", _read_exact(stream, 2))[0]
        elif length == 127:
            length = struct.unpack(">Q", _read_exact(stream, 8))[0]
        if length > _WS_MAX_MESSAGE or len(message) + length > _WS_MAX_MESSAGE:
            raise ConnectionError("websocket message too large")
        mask = _read_exact(stream, 4)
        payload = bytearray(_read_exact(stream, length)) if length else bytearray()
        for index in range(length):
            payload[index] ^= mask[index % 4]
        if opcode == 0x8:
            return None
        if opcode == 0x9:
            client._send_frame(0xA, bytes(payload[:125]))
            continue
        if opcode == 0xA:
            continue
        if opcode in {0x1, 0x2} and not started:
            started = True
        elif opcode != 0x0 or not started:
            raise ConnectionError("websocket protocol error")
        message.extend(payload)
        if fin:
            return message.decode("utf-8")


class _PendingCapture:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.replied: set[int] = set()


class BrowserContextReceiver:
    """Small authenticated HTTP receiver bound exclusively to IPv4 loopback."""

    def __init__(self, data_dir: Path, *, host: str = "127.0.0.1", port: int = 17842, token: str | None = None):
        if host != "127.0.0.1":
            raise ValueError("Browser receiver must bind to 127.0.0.1")
        if not 0 <= port <= 65535:
            raise ValueError("Invalid browser receiver port")
        self.data_dir = Path(data_dir).resolve()
        self.root = owned_path(self.data_dir, "browser-context")
        self.host = host
        self.port = port
        self.token = token or get_or_create_browser_token(self.data_dir)
        if len(self.token) < 32:
            raise ValueError("Browser companion token must contain at least 32 characters")
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._error_type: str | None = None
        self._ws_lock = threading.Lock()
        self._ws_clients: list[_WSClient] = []
        self._pending: dict[str, _PendingCapture] = {}

    def start(self) -> bool:
        if self._server is not None:
            return True
        self.root.mkdir(parents=True, exist_ok=True)
        self._prune_stale_snapshots()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "PersonalAgentBrowserBridge/1"
            sys_version = ""

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _valid_request_origin(self) -> str | None:
                origin = self.headers.get("Origin", "")
                expected_host = f"127.0.0.1:{self.server.server_port}"
                if self.path != "/v1/context" or self.headers.get("Host", "").casefold() != expected_host.casefold():
                    return None
                if not _EXTENSION_ORIGIN.fullmatch(origin):
                    return None
                return origin

            def _respond(self, status: int, origin: str | None = None) -> None:
                self.send_response(status)
                if origin:
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                    self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                    self.send_header("Vary", "Origin")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_OPTIONS(self) -> None:
                origin = self._valid_request_origin()
                self._respond(204 if origin else 403, origin)

            def do_POST(self) -> None:
                origin = self._valid_request_origin()
                if not origin:
                    self._respond(403)
                    return
                supplied = self.headers.get("Authorization", "")
                expected = "Bearer " + receiver.token
                if not hmac.compare_digest(supplied, expected):
                    self._respond(401, origin)
                    return
                if self.headers.get_content_type() != "application/json":
                    self._respond(415, origin)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    length = -1
                if length < 0 or length > _MAX_BODY_BYTES:
                    self._respond(413 if length > _MAX_BODY_BYTES else 400, origin)
                    return
                try:
                    raw = json.loads(self.rfile.read(length))
                    snapshot = normalize_browser_snapshot(raw)
                    if snapshot is None:
                        self._respond(400, origin)
                        return
                    receiver._write_snapshot(snapshot)
                except (OSError, ValueError, json.JSONDecodeError):
                    receiver._error_type = "snapshot_write_error"
                    self._respond(500, origin)
                    return
                receiver._error_type = None
                self._respond(204, origin)

            def do_GET(self) -> None:
                if self.path == _WS_PATH:
                    receiver._serve_websocket(self)
                    return
                self._respond(404)

        try:
            server = _LocalHTTPServer((self.host, self.port), Handler)
            server.daemon_threads = True
            server.timeout = 0.25
            self._server = server
            self.port = int(server.server_port)
            self._thread = threading.Thread(target=server.serve_forever, name="browser-context-http", daemon=True)
            self._thread.start()
            self._error_type = None
            return True
        except OSError as exc:
            self._error_type = type(exc).__name__
            self._server = None
            self._thread = None
            return False

    # -- live capture over WebSocket ------------------------------------------------------
    def _reject(self, handler, status: int) -> None:
        handler.send_response(status)
        handler.send_header("Content-Length", "0")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True

    def _serve_websocket(self, handler) -> None:
        """Handshake (loopback Host, extension Origin, bearer token as subprotocol), then the message loop."""
        headers = handler.headers
        if (headers.get("Host", "").casefold() != f"127.0.0.1:{handler.server.server_port}".casefold()
                or not _EXTENSION_ORIGIN.fullmatch(headers.get("Origin", ""))):
            self._reject(handler, 403)
            return
        offered = [item.strip() for item in headers.get("Sec-WebSocket-Protocol", "").split(",")]
        expected = _WS_SUBPROTOCOL_PREFIX + self.token
        accepted = next((item for item in offered if hmac.compare_digest(item, expected)), None)
        if accepted is None:
            self._reject(handler, 401)
            return
        key = headers.get("Sec-WebSocket-Key", "")
        try:
            valid_key = len(base64.b64decode(key, validate=True)) == 16
        except ValueError:
            valid_key = False
        if (headers.get("Upgrade", "").casefold() != "websocket"
                or "upgrade" not in headers.get("Connection", "").casefold()
                or headers.get("Sec-WebSocket-Version") != "13" or not valid_key):
            self._reject(handler, 400)
            return
        accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()).decode("ascii")
        handler.send_response(101, "Switching Protocols")
        handler.send_header("Upgrade", "websocket")
        handler.send_header("Connection", "Upgrade")
        handler.send_header("Sec-WebSocket-Accept", accept)
        handler.send_header("Sec-WebSocket-Protocol", accepted)
        handler.end_headers()
        handler.wfile.flush()
        handler.close_connection = True
        client = _WSClient(handler.connection)
        handler.connection.settimeout(_WS_IDLE_SECONDS)
        with self._ws_lock:
            self._ws_clients.append(client)
            surplus = self._ws_clients[:-_WS_MAX_CLIENTS]
            del self._ws_clients[:-_WS_MAX_CLIENTS]
        for old in surplus:
            old.close()
        try:
            while not client.closed:
                text = _read_ws_message(handler.rfile, client)
                if text is None:
                    break
                self._handle_ws_message(client, text)
        except (OSError, ConnectionError, ValueError, struct.error):
            pass
        finally:
            with self._ws_lock:
                if client in self._ws_clients:
                    self._ws_clients.remove(client)
            client.close()

    def _handle_ws_message(self, client: _WSClient, text: str) -> None:
        message = json.loads(text)
        if not isinstance(message, dict):
            return
        if message.get("type") == "ping":
            client.send_text('{"type":"pong"}')
            return
        if message.get("type") != "snapshot":
            return
        snapshot = normalize_browser_snapshot(message.get("snapshot"))
        if snapshot is not None:
            try:
                self._write_snapshot(snapshot)
                self._error_type = None
            except (OSError, ValueError):
                self._error_type = "snapshot_write_error"
        request_id = message.get("request_id")
        if isinstance(request_id, str):
            with self._ws_lock:
                pending = self._pending.get(request_id)
            if pending is not None:
                with pending.condition:
                    pending.replied.add(id(client))
                    pending.condition.notify_all()

    def request_capture(self, timeout: float = CAPTURE_TIMEOUT_SECONDS) -> dict[str, int]:
        """Ask every connected extension for a fresh snapshot and wait for the replies.

        Blocking and thread-safe: call it from a worker thread (asyncio.to_thread).  The fresh snapshots are
        stored like any other, so callers simply re-read the current contexts afterwards.
        """
        with self._ws_lock:
            clients = list(self._ws_clients)
        if not clients:
            return {"clients": 0, "responded": 0}
        request_id = secrets.token_hex(8)
        pending = _PendingCapture()
        with self._ws_lock:
            self._pending[request_id] = pending
        try:
            message = json.dumps({"type": "capture", "request_id": request_id})
            targets = []
            for client in clients:
                try:
                    client.send_text(message)
                    targets.append(client)
                except OSError:
                    client.close()
            wanted = {id(client) for client in targets}
            with pending.condition:
                pending.condition.wait_for(lambda: wanted <= pending.replied, timeout)
                responded = len(wanted & pending.replied)
            return {"clients": len(targets), "responded": responded}
        finally:
            with self._ws_lock:
                self._pending.pop(request_id, None)

    def websocket_clients(self) -> int:
        with self._ws_lock:
            return len(self._ws_clients)

    def _write_snapshot(self, snapshot: dict) -> None:
        path = owned_path(self.root, f"{snapshot['instance_id']}.json")
        temp = owned_path(self.root, f".{snapshot['instance_id']}.{secrets.token_hex(8)}.tmp")
        payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
        if len(payload.encode("utf-8")) > _MAX_BODY_BYTES:
            raise ValueError("Normalized browser snapshot exceeds the size limit")
        try:
            with temp.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def _prune_stale_snapshots(self) -> None:
        cutoff = datetime.now(timezone.utc) - _SNAPSHOT_TTL
        for path in self.root.glob("*.json"):
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                if modified < cutoff:
                    owned_path(self.root, path.name).unlink(missing_ok=True)
            except (OSError, ValueError):
                continue

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        with self._ws_lock:
            clients, self._ws_clients = self._ws_clients, []
        for client in clients:
            client.close()
        if server:
            server.shutdown()
            server.server_close()
        if thread and thread.is_alive():
            thread.join(timeout=2)
        if self.root.is_dir():
            for path in self.root.glob("*.json"):
                try:
                    if not path.is_symlink() and path.is_file():
                        owned_path(self.root, path.name).unlink(missing_ok=True)
                except (OSError, ValueError):
                    continue

    def health(self) -> dict[str, object]:
        return {
            "available": self._server is not None and self._error_type is None,
            "host": self.host,
            "port": self.port,
            "error_type": self._error_type,
            "token_path": str(browser_token_path(self.data_dir)),
            "live_clients": self.websocket_clients(),
            "active_snapshots": sum(1 for _ in self.root.glob("*.json")) if self.root.is_dir() else 0,
        }
