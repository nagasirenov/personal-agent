"""Live capture over the loopback WebSocket, the 30-minute snapshot lifetime and metadata-only snapshots."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import socket
import struct
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from personal_agent.adapters.browser import BrowserContextObserver
from personal_agent.browser_bridge import (
    BrowserContextReceiver, _SNAPSHOT_TTL, _WS_GUID, normalize_browser_snapshot,
)
from personal_agent.context import _select_browser_context

TOKEN = "t" * 43
ORIGIN = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
INSTANCE = "f27bc9ba-01aa-4f7d-98d1-374f7ef97b0c"


def snapshot(**extra) -> dict:
    data = {"schema_version": 1, "instance_id": INSTANCE, "browser": "Edge", "window_id": 1, "tab_id": 5,
            "status": "active", "url": "https://example.test/page", "title": "Page",
            "page_text": "BODY_TEXT", "selected_text": "", "dom_summary": {}}
    data.update(extra)
    return data


class FakeExtension:
    """A minimal RFC 6455 client, the way the extension's WebSocket behaves."""

    def __init__(self, port: int, *, token: str = TOKEN, origin: str = ORIGIN, host: str | None = None):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=3)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (f"GET /v1/ws HTTP/1.1\r\nHost: {host or f'127.0.0.1:{port}'}\r\nOrigin: {origin}\r\n"
                   f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                   f"Sec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: pa.{token}\r\n\r\n")
        self.sock.sendall(request.encode())
        self.reader = self.sock.makefile("rb")
        status = self.reader.readline().decode()
        self.status = int(status.split()[1])
        headers = {}
        while True:
            line = self.reader.readline().decode().strip()
            if not line:
                break
            name, _, value = line.partition(":")
            headers[name.casefold()] = value.strip()
        self.headers = headers
        self.expected_accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()

    def send(self, message: dict, opcode: int = 1) -> None:
        payload = json.dumps(message).encode()
        mask = os.urandom(4)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        if len(payload) < 126:
            header = bytes([0x80 | opcode, 0x80 | len(payload)])
        else:
            header = bytes([0x80 | opcode, 0x80 | 126]) + struct.pack(">H", len(payload))
        self.sock.sendall(header + mask + masked)

    def receive(self) -> dict:
        first, second = self.reader.read(2)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack(">H", self.reader.read(2))[0]
        return json.loads(self.reader.read(length))

    def close(self) -> None:
        try:
            self.reader.close()
            self.sock.close()
        except OSError:
            pass


class LiveCaptureTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.receiver = BrowserContextReceiver(Path(self._temp.name), port=0, token=TOKEN)
        self.assertTrue(self.receiver.start())
        self.addCleanup(self.receiver.stop)
        self.clients: list[FakeExtension] = []
        self.addCleanup(lambda: [client.close() for client in self.clients])

    def connect(self, **kwargs) -> FakeExtension:
        client = FakeExtension(self.receiver.port, **kwargs)
        self.clients.append(client)
        return client

    def wait_for_clients(self, count: int) -> None:
        for _ in range(200):
            if self.receiver.websocket_clients() == count:
                return
            threading.Event().wait(0.01)
        self.fail("websocket clients did not reach expected count")

    def test_handshake_accepts_correct_token_and_origin(self):
        client = self.connect()
        self.assertEqual(client.status, 101)
        self.assertEqual(client.headers["sec-websocket-accept"], client.expected_accept)
        self.assertEqual(client.headers["sec-websocket-protocol"], "pa." + TOKEN)
        self.wait_for_clients(1)
        client.send({"type": "ping"})
        self.assertEqual(client.receive(), {"type": "pong"})

    def test_bad_token_bad_origin_and_bad_host_are_rejected(self):
        self.assertEqual(self.connect(token="x" * 43).status, 401)
        self.assertEqual(self.connect(origin="https://evil.example").status, 403)
        self.assertEqual(self.connect(host="evil.example:80").status, 403)
        self.assertEqual(self.receiver.websocket_clients(), 0)

    def test_capture_request_returns_fresh_snapshot_tagged_with_request_id(self):
        client = self.connect()
        self.wait_for_clients(1)

        def extension() -> None:
            message = client.receive()
            assert message["type"] == "capture"
            client.send({"type": "snapshot", "request_id": message["request_id"],
                         "snapshot": snapshot(page_text="FRESH_TEXT")})

        thread = threading.Thread(target=extension)
        thread.start()
        reply = self.receiver.request_capture(timeout=3)
        thread.join(3)
        self.assertEqual(reply, {"clients": 1, "responded": 1})
        stored = json.loads((self.receiver.root / f"{INSTANCE.replace('-', '')}.json").read_text("utf-8"))
        self.assertEqual(stored["page_text"], "FRESH_TEXT")

    def test_no_client_connected_falls_back_without_waiting(self):
        self.assertEqual(self.receiver.request_capture(timeout=3), {"clients": 0, "responded": 0})

    def test_silent_client_times_out_and_snapshot_for_other_request_does_not_count(self):
        client = self.connect()
        self.wait_for_clients(1)

        def extension() -> None:
            client.receive()
            client.send({"type": "snapshot", "request_id": "someone-elses", "snapshot": snapshot()})

        thread = threading.Thread(target=extension)
        thread.start()
        started = datetime.now()
        reply = self.receiver.request_capture(timeout=0.4)
        thread.join(3)
        self.assertEqual(reply, {"clients": 1, "responded": 0})
        self.assertLess((datetime.now() - started).total_seconds(), 2)

    def test_closed_client_is_dropped_and_request_capture_works_from_event_loop_thread(self):
        client = self.connect()
        self.wait_for_clients(1)
        client.close()
        for _ in range(200):
            if self.receiver.websocket_clients() == 0:
                break
            threading.Event().wait(0.01)
        self.assertEqual(self.receiver.websocket_clients(), 0)

        async def run() -> dict:
            return await asyncio.to_thread(self.receiver.request_capture, 0.2)

        self.assertEqual(asyncio.run(run()), {"clients": 0, "responded": 0})

    def test_unmasked_or_oversized_frames_close_the_connection(self):
        client = self.connect()
        self.wait_for_clients(1)
        client.sock.sendall(bytes([0x81, 0x02]) + b"{}")  # client frames must be masked
        self.wait_for_clients(0)


class SnapshotLifetimeTests(unittest.TestCase):
    def test_snapshots_live_for_thirty_minutes(self):
        self.assertEqual(_SNAPSHOT_TTL, timedelta(minutes=30))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            now = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
            clock = [now]
            observer = BrowserContextObserver(root, clock=lambda: clock[0])
            raw = snapshot(captured_at=now.isoformat())
            (root / f"{INSTANCE.replace('-', '')}.json").write_text(json.dumps(
                normalize_browser_snapshot(raw, captured_at=now)), encoding="utf-8")
            clock[0] = now + timedelta(minutes=25)
            self.assertEqual(observer.current_context()["url"], "https://example.test/page")
            clock[0] = now + timedelta(minutes=31)
            self.assertIsNone(observer.current_context())

    def test_metadata_only_snapshot_keeps_url_and_title_without_text(self):
        normalized = normalize_browser_snapshot(snapshot(content_available=False, page_text="LEAK", selected_text="LEAK"))
        self.assertFalse(normalized["content_available"])
        self.assertEqual(normalized["url"], "https://example.test/page")
        self.assertEqual(normalized["page_text"], "")
        self.assertEqual(normalized["selected_text"], "")
        self.assertTrue(normalize_browser_snapshot(snapshot())["content_available"])  # old extensions

    def test_timeline_event_for_metadata_only_page_has_no_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            now = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
            observer = BrowserContextObserver(root, clock=lambda: now)
            normalized = normalize_browser_snapshot(snapshot(content_available=False), captured_at=now)
            (root / f"{INSTANCE.replace('-', '')}.json").write_text(json.dumps(normalized), encoding="utf-8")
            events = observer.poll()
            self.assertEqual(len(events), 1)
            self.assertFalse(events[0].data["content_available"])
            self.assertNotIn("page_text", events[0].data)


class SelectionFallbackTests(unittest.TestCase):
    def test_preferred_browser_without_snapshot_falls_back_to_freshest_other_browser(self):
        chrome = {"browser": "Chrome", "captured_at": "2026-10-28T10:00:00+00:00", "url": "https://c.test/"}
        chromium = {"browser": "Chromium", "captured_at": "2026-10-28T11:00:00+00:00", "url": "https://d.test/"}
        self.assertEqual(_select_browser_context([chrome, chromium], {"exe": "msedge.exe"})["url"], "https://d.test/")
        edge = {"browser": "Edge", "captured_at": "2026-10-28T09:00:00+00:00", "url": "https://e.test/"}
        self.assertEqual(_select_browser_context([chrome, chromium, edge], {"exe": "msedge.exe"})["url"], "https://e.test/")
        self.assertIsNone(_select_browser_context([], {"exe": "msedge.exe"}))
        self.assertIsNone(_select_browser_context([{"url": "https://x.test/"}], {"exe": "msedge.exe"}))


class ReadBrowserPageToolTests(unittest.IsolatedAsyncioTestCase):
    async def _turn(self, contexts, capture=None):
        from tests.test_tools import turn_for
        from tests.test_service_observation_integration import ObservationHarness
        harness = ObservationHarness()
        self.addAsyncCleanup(harness.close)
        turn = turn_for(harness)
        turn.browser_context_provider = lambda: contexts
        turn.browser_capture_provider = capture
        return turn

    def page(self, **extra) -> dict:
        captured = datetime.now(timezone.utc) - timedelta(minutes=12)
        data = {"browser": "Edge", "url": "https://example.test/a", "title": "A", "page_text": "TEXT",
                "captured_at": captured.isoformat(), "last_interaction_at": captured.isoformat(),
                "content_available": True}
        data.update(extra)
        return data

    async def test_stale_page_is_reported_with_age_and_no_fresh_capture(self):
        turn = await self._turn([self.page()])
        result = await turn.read_browser_page({})
        self.assertTrue(result["ok"])
        self.assertFalse(result["fresh_capture"])
        self.assertTrue(result["data"]["content_available"])
        self.assertGreaterEqual(result["data"]["age_seconds"], 700)
        self.assertIn("not live", result["note"])

    async def test_fresh_capture_is_requested_and_reported(self):
        calls = []
        contexts = [self.page()]

        def capture(timeout):
            calls.append(timeout)
            contexts[:] = [self.page(captured_at=datetime.now(timezone.utc).isoformat(), page_text="NEW")]
            return {"clients": 1, "responded": 1}

        turn = await self._turn(contexts, capture)
        result = await turn.read_browser_page({})
        self.assertEqual(len(calls), 1)
        self.assertTrue(result["fresh_capture"])
        self.assertEqual(result["data"]["page_text"], "NEW")
        self.assertLess(result["data"]["age_seconds"], 5)
        self.assertNotIn("note", result)

    async def test_capture_failure_falls_back_to_last_snapshot(self):
        def broken(timeout):
            raise RuntimeError("socket gone")

        turn = await self._turn([self.page()], broken)
        result = await turn.read_browser_page({})
        self.assertTrue(result["ok"])
        self.assertFalse(result["fresh_capture"])

    async def test_metadata_only_page_suggests_web_open(self):
        turn = await self._turn([self.page(content_available=False, page_text="")])
        result = await turn.read_browser_page({})
        self.assertTrue(result["ok"])
        self.assertFalse(result["data"]["content_available"])
        self.assertIn("web_open", result["note"])
        self.assertEqual(result["data"]["url"], "https://example.test/a")

    async def test_no_snapshot_still_reports_unavailable(self):
        turn = await self._turn([])
        result = await turn.read_browser_page({})
        self.assertEqual(result["error"]["code"], "browser_page_unavailable")


if __name__ == "__main__":
    unittest.main()
