from __future__ import annotations

import http.client
import json
import tempfile
import unittest
from pathlib import Path

from personal_agent.browser_bridge import BrowserContextReceiver, get_or_create_browser_token


class BrowserContextReceiverTests(unittest.TestCase):
    def _snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "instance_id": "f27bc9ba-01aa-4f7d-98d1-374f7ef97b0c",
            "browser": "Chrome",
            "window_id": 2,
            "tab_id": 12,
            "status": "active",
            "url": "https://example.test/article?access_token=long-secret-token#fragment",
            "title": "Example article",
            "page_text": "Public page text. password=hunter2 should be redacted.",
            "selected_text": "Selected passage",
            "dom_summary": {
                "headings": ["Heading"],
                "links": [{"text": "More", "url": "https://example.test/next"}],
                "link_count": 1,
                "button_count": 2,
                "form_count": 0,
                "text_length": 60,
            },
        }

    def _request(self, receiver: BrowserContextReceiver, body: bytes, *, origin: str, token: str) -> int:
        connection = http.client.HTTPConnection("127.0.0.1", receiver.port, timeout=2)
        try:
            connection.request("POST", "/v1/context", body=body, headers={
                "Host": f"127.0.0.1:{receiver.port}",
                "Origin": origin,
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            })
            return connection.getresponse().status
        finally:
            connection.close()

    def _preflight(self, receiver: BrowserContextReceiver, *, origin: str) -> tuple[int, str]:
        connection = http.client.HTTPConnection("127.0.0.1", receiver.port, timeout=2)
        try:
            connection.request("OPTIONS", "/v1/context", headers={
                "Host": f"127.0.0.1:{receiver.port}",
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            })
            response = connection.getresponse()
            return response.status, response.getheader("Access-Control-Allow-Origin", "")
        finally:
            connection.close()

    def test_authenticated_extension_snapshot_is_sanitized_and_removed_on_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp) / "state"
            token = "t" * 48
            receiver = BrowserContextReceiver(data_dir, port=0, token=token)
            self.assertTrue(receiver.start())
            body = json.dumps(self._snapshot()).encode("utf-8")
            origin = "chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

            self.assertEqual(self._preflight(receiver, origin=origin), (204, origin))
            self.assertEqual(self._request(receiver, body, origin=origin, token=token), 204)
            snapshots = list(receiver.root.glob("*.json"))
            self.assertEqual(len(snapshots), 1)
            saved = json.loads(snapshots[0].read_text(encoding="utf-8"))
            self.assertEqual(saved["browser"], "Chrome")
            self.assertNotIn("long-secret-token", saved["url"])
            self.assertNotIn("hunter2", saved["page_text"])
            self.assertEqual(saved["selected_text"], "Selected passage")

            receiver.stop()
            self.assertEqual(list(receiver.root.glob("*.json")), [])

    def test_rejects_bad_origin_and_bad_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            token = "t" * 48
            receiver = BrowserContextReceiver(Path(temp) / "state", port=0, token=token)
            self.assertTrue(receiver.start())
            body = json.dumps(self._snapshot()).encode("utf-8")

            self.assertEqual(self._request(receiver, body, origin="https://example.test", token=token), 403)
            self.assertEqual(self._request(
                receiver, body, origin="chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", token="wrong-token",
            ), 401)
            self.assertEqual(list(receiver.root.glob("*.json")), [])
            receiver.stop()

    def test_token_is_persisted_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp) / "state"
            first = get_or_create_browser_token(data_dir)
            self.assertEqual(get_or_create_browser_token(data_dir), first)
            self.assertGreaterEqual(len(first), 32)


if __name__ == "__main__":
    unittest.main()
