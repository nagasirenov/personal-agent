from pathlib import Path
import struct
import tempfile
import unittest

import httpx

from personal_agent.screenshot import _encode_png
from personal_agent.telegram import TelegramClient, TelegramError


class TelegramUploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_screenshot_upload_is_multipart_and_large_dimensions_use_document(self):
        requests = []
        async def handle(request):
            requests.append((request.url.path, request.headers["content-type"], await request.aread()))
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
        http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        client = TelegramClient("123:private", client=http)
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "screen.png"
                path.write_bytes(_encode_png(1, 1, bytes((0, 0, 255, 0))))
                await client.send_photo(101, path, caption="По запросу")
                self.assertTrue(requests[0][0].endswith("/sendPhoto"))
                self.assertIn("multipart/form-data", requests[0][1])
                self.assertIn(b'name="photo"', requests[0][2])
                path.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 10000, 100))
                await client.send_photo(101, path)
                self.assertTrue(requests[1][0].endswith("/sendDocument"))
                self.assertIn(b'name="document"', requests[1][2])
        finally:
            await client.close()

    async def test_upload_error_never_exposes_token(self):
        async def handle(request):
            raise httpx.ConnectError("failure with URL " + str(request.url), request=request)
        client = TelegramClient("123:private", client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "screen.png"
                path.write_bytes(_encode_png(1, 1, bytes((0, 0, 255, 0))))
                with self.assertRaises(TelegramError) as failure:
                    await client.send_photo(101, path)
                self.assertNotIn("123:private", str(failure.exception))
        finally:
            await client.close()


class TelegramFileUploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_document_video_audio_are_streamed_multipart_to_the_configured_api(self):
        seen = []

        async def handle(request):
            seen.append((str(request.url), await request.aread()))
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
        client = TelegramClient("123:private", client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
                                api_base="http://127.0.0.1:8081/")
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "clip.mp4"
                path.write_bytes(b"0" * 100)
                self.assertEqual(await client.send_document(5, path, caption="c"), "sendDocument")
                self.assertEqual(await client.send_video(5, path), "sendVideo")
                self.assertEqual(await client.send_audio(5, path), "sendAudio")
            self.assertTrue(all(url.startswith("http://127.0.0.1:8081/bot123:private/") for url, _ in seen))
            self.assertIn(b'name="document"', seen[0][1])
            self.assertIn(b'name="supports_streaming"', seen[1][1])
            self.assertIn(b'name="audio"', seen[2][1])
        finally:
            await client.close()
