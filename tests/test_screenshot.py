from __future__ import annotations

import struct
import zlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from personal_agent.screenshot import capture_screenshot, screenshot_health


class ScreenshotTests(unittest.TestCase):
    def test_capture_writes_only_agent_owned_screenshot_path(self) -> None:
        # Two pixels: red, then green, encoded as BGRX as returned by GDI.
        pixels = bytes((0, 0, 255, 0, 0, 255, 0, 0))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "owned"
            root.mkdir()
            outside = Path(temp) / "sentinel.txt"
            outside.write_text("unchanged", encoding="utf-8")
            with patch("personal_agent.screenshot._capture_desktop_bgrx", return_value=(2, 1, pixels)) as capture:
                target = capture_screenshot(root)
            capture.assert_called_once_with()
            self.assertEqual(target.parent, root.resolve() / "memory" / "screenshots")
            self.assertTrue(target.is_file())
            self.assertEqual(outside.read_text(encoding="utf-8"), "unchanged")
            payload = target.read_bytes()
            self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
            width, height = struct.unpack(">II", payload[16:24])
            self.assertEqual((width, height), (2, 1))
            self.assertEqual(zlib.decompress(payload[41:41 + struct.unpack('>I', payload[33:37])[0]]), b'\x00\xff\x00\x00\x00\xff\x00')

    def test_health_does_not_capture(self) -> None:
        with patch("personal_agent.screenshot._capture_desktop_bgrx") as capture:
            health = screenshot_health()
        capture.assert_not_called()
        self.assertEqual(health["method"], "win32_gdi")
        self.assertEqual(health["capture_mode"], "explicit_request_once")


if __name__ == "__main__":
    unittest.main()
