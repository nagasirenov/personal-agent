from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import imageio_ffmpeg

from personal_agent.animated_stickers import MAX_TGS_JSON_BYTES, StickerRenderError, render_sticker


def make_tgs(path: Path):
    position = {"a": 1, "k": [
        {"t": 0, "s": [12, 32], "e": [52, 32], "i": {"x": 1, "y": 1}, "o": {"x": 0, "y": 0}},
        {"t": 59, "s": [52, 32]},
    ]}
    data = {"v": "5.5.7", "fr": 30, "ip": 0, "op": 60, "w": 64, "h": 64,
            "layers": [{"ty": 4, "ind": 1, "ip": 0, "op": 60, "st": 0,
                        "ks": {"o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0},
                               "p": position, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}},
                        "shapes": [{"ty": "rc", "d": 1, "s": {"a": 0, "k": [16, 16]},
                                    "p": {"a": 0, "k": [0, 0]}, "r": {"a": 0, "k": 0}},
                                   {"ty": "fl", "c": {"a": 0, "k": [1, 0, 0, 1]}, "o": {"a": 0, "k": 100}}]}]}
    path.write_bytes(gzip.compress(json.dumps(data).encode()))


def make_webm(path: Path):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error',
                    '-f', 'lavfi', '-i', 'color=c=red:s=64x64:d=1:r=10',
                    '-f', 'lavfi', '-i', 'color=c=blue:s=64x64:d=1:r=10',
                    '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0',
                    '-c:v', 'libvpx-vp9', '-y', str(path)], check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)


def red_centroid(path: Path):
    with Image.open(path) as image:
        points = [x for y in range(image.height) for x in range(image.width)
                  if (lambda p: p[0] > 200 and p[1] < 80 and p[2] < 80)(image.getpixel((x, y)))]
    return sum(points) / len(points)


class AnimatedStickerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_actual_tgs_frames_show_motion(self):
        source = self.root / 'moving.tgs'
        make_tgs(source)
        result = await render_sticker(source, root=self.root, format='tgs')
        self.assertEqual(len(result.images), 4)
        self.assertEqual(result.timestamps, sorted(set(result.timestamps)))
        self.assertGreater(red_centroid(result.images[-1]), red_centroid(result.images[0]) + 20)

    async def test_actual_webm_frames_show_ordered_color_change(self):
        source = self.root / 'colors.webm'
        make_webm(source)
        result = await render_sticker(source, root=self.root, format='webm')
        self.assertEqual(len(result.images), 4)
        with Image.open(result.images[0]) as first, Image.open(result.images[-1]) as last:
            self.assertGreater(first.getpixel((16, 16))[0], 200)
            self.assertGreater(last.getpixel((16, 16))[2], 200)
        self.assertEqual(result.timestamps, sorted(set(result.timestamps)))

    async def test_malformed_and_oversized_gzip_leave_no_partial_frames(self):
        source = self.root / 'bad.tgs'
        for payload, code in ((b'not gzip', 'invalid_tgs'),
                              (gzip.compress(b' ' * (MAX_TGS_JSON_BYTES + 1)), 'tgs_json_too_large')):
            source.write_bytes(payload)
            with self.assertRaises(StickerRenderError) as caught:
                await render_sticker(source, root=self.root, format='tgs')
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(list((self.root / 'memory' / 'attachments').iterdir()), [])

    async def test_timeout_cleans_worker_and_output(self):
        source = self.root / 'valid.tgs'
        make_tgs(source)
        with patch('personal_agent.animated_stickers.RENDER_TIMEOUT', 0.0001):
            with self.assertRaises(StickerRenderError) as caught:
                await render_sticker(source, root=self.root, format='tgs')
        self.assertEqual(caught.exception.code, 'sticker_render_timeout')
        self.assertEqual(list((self.root / 'memory' / 'attachments').iterdir()), [])

    async def test_cancellation_cleans_worker_and_output(self):
        source = self.root / 'valid.tgs'
        make_tgs(source)
        task = asyncio.create_task(render_sticker(source, root=self.root, format='tgs'))
        await asyncio.sleep(.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(list((self.root / 'memory' / 'attachments').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
