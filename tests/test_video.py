"""Offline tests for video understanding: formats, limits, transcripts, frames, isolation, attachments."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from personal_agent.attachments import AttachmentPipeline
from personal_agent.models import Identity, IncomingMessage
from personal_agent.testing import ScriptedBrain, ScriptedTurn, make_synthetic_video
from personal_agent.tools import TurnTools
from personal_agent.video import (
    ANALYZED_SECONDS,
    Cue,
    MAX_DOWNLOAD_BYTES,
    MAX_DURATION_SECONDS,
    Sheet,
    VideoAnalyzer,
    VideoError,
    cleanup_videos,
    compose_sheets,
    error_code_for,
    format_transcript,
    parse_subtitles,
    pick_subtitle_track,
    select_formats,
    smallest_formats,
)
from tests.test_attachment_vision_workflows import packet_of
from tests.test_service_observation_integration import ObservationHarness, _message

PRIMARY_ID = 101
SECONDARY_ID = 202


@dataclass
class FakeConfig:
    data_dir: Path
    max_attachment_bytes: int = 20 * 1024 * 1024
    tz: timezone = timezone.utc
    ffmpeg_executable: str = ""


class FakeStore:
    def __init__(self) -> None:
        self.attachments: list[dict] = []
        self.events: list[object] = []
        self.messages: list[tuple[int, str, str]] = []
        self.state: dict[str, object] = {}

    def record_attachment(self, metadata: dict) -> None:
        self.attachments.append(dict(metadata))

    def record_event(self, event) -> None:
        self.events.append(event)

    def record_message(self, sender_id: int, role: str, text: str) -> None:
        self.messages.append((sender_id, role, text))

    def set_state(self, key: str, value) -> None:
        self.state[key] = value

    def get_state(self, key: str, default=None):
        return self.state.get(key, default)

    def recent_attachments(self, sender_id: int, limit: int = 20) -> list[dict]:
        return [dict(a) for a in self.attachments if a.get("sender_id") == sender_id][:limit]


class FakeTelegram:
    def __init__(self, payloads: dict[str, Path | bytes]) -> None:
        self.payloads = payloads
        self.downloads: list[tuple[str, Path, int]] = []

    async def download(self, file_id: str, destination: Path, max_bytes: int) -> int:
        self.downloads.append((file_id, destination, max_bytes))
        item = self.payloads[file_id]
        if isinstance(item, Path):
            shutil.copyfile(item, destination)
            size = destination.stat().st_size
        else:
            destination.write_bytes(item)
            size = len(item)
        if size > max_bytes:
            raise AssertionError("fixture exceeds requested download bound")
        return size


class FakeTranscriber:
    def __init__(self, text: str = "offline speech transcript", enabled: bool = True) -> None:
        self.text = text
        self._enabled = enabled
        self.paths: list[Path] = []

    def health(self) -> dict:
        return {"enabled": self._enabled}

    async def transcribe(self, path: Path) -> str:
        self.paths.append(path)
        return self.text


class VideoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = FakeStore()
        self.config = FakeConfig(self.root)
        self.transcriber = FakeTranscriber()
        self.analyzer = VideoAnalyzer(self.root, transcriber=self.transcriber)

    async def asyncTearDown(self) -> None:
        await self.analyzer.close()
        self.temp.cleanup()

    # 1. Format selection
    def test_format_selection(self) -> None:
        formats = [
            {"format_id": "1080p", "width": 1920, "height": 1080, "vcodec": "avc1", "acodec": "mp4a", "tbr": 4000, "protocol": "https"},
            {"format_id": "720p", "width": 1280, "height": 720, "vcodec": "avc1", "acodec": "mp4a", "tbr": 2000, "protocol": "https"},
            {"format_id": "480p", "width": 854, "height": 480, "vcodec": "avc1", "acodec": "mp4a", "tbr": 1000, "protocol": "https"},
            {"format_id": "360p", "width": 640, "height": 360, "vcodec": "avc1", "acodec": "mp4a", "tbr": 600, "protocol": "https"},
            {"format_id": "storyboard", "width": 320, "height": 180, "format_note": "storyboard", "protocol": "https"},
        ]
        chosen = select_formats(formats, duration=60, max_bytes=MAX_DOWNLOAD_BYTES)
        self.assertEqual(chosen, "480p")

        # Vertical video judged by short side
        vertical = [
            {"format_id": "v1080", "width": 1080, "height": 1920, "vcodec": "avc1", "acodec": "mp4a", "protocol": "https"},
            {"format_id": "v480", "width": 480, "height": 854, "vcodec": "avc1", "acodec": "mp4a", "protocol": "https"},
        ]
        self.assertEqual(select_formats(vertical, duration=30), "v480")

        # Video without audio paired with best audio
        video_only = [
            {"format_id": "v480_no_audio", "width": 854, "height": 480, "vcodec": "avc1", "acodec": "none", "protocol": "https"},
            {"format_id": "audio_best", "vcodec": "none", "acodec": "mp4a", "abr": 128, "protocol": "https"},
        ]
        self.assertEqual(select_formats(video_only, duration=60), "v480_no_audio+audio_best")

        # Size constraint: when 480p is too big, falls back to 360p or returns None
        capped = select_formats(formats, duration=60, max_bytes=4_000_000)  # 600 kbps * 60s / 8 = 4.5 MB -> 480p is 7.5 MB
        self.assertIn(capped, ("360p", None))
        none_fits = select_formats(formats, duration=1000, max_bytes=100)
        self.assertIsNone(none_fits)

    # 1b. Lowest-quality fallback when a download is refused for size
    def test_smallest_formats_and_worker_fallback(self) -> None:
        import sys
        import types
        from personal_agent import video as video_module
        formats = [
            {"format_id": "hd", "width": 1280, "height": 720, "vcodec": "avc1", "acodec": "mp4a", "tbr": 8000, "protocol": "https"},
            {"format_id": "low", "width": 256, "height": 144, "vcodec": "avc1", "acodec": "mp4a", "tbr": 800, "protocol": "https"},
            {"format_id": "v", "width": 426, "height": 240, "vcodec": "avc1", "acodec": "none", "tbr": 300, "protocol": "https"},
            {"format_id": "a", "vcodec": "none", "acodec": "mp4a", "abr": 48, "tbr": 48, "protocol": "https"},
        ]
        self.assertIsNone(select_formats(formats, duration=600, max_bytes=1_000_000))
        self.assertEqual(smallest_formats(formats, duration=600, max_bytes=1_000_000), [])
        plans = smallest_formats(formats, duration=600, max_bytes=100_000_000)
        self.assertEqual((plans[0]["format"], plans[0]["height"], plans[0]["tier"]), ("low", 144, "smallest_muxed"))
        silent = [f for f in formats if f["format_id"] in {"v", "a"}]
        plan = smallest_formats(silent, duration=600, max_bytes=100_000_000)[0]
        self.assertEqual((plan["format"], plan["tier"]), ("v+a", "worst_video_worst_audio"))

        chosen: list[str] = []

        class FakeDL(Exception):
            pass

        class FakeYDL:
            def __init__(self, options): self.params = dict(options)
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def extract_info(self, url, download=False):
                return {"title": "t", "duration": 600.0, "formats": formats, "thumbnail": "x"}
            def sanitize_info(self, info): return info
            def process_ie_result(self, info, download=True):
                chosen.append(self.params.get("format", ""))
                if not self.params.get("skip_download"):
                    Path(str(self.params["outtmpl"]["default"]).replace("%(ext)s", "mp4")).write_bytes(b"x")

        fake = types.ModuleType("yt_dlp")
        fake.YoutubeDL, fake.utils = FakeYDL, types.ModuleType("yt_dlp.utils")
        fake.utils.DownloadError, fake.utils.download_range_func = FakeDL, lambda *a, **k: None
        args = {"url": "https://example.com/v", "out_dir": str(self.root), "max_bytes": 1_000_000, "max_duration": 900,
                "analyzed_seconds": 600, "window": None, "subtitles": False, "prefix": "video", "timeout": 5, "smallest": True}
        with patch.dict(sys.modules, {"yt_dlp": fake, "yt_dlp.utils": fake.utils}):
            small = video_module._worker_fetch({**args, "max_bytes": 100_000_000})
            tiny = video_module._worker_fetch({**args, "max_bytes": 1000})
        self.assertEqual(chosen[0], "low")
        self.assertEqual(small["quality"]["height"], 144)
        self.assertTrue(small["quality"]["quality_fallback"])
        self.assertTrue(tiny["metadata_only"])  # nothing fits: metadata, subtitles and thumbnail only

    async def test_too_large_download_retries_with_smallest_quality(self) -> None:
        directory = self.root / "entry"
        directory.mkdir()
        calls: list[dict] = []

        async def worker(args: dict) -> dict:
            calls.append(args)
            if not args.get("smallest"):
                raise VideoError("video_too_large")
            (directory / "video.mp4").write_bytes(b"x")
            return {"ok": True, "info": {"title": "t"}, "quality": {"quality_fallback": True, "height": 144, "tier": "smallest_muxed"},
                    "notes": []}

        with patch.object(self.analyzer, "_run_worker", side_effect=worker):
            meta = await self.analyzer._fetch_with_ytdlp(directory, "https://example.com/v")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[1]["smallest"])
        self.assertEqual(meta["quality"]["height"], 144)
        self.assertTrue(meta["quality"]["quality_fallback"])

    # 2. Duration and size limits
    async def test_duration_and_size_limits(self) -> None:
        meta_normal = {"duration": 120.0, "downloaded_range": [0.0, 120.0]}
        low, high, notes = self.analyzer.resolve_window(meta_normal, None, None)
        self.assertEqual((low, high), (0.0, 120.0))
        self.assertEqual(notes, [])

        # Long video (> MAX_DURATION_SECONDS) limits default window to first ANALYZED_SECONDS
        meta_long = {"duration": 3600.0, "downloaded_range": [0.0, 3600.0]}
        low, high, notes = self.analyzer.resolve_window(meta_long, None, None)
        self.assertEqual((low, high), (0.0, ANALYZED_SECONDS))
        self.assertTrue(any("video_longer_than" in n for n in notes))

        # Explicit start past duration raises VideoError
        with self.assertRaises(VideoError) as raised:
            self.analyzer.resolve_window(meta_normal, 150.0, None)
        self.assertEqual(raised.exception.code, "invalid_window")

        # Negative bounds or invalid values raise VideoError
        with self.assertRaises(VideoError):
            self.analyzer.resolve_window(meta_normal, -5.0, 10.0)

        # Direct download byte limit enforcement
        class ChunkClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def stream(self, method, url):
                class StreamResp:
                    status_code = 200
                    def raise_for_status(self): pass
                    async def aiter_bytes(self):
                        yield b"X" * (MAX_DOWNLOAD_BYTES + 1024)
                    async def __aenter__(self): return self
                    async def __aexit__(self, *args): pass
                return StreamResp()

        target = self.root / "oversize.mp4"
        with patch("httpx.AsyncClient", return_value=ChunkClient()), \
             patch("personal_agent.video._public_url", new=AsyncMock(return_value=True)):
            with self.assertRaises(VideoError) as exc:
                await self.analyzer._download_direct("https://public-cdn.example.com/big.mp4", target)
            self.assertEqual(exc.exception.code, "video_too_large")

    # 3. Subtitles priority over Whisper
    async def test_subtitles_priority_over_whisper(self) -> None:
        info = {
            "language": "en",
            "subtitles": {"en": [{"ext": "vtt"}]},
            "automatic_captions": {"en-orig": [{"ext": "vtt"}]},
        }
        kind, lang = pick_subtitle_track(info)
        self.assertEqual(kind, "manual")
        self.assertEqual(lang, "en")

        entry_dir = self.analyzer.entry_dir(PRIMARY_ID, "test-sub-entry")
        entry_dir.mkdir(parents=True)
        sub_path = entry_dir / "subs.vtt"
        sub_path.write_text(
            "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nHello subtitle world\n",
            encoding="utf-8",
        )
        meta = {
            "video_id": "test-sub-entry", "sender_id": PRIMARY_ID, "file": "video.mp4",
            "has_audio": True, "duration": 10.0,
            "subtitle": {"file": "subs.vtt", "language": "en", "kind": "manual"},
        }
        record, notes = await self.analyzer._ensure_transcript(entry_dir, meta)
        self.assertEqual(record["source"], "subtitles")
        self.assertEqual(record["language"], "en")
        self.assertEqual(len(record["cues"]), 1)
        self.assertEqual(record["cues"][0][1], "Hello subtitle world")
        # Transcriber (Whisper) was NOT called
        self.assertEqual(len(self.transcriber.paths), 0)

    # 4. Whisper fallback
    async def test_whisper_fallback(self) -> None:
        entry_dir = self.analyzer.entry_dir(PRIMARY_ID, "test-whisper-entry")
        entry_dir.mkdir(parents=True)
        synthetic_video = self.root / "silent_synth.mp4"
        make_synthetic_video(synthetic_video, segments=(("red", 1.0),), audio="tone")
        shutil.copyfile(synthetic_video, entry_dir / "video.mp4")

        meta = {
            "video_id": "test-whisper-entry", "sender_id": PRIMARY_ID, "file": "video.mp4",
            "has_audio": True, "duration": 1.0, "subtitle": None,
        }
        record, notes = await self.analyzer._ensure_transcript(entry_dir, meta)
        self.assertEqual(record["source"], "whisper")
        self.assertEqual(record["text"], "offline speech transcript")
        self.assertEqual(len(self.transcriber.paths), 1)

        # Video with no audio track
        meta_no_audio = {
            "video_id": "test-whisper-no-audio", "sender_id": PRIMARY_ID, "file": "video.mp4",
            "has_audio": False, "duration": 1.0, "subtitle": None,
        }
        no_audio_dir = self.analyzer.entry_dir(PRIMARY_ID, "test-whisper-no-audio")
        no_audio_dir.mkdir(parents=True)
        rec_no_audio, _ = await self.analyzer._ensure_transcript(no_audio_dir, meta_no_audio)
        self.assertEqual(rec_no_audio["source"], "none")
        self.assertEqual(rec_no_audio["reason"], "no_audio_track")

        # Whisper disabled / unconfigured
        disabled_transcriber = FakeTranscriber(enabled=False)
        analyzer_disabled = VideoAnalyzer(self.root, transcriber=disabled_transcriber)
        meta_unconf = {
            "video_id": "test-whisper-unconf", "sender_id": PRIMARY_ID, "file": "video.mp4",
            "has_audio": True, "duration": 1.0, "subtitle": None,
        }
        unconf_dir = self.analyzer.entry_dir(PRIMARY_ID, "test-whisper-unconf")
        unconf_dir.mkdir(parents=True)
        rec_unconf, _ = await analyzer_disabled._ensure_transcript(unconf_dir, meta_unconf)
        self.assertEqual(rec_unconf["source"], "none")
        self.assertEqual(rec_unconf["reason"], "whisper_not_configured")
        await analyzer_disabled.close()

    # 5. Frame grid (compose_sheets)
    def test_contact_sheets_grid(self) -> None:
        from PIL import Image
        frames_dir = self.root / "synth_frames"
        frames_dir.mkdir(parents=True)
        frames: list[tuple[float, Path]] = []
        for i in range(12):
            img_path = frames_dir / f"frame_{i:02d}.jpg"
            img = Image.new("RGB", (320, 240), color=(i * 20, 100, 150))
            img.save(img_path, "JPEG")
            frames.append((float(i), img_path))

        out_dir = self.root / "sheets_out"
        sheets = compose_sheets(frames, out_dir, prefix="test-sheet")
        self.assertGreaterEqual(len(sheets), 2)
        total_times = sum(len(s.times) for s in sheets)
        self.assertEqual(total_times, 12)
        for sheet in sheets:
            self.assertTrue(sheet.path.is_file())
            self.assertLessEqual(sheet.path.stat().st_size, 3 * 1024 * 1024)
            with Image.open(sheet.path) as loaded:
                self.assertGreater(loaded.width, 300)
                self.assertGreater(loaded.height, 200)

    # 6. Cancellation terminates child process
    async def test_cancellation_terminates_child_process(self) -> None:
        import sys
        # Launch a python child that sleeps indefinitely
        task = asyncio.create_task(
            self.analyzer._run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=30.0)
        )
        await asyncio.sleep(0.1)
        self.assertEqual(len(self.analyzer.processes), 1)
        process = next(iter(self.analyzer.processes))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        # Process must be terminated and cleaned up from tracker
        self.assertEqual(len(self.analyzer.processes), 0)
        await asyncio.sleep(0.05)
        self.assertIsNotNone(process.returncode)

    # 7. Rejection of private IP and file scheme
    async def test_rejection_private_ip_and_file_scheme(self) -> None:
        denied_urls = [
            "file:///C:/Windows/System32/calc.exe",
            "file:///etc/passwd",
            "http://127.0.0.1/video.mp4",
            "http://10.0.0.1/video.mp4",
            "http://192.168.1.1/video.mp4",
            "http://169.254.169.254/latest/meta-data",
            "ftp://example.com/video.mp4",
            "javascript:alert(1)",
        ]
        for url in denied_urls:
            with self.assertRaises(VideoError, msg=f"Should deny: {url}") as raised:
                await self.analyzer._validate_url(url)
            self.assertEqual(raised.exception.code, "web_url_denied")

    # 8. Sender isolation and data retention
    async def test_sender_isolation_and_retention(self) -> None:
        synthetic = self.root / "shared_synth.mp4"
        make_synthetic_video(synthetic, segments=(("red", 1.0),))

        # Prepare for PRIMARY_ID
        meta_primary = await self.analyzer.prepare_file(PRIMARY_ID, synthetic, "video-primary-001", name="clip.mp4")
        self.assertEqual(meta_primary["video_id"], "video-primary-001")

        # SECONDARY_ID cannot load or list PRIMARY_ID's video
        self.assertIsNone(self.analyzer.load_entry(SECONDARY_ID, "video-primary-001"))
        self.assertEqual(self.analyzer.list_entries(SECONDARY_ID), [])
        with self.assertRaises(VideoError) as raised:
            await self.analyzer.watch(SECONDARY_ID, media_id="video-primary-001")
        self.assertEqual(raised.exception.code, "media_not_found")

        # Retention cleanup: set mtime back by 10 days
        primary_dir = self.analyzer.entry_dir(PRIMARY_ID, "video-primary-001")
        old_time = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
        meta_file = primary_dir / "meta.json"
        os.utime(meta_file, (old_time, old_time))
        os.utime(primary_dir, (old_time, old_time))

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        removed = cleanup_videos(self.root, cutoff)
        self.assertEqual(removed, 1)
        self.assertFalse(primary_dir.exists())

    # 9. Telegram video attachment end-to-end
    async def test_telegram_video_attachment_workflow(self) -> None:
        synth_file = self.root / "telegram_synth.mp4"
        make_synthetic_video(
            synth_file,
            segments=(("red", 1.0), ("green", 1.0), ("blue", 1.0)),
            labels=["START", "MID", "END"],
            audio="tone",
        )

        pipeline = AttachmentPipeline(
            self.config, self.store,
            FakeTelegram({"vid-tg-1": synth_file}),
            self.transcriber,
            video_analyzer=self.analyzer,
        )

        message = IncomingMessage(
            update_id=1, sender_id=PRIMARY_ID, chat_id=PRIMARY_ID,
            text="Посмотри видео", message_id=1,
            attachment={"kind": "video", "file_id": "vid-tg-1", "file_size": synth_file.stat().st_size},
            is_text=False,
        )
        result = await pipeline.prepare(message)
        self.assertTrue(result.images)
        self.assertTrue(all(p.is_file() for p in result.images))
        self.assertEqual(result.observations[0]["kind"], "attachment_video")
        self.assertEqual(result.observations[0]["sender_id"], PRIMARY_ID)
        media_id = result.observations[0]["media_id"]
        self.assertTrue(media_id)

        # TurnTools checks: recent_media exposes the video on the next turn
        turn = TurnTools(
            config=self.config, store=self.store, context=None, knowledge=None,
            action_broker=None, profile_store=None, telegram=None,
            sender_id=PRIMARY_ID, chat_id=PRIMARY_ID, request_id="req-turn-2",
            supports_images=lambda: True, video_analyzer=self.analyzer,
        )
        catalog = turn.public_media_catalog()
        video_entry = next((e for e in catalog if e.get("media_id") == media_id), None)
        self.assertIsNotNone(video_entry)
        self.assertEqual(video_entry["kind"], "video")

        # Rewatching the end: watch_video with start_seconds=2.0, end_seconds=3.0
        rewatch_res, rewatch_images = await turn.watch_video({
            "media_id": media_id,
            "start_seconds": 2.0,
            "end_seconds": 3.0,
        })
        self.assertTrue(rewatch_res["ok"])
        self.assertEqual(rewatch_res["window"]["start_seconds"], 2.0)
        self.assertEqual(rewatch_res["window"]["end_seconds"], 3.0)
        self.assertTrue(rewatch_images)

    # 10. A Telegram video reaches the model in the same turn, and the next turn can rewatch a part.
    async def test_supervisor_shows_video_now_and_rewatches_the_end_later(self) -> None:
        harness = ObservationHarness(ScriptedBrain(image_input_supported=True, response="Вижу ролик."))
        fixture = harness.root / "telegram-video.mp4"
        make_synthetic_video(
            fixture,
            segments=(("red", 1.0), ("green", 1.0), ("blue", 1.0)),
            labels=["START", "MID", "END"],
            audio="tone",
        )

        async def download(file_id, destination, max_bytes):
            shutil.copyfile(fixture, destination)
            return destination.stat().st_size

        harness.telegram.download = download
        try:
            await harness.supervisor.submit(replace(_message("Посмотри, что в этом видео."), attachment={
                "kind": "video", "file_id": "synthetic-video", "file_name": "clip.mp4",
                "file_size": fixture.stat().st_size}), Identity.PRIMARY)
            await harness.wait_idle()

            first = harness.brain.generate_calls[-1]
            self.assertTrue(1 <= len(first["images"]) <= 4)
            video_root = (harness.root / "memory" / "videos").resolve()
            for image in first["images"]:
                self.assertTrue(Path(image).resolve().is_relative_to(video_root))
            self.assertIn('"kind": "attachment_video"', first["prompt"])
            self.assertIn("Посмотри, что в этом видео.", first["prompt"])

            entries = harness.supervisor.video_analyzer.list_entries(PRIMARY_ID)
            self.assertEqual(len(entries), 1)
            media_id = entries[0]["video_id"]

            brain = ScriptedBrain([ScriptedTurn(calls=[
                ("watch_video", {"media_id": media_id, "start_seconds": 2.0, "end_seconds": 3.0})
            ], reply="В самом конце синий кадр с надписью END.")], image_input_supported=True)
            harness.brain = harness.supervisor.brain = brain
            await harness.supervisor.submit(_message("А что в самом конце ролика?", update_id=2), Identity.PRIMARY)
            await harness.wait_idle()

            catalog = packet_of(brain.generate_calls[-1]["prompt"])["recent_media"]
            self.assertEqual([(item["media_id"], item["kind"]) for item in catalog], [(media_id, "video")])
            record = brain.records[0]
            self.assertTrue(record.ok, record.error_code)
            self.assertEqual(record.data["window"]["start_seconds"], 2.0)
            self.assertEqual(record.data["window"]["end_seconds"], 3.0)
            self.assertTrue(record.outcome.images)
            self.assertTrue(all(not Path(image).resolve().is_relative_to(
                (harness.root / "memory" / "attachments").resolve()) for image in record.outcome.images))
        finally:
            await harness.close()


if __name__ == "__main__":
    unittest.main()
