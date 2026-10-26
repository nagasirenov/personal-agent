import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from personal_agent.config import Config, load_config
from personal_agent.context import requested_window
from personal_agent.media import LocalTranscriber, TranscriptionUnavailable
from personal_agent.security import InstanceLock, owned_path


class ConfigTests(unittest.TestCase):
    def test_exact_numeric_ids_and_explicit_web_switch(self):
        with tempfile.TemporaryDirectory() as root:
            for primary, secondary in ((1, 1), (True, 2), (0, 2), ("1", 2)):
                with self.assertRaises(ValueError):
                    Config(Path(root), primary, secondary).validate()
            with self.assertRaises(ValueError):
                Config(Path(root), 1, 2, backend="codex_chatgpt_web").validate()

    def test_token_env_is_redacted_and_paths_relative_to_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "config.toml"
            path.write_text('data_dir="state"\nprimary_id=123\nsecondary_id=456\n', encoding="utf-8")
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123:secret"}):
                config = load_config(path, live=True)
            self.assertEqual(config.data_dir, root / "state")
            self.assertNotIn("secret", repr(config))
            path.write_text('primary_id=123\nsecondary_id=456\ntoken="do-not-store"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_time_windows_use_moscow_midnight(self):
        now = datetime(2026, 10, 26, 20, 0, tzinfo=timezone.utc)
        tz = Config(Path("unused"), 1, 2).tz
        start, end = requested_window("Чем я занимался вчера?", now, tz)
        self.assertEqual(start.astimezone(timezone.utc).isoformat(), "2026-10-24T21:00:00+00:00")
        self.assertEqual(end - start, __import__("datetime").timedelta(days=1))

    def test_lock_is_exclusive_and_reusable(self):
        with tempfile.TemporaryDirectory() as root:
            lock = InstanceLock(Path(root))
            try:
                with self.assertRaises(RuntimeError):
                    InstanceLock(Path(root))
            finally:
                lock.close()
            InstanceLock(Path(root)).close()

    def test_owned_path_rejects_escape(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                owned_path(Path(root), "..", "outside")


class LocalMediaTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_voice_pipeline_fixed_argv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, model = root / "voice.ogg", root / "model.bin"
            source.write_bytes(b"fixture")
            model.write_bytes(b"fixture")
            transcriber = LocalTranscriber(root, "whisper-cli", str(model), "ffmpeg")
            calls = []

            async def fake_run(*argv):
                calls.append(argv)
                if argv[0] == "whisper-cli":
                    Path(argv[argv.index("-of") + 1] + ".txt").write_text("Проверка голоса", encoding="utf-8")
                else:
                    Path(argv[-1]).write_bytes(b"wav")

            transcriber._run = fake_run
            text = await transcriber.transcribe(source)
            self.assertEqual(text, "Проверка голоса")
            self.assertIn("file,pipe", calls[0])
            self.assertIn(str(model), calls[1])
            self.assertFalse(list((root / "memory" / "transcripts").glob("*.wav")))
            await transcriber.close()

    async def test_missing_model_is_explicit_and_source_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "voice.ogg"
            source.write_bytes(b"voice")
            transcriber = LocalTranscriber(root)
            with self.assertRaises(TranscriptionUnavailable):
                await transcriber.transcribe(source)
            with self.assertRaises(ValueError):
                await transcriber.transcribe(root.parent / "outside.ogg")
