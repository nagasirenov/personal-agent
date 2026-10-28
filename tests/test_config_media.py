import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from personal_agent.config import Config, load_config
from personal_agent.context import ACTIVITY_WINDOW, ContextBuilder
from personal_agent.memory import MemoryStore
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

    def test_context_activity_window_is_fixed_and_not_chosen_by_wording(self):
        now = datetime(2026, 10, 26, 20, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp, MemoryStore(Path(temp)) as store:
            builder = ContextBuilder(store, Config(Path(temp), 1, 2).tz)
            windows = set()
            for text in ("Чем я занимался вчера?", "что было сегодня", "последние 5 часов", "?"):
                packet = builder.build(text, 1, now=now)
                raw = packet.split("\nUNTRUSTED LOCAL EVIDENCE (JSON):\n", 1)[1].split("\n\nCURRENT USER MESSAGE (JSON):\n", 1)[0]
                window = __import__("json").loads(raw)["window"]
                windows.add((window["start"], window["end"]))
        self.assertEqual(windows, {((now - ACTIVITY_WINDOW).isoformat(), now.isoformat())})
        self.assertEqual(ACTIVITY_WINDOW, __import__("datetime").timedelta(hours=2))

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
    async def test_transcription_timeout_reports_unavailable_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as temp:
            transcriber = LocalTranscriber(Path(temp), ffmpeg="ffmpeg")
            process = MagicMock()
            process.wait = AsyncMock()

            async def timeout(awaitable, timeout):
                awaitable.close()
                raise TimeoutError

            with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
                 patch("personal_agent.media.asyncio.wait_for", side_effect=timeout), \
                 patch("personal_agent.media.terminate_tree", new=AsyncMock()) as terminate:
                with self.assertRaisesRegex(TranscriptionUnavailable, "заняло слишком много времени"):
                    await transcriber._run("ffmpeg")
            terminate.assert_awaited_once_with(process)
            self.assertFalse(transcriber.processes)

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
