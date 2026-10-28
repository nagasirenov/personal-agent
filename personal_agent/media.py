"""Local-only voice transcription; fixed argv, local model, no downloads."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
from uuid import uuid4

from .security import owned_path


class TranscriptionUnavailable(RuntimeError):
    pass


async def terminate_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec(
            "taskkill.exe", "/PID", str(process.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            await asyncio.wait_for(killer.wait(), 3)
        except TimeoutError:
            killer.kill()
    else:
        import signal
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), 3)
    except TimeoutError:
        process.kill()


class LocalTranscriber:
    def __init__(self, root: Path, whisper: str = "", model: str = "", ffmpeg: str = ""):
        if not ffmpeg:
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg = get_ffmpeg_exe()
            except (ImportError, RuntimeError):
                pass
        self.root, self.whisper, self.model, self.ffmpeg = root.resolve(), whisper, model, ffmpeg
        self.processes: set[asyncio.subprocess.Process] = set()
        self._closed = False

    def health(self) -> dict:
        ready = bool(self.whisper and self.ffmpeg and self.model and Path(self.model).is_file())
        return {"enabled": ready, "engine": "whisper.cpp", "cloud": False,
                "detail": "configured; executable checked at use" if ready else "set whisper_executable, whisper_model and ffmpeg_executable"}

    async def _run(self, *argv: str) -> None:
        if self._closed:
            raise TranscriptionUnavailable("Transcriber is stopped")
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv, cwd=str(self.root), stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, **kwargs,
            )
            self.processes.add(process)
            if await asyncio.wait_for(process.wait(), timeout=180) != 0:
                raise TranscriptionUnavailable("Local transcription tool failed")
        except FileNotFoundError:
            raise TranscriptionUnavailable("Local transcription executable not found") from None
        except TimeoutError:
            raise TranscriptionUnavailable("Локальное распознавание голосового заняло слишком много времени; файл сохранён.") from None
        finally:
            if process is not None:
                await terminate_tree(process)
                self.processes.discard(process)

    async def transcribe(self, source: Path) -> str:
        source = source.resolve()
        if not source.is_relative_to(self.root) or not source.is_file():
            raise ValueError("Voice source must be a local agent-owned attachment")
        if not self.health()["enabled"]:
            raise TranscriptionUnavailable("Локальный STT не настроен; голосовое сохранено без расшифровки.")
        temp_dir = owned_path(self.root, "memory", "transcripts")
        temp_dir.mkdir(parents=True, exist_ok=True)
        stem = str(uuid4())
        wav = owned_path(self.root, "memory", "transcripts", stem + ".wav")
        output_base = owned_path(self.root, "memory", "transcripts", stem)
        output_text = output_base.with_suffix(".txt")
        try:
            await self._run(self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                            "-protocol_whitelist", "file,pipe", "-i", str(source), "-vn", "-ac", "1",
                            "-ar", "16000", "-c:a", "pcm_s16le", str(wav))
            await self._run(self.whisper, "-m", self.model, "-f", str(wav), "-l", "auto", "-otxt", "-of", str(output_base))
            if not output_text.is_file() or output_text.stat().st_size > 256000:
                raise TranscriptionUnavailable("Local STT did not produce a bounded transcript")
            return output_text.read_text(encoding="utf-8").strip()
        finally:
            wav.unlink(missing_ok=True)

    async def close(self) -> None:
        self._closed = True
        await asyncio.gather(*(terminate_tree(p) for p in list(self.processes)), return_exceptions=True)
