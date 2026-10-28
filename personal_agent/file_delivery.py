"""Sending files, photos and videos to the current Telegram chat, within the Bot API upload cap.

The cap logic lives here and nowhere else (``TelegramClient.upload_cap_bytes`` supplies the number):

* anything within the cap is uploaded as is (photo / video / audio / document by type);
* a video over the cap is re-encoded with ffmpeg to fit and sent as a video (never split);
* any other file over the cap is uploaded to Google Drive through ``rclone`` and a link is sent; when rclone
  or the remote is not configured it is split into ``name.partNN`` documents with a join hint.

Everything returns receipts (dicts); nothing here raises for an ordinary delivery failure.  Child processes
run cancellable with a timeout, and their output (rclone may print config or tokens) never reaches a receipt, only stable error codes do.
"""
from __future__ import annotations

import asyncio
import glob
import math
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from .media import terminate_tree
from .telegram import CLOUD_UPLOAD_CAP, TelegramError

PHOTO_LIMIT = 10 * 1024 * 1024
MAX_FILES_PER_TURN = 10
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".wav"}
_SAFETY_MARGIN = 1024 * 1024


class DeliveryError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def upload_cap(telegram) -> int:
    """The bot upload limit of the Bot API in use (50 MB hosted, 2000 MB for a local server)."""
    cap = getattr(telegram, "upload_cap_bytes", CLOUD_UPLOAD_CAP)
    return cap if type(cap) is int and cap > 0 else CLOUD_UPLOAD_CAP


def part_size(cap: int) -> int:
    return max(1, cap - min(_SAFETY_MARGIN, cap // 50))


def choose_method(path: Path, preference: str = "auto") -> str:
    """Bot API method for the file: sendPhoto, sendVideo, sendAudio or sendDocument."""
    suffix = path.suffix.lower()
    if preference == "auto":
        preference = ("photo" if suffix in IMAGE_SUFFIXES else "video" if suffix in VIDEO_SUFFIXES
                      else "audio" if suffix in AUDIO_SUFFIXES else "document")
    return {"photo": "sendPhoto", "video": "sendVideo", "audio": "sendAudio"}.get(preference, "sendDocument")


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.0f} {unit}" if unit == "Б" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} Б"


# -- child processes ------------------------------------------------------------------------------
def _child_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in {"TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"}}
    env.update({"PYTHONIOENCODING": "utf-8", "RCLONE_ASK_PASSWORD": "false"})
    return env


async def run_process(argv: list[str], *, timeout: float) -> tuple[int, bytes, bytes]:
    """Run a child without a console window; timeout and cancellation kill the whole process tree."""
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *argv, env=_child_env(), stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=1_000_000, **kwargs)
        out, err = await asyncio.wait_for(process.communicate(), timeout)
        return process.returncode or 0, out[-100_000:], err[-100_000:]
    except FileNotFoundError:
        raise DeliveryError("tool_not_found") from None
    except TimeoutError:
        raise DeliveryError("tool_timeout") from None
    finally:
        if process is not None:
            await terminate_tree(process)


# -- video re-encode ------------------------------------------------------------------------------
async def probe_video(ffmpeg: str, path: Path) -> tuple[float, int | None, bool]:
    _, _, err = await run_process([ffmpeg, "-nostdin", "-hide_banner", "-i", str(path)], timeout=60)
    text = err.decode("utf-8", errors="replace")
    timing = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not timing:
        raise DeliveryError("video_duration_unknown")
    hours, minutes, seconds = map(float, timing.groups())
    duration = hours * 3600 + minutes * 60 + seconds
    size = re.search(r"Video: [^\n]*?\b(\d{2,5})x(\d{2,5})\b", text)
    return duration, (int(size.group(2)) if size else None), bool(re.search(r"Stream #[^\n]*Audio:", text))


def _height_for(video_kbps: float, source_height: int | None) -> int | None:
    limit = 240 if video_kbps < 200 else 360 if video_kbps < 500 else 480 if video_kbps < 1200 else \
        720 if video_kbps < 2500 else 1080
    if source_height is None:
        return None
    height = min(source_height, limit)
    return height - height % 2


async def reencode_video(ffmpeg: str, source: Path, target: Path, *, target_bytes: int, hard_cap: int) -> Path:
    """H.264/AAC MP4 at a bitrate computed from the duration so the file fits, downscaled as needed."""
    if not ffmpeg:
        raise DeliveryError("ffmpeg_unavailable")
    duration, source_height, has_audio = await probe_video(ffmpeg, source)
    if duration <= 0:
        raise DeliveryError("video_duration_unknown")
    audio_kbps = 96 if has_audio else 0
    total_kbps = target_bytes * 8 * 0.93 / duration / 1000
    video_kbps = max(40.0, total_kbps - audio_kbps)
    for _ in range(3):
        height = _height_for(video_kbps, source_height)
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                   "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast",
                   "-b:v", f"{int(video_kbps)}k", "-maxrate", f"{int(video_kbps * 1.3)}k",
                   "-bufsize", f"{int(video_kbps * 2)}k", "-pix_fmt", "yuv420p"]
        if height:
            command += ["-vf", f"scale=-2:{height}"]
        command += ["-c:a", "aac", "-b:a", f"{audio_kbps or 96}k", "-movflags", "+faststart", str(target)]
        code, _, _ = await run_process(command, timeout=min(4 * 3600, max(900.0, duration * 3)))
        if code != 0 or not target.is_file():
            raise DeliveryError("reencode_failed")
        size = target.stat().st_size
        if size <= hard_cap:
            return target
        video_kbps *= max(0.3, 0.9 * (target_bytes / size))
    raise DeliveryError("reencode_too_large")


def default_ffmpeg(configured: str = "") -> str:
    if configured:
        return configured
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return ""


# -- Google Drive through rclone ------------------------------------------------------------------
def find_rclone(configured: str = "") -> str:
    if configured:
        return configured if Path(configured).is_file() else ""
    found = shutil.which("rclone")
    if found:
        return found
    base = os.environ.get("LOCALAPPDATA")
    if base:
        pattern = str(Path(base) / "Microsoft" / "WinGet" / "Packages" / "Rclone.Rclone*" / "**" / "rclone.exe")
        candidates = [Path(item) for item in glob.glob(pattern, recursive=True)]
        if candidates:
            return str(max(candidates, key=lambda item: item.stat().st_mtime))
    return ""


async def drive_upload(rclone: str, remote: str, path: Path, day: str) -> str:
    """Upload with `rclone copyto`, then return the shareable link from `rclone link`."""
    destination = f"{remote.rstrip('/')}/{day}/{path.name}"
    code, _, _ = await run_process([rclone, "copyto", str(path), destination], timeout=6 * 3600)
    if code != 0:
        raise DeliveryError("drive_upload_failed")
    code, out, _ = await run_process([rclone, "link", destination], timeout=120)
    link = out.decode("utf-8", errors="replace").strip().splitlines()[-1].strip() if out.strip() else ""
    if code != 0 or not re.fullmatch(r"https?://\S+", link):
        raise DeliveryError("drive_link_failed")
    return link


# -- the sender -----------------------------------------------------------------------------------
class FileSender:
    """Delivers files to one chat; every call returns a receipt."""

    def __init__(self, telegram, chat_id: int, work_dir: Path, *, ffmpeg: str = "", rclone: str = "", remote: str = "",
                 tz=None) -> None:
        self.telegram, self.chat_id, self.work_dir = telegram, chat_id, Path(work_dir)
        self.ffmpeg, self.rclone, self.remote, self.tz = ffmpeg, rclone, remote, tz

    @property
    def drive_ready(self) -> bool:
        return bool(self.rclone and self.remote)

    async def _send(self, method: str, path: Path, caption: str) -> str:
        sender = {"sendPhoto": self.telegram.send_photo, "sendVideo": getattr(self.telegram, "send_video", None),
                  "sendAudio": getattr(self.telegram, "send_audio", None),
                  "sendDocument": getattr(self.telegram, "send_document", None)}[method]
        if sender is None:
            raise TelegramError("unsupported delivery method")
        used = await sender(self.chat_id, path, caption=caption)
        return used if isinstance(used, str) else method

    async def send(self, path: Path, preference: str = "auto", caption: str = "") -> dict:
        size = path.stat().st_size
        receipt: dict = {"kind": "send_file", "ok": False, "name": path.name, "size": size, "method": None,
                         "parts": 1, "reencoded": False}
        if size <= 0:
            receipt["error"] = {"code": "empty_file"}
            return receipt
        cap = upload_cap(self.telegram)
        method = choose_method(path, preference)
        if method == "sendPhoto" and (size > PHOTO_LIMIT or path.suffix.lower() not in IMAGE_SUFFIXES):
            method = "sendDocument"
        try:
            if size <= cap:
                receipt["method"] = await self._send(method, path, caption)
                receipt["ok"] = True
            elif method == "sendVideo":
                await self._send_reencoded(path, caption, cap, receipt)
            else:
                await self._send_big(path, caption, cap, receipt)
        except DeliveryError as exc:
            receipt["error"] = {"code": exc.code}
        except TelegramError:
            receipt["error"] = {"code": "telegram_upload_failed"}
        except OSError:
            receipt["error"] = {"code": "file_unreadable"}
        return receipt

    def _scratch(self) -> Path:
        directory = self.work_dir / uuid.uuid4().hex[:12]
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    async def _send_reencoded(self, path: Path, caption: str, cap: int, receipt: dict) -> None:
        scratch = self._scratch()
        try:
            output = await reencode_video(self.ffmpeg, path, scratch / (path.stem[:80] + ".mp4"),
                                          target_bytes=int(cap * 0.96), hard_cap=cap)
            receipt.update(reencoded=True, sent_size=output.stat().st_size)
            receipt["method"] = await self._send("sendVideo", output, caption)
            receipt["ok"] = True
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    async def _send_big(self, path: Path, caption: str, cap: int, receipt: dict) -> None:
        if self.drive_ready:
            day = datetime.now(self.tz).strftime("%Y-%m-%d")
            try:
                link = await drive_upload(self.rclone, self.remote, path, day)
            except DeliveryError as exc:
                receipt["drive_error"] = exc.code  # fall through to the split below
            else:
                text = f"{path.name} ({human_size(receipt['size'])}): {link}"
                await self.telegram.send_text(self.chat_id, (caption + "\n" if caption else "") + text)
                receipt.update(ok=True, method="drive_link", link=link)
                return
        receipt["drive_unavailable"] = True
        await self._send_parts(path, cap, receipt)

    async def _send_parts(self, path: Path, cap: int, receipt: dict) -> None:
        size = path.stat().st_size
        chunk = part_size(cap)
        count = math.ceil(size / chunk)
        names = [f"{path.name}.part{index + 1:02d}" for index in range(count)]
        joined = f"copy /b {'+'.join(names)} {path.name}"
        if len(joined) > 600:
            joined = f'copy /b "{path.name}.part*" "{path.name}"'
        receipt.update(parts=count, parts_sent=0, method="sendDocument")
        scratch = self._scratch()
        try:
            with path.open("rb") as source:
                for index, name in enumerate(names):
                    part = scratch / name
                    await asyncio.to_thread(_copy_part, source, part, chunk)
                    note = f"Часть {index + 1}/{count} файла {path.name}. Склеить в Windows: {joined}"
                    await self._send("sendDocument", part, note[:1000])
                    part.unlink(missing_ok=True)
                    receipt["parts_sent"] = index + 1
            receipt["ok"] = True
            receipt["join_command"] = joined
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def _copy_part(source, target: Path, limit: int) -> None:
    remaining = limit
    with target.open("wb") as out:
        while remaining > 0:
            block = source.read(min(8 * 1024 * 1024, remaining))
            if not block:
                break
            out.write(block)
            remaining -= len(block)


__all__ = ["DeliveryError", "FileSender", "MAX_FILES_PER_TURN", "choose_method", "default_ffmpeg", "find_rclone",
           "human_size", "upload_cap"]
