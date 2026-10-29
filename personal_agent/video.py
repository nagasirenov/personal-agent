"""Bounded video understanding: yt-dlp in a cancellable child, ffmpeg frames, subtitles or local speech-to-text.

A public http(s) link (YouTube, TikTok, ...) or a local video file becomes: bounded metadata, a transcript
(platform subtitles first, local whisper.cpp otherwise) and a few JPEG contact sheets with timestamp labels.
Everything is cached per sender under ``<data_dir>/memory/videos/<sender_id>/<video_id>/`` so follow-up
questions reuse it.  Page, caption, subtitle and frame content is untrusted data, never instructions.

yt-dlp runs in a separate Python process with no cookies, no config files and no browser access; the
generic extractor is disabled so an arbitrary page cannot make it fetch internal addresses.  Direct links
to media files are downloaded by this module with the same per-hop public-address validation as web_open.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .media import terminate_tree
from .security import owned_path
from .web_lookup import MAX_REDIRECTS, WebLookupDenied, _public_url, validate_web_url

MAX_DURATION_SECONDS = 1800          # longer videos: only a window is downloaded and looked at
ANALYZED_SECONDS = 600               # window length for over-long videos and for audio transcription
DEFAULT_SEGMENT_SECONDS = 120        # window when only start_seconds is given
MAX_DOWNLOAD_BYTES = 300 * 1024 * 1024
FETCH_TIMEOUT = 180.0
FRAME_TIMEOUT = 90.0
TRANSCRIBE_TIMEOUT = 100.0
MIN_FRAMES, MAX_FRAMES, FRAMES_PER_SHEET = 8, 24, 8
FRAME_MAX_SIDE = 640
TILE_MAX_SIDE = 448
MAX_SHEET_BYTES = 3 * 1024 * 1024
DESCRIPTION_LIMIT = 1500
TRANSCRIPT_LIMIT = 12_000
SHORT_SIDE_TARGET = 480
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi", ".gif"}
_SECRET_ENV = {"TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"}
_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_UNTRUSTED = "untrusted_video_content"


class VideoError(RuntimeError):
    def __init__(self, code: str, public_message: str = "Не удалось обработать видео.") -> None:
        super().__init__(code)
        self.code, self.public_message = code, public_message


# -- pure helpers --------------------------------------------------------------------------------
def clock(seconds: float) -> str:
    tenths = round(max(0.0, float(seconds)) * 10)
    seconds = tenths / 10
    whole, tenth = divmod(tenths, 10)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    base = f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
    return base + (f".{tenth}" if seconds < 600 else "")


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _short_side(fmt: dict) -> float | None:
    width, height = _num(fmt.get("width")), _num(fmt.get("height"))
    if width and height:
        return min(width, height)
    return height or width


def _estimated_bytes(fmt: dict, duration: float | None) -> float | None:
    size = _num(fmt.get("filesize")) or _num(fmt.get("filesize_approx"))
    if size:
        return size
    rate = _num(fmt.get("tbr"))
    return rate * 1000 / 8 * duration if rate and duration else None


_CODEC_RANK = (("avc1", 0), ("h264", 0), ("vp9", 1), ("vp09", 1), ("av01", 2), ("hev", 3), ("h265", 3), ("hvc", 3))
_PROTOCOL_RANK = {"https": 0, "http": 0, "m3u8_native": 2, "m3u8": 2}


def _codec_rank(vcodec: object) -> int:
    text = str(vcodec or "").lower()
    return next((rank for prefix, rank in _CODEC_RANK if text.startswith(prefix)), 4)


def _usable_format(fmt: dict) -> bool:
    if not isinstance(fmt, dict):
        return False
    note = str(fmt.get("format_note", "")).lower()
    return (isinstance(fmt.get("format_id"), str) and fmt.get("ext") != "mhtml"
            and "storyboard" not in note and "dubbed" not in note  # keep the original audio, not auto-dubs
            and str(fmt.get("protocol", "https")) in _PROTOCOL_RANK)


def smallest_formats(formats: list[dict], *, duration: float | None = None,
                     max_bytes: int = MAX_DOWNLOAD_BYTES) -> list[dict]:
    """Fallback plans, best first, for when select_formats found nothing that fits: the lowest quality.

    Each plan is {"format", "height", "tier"} (and "audio_format" for the video-only tier, whose audio is
    fetched separately).  Tiers: the lowest-resolution muxed video, or the worst video plus the worst audio
    (whichever is smaller), then the smallest video-only stream.  Empty when even that does not fit.
    """
    pool = [f for f in formats if _usable_format(f)]
    videos = [f for f in pool if str(f.get("vcodec", "unknown")) != "none" and (f.get("width") or f.get("height")
                                                                               or f.get("vcodec"))]
    audios = [f for f in pool if str(f.get("vcodec", "none")) == "none" and str(f.get("acodec", "none")) != "none"]

    def size(fmt: dict | None) -> float:
        return (_estimated_bytes(fmt, duration) or 0) if fmt else 0

    def fits(fmt: dict, extra: float = 0) -> bool:
        known = _estimated_bytes(fmt, duration)
        return known is None or known + extra <= max_bytes

    def small(fmt: dict):
        side = _short_side(fmt)
        return (side is None, side or 0, _PROTOCOL_RANK.get(str(fmt.get("protocol")), 3), _num(fmt.get("tbr")) or 0,
                _codec_rank(fmt.get("vcodec")))

    def low_audio(fmt: dict):
        return (_PROTOCOL_RANK.get(str(fmt.get("protocol")), 3), _num(fmt.get("abr")) or _num(fmt.get("tbr")) or 0)

    muxed = sorted((f for f in videos if str(f.get("acodec", "unknown")) != "none"), key=small)
    silent = sorted((f for f in videos if str(f.get("acodec", "unknown")) == "none"), key=small)
    audio = min((f for f in audios if fits(f)), key=low_audio, default=None)
    plans: list[tuple[tuple, dict]] = []
    if muxed and fits(muxed[0]):
        plans.append((small(muxed[0]), {"format": muxed[0]["format_id"], "height": _short_side(muxed[0]),
                                        "tier": "smallest_muxed"}))
    if silent and audio and fits(silent[0], size(audio)):
        plans.append((small(silent[0]), {"format": f"{silent[0]['format_id']}+{audio['format_id']}",
                                         "height": _short_side(silent[0]), "tier": "worst_video_worst_audio"}))
    result = [plan for _, plan in sorted(plans, key=lambda item: item[0])]
    if not result and silent and fits(silent[0]):
        extra = audio if audio is not None and fits(audio, size(silent[0])) else None
        result.append({"format": silent[0]["format_id"], "height": _short_side(silent[0]), "tier": "video_only",
                       **({"audio_format": extra["format_id"]} if extra else {})})
    return result


def select_formats(formats: list[dict], *, duration: float | None = None,
                   max_bytes: int = MAX_DOWNLOAD_BYTES) -> str | None:
    """yt-dlp format expression for the best small version: a short side of at most 480 px, with audio.

    Vertical videos (TikTok) are judged by their short side.  Returns None when nothing fits max_bytes.
    """
    pool = [f for f in formats if _usable_format(f)]
    videos = [f for f in pool if str(f.get("vcodec", "unknown")) != "none" and (f.get("width") or f.get("height")
                                                                               or f.get("vcodec"))]
    audios = [f for f in pool if str(f.get("vcodec", "none")) == "none" and str(f.get("acodec", "none")) != "none"]

    def has_audio(fmt: dict) -> bool:
        return str(fmt.get("acodec", "unknown")) != "none"

    def small_enough(fmt: dict, extra: float = 0) -> bool:
        size = _estimated_bytes(fmt, duration)
        return size is None or size + extra <= max_bytes

    def video_key(fmt: dict):
        side = _short_side(fmt)
        within = side is not None and side <= SHORT_SIDE_TARGET
        # Within the target take the biggest picture; above it, the smallest one; unknown size goes last
        # (TikTok's "download" format is an untested watermarked copy of unknown resolution).
        order = -(side or 0) if within else (side or 0)
        return (2 if side is None else 0 if within else 1, order, not has_audio(fmt), _PROTOCOL_RANK.get(str(fmt.get("protocol")), 3),
                _codec_rank(fmt.get("vcodec")), _num(fmt.get("tbr")) or 0)

    def audio_key(fmt: dict):
        rate = _num(fmt.get("abr")) or _num(fmt.get("tbr")) or 0
        return (_PROTOCOL_RANK.get(str(fmt.get("protocol")), 3), 0 if rate >= 40 else 1, rate,
                0 if fmt.get("ext") in {"m4a", "webm", "mp4"} else 1)

    best_audio = min((f for f in audios if small_enough(f)), key=audio_key, default=None)
    for video in sorted((f for f in videos if small_enough(f)), key=video_key):
        extra = 0.0 if has_audio(video) or best_audio is None else (_estimated_bytes(best_audio, duration) or 0)
        if not small_enough(video, extra):
            continue
        if has_audio(video) or best_audio is None:
            return video["format_id"]
        return f"{video['format_id']}+{best_audio['format_id']}"
    return None


def _lang_matches(key: str, wanted: str) -> bool:
    key = key.lower().replace("_", "-")
    return key.split("-", 1)[0][:2] == wanted if wanted else False


def pick_subtitle_track(info: dict) -> tuple[str, str] | None:
    """("manual"|"auto", language key): the original language first, then Russian, then English."""
    original = str(info.get("language") or "").lower()[:2]
    wanted = [code for code in dict.fromkeys([original, "ru", "en"]) if code]
    manual = {k: v for k, v in (info.get("subtitles") or {}).items() if v and k != "live_chat"}
    auto = {k: v for k, v in (info.get("automatic_captions") or {}).items() if v}
    # Platform auto-captions list dozens of machine translations; only the spoken original is wanted.
    originals = {k: v for k, v in auto.items() if k.lower().endswith("-orig")}
    if not originals and original:
        originals = {k: v for k, v in auto.items() if _lang_matches(k, original) and "-" not in k}
    for pool, kind in ((manual, "manual"), (originals, "auto")):
        for code in wanted:
            for key in sorted(pool, key=len):
                if _lang_matches(key, code):
                    return kind, key
    if originals:
        return "auto", sorted(originals, key=len)[0]
    for code in wanted:
        for key in sorted(auto, key=len):
            if _lang_matches(key, code) and "-" not in key:
                return "auto", key
    return None


@dataclass(frozen=True)
class Cue:
    start: float
    text: str


_TIMING = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*(?:(?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})")
_TAGS = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def parse_subtitles(text: str, *, limit: int = 20000) -> list[Cue]:
    """WebVTT/SRT cues; rolling auto-caption lines that repeat the previous cue are dropped."""
    cues: list[Cue] = []
    previous: set[str] = set()
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    for block in blocks:
        lines = block.split("\n")
        for index, line in enumerate(lines):
            match = _TIMING.search(line)
            if not match:
                continue
            hours, minutes, seconds, millis = match.groups()
            start = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(millis.ljust(3, "0")) / 1000
            body = [re.sub(r"\s+", " ", _TAGS.sub("", entry).replace("&nbsp;", " ").replace("&amp;", "&")
                           .replace("&lt;", "<").replace("&gt;", ">")).strip() for entry in lines[index + 1:]]
            body = [entry for entry in body if entry]
            fresh = [entry for entry in body if entry not in previous]
            previous = set(body)
            if fresh:
                joined = " ".join(fresh)
                if cues and abs(cues[-1].start - start) < 0.05:
                    cues[-1] = Cue(cues[-1].start, f"{cues[-1].text} {joined}")  # same instant: one line
                elif not cues or cues[-1].text != joined:
                    cues.append(Cue(start, joined))
            break
        if len(cues) >= limit:
            break
    return cues


def format_transcript(cues: list[Cue], start: float | None = None, end: float | None = None,
                      limit: int = TRANSCRIPT_LIMIT) -> tuple[str, bool]:
    """Timestamped transcript lines for the window, with the middle dropped when it exceeds the limit."""
    chosen = [c for c in cues if (start is None or c.start >= start - 0.5) and (end is None or c.start < end)]
    lines = [f"[{clock(c.start)}] {c.text}" for c in chosen]
    text = "\n".join(lines)
    if len(text) <= limit:
        return text, False
    head, tail, used = [], [], 0
    budget_head, budget_tail = int(limit * 0.6), int(limit * 0.35)
    for line in lines:
        if used + len(line) + 1 > budget_head:
            break
        head.append(line)
        used += len(line) + 1
    used = 0
    for line in reversed(lines[len(head):]):
        if used + len(line) + 1 > budget_tail:
            break
        tail.append(line)
        used += len(line) + 1
    tail.reverse()
    omitted = len(lines) - len(head) - len(tail)
    return "\n".join([*head, f"[... {omitted} lines omitted ...]", *tail]), True


def bounded_metadata(info: dict) -> dict:
    def text(key: str, limit: int) -> str | None:
        value = info.get(key)
        return re.sub(r"[ \t]+", " ", value).strip()[:limit] if isinstance(value, str) and value.strip() else None

    result: dict = {"title": text("title", 300), "uploader": text("uploader", 200) or text("channel", 200),
                    "duration_seconds": _num(info.get("duration")), "description": text("description", DESCRIPTION_LIMIT),
                    "webpage_url": text("webpage_url", 500), "site": text("extractor_key", 60) or text("extractor", 60)}
    upload = info.get("upload_date")
    if isinstance(upload, str) and re.fullmatch(r"\d{8}", upload):
        result["upload_date"] = f"{upload[:4]}-{upload[4:6]}-{upload[6:]}"
    for key in ("view_count", "like_count", "comment_count"):
        value = info.get(key)
        if type(value) is int and value >= 0:
            result[key] = value
    return {key: value for key, value in result.items() if value is not None}


def sample_times(start: float, end: float, count: int) -> list[float]:
    span = max(0.0, end - start)
    count = max(1, count)
    return [round(start + (index + 0.5) * span / count, 2) for index in range(count)]


def frame_count_for(span: float) -> int:
    return max(MIN_FRAMES, min(MAX_FRAMES, round(span * 1.5)))


def error_code_for(message: str) -> str:
    low = message.lower()
    for needle, code in (("unsupported url", "unsupported_url"), ("sign in", "login_required"),
                         ("log in", "login_required"),
                         ("private video", "video_private"), ("ip address is blocked", "site_blocked"),
                         ("blocked", "site_blocked"), ("not available in your country", "geo_blocked"),
                         ("video unavailable", "video_unavailable"),
                         ("has been removed", "video_unavailable"), ("does not exist", "video_unavailable"),
                         ("not found", "video_unavailable"), ("larger than", "video_too_large"),
                         ("too large", "video_too_large"), ("429", "rate_limited"),
                         ("too many requests", "rate_limited"), ("timed out", "network_timeout"),
                         ("live event", "live_stream")):
        if needle in low:
            return code
    return "extraction_failed"


# -- contact sheets -------------------------------------------------------------------------------
@dataclass(frozen=True)
class Sheet:
    path: Path
    times: list[float]
    cols: int = 1
    rows: int = 1


def compose_sheets(frames: list[tuple[float, Path]], directory: Path, prefix: str = "sheet") -> list[Sheet]:
    """Grid contact sheets (up to 8 labelled tiles each); plain frames when Pillow is unavailable."""
    if not frames:
        return []
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        step = max(1, math.ceil(len(frames) / 4))
        return [Sheet(path, [time_]) for time_, path in frames[::step][:4]]
    directory.mkdir(parents=True, exist_ok=True)
    sheets_needed = max(1, math.ceil(len(frames) / FRAMES_PER_SHEET))
    per_sheet = math.ceil(len(frames) / sheets_needed)
    sheets: list[Sheet] = []
    font = ImageFont.load_default(size=18)
    for number in range(sheets_needed):
        chunk = frames[number * per_sheet:(number + 1) * per_sheet]
        if not chunk:
            break
        tiles = []
        for _, path in chunk:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((TILE_MAX_SIDE, TILE_MAX_SIDE))
                tiles.append(image.copy())
        tile_w, tile_h = max(t.width for t in tiles), max(t.height for t in tiles)
        cols = min(len(tiles), 4 if tile_h > tile_w else 3)
        rows = math.ceil(len(tiles) / cols)
        gap = 4
        sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * gap, rows * tile_h + (rows + 1) * gap), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        for index, ((time_, _), tile) in enumerate(zip(chunk, tiles)):
            x = gap + (index % cols) * (tile_w + gap) + (tile_w - tile.width) // 2
            y = gap + (index // cols) * (tile_h + gap) + (tile_h - tile.height) // 2
            sheet.paste(tile, (x, y))
            label = f"{index + 1}  {clock(time_)}"
            box = draw.textbbox((x + 4, y + 4), label, font=font)
            draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
            draw.text((x + 4, y + 4), label, fill=(255, 255, 255), font=font)
        target = directory / f"{prefix}-{number + 1}.jpg"
        quality = 85
        while True:
            sheet.save(target, "JPEG", quality=quality)
            if target.stat().st_size <= MAX_SHEET_BYTES or quality <= 40:
                break
            quality -= 15
        sheets.append(Sheet(target, [t for t, _ in chunk], cols, rows))
    return sheets


# -- subprocess plumbing --------------------------------------------------------------------------
def _child_env(extra_path: str = "") -> dict:
    env = {k: v for k, v in os.environ.items() if k.upper() not in _SECRET_ENV}
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_path:
        env["PATH"] = extra_path + os.pathsep + env.get("PATH", "")
    return env


def _popen_kwargs() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}


# -- the analyzer ---------------------------------------------------------------------------------
@dataclass
class VideoView:
    media_id: str
    metadata: dict
    duration: float | None
    window: tuple[float, float]
    sheets: list[Sheet]
    transcript_text: str
    transcript_source: str          # subtitles | whisper | none
    transcript_detail: dict
    notes: list[str] = field(default_factory=list)
    from_cache: bool = False
    quality: dict | None = None

    @property
    def images(self) -> list[Path]:
        return [sheet.path for sheet in self.sheets]

    def to_result(self) -> dict:
        transcript = {"source": self.transcript_source, **self.transcript_detail}
        if self.transcript_text:
            transcript["text"] = self.transcript_text
        quality = {"quality_fallback": True, "height": self.quality.get("height"),
                   "quality_tier": self.quality.get("tier")} if self.quality else {}
        return {"kind": "watch_video", "ok": True, "media_id": self.media_id, "content_trust": _UNTRUSTED, **quality,
                "metadata": self.metadata,
                "window": {"start_seconds": round(self.window[0], 1), "end_seconds": round(self.window[1], 1),
                           "start": clock(self.window[0]), "end": clock(self.window[1])},
                "transcript": transcript,
                "frame_sheets": [
                    {"image_index": index + 1, "grid": f"{sheet.cols}x{sheet.rows}",
                     "frames": [{"tile": tile + 1, "time_seconds": round(t, 1), "time": clock(t)}
                                for tile, t in enumerate(sheet.times)]}
                    for index, sheet in enumerate(self.sheets)],
                "notes": self.notes, "from_cache": self.from_cache}


class VideoAnalyzer:
    def __init__(self, data_dir: Path, *, ffmpeg: str = "", transcriber=None, worker_argv: list[str] | None = None,
                 fetch_timeout: float = FETCH_TIMEOUT, max_download_bytes: int = MAX_DOWNLOAD_BYTES,
                 max_duration: float = MAX_DURATION_SECONDS) -> None:
        self.root = Path(data_dir).resolve()
        if not ffmpeg:
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg = get_ffmpeg_exe()
            except (ImportError, RuntimeError):
                pass
        self.ffmpeg, self.transcriber = ffmpeg, transcriber
        self.worker_argv = worker_argv or [sys.executable, "-m", "personal_agent.video", "--worker"]
        self.fetch_timeout, self.max_download_bytes, self.max_duration = fetch_timeout, max_download_bytes, max_duration
        self.processes: set[asyncio.subprocess.Process] = set()
        self._locks: dict[str, asyncio.Lock] = {}
        self._closed = False

    # paths ------------------------------------------------------------------------------------
    def sender_root(self, sender_id: int) -> Path:
        if type(sender_id) is not int or sender_id == 0:
            raise VideoError("invalid_sender")
        return owned_path(self.root, "memory", "videos", str(sender_id))

    def entry_dir(self, sender_id: int, video_id: str) -> Path:
        if not isinstance(video_id, str) or not _ID_RE.fullmatch(video_id):
            raise VideoError("invalid_media_id")
        return owned_path(self.root, "memory", "videos", str(sender_id), video_id)

    def load_entry(self, sender_id: int, video_id: str) -> dict | None:
        """The cached entry of exactly this sender, or None."""
        try:
            directory = self.entry_dir(sender_id, video_id)
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        except (VideoError, OSError, ValueError):
            return None
        if not isinstance(meta, dict) or meta.get("sender_id") != sender_id or meta.get("video_id") != video_id:
            return None
        return meta

    def list_entries(self, sender_id: int, limit: int = 6) -> list[dict]:
        base = self.sender_root(sender_id)
        entries = []
        if base.is_dir():
            for directory in base.iterdir():
                if directory.is_dir() and not directory.is_symlink() and _ID_RE.fullmatch(directory.name):
                    meta = self.load_entry(sender_id, directory.name)
                    if meta:
                        entries.append(meta)
        entries.sort(key=lambda m: str(m.get("created_at", "")), reverse=True)
        return entries[:limit]

    def _save_meta(self, directory: Path, meta: dict) -> None:
        temporary = directory / "meta.json.tmp"
        temporary.write_text(json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8")
        temporary.replace(directory / "meta.json")

    def _lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    # process helpers --------------------------------------------------------------------------
    async def _run(self, argv: list[str], *, timeout: float, stdin_data: bytes | None = None,
                   cwd: Path | None = None, max_output: int = 1_000_000,
                   extra_path: str = "") -> tuple[int, bytes, bytes]:
        if self._closed:
            raise VideoError("video_analyzer_closed")
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv, cwd=str(cwd or self.root), env=_child_env(extra_path),
                stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=1_000_000, **_popen_kwargs())
            self.processes.add(process)
            out, err = await asyncio.wait_for(process.communicate(stdin_data), timeout)
            return process.returncode or 0, out[:max_output], err[-max_output:]
        except FileNotFoundError:
            raise VideoError("tool_not_found", "Нужная для видео программа не найдена.") from None
        except TimeoutError:
            raise VideoError("video_timeout", "Обработка видео заняла слишком много времени.") from None
        finally:
            if process is not None:
                self.processes.discard(process)
                await terminate_tree(process)

    async def close(self) -> None:
        self._closed = True
        await asyncio.gather(*(terminate_tree(p) for p in list(self.processes)), return_exceptions=True)

    # acquiring the file -------------------------------------------------------------------------
    @staticmethod
    def video_id_for_url(url: str) -> str:
        return "v-" + hashlib.sha256(url.split("#", 1)[0].encode("utf-8")).hexdigest()[:20]

    async def _validate_url(self, url: object) -> str:
        try:
            checked = validate_web_url(url)
        except WebLookupDenied:
            raise VideoError("web_url_denied", "Эта ссылка не подходит: нужна публичная http(s)-ссылка.") from None
        if not await _public_url(checked):
            raise VideoError("web_url_denied", "Эта ссылка не подходит: нужна публичная http(s)-ссылка.")
        return checked

    def ffmpeg_dir(self) -> str:
        """A directory holding the ffmpeg binary under the plain name ``ffmpeg(.exe)``.

        yt-dlp's partial-download check only looks for that name on PATH, while imageio-ffmpeg ships a
        versioned file name; a hard link (or copy) inside the data directory bridges the two.
        """
        if not self.ffmpeg:
            return ""
        source = Path(self.ffmpeg)
        if source.stem.lower() == "ffmpeg":
            return str(source.parent)
        directory = owned_path(self.root, "tools", "ffmpeg")
        target = directory / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        try:
            if not target.is_file() or target.stat().st_size != source.stat().st_size:
                directory.mkdir(parents=True, exist_ok=True)
                target.unlink(missing_ok=True)
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
                    target.chmod(0o755)
            return str(directory)
        except OSError:
            return ""

    def _worker_args(self, directory: Path, url: str, *, window: tuple[float, float] | None = None,
                     subtitles: bool = True, prefix: str = "video") -> dict:
        return {"mode": "fetch", "url": url, "out_dir": str(directory), "ffmpeg": self.ffmpeg_dir() or self.ffmpeg,
                "max_bytes": self.max_download_bytes, "max_duration": self.max_duration,
                "analyzed_seconds": ANALYZED_SECONDS, "window": list(window) if window else None,
                "subtitles": subtitles, "prefix": prefix, "timeout": self.fetch_timeout}

    async def _run_worker(self, args: dict) -> dict:
        code, out, _ = await self._run(self.worker_argv, timeout=self.fetch_timeout + 10,
                                       stdin_data=json.dumps(args).encode("utf-8"),
                                       cwd=Path(__file__).resolve().parents[1], max_output=200_000,
                                       extra_path=args.get("ffmpeg") if Path(str(args.get("ffmpeg"))).is_dir() else "")
        try:
            result = json.loads(out.decode("utf-8", errors="replace").strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise VideoError("video_fetch_failed", "Не удалось получить видео по ссылке.") from None
        if not isinstance(result, dict) or not result.get("ok"):
            raw = str(result.get("code", "extraction_failed")) if isinstance(result, dict) else "extraction_failed"
            raise VideoError(raw if re.fullmatch(r"[a-z_]{1,60}", raw) else "extraction_failed",
                             "Не удалось получить видео по ссылке.")
        return result

    async def _download_direct(self, url: str, target: Path) -> None:
        """A plain link to a media file: per-hop public-address validation, hard size cap, no cookies."""
        import httpx
        from .web_lookup import PinnedAsyncHTTPTransport
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PersonalAgent/0.1"}
        async with httpx.AsyncClient(transport=PinnedAsyncHTTPTransport(), timeout=20.0, trust_env=False, follow_redirects=False, headers=headers) as client:
            for hop in range(MAX_REDIRECTS + 1):
                if not await _public_url(url):
                    raise VideoError("web_url_denied")
                async with client.stream("GET", url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location", "")
                        if not location or hop == MAX_REDIRECTS:
                            raise VideoError("web_redirect_denied")
                        url = urljoin(url, location)
                        continue
                    response.raise_for_status()
                    total = 0
                    with target.open("wb") as handle:
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > self.max_download_bytes:
                                raise VideoError("video_too_large", "Видео слишком большое.")
                            handle.write(chunk)
                    return
        raise VideoError("web_redirect_denied")

    async def prepare_url(self, sender_id: int, url: str) -> dict:
        """Metadata, video file and subtitles for a public link (cached per sender)."""
        url = await self._validate_url(url)
        video_id = self.video_id_for_url(url)
        directory = self.entry_dir(sender_id, video_id)
        async with self._lock(f"{sender_id}:{video_id}"):
            meta = self.load_entry(sender_id, video_id)
            if meta and (directory / meta.get("file", "-")).is_file():
                return meta
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True)
            try:
                suffix = Path(urlsplit(url).path).suffix.lower()
                if suffix in VIDEO_SUFFIXES:
                    target = directory / f"video{suffix}"
                    await asyncio.wait_for(self._download_direct(url, target), self.fetch_timeout)
                    meta = {"metadata": {"title": Path(urlsplit(url).path).name[:200], "webpage_url": url[:500]},
                            "file": target.name, "subtitle": None, "downloaded_range": None, "notes": []}
                else:
                    meta = await self._fetch_with_ytdlp(directory, url)
                meta.update({"sender_id": sender_id, "video_id": video_id, "source": {"kind": "url", "url": url},
                             "created_at": datetime.now(timezone.utc).isoformat(), "views": {}, "sections": []})
                if not meta.get("metadata_only"):
                    await self._probe_into(directory, meta)
                    if meta.get("audio_file"):
                        meta["has_audio"] = True
                    if meta.get("quality"):
                        if meta.get("width") and meta.get("height"):
                            meta["quality"]["height"] = min(meta["width"], meta["height"])
                self._save_meta(directory, meta)
                return meta
            except TimeoutError:
                shutil.rmtree(directory, ignore_errors=True)
                raise VideoError("video_timeout", "Обработка видео заняла слишком много времени.") from None
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise

    async def _fetch_with_ytdlp(self, directory: Path, url: str) -> dict:
        args = self._worker_args(directory, url)
        try:
            result = await self._run_worker(args)
        except VideoError as exc:
            if exc.code != "video_too_large":
                raise
            # Refused or aborted for size: wipe the partial files and retry with the lowest quality.
            for leftover in directory.glob("*"):
                if leftover.is_file():
                    leftover.unlink(missing_ok=True)
            result = await self._run_worker({**args, "smallest": True})
        video_file = self._find_file(directory, "video.")
        metadata_only = bool(result.get("metadata_only"))
        if video_file is None and not metadata_only:
            raise VideoError("video_too_large" if result.get("skipped_by_size") else "download_failed",
                             "Видео не удалось скачать.")
        subtitle = None
        found = sorted(directory.glob("subs.*"))
        if found and result.get("subtitle"):
            subtitle = {"file": found[0].name, "language": str(result["subtitle"].get("language", ""))[:20],
                        "kind": result["subtitle"].get("kind") if result["subtitle"].get("kind") in {"manual", "auto"} else "auto"}
        quality = result.get("quality") if isinstance(result.get("quality"), dict) else None
        meta = {"metadata": bounded_metadata(result.get("info") or {}), "file": video_file.name if video_file else "",
                "subtitle": subtitle, "downloaded_range": result.get("downloaded_range"),
                "format_id": str(result.get("format_id", ""))[:80],
                "notes": [n for n in result.get("notes", []) if isinstance(n, str)][:6]}
        if quality:
            meta["quality"] = {"quality_fallback": True, "height": quality.get("height"), "tier": str(quality.get("tier"))}
        audio = self._find_file(directory, "audio.") if result.get("audio_fetched") else None
        if audio:
            meta["audio_file"] = audio.name
        if metadata_only:
            thumb = self._find_file(directory, "thumb.")
            duration = _num(meta["metadata"].get("duration_seconds")) or _num((result.get("info") or {}).get("duration")) or 1.0
            meta.update({"metadata_only": True, "file": thumb.name if thumb else "", "duration": duration,
                         "downloaded_range": [0.0, duration], "has_audio": False})
        return meta

    @staticmethod
    def _find_file(directory: Path, prefix: str) -> Path | None:
        candidates = [p for p in directory.glob(prefix + "*") if p.is_file() and p.suffix.lower() not in
                      {".part", ".ytdl", ".vtt", ".srt", ".json", ".tmp"} and not p.name.endswith(".part")]
        return max(candidates, key=lambda p: p.stat().st_size, default=None)

    async def prepare_file(self, sender_id: int, source: Path, video_id: str, *, name: str = "") -> dict:
        """A local video that already lives inside the agent's data directory (a Telegram attachment)."""
        source = Path(source).resolve()
        if not source.is_relative_to(self.root) or not source.is_file():
            raise VideoError("invalid_video_source")
        directory = self.entry_dir(sender_id, video_id)
        async with self._lock(f"{sender_id}:{video_id}"):
            meta = self.load_entry(sender_id, video_id)
            if meta and (directory / meta.get("file", "-")).is_file():
                return meta
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True)
            try:
                suffix = source.suffix.lower() if source.suffix.lower() in VIDEO_SUFFIXES else ".mp4"
                shutil.copyfile(source, directory / f"video{suffix}")
                meta = {"sender_id": sender_id, "video_id": video_id, "source": {"kind": "file"},
                        "metadata": {"title": (name or "video")[:200]}, "file": f"video{suffix}", "subtitle": None,
                        "downloaded_range": None, "notes": [], "views": {}, "sections": [],
                        "created_at": datetime.now(timezone.utc).isoformat()}
                await self._probe_into(directory, meta)
                self._save_meta(directory, meta)
                return meta
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise

    # probing / ffmpeg ---------------------------------------------------------------------------
    async def _ffmpeg(self, *args: str, timeout: float = 60.0) -> tuple[int, str]:
        if not self.ffmpeg:
            raise VideoError("ffmpeg_unavailable", "ffmpeg недоступен.")
        code, _, err = await self._run([self.ffmpeg, "-nostdin", "-hide_banner", "-protocol_whitelist", "file,pipe",
                                        *args], timeout=timeout, max_output=64_000)
        return code, err.decode("utf-8", errors="replace")

    async def _probe_into(self, directory: Path, meta: dict) -> None:
        path = directory / meta["file"]
        _, info = await self._ffmpeg("-i", str(path), timeout=30)
        video = re.search(r"Video: [^\n]*?\b(\d{2,5})x(\d{2,5})\b", info)
        if not video:
            raise VideoError("not_a_video", "Это не похоже на видеофайл.")
        duration = None
        timing = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", info)
        if timing:
            hour, minute, second = map(float, timing.groups())
            duration = hour * 3600 + minute * 60 + second
        if not duration or duration <= 0:
            duration = _num(meta["metadata"].get("duration_seconds"))
        if not duration:
            _, decoded = await self._ffmpeg("-i", str(path), "-map", "0:v:0", "-an", "-t", str(self.max_duration),
                                            "-f", "null", "-", timeout=60)
            times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", decoded)
            if times:
                hour, minute, second = map(float, times[-1])
                duration = hour * 3600 + minute * 60 + second
        if not duration or duration <= 0:
            raise VideoError("video_duration_unknown", "Не удалось определить длительность видео.")
        coverage = meta.get("downloaded_range")
        if coverage and _num(meta["metadata"].get("duration_seconds")):
            duration_total = float(meta["metadata"]["duration_seconds"])  # a partial download: the file is shorter
        else:
            duration_total = duration
        meta["duration"] = round(duration_total, 2)
        meta["metadata"].setdefault("duration_seconds", round(duration_total, 2))
        meta["width"], meta["height"] = int(video.group(1)), int(video.group(2))
        meta["has_audio"] = bool(re.search(r"Stream #[^\n]*Audio:", info))
        meta["downloaded_range"] = coverage if coverage else [0.0, round(duration, 2)]

    async def _extract_frames(self, video: Path, times: list[float], offset: float, directory: Path) -> list[tuple[float, Path]]:
        directory.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(4)
        scale = (f"scale=w='min(iw,{FRAME_MAX_SIDE})':h='min(ih,{FRAME_MAX_SIDE})':"
                 "force_original_aspect_ratio=decrease:force_divisible_by=2")

        async def one(index: int, moment: float):
            output = directory / f"f{index:02d}.jpg"
            async with semaphore:
                for attempt in (0.0, 0.7, 1.5):
                    local = max(0.0, moment - offset - attempt)
                    code, _ = await self._ffmpeg("-loglevel", "error", "-ss", f"{local:.3f}", "-i", str(video), "-an",
                                                 "-sn", "-frames:v", "1", "-vf", scale, "-q:v", "3", "-y", str(output),
                                                 timeout=30)
                    if code == 0 and output.is_file() and output.stat().st_size > 0:
                        return moment, output
            return None

        results = await asyncio.gather(*(one(i, t) for i, t in enumerate(times)))
        return [item for item in results if item]

    # transcript ---------------------------------------------------------------------------------
    async def _ensure_transcript(self, directory: Path, meta: dict) -> tuple[dict, list[str]]:
        """Transcript record {source, ...}; definitive outcomes are cached, transient failures are not."""
        cached = meta.get("transcript")
        if isinstance(cached, dict):
            return cached, []
        notes: list[str] = []
        subtitle = meta.get("subtitle")
        if subtitle:
            try:
                cues = parse_subtitles((directory / subtitle["file"]).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                cues = []
            if cues:
                record = {"source": "subtitles", "language": subtitle.get("language"), "kind": subtitle.get("kind"),
                          "cues": [[c.start, c.text] for c in cues]}
                meta["transcript"] = record
                self._save_meta(directory, meta)
                return record, notes
        if meta.get("metadata_only"):
            return {"source": "none", "reason": "video_not_downloaded"}, notes
        if not meta.get("has_audio"):
            record = {"source": "none", "reason": "no_audio_track"}
            meta["transcript"] = record
            self._save_meta(directory, meta)
            return record, notes
        transcriber = self.transcriber
        health = getattr(transcriber, "health", None)
        if (transcriber is None or not hasattr(transcriber, "transcribe")
                or (callable(health) and not health().get("enabled"))):
            return {"source": "none", "reason": "whisper_not_configured"}, notes
        audio = directory / "audio.wav"
        try:
            code, _ = await self._ffmpeg("-loglevel", "error", "-i", str(directory / (meta.get("audio_file") or meta["file"])), "-vn", "-t",
                                         str(ANALYZED_SECONDS), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y",
                                         str(audio), timeout=60)
            if code != 0 or not audio.is_file():
                return {"source": "none", "reason": "audio_extraction_failed"}, notes
            text = await asyncio.wait_for(transcriber.transcribe(audio), TRANSCRIBE_TIMEOUT)
        except TimeoutError:
            return {"source": "none", "reason": "transcription_timeout"}, notes
        except VideoError:
            return {"source": "none", "reason": "audio_extraction_failed"}, notes
        except Exception as exc:
            if type(exc).__name__ == "TranscriptionUnavailable":
                return {"source": "none", "reason": "transcription_unavailable"}, notes
            raise
        finally:
            audio.unlink(missing_ok=True)
        text = re.sub(r"\s+", " ", text or "").strip()
        record = {"source": "whisper", "text": text[:60_000], "language": "auto",
                  "covers_seconds": min(ANALYZED_SECONDS, meta.get("duration", ANALYZED_SECONDS))} if text else \
            {"source": "none", "reason": "no_speech_detected"}
        meta["transcript"] = record
        self._save_meta(directory, meta)
        return record, notes

    # public entry ---------------------------------------------------------------------------------
    def resolve_window(self, meta: dict, start: float | None, end: float | None) -> tuple[float, float, list[str]]:
        duration = float(meta["duration"])
        covered_end = float(meta["downloaded_range"][1]) if meta.get("downloaded_range") else duration
        notes: list[str] = []
        for value in (start, end):
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)
                                      or not math.isfinite(value) or value < 0):
                raise VideoError("invalid_window")
        if start is None and end is None:
            low, high = 0.0, min(duration, covered_end if duration > self.max_duration else duration)
            if duration > self.max_duration:
                notes.append(f"video_longer_than_{int(self.max_duration // 60)}_min_only_first_"
                             f"{int(ANALYZED_SECONDS // 60)}_min_shown_use_start_seconds_for_other_parts")
                high = min(duration, ANALYZED_SECONDS)
        else:
            low = float(start or 0.0)
            high = float(end) if end is not None else min(duration, low + DEFAULT_SEGMENT_SECONDS)
            if low >= duration:
                raise VideoError("invalid_window", "Начало отрезка за пределами видео.")
            high = min(high, duration, low + ANALYZED_SECONDS)
            if high - low < 1.0:
                high = min(duration, low + 1.0)
                low = max(0.0, high - 1.0)
            if high <= low:
                raise VideoError("invalid_window")
        return low, high, notes

    async def _file_for_window(self, sender_id: int, directory: Path, meta: dict, low: float, high: float,
                               notes: list[str]) -> tuple[Path, float]:
        """(video file, its start offset on the original timeline) covering [low, high]."""
        have_start, have_end = meta["downloaded_range"]
        if low >= have_start - 0.5 and high <= have_end + 0.5:
            return directory / meta["file"], float(have_start)
        for section in meta.get("sections", []):
            if low >= section["start"] - 0.5 and high <= section["end"] + 0.5 and (directory / section["file"]).is_file():
                return directory / section["file"], float(section["start"])
        if meta["source"].get("kind") != "url" or not meta["source"].get("url"):
            raise VideoError("window_not_available", "Этот отрезок недоступен в сохранённом файле.")
        prefix = f"section-{int(low)}-{int(high)}"
        async with self._lock(f"{sender_id}:{meta['video_id']}:fetch"):
            args = self._worker_args(directory, meta["source"]["url"], window=(low, high), subtitles=False, prefix=prefix)
            await self._run_worker(args)
            section_file = self._find_file(directory, prefix + ".")
            if section_file is None:
                raise VideoError("download_failed", "Не удалось скачать этот отрезок.")
            meta.setdefault("sections", []).append({"file": section_file.name, "start": low, "end": high})
            self._save_meta(directory, meta)
            notes.append("segment_downloaded_separately")
            return section_file, low

    async def watch(self, sender_id: int, *, url: str | None = None, media_id: str | None = None,
                    start: float | None = None, end: float | None = None) -> VideoView:
        if (url is None) == (media_id is None):
            raise VideoError("invalid_arguments")
        if url is not None:
            meta = await self.prepare_url(sender_id, url)
        else:
            meta = self.load_entry(sender_id, media_id)
            if meta is None:
                raise VideoError("media_not_found", "Такого видео нет в сохранённых.")
        return await self.view(sender_id, meta, start, end)

    async def view(self, sender_id: int, meta: dict, start: float | None = None, end: float | None = None) -> VideoView:
        video_id = meta["video_id"]
        directory = self.entry_dir(sender_id, video_id)
        low, high, notes = self.resolve_window(meta, start, end)
        notes = [*meta.get("notes", []), *notes]
        key = f"{low:.1f}-{high:.1f}"
        async with self._lock(f"{sender_id}:{video_id}:view"):
            meta = self.load_entry(sender_id, video_id) or meta
            transcript_task = asyncio.create_task(self._ensure_transcript(directory, meta))
            try:
                cached = meta.get("views", {}).get(key)
                sheets: list[Sheet] = []
                from_cache = False
                if cached and all((directory / item["file"]).is_file() for item in cached):
                    sheets = [Sheet(directory / item["file"], item["times"], item["cols"], item["rows"]) for item in cached]
                    from_cache = True
                elif meta.get("metadata_only"):
                    thumb = directory / str(meta.get("file", ""))
                    sheets = [Sheet(thumb, [0.0])] if meta.get("file") and thumb.is_file() else []
                    notes.append("video_not_downloaded_only_thumbnail_and_metadata")
                else:
                    source, offset = await self._file_for_window(sender_id, directory, meta, low, high, notes)
                    times = sample_times(low, high, frame_count_for(high - low))
                    times = [min(t, float(meta["duration"]) - 0.1) for t in times]
                    frames_dir = directory / "frames" / re.sub(r"[^0-9-]", "_", key)
                    shutil.rmtree(frames_dir, ignore_errors=True)
                    frames = await asyncio.wait_for(self._extract_frames(source, times, offset, frames_dir), FRAME_TIMEOUT)
                    sheets = compose_sheets(frames, frames_dir, "sheet")
                    if not sheets:
                        notes.append("frames_unavailable")
                    else:
                        meta.setdefault("views", {})[key] = [
                            {"file": str(s.path.relative_to(directory)).replace("\\", "/"), "times": s.times,
                             "cols": s.cols, "rows": s.rows} for s in sheets]
                        self._save_meta(directory, meta)
                record, extra_notes = await transcript_task
            except BaseException:
                transcript_task.cancel()
                await asyncio.gather(transcript_task, return_exceptions=True)
                raise
        notes.extend(extra_notes)
        text, detail = "", {}
        if record["source"] == "subtitles":
            cues = [Cue(float(s), str(t)) for s, t in record["cues"]]
            whole = (low, high) != (0.0, float(meta["duration"]))
            text, truncated = format_transcript(cues, low if whole else None, high if whole else None)
            detail = {"language": record.get("language"), "kind": record.get("kind"), "truncated": truncated,
                      "timestamped": True}
            if record.get("kind") == "auto":
                notes.append("subtitles_are_automatic_captions_may_contain_errors")
            if truncated:
                notes.append("transcript_truncated_use_start_seconds_to_read_a_part")
            if not text:
                notes.append("no_speech_in_this_window")
        elif record["source"] == "whisper":
            body = record.get("text", "")
            text = body if len(body) <= TRANSCRIPT_LIMIT else body[:TRANSCRIPT_LIMIT] + " [truncated]"
            detail = {"language": "auto", "timestamped": False, "covers_seconds": record.get("covers_seconds"),
                      "truncated": len(body) > TRANSCRIPT_LIMIT}
            notes.append("transcript_from_local_speech_recognition_without_timestamps")
            if meta["duration"] > ANALYZED_SECONDS:
                notes.append("speech_recognized_only_for_the_first_10_minutes")
            if (low, high) != (0.0, float(meta["duration"])):
                notes.append("transcript_covers_the_whole_clip_not_only_this_window")
        else:
            detail = {"reason": record.get("reason")}
            if record.get("reason") == "whisper_not_configured":
                notes.append("no_platform_subtitles_and_local_speech_recognition_is_not_configured")
            elif record.get("reason") == "no_audio_track":
                notes.append("video_has_no_audio_track")
        return VideoView(video_id, meta.get("metadata", {}), float(meta["duration"]), (low, high), sheets, text,
                         record["source"], detail, list(dict.fromkeys(notes)), from_cache, meta.get("quality"))


# -- retention ------------------------------------------------------------------------------------
def cleanup_videos(root: Path, cutoff: datetime) -> int:
    """Delete cached video entries older than cutoff (by meta.json / directory mtime); returns the count."""
    base = owned_path(root, "memory", "videos")
    removed = 0
    if not base.is_dir():
        return 0
    for sender_dir in base.iterdir():
        if sender_dir.is_symlink() or not sender_dir.is_dir():
            continue
        for entry in sender_dir.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                continue
            try:
                meta = entry / "meta.json"
                modified = datetime.fromtimestamp((meta if meta.is_file() else entry).stat().st_mtime, timezone.utc)
                if modified < cutoff:
                    owned_path(root, "memory", "videos", sender_dir.name, entry.name)
                    shutil.rmtree(entry)
                    removed += 1
            except (OSError, ValueError):
                continue
        with suppress(OSError):
            sender_dir.rmdir()
    return removed


# -- worker process (yt-dlp) ----------------------------------------------------------------------
def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        with suppress(OSError):
            if item.is_file():
                total += item.stat().st_size
    return total


def _worker_fetch(args: dict) -> dict:
    state = {"done": False}
    try:
        return _worker_fetch_inner(args, state)
    finally:
        state["done"] = True


def _worker_fetch_inner(args: dict, state: dict) -> dict:
    import yt_dlp
    from yt_dlp.utils import DownloadError, download_range_func

    out_dir = Path(args["out_dir"])
    prefix = args.get("prefix", "video")
    max_bytes = int(args["max_bytes"])
    notes: list[str] = []
    window = args.get("window")

    def watchdog() -> None:
        while not state["done"]:
            # Separate audio and video streams are merged into a third file: allow for that peak.
            if _directory_size(out_dir) > 2 * max_bytes + 64 * 1024 * 1024:
                sys.stdout = sys.__stdout__
                print(json.dumps({"ok": False, "code": "video_too_large"}), flush=True)
                os._exit(3)
            time.sleep(0.4)

    threading.Thread(target=watchdog, daemon=True).start()

    class Quiet:
        def debug(self, msg): pass
        def info(self, msg): pass
        def warning(self, msg): pass
        def error(self, msg): pass

    options = {
        "quiet": True, "no_warnings": True, "noprogress": True, "logger": Quiet(), "ignoreconfig": True,
        "cachedir": False, "noplaylist": True, "playlist_items": "1", "allowed_extractors": ["default", "-generic"],
        "socket_timeout": 20, "retries": 2, "fragment_retries": 2, "extractor_retries": 1,
        "cookiefile": None, "cookiesfrombrowser": None, "usenetrc": False, "netrc_cmd": None,
        "outtmpl": {"default": str(out_dir / (prefix + ".%(ext)s")), "subtitle": str(out_dir / "subs.%(ext)s")},
        "windowsfilenames": True, "restrictfilenames": True, "overwrites": True, "updatetime": False,
        "writethumbnail": False, "writeinfojson": False, "live_from_start": False,
        "merge_output_format": "mp4/mkv", "concurrent_fragment_downloads": 2,
    }
    if args.get("ffmpeg"):
        options["ffmpeg_location"] = args["ffmpeg"]
    with yt_dlp.YoutubeDL(options) as ydl:
        try:
            info = ydl.extract_info(args["url"], download=False)
        except DownloadError as exc:
            return {"ok": False, "code": error_code_for(str(exc))}
        if not isinstance(info, dict):
            return {"ok": False, "code": "extraction_failed"}
        if info.get("_type") == "playlist" and info.get("entries"):
            info = next((e for e in info["entries"] if isinstance(e, dict)), info)
        if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"}:
            return {"ok": False, "code": "live_stream"}
        duration = _num(info.get("duration"))
        ranged = bool(window)
        formats = info.get("formats") or []
        budget = (window[1] - window[0]) if ranged else duration
        smallest = bool(args.get("smallest"))
        format_id = None if smallest else select_formats(formats, duration=budget, max_bytes=max_bytes)
        if not ranged and duration and duration > args["max_duration"] and format_id is None:
            # Even the smallest version of the whole video is over the cap: fetch only the first part.
            # (Whole-file downloads are preferred: a partial ffmpeg fetch of a DASH stream is throttled.)
            window = [0, args["analyzed_seconds"]]
            ranged = True
            notes.append("video_longer_than_limit_only_first_part_downloaded")
            format_id = None if smallest else select_formats(formats, duration=window[1], max_bytes=max_bytes)
        quality: dict | None = None
        audio_format: str | None = None
        metadata_only = False
        if format_id is None:
            if not info.get("formats") and info.get("url"):
                format_id = None  # a single direct stream; let yt-dlp take it as is
            elif not formats:
                return {"ok": False, "code": "extraction_failed", "skipped_by_size": True}
            else:
                # Nothing fits at the usual quality: retry with the smallest one instead of failing.
                plans = smallest_formats(formats, duration=window[1] if ranged else duration, max_bytes=max_bytes)
                if not plans and not ranged and duration and duration > args["analyzed_seconds"]:
                    window = [0, args["analyzed_seconds"]]
                    ranged = True
                    notes.append("video_longer_than_limit_only_first_part_downloaded")
                    plans = smallest_formats(formats, duration=window[1], max_bytes=max_bytes)
                if plans:
                    format_id, audio_format = plans[0]["format"], plans[0].get("audio_format")
                    quality = {"quality_fallback": True, "height": int(plans[0]["height"] or 0) or None, "tier": plans[0]["tier"]}
                    notes.append("lowest_quality_used_because_the_video_is_too_large")
                else:
                    metadata_only = True
                    quality = {"quality_fallback": True, "height": None, "tier": "metadata_only"}
                    notes.append("video_not_downloaded_too_large_only_metadata_subtitles_and_thumbnail")
        choice = pick_subtitle_track(info) if args.get("subtitles") else None
        updates: dict = {}
        if format_id:
            updates["format"] = format_id
        if ranged:
            updates["download_ranges"] = download_range_func([], [(float(window[0]), float(window[1]))])
            updates["force_keyframes_at_cuts"] = False
        if choice:
            kind, language = choice
            updates.update({"writesubtitles": kind == "manual", "writeautomaticsub": kind == "auto",
                            "subtitleslangs": [language], "subtitlesformat": "vtt/srt/best"})
        if metadata_only:
            updates.update({"skip_download": True, "writethumbnail": True})
            ydl.params["outtmpl"] = {**ydl.params["outtmpl"], "thumbnail": str(out_dir / "thumb.%(ext)s")}
        ydl.params.update(updates)
        try:
            ydl.process_ie_result(ydl.sanitize_info(info), download=True)
        except DownloadError as exc:
            return {"ok": False, "code": error_code_for(str(exc))}
        if audio_format:
            # The smallest video-only stream has no sound: fetch the audio track separately.
            ydl.params.update({"format": audio_format, "writesubtitles": False, "writeautomaticsub": False,
                               "outtmpl": {**ydl.params["outtmpl"], "default": str(out_dir / "audio.%(ext)s")}})
            try:
                ydl.process_ie_result(ydl.sanitize_info(info), download=True)
            except DownloadError:
                audio_format = None
                notes.append("audio_track_not_fetched")
    return {"ok": True, "quality": quality, "metadata_only": metadata_only, "audio_fetched": bool(audio_format), "info": {k: v for k, v in info.items() if k in {
        "title", "uploader", "channel", "duration", "upload_date", "description", "view_count", "like_count",
        "comment_count", "webpage_url", "extractor_key", "extractor", "language"}},
        "format_id": format_id or "", "notes": notes,
        "downloaded_range": [float(window[0]), float(window[1])] if ranged else None,
        "subtitle": {"language": choice[1], "kind": choice[0]} if choice else None}


def _worker() -> None:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # nothing but the final JSON line may reach the parent's stdout
    try:
        args = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        result = _worker_fetch(args)
    except Exception as exc:
        result = {"ok": False, "code": error_code_for(str(exc))}
    real_stdout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    real_stdout.flush()


if __name__ == "__main__" and len(sys.argv) == 2 and sys.argv[1] == "--worker":
    _worker()
