"""Bounded local sticker rendering in a cancellable child process."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from uuid import uuid4

from .media import terminate_tree
from .security import owned_path

MAX_SOURCE_BYTES = 10 * 1024 * 1024
MAX_TGS_JSON_BYTES = 2 * 1024 * 1024
MAX_DURATION = 10.0
MAX_FRAMES = 4
RENDER_TIMEOUT = 30.0


class StickerRenderError(RuntimeError):
    def __init__(self, code: str, public_message: str = "Не удалось разобрать анимированный стикер."):
        super().__init__(code)
        self.code, self.public_message = code, public_message


@dataclass(frozen=True)
class RenderedSticker:
    images: list[Path]
    timestamps: list[float]
    duration: float


def _finite(value, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maximum:
        raise StickerRenderError("sticker_limits")
    return float(value)


def _load_tgs(source: Path) -> dict:
    try:
        with gzip.open(source, "rb") as stream:
            raw = stream.read(MAX_TGS_JSON_BYTES + 1)
        if len(raw) > MAX_TGS_JSON_BYTES:
            raise StickerRenderError("tgs_json_too_large")
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("layers"), list):
            raise StickerRenderError("invalid_tgs")
        _finite(data.get("w"), maximum=2048)
        _finite(data.get("h"), maximum=2048)
        fps = _finite(data.get("fr"), maximum=120)
        first, last = data.get("ip", 0), data.get("op")
        if not isinstance(first, (int, float)) or not isinstance(last, (int, float)):
            raise StickerRenderError("invalid_tgs")
        _finite((last - first) / fps, maximum=MAX_DURATION)
        stack, count = [data], 0
        while stack:
            item = stack.pop()
            count += 1
            if count > 100_000:
                raise StickerRenderError("tgs_complexity_limit")
            if isinstance(item, dict):
                # Telegram vector stickers need no external resources or expressions.
                if isinstance(item.get("u"), str) and item["u"] or isinstance(item.get("p"), str) and item["p"] or isinstance(item.get("x"), str):
                    raise StickerRenderError("tgs_external_resource")
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
        return data
    except StickerRenderError:
        raise
    except (OSError, ValueError, EOFError, RecursionError, TypeError):
        raise StickerRenderError("invalid_tgs") from None


def _save_frame(image, destination: Path):
    from PIL import Image
    image = image.convert("RGBA")
    # A stable light backdrop makes transparent sticker details visible to vision.
    background = Image.new("RGBA", image.size, (240, 240, 240, 255))
    background.alpha_composite(image)
    background.convert("RGB").save(destination, "PNG")


def _render_tgs(source: Path, destination: Path) -> dict:
    from rlottie_python import LottieAnimation
    data = _load_tgs(source)
    animation = LottieAnimation.from_data(json.dumps(data), resource_path=str(destination))
    try:
        total = animation.lottie_animation_get_totalframe()
        fps = animation.lottie_animation_get_framerate()
        duration = _finite(total / fps, maximum=MAX_DURATION)
        width, height = animation.lottie_animation_get_size()
        scale = min(1, 512 / max(width, height))
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
        indices = sorted(set(round((total - 1) * i / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)))
        for index, frame in enumerate(indices):
            image = animation.render_pillow_frame(frame, width=width, height=height)
            _save_frame(image, destination / f"frame-{index}.png")
        return {"timestamps": [frame / fps for frame in indices], "duration": duration}
    finally:
        animation.lottie_animation_destroy()


def _ffmpeg_run(argv: list[str], *, expected_failure=False) -> str:
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            timeout=20, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode and not expected_failure:
        raise StickerRenderError("webm_decode_failed")
    return result.stderr[-64_000:].decode("utf-8", errors="replace")


def _render_webm(source: Path, destination: Path, ffmpeg: str) -> dict:
    from PIL import Image
    if source.read_bytes()[:4] != b"\x1aE\xdf\xa3":
        raise StickerRenderError("invalid_webm")
    if not ffmpeg:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
    base = [ffmpeg, "-nostdin", "-hide_banner", "-threads", "1", "-filter_threads", "1",
            "-protocol_whitelist", "file,pipe", "-i", str(source)]
    info = _ffmpeg_run(base, expected_failure=True)
    dimensions = re.search(r"Video: [^\n]*?\b(\d{1,5})x(\d{1,5})\b", info)
    if not dimensions or max(map(int, dimensions.groups())) > 4096:
        raise StickerRenderError("invalid_webm_dimensions")
    # libvpx preserves the alpha plane of Telegram VP8/VP9 stickers.
    codec = re.search(r"Video: (vp8|vp9)\b", info)
    if codec:
        insertion = base.index("-i")
        base[insertion:insertion] = ["-c:v", "libvpx-vp9" if codec[1] == "vp9" else "libvpx"]
    timing = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if timing:
        hour, minute, second = map(float, timing.groups())
        duration = hour * 3600 + minute * 60 + second
    else:
        # Some valid WebM files omit container duration; bounded decode measures it.
        decoded = _ffmpeg_run(base + ["-map", "0:v:0", "-an", "-t", "10.1", "-f", "null", "-"])
        times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", decoded)
        if not times:
            raise StickerRenderError("webm_duration_unavailable")
        hour, minute, second = map(float, times[-1])
        duration = hour * 3600 + minute * 60 + second
    duration = _finite(duration, maximum=MAX_DURATION)
    # Avoid seeking exactly to the EOF of a short clip.
    timestamps = [max(0, duration - min(0.1, duration / 4)) * i / (MAX_FRAMES - 1) for i in range(MAX_FRAMES)]
    for index, timestamp in enumerate(timestamps):
        output = destination / f"frame-{index}.png"
        _ffmpeg_run(base + ["-loglevel", "error", "-ss", str(timestamp), "-map", "0:v:0", "-an", "-frames:v", "1",
                            "-vf", "scale=512:512:force_original_aspect_ratio=decrease", "-y", str(output)])
        if not output.is_file():
            if index == 0:
                raise StickerRenderError("webm_no_frames")
            return {"timestamps": timestamps[:index], "duration": duration}
        with Image.open(output) as image:
            _save_frame(image, output)
    return {"timestamps": timestamps, "duration": duration}


def _worker(format: str, source: Path, destination: Path, ffmpeg: str):
    try:
        result = _render_tgs(source, destination) if format == "tgs" else _render_webm(source, destination, ffmpeg)
        print(json.dumps({"ok": True, **result}), flush=True)
    except StickerRenderError as exc:
        print(json.dumps({"ok": False, "code": exc.code}), flush=True)
    except Exception:
        print(json.dumps({"ok": False, "code": "sticker_render_failed"}), flush=True)


async def render_sticker(source: Path, *, root: Path, format: str, ffmpeg: str = "") -> RenderedSticker:
    source, root = Path(source).resolve(), Path(root).resolve()
    if format not in {"tgs", "webm"} or not source.is_relative_to(root) or not source.is_file():
        raise StickerRenderError("invalid_sticker_source")
    if not 0 < source.stat().st_size <= MAX_SOURCE_BYTES:
        raise StickerRenderError("sticker_source_too_large")
    destination = owned_path(root, "memory", "attachments", str(uuid4()) + "-frames")
    destination.mkdir(parents=True)
    environment = {k: v for k, v in os.environ.items() if k.upper() not in {"TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"}}
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    process = None
    complete = False
    launch = asyncio.create_task(asyncio.create_subprocess_exec(
        sys.executable, "-m", "personal_agent.animated_stickers", "--worker", format, str(source), str(destination), ffmpeg,
        cwd=str(Path(__file__).resolve().parents[1]), env=environment, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=16_384, **kwargs))
    try:
        process = await asyncio.shield(launch)
        output, _ = await asyncio.wait_for(process.communicate(), RENDER_TIMEOUT)
        if process.returncode or len(output) > 4096:
            raise StickerRenderError("sticker_render_failed")
        result = json.loads(output)
        if not result.get("ok"):
            code = result.get("code", "sticker_render_failed")
            raise StickerRenderError(code if re.fullmatch(r"[a-z_]{1,80}", str(code)) else "sticker_render_failed")
        duration = _finite(result["duration"], maximum=MAX_DURATION)
        timestamps = result["timestamps"]
        if not isinstance(timestamps, list) or not 1 <= len(timestamps) <= MAX_FRAMES:
            raise StickerRenderError("sticker_invalid_frames")
        images = [owned_path(destination, f"frame-{i}.png") for i in range(len(timestamps))]
        if any(not isinstance(t, (float, int)) or not math.isfinite(t) or not 0 <= t < duration for t in timestamps):
            raise StickerRenderError("sticker_invalid_frames")
        if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
            raise StickerRenderError("sticker_invalid_frames")
        if any(not p.is_file() or p.stat().st_size > 1024 * 1024 for p in images):
            raise StickerRenderError("sticker_invalid_frames")
        complete = True
        return RenderedSticker(images, timestamps, duration)
    except asyncio.CancelledError:
        if process is None:
            process = await launch
        raise
    except TimeoutError:
        raise StickerRenderError("sticker_render_timeout") from None
    except StickerRenderError:
        raise
    except Exception:
        raise StickerRenderError("sticker_render_failed") from None
    finally:
        if process is not None:
            await terminate_tree(process)
        if not complete:
            if not destination.resolve().is_relative_to((root / "memory" / "attachments").resolve()):
                raise StickerRenderError("sticker_invalid_output")
            shutil.rmtree(destination)


if __name__ == "__main__" and len(sys.argv) == 6 and sys.argv[1] == "--worker":
    _worker(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5])
