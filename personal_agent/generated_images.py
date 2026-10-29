"""Pictures made by the built-in image generator, kept per account/chat so they can be sent or viewed again."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

KEEP_PER_SENDER = 20
_STEM = re.compile(r"[A-Za-z0-9_-]{1,80}-[0-9]{1,2}")


@dataclass(frozen=True)
class SavedImage:
    path: Path
    media_id: str
    revised_prompt: str
    size: int


def generated_root(data_dir: Path, sender_id: int) -> Path:
    """data_dir/memory/generated/<sender or chat id>: one folder per storage scope, never shared."""
    return Path(data_dir) / "memory" / "generated" / str(int(sender_id))


def media_id_for(stem: str) -> str:
    request, _, number = stem.rpartition("-")
    return f"gen-{request[:8]}-{number}"


def save_generated(data_dir: Path, sender_id: int, request_id: str, images) -> list[SavedImage]:
    """Write validated GeneratedImage objects; returns what reached the disk (failures are skipped)."""
    folder = generated_root(data_dir, sender_id)
    safe = re.sub(r"[^A-Za-z0-9_-]", "", request_id)[:60] or "request"
    saved: list[SavedImage] = []
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        return saved
    for number, image in enumerate(images, 1):
        stem = f"{safe}-{number}"
        path = folder / (stem + image.suffix)
        try:
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_bytes(image.data)
            os.replace(temporary, path)
            path.with_suffix(".json").write_text(json.dumps({
                "time": datetime.now(timezone.utc).isoformat(), "revised_prompt": image.revised_prompt[:500],
            }, ensure_ascii=False), encoding="utf-8")
        except OSError:
            continue
        saved.append(SavedImage(path, media_id_for(stem), image.revised_prompt, len(image.data)))
    _prune(folder)
    return saved


def _prune(folder: Path) -> None:
    try:
        pictures = sorted((p for p in folder.iterdir() if p.suffix in {".png", ".jpg"} and p.is_file()),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        for old in pictures[KEEP_PER_SENDER:]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)
    except OSError:
        pass


def list_generated(data_dir: Path, sender_id: int, limit: int = 6) -> list[dict]:
    """Newest first: media_id, time, path and a short prompt for this scope's pictures."""
    folder = generated_root(data_dir, sender_id)
    entries: list[dict] = []
    try:
        pictures = [p for p in folder.iterdir()
                    if p.suffix in {".png", ".jpg"} and _STEM.fullmatch(p.stem) and p.is_file()]
        pictures.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return entries
    root = folder.resolve()
    for path in pictures[:limit]:
        real = path.resolve()
        if not real.is_relative_to(root):
            continue
        try:
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            meta = meta if isinstance(meta, dict) else {}
        except (OSError, ValueError):
            meta = {}
        stamp = meta.get("time") if isinstance(meta.get("time"), str) else \
            datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        prompt = meta.get("revised_prompt") if isinstance(meta.get("revised_prompt"), str) else ""
        entries.append({"media_id": media_id_for(path.stem), "time": stamp, "path": real,
                        "prompt": prompt[:120]})
    return entries
