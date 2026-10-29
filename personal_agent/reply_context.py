"""Short markers that keep a message's reply/forward context in the stored conversation history."""
from __future__ import annotations

import re

from .groups import clean_name
from .models import IncomingMessage

SNIPPET_CHARS = 200
_MEDIA_LABELS = {"photo": "фото", "sticker": "стикер", "document": "документ", "video": "видео", "voice": "голосовое",
                 "audio": "аудио", "animation": "гифку", "video_note": "видеокружок"}
_PREFIX = re.compile(r"\A(?:\[(?:Переслано от|В ответ на)[^\n]*\]\n)+")


def _snippet(text: object, limit: int = SNIPPET_CHARS) -> str:
    flat = " ".join(str(text or "").split())
    return flat[:limit].rstrip() + "…" if len(flat) > limit else flat


def memory_prefix(message: IncomingMessage) -> str:
    """Lines like "[Переслано от: Иван]" / "[В ответ на сообщение бота: «…»]" (each ends with a newline), or ""."""
    lines: list[str] = []
    forwarded = message.forwarded_from
    if isinstance(forwarded, dict) and forwarded.get("name"):
        lines.append(f"[Переслано от: {clean_name(forwarded['name'], 120)}]")
    reply = message.reply_to
    if isinstance(reply, dict):
        author = reply.get("author") if isinstance(reply.get("author"), dict) else {}
        text = _snippet(reply.get("text") or reply.get("quote"))
        media = reply.get("media") if isinstance(reply.get("media"), dict) else {}
        label = _MEDIA_LABELS.get(str(media.get("kind", "")), "файл") if media else ""
        name = clean_name(author.get("name"), 64)
        if text:
            who = (" бота" if author.get("is_bot_self") else "" if author.get("is_self") else f" {name}" if name else "")
            lines.append(f"[В ответ на {'своё ' if author.get('is_self') else ''}сообщение{who}: «{text}»]")
        elif label:
            whose = " бота" if author.get("is_bot_self") else f" от {name}" if name and not author.get("is_self") else ""
            lines.append(f"[В ответ на {label}{whose}]")
        else:
            lines.append("[В ответ на сообщение]")
    return "".join(line + "\n" for line in lines)


def strip_memory_prefix(text: str) -> str:
    """The user's own words without the markers added by memory_prefix."""
    return _PREFIX.sub("", text, count=1)
