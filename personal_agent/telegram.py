"""Small cancellable Bot API transport and an identity-first polling gateway."""
from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import httpx

from .groups import BotIdentity, GroupAdmission, admitted_group, clean_name
from .models import EMERGENCY_PHRASE, GROUP_CHAT_TYPES, Identity, IncomingMessage
from .security import TrustGate


class TelegramError(RuntimeError):
    pass


DEFAULT_API_BASE = "https://api.telegram.org"
CLOUD_UPLOAD_CAP = 50 * 1024 * 1024        # the hosted Bot API refuses bot uploads above 50 MB
LOCAL_UPLOAD_CAP = 2000 * 1024 * 1024      # a self-hosted Bot API server accepts up to 2000 MB
_UPLOAD_TIMEOUT = httpx.Timeout(connect=30.0, read=900.0, write=900.0, pool=30.0)
ALLOWED_UPDATES = ("message", "my_chat_member", "chat_member")


class TelegramClient:
    def __init__(self, token: str, *, client: httpx.AsyncClient | None = None, api_base: str = DEFAULT_API_BASE):
        if not token or ":" not in token or any(c.isspace() for c in token):
            raise ValueError("Invalid Telegram token format")
        api_base = (api_base or DEFAULT_API_BASE).rstrip("/")
        self.api_base = api_base
        self._base = api_base + "/bot" + token + "/"
        self._file_base = api_base + "/file/bot" + token + "/"
        # Ignore proxy env vars to avoid forwarding bot credentials accidentally.
        self._client = client or httpx.AsyncClient(timeout=35, follow_redirects=False, trust_env=False)

    @property
    def upload_cap_bytes(self) -> int:
        """The single place that knows the bot upload limit of the Bot API in use."""
        return CLOUD_UPLOAD_CAP if self.api_base == DEFAULT_API_BASE else LOCAL_UPLOAD_CAP

    async def call(self, method: str, payload: dict[str, Any]) -> Any:
        try:
            response = await self._client.post(self._base + method, json=payload)
            response.raise_for_status()
            body = response.json()
            if not body.get("ok"):
                raise TelegramError(f"Telegram rejected {method} (code {body.get('error_code', 'unknown')})")
            return body["result"]
        except (httpx.HTTPError, ValueError, KeyError):
            # httpx exception messages include the bot token in their request URL.
            raise TelegramError(f"Telegram {method} failed; check connection and credentials") from None

    async def get_updates(self, offset: int, timeout: int = 25) -> list[dict]:
        result = await self.call("getUpdates", {"offset": offset, "timeout": timeout,
                                                "allowed_updates": list(ALLOWED_UPDATES)})
        if not isinstance(result, list):
            raise TelegramError("Invalid update response")
        return result

    async def get_me(self) -> dict:
        result = await self.call("getMe", {})
        if not isinstance(result, dict):
            raise TelegramError("Invalid getMe response")
        return result

    async def get_chat_member(self, chat_id: int, user_id: int) -> dict:
        result = await self.call("getChatMember", {"chat_id": chat_id, "user_id": user_id})
        if not isinstance(result, dict):
            raise TelegramError("Invalid getChatMember response")
        return result

    async def leave_chat(self, chat_id: int) -> None:
        await self.call("leaveChat", {"chat_id": chat_id})

    async def send_text(self, chat_id: int, text: str, *, reply_to: int | None = None) -> None:
        for index, start in enumerate(range(0, len(text), 1800)):
            payload = {"chat_id": chat_id, "text": text[start:start + 1800],
                       "link_preview_options": {"is_disabled": True}}
            if reply_to and index == 0:
                payload.update(reply_to_message_id=reply_to, allow_sending_without_reply=True)
            await self.call("sendMessage", payload)

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Show "typing..." in the chat for about 5 seconds (or until a message arrives)."""
        await self.call("sendChatAction", {"chat_id": chat_id, "action": action})

    async def send_photo(self, chat_id: int, path: Path, caption: str = "", *, reply_to: int | None = None) -> str:
        """Upload an image; returns the Bot API method actually used (sendPhoto or sendDocument)."""
        import mimetypes
        import struct
        size = path.stat().st_size
        if size > 50 * 1024 * 1024:
            raise TelegramError("Screenshot exceeds Telegram upload limit")
        with path.open("rb") as image:
            header = image.read(24)
        width, height = struct.unpack(">II", header[16:24]) if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) == 24 else (1, 1)
        as_document = size > 10 * 1024 * 1024 or width + height > 10000 or max(width, height) > 20 * max(1, min(width, height))
        method, field = ("sendDocument", "document") if as_document else ("sendPhoto", "photo")
        try:
            with path.open("rb") as image:
                response = await self._client.post(self._base + method,
                    data={"chat_id": str(chat_id), "caption": caption[:1000],
                          **({"reply_to_message_id": str(reply_to), "allow_sending_without_reply": "true"}
                             if reply_to else {})},
                    files={field: (path.name, image, mimetypes.guess_type(path.name)[0] or "image/png")})
            response.raise_for_status()
            result = response.json()
            if not result.get("ok"):
                raise TelegramError(f"Telegram rejected {method} (code {result.get('error_code', 'unknown')})")
        except (httpx.HTTPError, ValueError):
            raise TelegramError("Telegram screenshot upload failed") from None
        return method

    async def _upload(self, method: str, field: str, chat_id: int, path: Path, caption: str, extra: dict) -> str:
        """Multipart upload streamed from disk (the file is never read into memory)."""
        import mimetypes
        data = {"chat_id": str(chat_id), **extra}
        if caption:
            data["caption"] = caption[:1000]
        try:
            with path.open("rb") as handle:
                response = await self._client.post(
                    self._base + method, data=data, timeout=_UPLOAD_TIMEOUT,
                    files={field: (path.name, handle, mimetypes.guess_type(path.name)[0] or "application/octet-stream")})
            response.raise_for_status()
            body = response.json()
            if not body.get("ok"):
                raise TelegramError(f"Telegram rejected {method} (code {body.get('error_code', 'unknown')})")
        except (httpx.HTTPError, ValueError, OSError):
            # httpx exception messages include the bot token in their request URL.
            raise TelegramError(f"Telegram {method} upload failed") from None
        return method

    async def send_document(self, chat_id: int, path: Path, caption: str = "") -> str:
        return await self._upload("sendDocument", "document", chat_id, path, caption, {})

    async def send_video(self, chat_id: int, path: Path, caption: str = "") -> str:
        return await self._upload("sendVideo", "video", chat_id, path, caption, {"supports_streaming": "true"})

    async def send_audio(self, chat_id: int, path: Path, caption: str = "") -> str:
        return await self._upload("sendAudio", "audio", chat_id, path, caption, {})

    async def download(self, file_id: str, destination: Path, max_bytes: int) -> int:
        info = await self.call("getFile", {"file_id": file_id})
        remote = info.get("file_path", "")
        path = PurePosixPath(remote)
        if (not remote or path.is_absolute() or ".." in path.parts or "\\" in remote
                or ":" in remote or "?" in remote or "#" in remote):
            raise TelegramError("Unsafe attachment path from Telegram")
        if info.get("file_size", 0) > max_bytes:
            raise TelegramError("Attachment exceeds the local size limit")
        partial = destination.with_suffix(destination.suffix + ".partial")
        total = 0
        try:
            async with self._client.stream("GET", self._file_base + quote(remote, safe="/")) as response:
                response.raise_for_status()
                with partial.open("xb") as handle:
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise TelegramError("Attachment exceeds the local size limit")
                        handle.write(chunk)
            partial.replace(destination)
            return total
        except httpx.HTTPError:
            raise TelegramError("Telegram attachment download failed") from None
        finally:
            partial.unlink(missing_ok=True)

    async def close(self) -> None:
        await self._client.aclose()


def _addressed_to_bot(message: dict, text: str, bot: BotIdentity | None) -> bool:
    """A group message is for the bot when it @mentions it, replies to it, or starts with its name/command."""
    if bot is None:
        return False
    reply = message.get("reply_to_message")
    if isinstance(reply, dict) and isinstance(reply.get("from"), dict) and reply["from"].get("id") == bot.id:
        return True
    username = bot.username.casefold()
    entities = message.get("entities", message.get("caption_entities", []))
    for entity in entities if isinstance(entities, list) else []:
        if not isinstance(entity, dict):
            continue
        kind, offset, length = entity.get("type"), entity.get("offset"), entity.get("length")
        if type(offset) is not int or type(length) is not int:
            continue
        piece = _utf16_slice(text, offset, length).casefold()
        if kind == "mention" and username and piece == "@" + username:
            return True
        if kind == "bot_command" and username and piece.endswith("@" + username):
            return True
        if kind == "text_mention" and isinstance(entity.get("user"), dict) and entity["user"].get("id") == bot.id:
            return True
    lowered = text.lstrip().casefold()
    for name in {bot.first_name.casefold(), username}:
        if name and lowered.startswith(name) and (len(lowered) == len(name) or not lowered[len(name)].isalnum()):
            return True
    return bool(username) and re.search(r"(?<!\w)@" + re.escape(username) + r"(?!\w)", lowered) is not None


def _utf16_slice(text: str, offset: int, length: int) -> str:
    """Telegram entity offsets count UTF-16 code units."""
    units = text.encode("utf-16-le")
    return units[offset * 2:(offset + length) * 2].decode("utf-16-le", errors="ignore")


def _display_name(sender: dict) -> str:
    name = " ".join(part for part in (sender.get("first_name"), sender.get("last_name")) if isinstance(part, str))
    return clean_name(name) or clean_name(sender.get("username")) or "участник"


_MEDIA_KINDS = ("document", "voice", "audio", "video", "video_note", "animation", "sticker", "photo")
REPLY_TEXT_LIMIT = 2000
REPLY_QUOTE_LIMIT = 1000
_TEXT_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _extract_media(message: dict) -> dict | None:
    """The first downloadable attachment of a Telegram message (photo: the largest size), or None."""
    for kind in _MEDIA_KINDS:
        media = message.get(kind)
        if kind == "photo" and isinstance(media, list) and media:
            media = media[-1]
        if isinstance(media, dict) and media.get("file_id"):
            attachment = dict(media, kind=kind)
            if kind == "sticker":
                is_animated = bool(media.get("is_animated"))
                is_video = bool(media.get("is_video"))
                attachment.update({
                    "is_animated": is_animated,
                    "is_video": is_video,
                    "sticker_format": "webm" if is_video else "tgs" if is_animated else "webp",
                })
            return attachment
    return None


def _clean_text(value: object, limit: int) -> str:
    """Untrusted message text: bounded, without control characters (line breaks are kept)."""
    return _TEXT_CONTROL.sub(" ", value).strip()[:limit] if isinstance(value, str) else ""


def _iso_utc(value: object) -> str | None:
    if type(value) is not int or value <= 0:
        return None
    try:
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _origin_info(origin: dict) -> dict | None:
    """Who originally wrote a forwarded/externally-replied message, from a Bot API 7+ MessageOrigin."""
    kind = origin.get("type")
    info: dict | None = None
    if kind == "user" and isinstance(origin.get("sender_user"), dict):
        user = origin["sender_user"]
        info = {"type": "user", "name": _display_name(user)}
        username = clean_name(user.get("username"), 64)
        if username:
            info["username"] = username
    elif kind == "hidden_user":
        info = {"type": "hidden_user", "name": clean_name(origin.get("sender_user_name"), 64) or "скрытый пользователь"}
    elif kind in ("chat", "channel"):
        chat = origin.get("sender_chat" if kind == "chat" else "chat")
        chat = chat if isinstance(chat, dict) else {}
        info = {"type": kind, "name": _chat_label(clean_name(chat.get("title"), 100), origin.get("author_signature"))}
        username = clean_name(chat.get("username"), 64)
        if username:
            info["username"] = username
    if info is None:
        return None
    date = _iso_utc(origin.get("date"))
    if date:
        info["date"] = date
    return info


def _chat_label(title: str, signature: object) -> str:
    signature = clean_name(signature, 64)
    return f"{title or 'чат'} ({signature})" if signature else (title or "чат")


def _forward_info(message: dict) -> dict | None:
    """The original author of a forwarded message: forward_origin, else the pre-Bot-API-7 fields."""
    origin = message.get("forward_origin")
    if isinstance(origin, dict):
        return _origin_info(origin)
    info: dict | None = None
    if isinstance(message.get("forward_from"), dict):
        user = message["forward_from"]
        info = {"type": "user", "name": _display_name(user)}
        username = clean_name(user.get("username"), 64)
        if username:
            info["username"] = username
    elif isinstance(message.get("forward_from_chat"), dict):
        chat = message["forward_from_chat"]
        info = {"type": "channel" if chat.get("type") == "channel" else "chat",
                "name": _chat_label(clean_name(chat.get("title"), 100), message.get("forward_signature"))}
        username = clean_name(chat.get("username"), 64)
        if username:
            info["username"] = username
    elif isinstance(message.get("forward_sender_name"), str):
        info = {"type": "hidden_user", "name": clean_name(message["forward_sender_name"], 64) or "скрытый пользователь"}
    if info is None:
        return None
    date = _iso_utc(message.get("forward_date"))
    if date:
        info["date"] = date
    return info


def _reply_info(message: dict, sender_id: int, bot: BotIdentity | None) -> dict | None:
    """The message the user replied to (reply_to_message, else external_reply): author, text, media, quote."""
    reply = message.get("reply_to_message")
    external = not isinstance(reply, dict)
    if external:
        reply = message.get("external_reply")
    if not isinstance(reply, dict):
        return None
    info: dict = {}
    if type(reply.get("message_id")) is int:
        info["message_id"] = reply["message_id"]
    author: dict = {}
    if external:
        origin = _origin_info(reply["origin"]) if isinstance(reply.get("origin"), dict) else None
        if origin:
            author["name"] = origin["name"]
        raw = reply["origin"].get("sender_user") if isinstance(reply.get("origin"), dict) else None
        raw_id = raw.get("id") if isinstance(raw, dict) else None
    else:
        raw = reply.get("from")
        raw = raw if isinstance(raw, dict) else None
        raw_id = raw.get("id") if raw else None
        if raw:
            author["name"] = _display_name(raw)
    if bot is not None and raw_id is not None and raw_id == bot.id:
        author["is_bot_self"] = True
    elif raw_id is not None and raw_id == sender_id:
        author["is_self"] = True
    if author:
        info["author"] = author
    text = _clean_text(reply.get("text", reply.get("caption")), REPLY_TEXT_LIMIT)
    if text:
        info["text"] = text
    media = _extract_media(reply)
    if media:
        info["media"] = {key: media[key] for key in ("kind", "file_id", "file_size", "sticker_format", "is_animated",
                                                      "is_video", "emoji", "set_name") if media.get(key) is not None}
    quote = message.get("quote")
    quote = _clean_text(quote.get("text"), REPLY_QUOTE_LIMIT) if isinstance(quote, dict) else ""
    if quote:
        info["quote"] = quote
    forwarded = _forward_info(reply)
    if forwarded:
        info["forwarded_from"] = forwarded
    if not (text or media or quote or forwarded) and "forum_topic_created" in reply:
        return None  # a forum topic's service message is not something the user answered
    return info or None


def parse_message(update: dict, bot: BotIdentity | None = None) -> IncomingMessage | None:
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    sender, chat = message.get("from", {}), message.get("chat", {})
    if not isinstance(sender, dict) or not isinstance(chat, dict) or sender.get("is_bot"):
        return None
    chat_type = chat.get("type")
    user_id, chat_id = sender.get("id"), chat.get("id")
    if type(user_id) is not int or type(chat_id) is not int:
        return None
    if chat_type == "private":
        if user_id != chat_id:
            return None
    elif chat_type in GROUP_CHAT_TYPES:
        if chat_id >= 0:
            return None
    else:
        return None  # channels and anything else are never served
    attachment = _extract_media(message)
    text = message.get("text", message.get("caption", ""))
    text = text if isinstance(text, str) else ""
    group = chat_type in GROUP_CHAT_TYPES
    return IncomingMessage(
        update_id=update["update_id"], sender_id=user_id, chat_id=chat_id,
        text=text, message_id=message.get("message_id", 0), attachment=attachment,
        is_text=isinstance(message.get("text"), str),
        chat_type=chat_type, chat_title=clean_name(chat.get("title"), 100) if group else "",
        sender_name=_display_name(sender) if group else "",
        addressed=_addressed_to_bot(message, text, bot) if group else True,
        reply_to=_reply_info(message, user_id, bot), forwarded_from=_forward_info(message),
    )


class TelegramGateway:
    def __init__(self, client, trust: TrustGate, store, on_message: Callable[[IncomingMessage, Identity], Awaitable[None]], on_emergency: Callable[[], Awaitable[None]], stop: asyncio.Event, *, groups: GroupAdmission | None = None):
        self.client, self.trust, self.store = client, trust, store
        self.on_message, self.on_emergency, self.stop = on_message, on_emergency, stop
        self.groups = groups or GroupAdmission(client, store, trust.primary_id)

    def emergency(self, message: IncomingMessage | None) -> bool:
        # Only the owner's private chat: a group message (even the owner's own) never stops the agent.
        return bool(message and message.is_text and message.chat_type == "private"
                    and self.trust.identify(message.sender_id) == Identity.PRIMARY
                    and message.text.strip() == EMERGENCY_PHRASE)

    async def _handle_group_update(self, update: dict) -> bool:
        """Membership updates and group service messages; True when the update was fully handled here."""
        if "my_chat_member" in update:
            await self.groups.on_my_chat_member(update)
            return True
        if "chat_member" in update:
            await self.groups.on_chat_member(update)
            return True
        raw = update.get("message")
        chat = raw.get("chat") if isinstance(raw, dict) else None
        if not isinstance(chat, dict) or chat.get("type") not in GROUP_CHAT_TYPES or type(chat.get("id")) is not int:
            return False
        left = raw.get("left_chat_member")
        if isinstance(left, dict):
            if left.get("id") == self.trust.primary_id:
                await self.groups.on_owner_left_message(chat["id"], chat.get("title") or "")
            elif self.groups.bot is not None and left.get("id") == self.groups.bot.id:
                self.groups.forget(chat["id"])
            return True
        return False

    async def _deliver(self, message: IncomingMessage) -> None:
        if message.chat_type == "private":
            identity = self.trust.identify(message.sender_id)
            if identity is not None:
                await self.on_message(message, identity)
            return
        if not message.text and not message.attachment:
            return  # service messages (title changes, pins, joins) carry nothing to answer or remember
        # Group: nothing is stored or answered until the owner is confirmed as a member.
        if await self.groups.ensure_identity() is None:
            return  # without our own identity we cannot tell who is addressed; stay silent
        if await self.groups.check(message.chat_id, message.chat_title):
            await self.on_message(message, Identity.GROUP)

    async def dispatch_batch(self, updates: list[dict]) -> None:
        # Scan the batch before any ordinary message can start a Brain request.
        parsed = [(update, parse_message(update, self.groups.bot)) for update in updates]
        for update, message in parsed:
            if self.emergency(message):
                self.stop.set()
                self.store.set_state("telegram_offset", update["update_id"] + 1)
                await self.on_emergency()
                try:
                    await asyncio.wait_for(self.client.send_text(message.chat_id, "Агент остановлен."), timeout=0.5)
                except (Exception, asyncio.CancelledError):
                    pass
                return
        for update, message in parsed:
            offset = update.get("update_id")
            if type(offset) is not int:
                continue
            if self.stop.is_set():
                return
            if message is not None and message.is_group and self.groups.bot is None:
                # Learn the bot's name before judging whether the message is addressed to it.
                if await self.groups.ensure_identity() is not None:
                    message = parse_message(update, self.groups.bot)
            if not await self._handle_group_update(update) and message is not None:
                await self._deliver(message)
            self.store.set_state("telegram_offset", offset + 1)

    async def run(self) -> None:
        backoff = 1
        await self.groups.ensure_identity()  # our own @username, to recognise group messages addressed to us
        while not self.stop.is_set():
            offset = self.store.get_state("telegram_offset", 0)
            try:
                updates = await self.client.get_updates(offset)
                self.store.set_state("telegram_health", {"ok": True, "last_poll_at": datetime.now(timezone.utc).isoformat()})
                await self.dispatch_batch(updates)
                backoff = 1
            except TelegramError:
                self.store.set_state("telegram_health", {"ok": False, "last_error_at": datetime.now(timezone.utc).isoformat(), "detail": "connection failed; retrying"})
                try:
                    await asyncio.wait_for(self.stop.wait(), backoff)
                except TimeoutError:
                    pass
                backoff = min(backoff * 2, 30)
