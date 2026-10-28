"""Small cancellable Bot API transport and an identity-first polling gateway."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import httpx

from .models import EMERGENCY_PHRASE, Identity, IncomingMessage
from .security import TrustGate


class TelegramError(RuntimeError):
    pass


DEFAULT_API_BASE = "https://api.telegram.org"
CLOUD_UPLOAD_CAP = 50 * 1024 * 1024        # the hosted Bot API refuses bot uploads above 50 MB
LOCAL_UPLOAD_CAP = 2000 * 1024 * 1024      # a self-hosted Bot API server accepts up to 2000 MB
_UPLOAD_TIMEOUT = httpx.Timeout(connect=30.0, read=900.0, write=900.0, pool=30.0)


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
        result = await self.call("getUpdates", {"offset": offset, "timeout": timeout, "allowed_updates": ["message"]})
        if not isinstance(result, list):
            raise TelegramError("Invalid update response")
        return result

    async def send_text(self, chat_id: int, text: str) -> None:
        for start in range(0, len(text), 1800):
            await self.call("sendMessage", {
                "chat_id": chat_id, "text": text[start:start + 1800],
                "link_preview_options": {"is_disabled": True},
            })

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Show "typing..." in the chat for about 5 seconds (or until a message arrives)."""
        await self.call("sendChatAction", {"chat_id": chat_id, "action": action})

    async def send_photo(self, chat_id: int, path: Path, caption: str = "") -> str:
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
                    data={"chat_id": str(chat_id), "caption": caption[:1000]},
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


def parse_message(update: dict) -> IncomingMessage | None:
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    sender, chat = message.get("from", {}), message.get("chat", {})
    if chat.get("type") != "private" or sender.get("is_bot"):
        return None
    user_id, chat_id = sender.get("id"), chat.get("id")
    if type(user_id) is not int or type(chat_id) is not int or user_id != chat_id:
        return None
    attachment = None
    for kind in ("document", "voice", "audio", "video", "video_note", "animation", "sticker", "photo"):
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
            break
    text = message.get("text", message.get("caption", ""))
    return IncomingMessage(
        update_id=update["update_id"], sender_id=user_id, chat_id=chat_id,
        text=text if isinstance(text, str) else "", message_id=message.get("message_id", 0), attachment=attachment,
        is_text=isinstance(message.get("text"), str),
    )


class TelegramGateway:
    def __init__(self, client, trust: TrustGate, store, on_message: Callable[[IncomingMessage, Identity], Awaitable[None]], on_emergency: Callable[[], Awaitable[None]], stop: asyncio.Event):
        self.client, self.trust, self.store = client, trust, store
        self.on_message, self.on_emergency, self.stop = on_message, on_emergency, stop

    def emergency(self, message: IncomingMessage | None) -> bool:
        return bool(message and message.is_text and self.trust.identify(message.sender_id) == Identity.PRIMARY
                    and message.text.strip() == EMERGENCY_PHRASE)

    async def dispatch_batch(self, updates: list[dict]) -> None:
        # Scan the batch before any ordinary message can start a Brain request.
        parsed = [(update, parse_message(update)) for update in updates]
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
            identity = self.trust.identify(message.sender_id) if message else None
            if identity is not None:
                await self.on_message(message, identity)
            self.store.set_state("telegram_offset", offset + 1)

    async def run(self) -> None:
        backoff = 1
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
