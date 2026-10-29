"""Group chats: the owner must be a current member, otherwise the bot leaves.

The bot serves a group or supergroup only while the owner (config.primary_id) is a member of it.  The check
is made when the bot is added, before group messages are handled (with a short cache), and again whenever
Telegram reports that the owner left.  An owner who is absent, or a check that fails, means "not admitted":
the bot stays silent, leaves the chat and tells the owner in private.  Channels are never served.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .models import GROUP_CHAT_TYPES, TimelineEvent

GROUPS_STATE = "telegram_groups"
ADMISSION_TTL_SECONDS = 300.0
_OWNER_STATUSES = {"creator", "administrator", "member"}
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
NOT_A_MEMBER_NOTICE = "Меня добавили в чат «{title}», но тебя там нет — я вышел."
CHECK_FAILED_NOTICE = "Не смог проверить, что ты состоишь в чате «{title}», поэтому я вышел из него."


def clean_name(value: object, limit: int = 64) -> str:
    """A display name or title from Telegram: single line, bounded, no control characters."""
    return _CONTROL.sub(" ", value).strip()[:limit] if isinstance(value, str) else ""


def is_group_id(value: object) -> bool:
    return type(value) is int and value < 0


def owner_is_member(member: object) -> bool:
    """A getChatMember result counts only when the owner is in the chat right now."""
    if not isinstance(member, dict):
        return False
    status = member.get("status")
    if status in _OWNER_STATUSES:
        return True
    return status == "restricted" and member.get("is_member") is True


def _registry(store) -> dict:
    value = store.get_state(GROUPS_STATE, {})
    return value if isinstance(value, dict) else {}


def admitted_group(store, chat_id: object) -> dict | None:
    """The stored record of a group the bot currently serves, else None."""
    record = _registry(store).get(str(chat_id))
    return record if isinstance(record, dict) and record.get("admitted") is True else None


def group_title(store, chat_id: object) -> str:
    record = _registry(store).get(str(chat_id))
    return clean_name(record.get("title"), 100) if isinstance(record, dict) else ""


@dataclass(frozen=True)
class BotIdentity:
    id: int
    username: str
    first_name: str


class GroupAdmission:
    """Owner-membership gate and bookkeeping for group chats (Bot API calls go through `client`)."""

    def __init__(self, client, store, primary_id: int, *, ttl: float = ADMISSION_TTL_SECONDS, clock=time.monotonic):
        self.client, self.store, self.primary_id = client, store, primary_id
        self.ttl, self._clock = ttl, clock
        self.bot: BotIdentity | None = None
        self._cache: dict[int, tuple[bool, float]] = {}

    # -- bot identity ---------------------------------------------------------------------------------
    async def ensure_identity(self) -> BotIdentity | None:
        if self.bot is None:
            try:
                me = await self.client.get_me()
                if isinstance(me, dict) and type(me.get("id")) is int:
                    self.bot = BotIdentity(me["id"], str(me.get("username") or ""), str(me.get("first_name") or ""))
            except Exception:
                self.bot = None
        return self.bot

    # -- registry -------------------------------------------------------------------------------------
    def _set(self, chat_id: int, title: str, admitted: bool) -> None:
        registry = dict(_registry(self.store))
        key = str(chat_id)
        previous = registry.get(key) if isinstance(registry.get(key), dict) else {}
        registry[key] = {**previous, "title": clean_name(title, 100) or previous.get("title", ""), "admitted": admitted}
        self.store.set_state(GROUPS_STATE, registry)

    def forget(self, chat_id: int) -> None:
        self._cache.pop(chat_id, None)
        registry = dict(_registry(self.store))
        if registry.pop(str(chat_id), None) is not None:
            self.store.set_state(GROUPS_STATE, registry)

    # -- the check ------------------------------------------------------------------------------------
    async def _owner_present(self, chat_id: int) -> bool | None:
        """True/False from Telegram; None when the check itself failed."""
        try:
            return owner_is_member(await self.client.get_chat_member(chat_id, self.primary_id))
        except Exception:
            return None

    async def check(self, chat_id: int, title: str = "", *, force: bool = False) -> bool:
        """Admit the chat when the owner is a member; otherwise leave it and tell the owner (fail closed)."""
        cached = self._cache.get(chat_id)
        if not force and cached is not None and self._clock() < cached[1]:
            return cached[0]
        present = await self._owner_present(chat_id)
        if present is True:
            newly = admitted_group(self.store, chat_id) is None
            self._set(chat_id, title, True)
            self._cache[chat_id] = (True, self._clock() + self.ttl)
            if newly:
                self.store.record_event(TimelineEvent("telegram", "GROUP_ADMITTED", {
                    "chat_id": chat_id, "title": clean_name(title, 100)}))
            return True
        await self.reject(chat_id, title, "owner_not_member" if present is False else "membership_check_failed")
        return False

    async def reject(self, chat_id: int, title: str, reason: str) -> None:
        """Stop serving the chat: leave it, record the event and tell the owner (once per rejection)."""
        title = clean_name(title, 100) or group_title(self.store, chat_id)
        was_admitted = admitted_group(self.store, chat_id) is not None
        self._cache[chat_id] = (False, self._clock() + self.ttl)
        self._set(chat_id, title, False)
        left = True
        try:
            await self.client.leave_chat(chat_id)
        except Exception:
            left = False
        self.store.record_event(TimelineEvent("telegram", "GROUP_REJECTED", {
            "chat_id": chat_id, "title": title, "reason": reason, "left": left, "was_admitted": was_admitted}))
        template = CHECK_FAILED_NOTICE if reason == "membership_check_failed" else NOT_A_MEMBER_NOTICE
        notice = template.format(title=title or str(chat_id))
        try:
            await self.client.send_text(self.primary_id, notice)
            self.store.record_message(self.primary_id, "assistant", notice)
        except Exception:
            pass

    # -- updates --------------------------------------------------------------------------------------
    async def on_my_chat_member(self, update: dict) -> None:
        """The bot's own status in a chat changed: added (check at once) or removed (forget)."""
        change = update.get("my_chat_member")
        chat = change.get("chat") if isinstance(change, dict) else None
        if not isinstance(chat, dict) or type(chat.get("id")) is not int:
            return
        status = (change.get("new_chat_member") or {}).get("status")
        if chat.get("type") not in GROUP_CHAT_TYPES:
            return  # channels and private chats are never served here
        if status in {"member", "administrator"}:
            await self.check(chat["id"], chat.get("title") or "", force=True)
        elif status in {"left", "kicked"}:
            self.forget(chat["id"])
        elif status == "restricted":
            await self.check(chat["id"], chat.get("title") or "", force=True)

    async def on_chat_member(self, update: dict) -> None:
        """Somebody's membership changed; only the owner leaving matters."""
        change = update.get("chat_member")
        chat = change.get("chat") if isinstance(change, dict) else None
        if not isinstance(chat, dict) or chat.get("type") not in GROUP_CHAT_TYPES or type(chat.get("id")) is not int:
            return
        member = change.get("new_chat_member") or {}
        user = member.get("user") if isinstance(member, dict) else None
        if not isinstance(user, dict) or user.get("id") != self.primary_id:
            return
        if owner_is_member(member) or admitted_group(self.store, chat["id"]) is None:
            return  # joined/promoted (the next message re-checks anyway), or the chat is not being served
        await self.reject(chat["id"], chat.get("title") or "", "owner_left")

    async def on_owner_left_message(self, chat_id: int, title: str) -> None:
        if admitted_group(self.store, chat_id) is not None:
            await self.reject(chat_id, title, "owner_left")
