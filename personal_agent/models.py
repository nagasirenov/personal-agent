"""Shared data types. Observations never carry authorization."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class TimelineEvent:
    source: str
    type: str
    data: dict[str, Any]
    project: str | None = None
    time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid4()))


class Identity(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    GROUP = "group"  # an admitted group chat: SECONDARY rights for everyone in it, the owner included


@dataclass(frozen=True)
class IncomingMessage:
    update_id: int
    sender_id: int
    chat_id: int
    text: str
    message_id: int = 0
    attachment: dict[str, Any] | None = None
    is_text: bool = True
    # Group chats.  In a group `sender_id` is the author; the supervisor re-scopes the message to the chat
    # (sender_id=chat_id, actor_id=author) so that all per-"user" storage becomes per-chat storage.
    chat_type: str = "private"
    chat_title: str = ""
    sender_name: str = ""
    addressed: bool = True
    actor_id: int | None = None
    # What the message answers / who wrote the forwarded original (plain JSON-compatible data, untrusted).
    reply_to: dict[str, Any] | None = None
    forwarded_from: dict[str, Any] | None = None

    @property
    def is_group(self) -> bool:
        return self.chat_type in GROUP_CHAT_TYPES


GROUP_CHAT_TYPES = ("group", "supergroup")


EMERGENCY_PHRASE = "АВАРИЙНОЕ ВЫКЛЮЧЕНИЕ"
