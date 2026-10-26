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


@dataclass(frozen=True)
class IncomingMessage:
    update_id: int
    sender_id: int
    chat_id: int
    text: str
    message_id: int = 0
    attachment: dict[str, Any] | None = None
    is_text: bool = True


EMERGENCY_PHRASE = "АВАРИЙНОЕ ВЫКЛЮЧЕНИЕ"
