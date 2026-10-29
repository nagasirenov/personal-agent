"""Explicit local configuration; no auto-discovery of user credentials."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from datetime import timedelta, timezone, tzinfo


@dataclass(frozen=True)
class Config:
    data_dir: Path
    primary_id: int
    secondary_id: int
    token: str = field(default="", repr=False)
    backend: str = "codex_subscription"
    model: str = "gpt-6-sol"
    reasoning: str = "medium"
    web_enabled: bool = False
    web_base_url: str = "http://127.0.0.1:17841/v1"
    codex_executable: str = "codex"
    project_roots: dict[str, Path] = field(default_factory=dict)
    pc_access: str = "read"
    observe_interval: float = 5.0
    timezone_name: str = "Europe/Moscow"
    max_attachment_bytes: int = 20 * 1024 * 1024
    max_pending_tasks: int = 16
    brain_timeout: float = 180.0
    whisper_executable: str = ""
    whisper_model: str = ""
    ffmpeg_executable: str = ""
    # Bot API server: the hosted one caps bot uploads at 50 MB; a local server (telegram-bot-api) allows 2000 MB.
    telegram_api_base: str = "https://api.telegram.org"
    # Files too big for Telegram go to Google Drive through rclone: "gdrive:PersonalAgent" (empty = split instead).
    drive_remote: str = ""
    rclone_executable: str = ""
    # Let the model draw pictures with the built-in image generator when asked (all roles, including groups).
    image_generation: bool = True

    @property
    def tz(self) -> tzinfo:
        try:
            return ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError:
            # Windows often has no IANA tz database; Moscow is UTC+3 since 2014.
            if self.timezone_name == "Europe/Moscow":
                return timezone(timedelta(hours=3), "Europe/Moscow")
            if self.timezone_name == "UTC":
                return timezone.utc
            raise ValueError("Install tzdata for the configured timezone") from None

    def validate(self, *, live: bool = False) -> None:
        if (type(self.primary_id) is not int or type(self.secondary_id) is not int
                or self.primary_id <= 0 or self.secondary_id <= 0
                or self.primary_id == self.secondary_id):
            raise ValueError("PRIMARY and SECONDARY must be distinct positive integer Telegram IDs")
        if self.backend not in {"mock", "codex_subscription", "codex_chatgpt_web"}:
            raise ValueError("Unknown Brain backend")
        if self.backend == "codex_chatgpt_web" and not self.web_enabled:
            raise ValueError("Experimental web backend is disabled; set web_enabled explicitly")
        if self.reasoning not in {"low", "medium", "high"}:
            raise ValueError("reasoning must be low, medium or high")
        if self.pc_access not in {"read", "full"}:
            raise ValueError("pc_access must be read or full")
        if not 1 <= self.observe_interval <= 300:
            raise ValueError("observe_interval must be in [1, 300]")
        if not 1 <= self.max_pending_tasks <= 128 or not 1 <= self.brain_timeout <= 600:
            raise ValueError("Invalid task bounds")
        if not 1 <= self.max_attachment_bytes <= 20 * 1024 * 1024:
            raise ValueError("Attachment limit must be at most 20 MiB for the hosted Bot API")
        if not re.fullmatch(r"https?://[^\s/?#]+(?:/[^\s?#]*)?", self.telegram_api_base):
            raise ValueError("telegram_api_base must be an http(s) URL")
        if self.drive_remote and not re.fullmatch(r"[^\s:]+:[^\r\n]*", self.drive_remote):
            raise ValueError("drive_remote must look like remote:folder")
        root = self.data_dir.resolve()
        protected = {Path(root.anchor), Path.home().resolve(), Path.cwd().resolve()}
        for key in ("WINDIR", "PROGRAMFILES", "USERPROFILE"):
            if os.environ.get(key):
                protected.add(Path(os.environ[key]).resolve())
        if root in protected:
            raise ValueError("data_dir must be a dedicated agent-owned directory")
        self.tz
        if live and (not self.token or ":" not in self.token):
            raise ValueError("Set TELEGRAM_BOT_TOKEN locally before starting Telegram")


def load_config(path: Path, *, live: bool = False) -> Config:
    path = path.resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    allowed = set(Config.__dataclass_fields__) - {"token"}
    unknown = set(raw) - allowed - {"token_env"}
    if unknown:
        raise ValueError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
    def resolve(value: str) -> Path:
        candidate = Path(os.path.expandvars(value)).expanduser()
        return (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    raw["data_dir"] = resolve(raw.get("data_dir", ".state"))
    raw["project_roots"] = {name: resolve(value) for name, value in raw.get("project_roots", {}).items()}
    raw["token"] = os.environ.get(raw.pop("token_env", "TELEGRAM_BOT_TOKEN"), "")
    config = Config(**raw)
    config.validate(live=live)
    return config
