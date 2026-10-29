"""Opt-in real check of built-in image generation: gpt-6-luna / medium, temp data dir, offline Telegram.

Usage: python scripts/verify_image_generation_real.py --real
Checks: a drawing request sends exactly one valid PNG photo; a plain question generates no image.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from personal_agent.backends import CodexSubscriptionBackend, resolve_codex_command
from personal_agent.config import Config
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage
from personal_agent.supervisor import TaskSupervisor
from personal_agent.testing import block_host_desktop

PRIMARY, SECONDARY = 101, 202


class OfflineTelegram:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.photos: list[tuple[str, bytes]] = []

    async def send_text(self, chat_id, text, *, reply_to=None):
        self.texts.append(text)

    async def send_chat_action(self, chat_id, action="typing"):
        pass

    async def send_photo(self, chat_id, path, caption="", *, reply_to=None):
        self.photos.append((path.name, path.read_bytes()))
        return "sendPhoto"


async def ask(supervisor, text: str, update_id: int) -> None:
    await supervisor.submit(IncomingMessage(update_id, PRIMARY, PRIMARY, text, message_id=update_id), Identity.PRIMARY)
    await asyncio.wait_for(asyncio.gather(*list(supervisor.tasks)), 240)


async def main() -> int:
    restore = block_host_desktop()
    failures = 0
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        config = Config(root, PRIMARY, SECONDARY, model="gpt-6-luna", reasoning="medium", timezone_name="UTC",
                        brain_timeout=240.0)
        store = MemoryStore(root)
        brain = CodexSubscriptionBackend(model="gpt-6-luna", reasoning_effort="medium",
                                         codex_command=resolve_codex_command("codex"), work_root=root)
        telegram = OfflineTelegram()
        supervisor = TaskSupervisor(config, store, brain, telegram, asyncio.Event())
        try:
            await ask(supervisor, "нарисуй простую картинку: синий квадрат на жёлтом фоне", 1)
            valid = [name for name, data in telegram.photos if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 1000]
            ok = len(telegram.photos) == 1 and len(valid) == 1
            print(f"draw: photos={len(telegram.photos)} valid_png={len(valid)} text={telegram.texts[-1:]!r} -> {'PASS' if ok else 'FAIL'}")
            failures += not ok
            before = len(telegram.photos)
            await ask(supervisor, "сколько будет 12 умножить на 7?", 2)
            ok = len(telegram.photos) == before and bool(telegram.texts)
            print(f"plain: new_photos={len(telegram.photos) - before} text={telegram.texts[-1:]!r} -> {'PASS' if ok else 'FAIL'}")
            failures += not ok
        finally:
            await supervisor.shutdown()
            await brain.close()
            store.close()
            restore()
    return failures


if __name__ == "__main__":
    if "--real" not in sys.argv:
        raise SystemExit("Pass --real to call the real gpt-6-luna model.")
    raise SystemExit(1 if asyncio.run(main()) else 0)
