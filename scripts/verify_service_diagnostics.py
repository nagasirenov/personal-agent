"""Offline Telegram, synthetic service-state probe with real gpt-6-luna / medium."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personal_agent.backends import CodexSubscriptionBackend
from personal_agent.config import Config
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage
from personal_agent.supervisor import TaskSupervisor
from personal_agent.testing import traced_handler


MODEL, EFFORT = "gpt-6-luna", "medium"


class OfflineTelegram:
    def __init__(self):
        self.sent = []

    async def send_text(self, chat_id, text):
        self.sent.append((chat_id, text))


class NoTranscriber:
    async def close(self):
        pass


class RealLuna(CodexSubscriptionBackend):
    def __init__(self, **kwargs):
        if kwargs.get("model") != MODEL or kwargs.get("reasoning_effort") != EFFORT:
            raise ValueError("Only gpt-6-luna medium is permitted in this test")
        super().__init__(**kwargs)
        self.tool_calls = []
        self.prompts = []

    async def generate(self, prompt, request_id=None, *, images=None, tools=None, tool_handler=None):
        self.prompts.append(prompt)
        return await super().generate(prompt, request_id=request_id, images=images, tools=tools,
                                      tool_handler=traced_handler(tool_handler, self.tool_calls))

    async def health(self):
        # The probe tests model tool choice, not real daemon or Telegram status.
        return {"ok": False, "error_code": "synthetic_bridge_unavailable"}


async def main():
    if "--real" not in sys.argv:
        raise SystemExit("Run with --real for synthetic gpt-6-luna / medium test")
    with tempfile.TemporaryDirectory(prefix="personal-agent-diagnostics-") as temporary:
        root = Path(temporary)
        config = Config(data_dir=root, primary_id=101, secondary_id=202,
                        backend="codex_subscription", model=MODEL, reasoning=EFFORT,
                        brain_timeout=150, timezone_name="UTC")
        store = MemoryStore(root)
        telegram = OfflineTelegram()
        brain = RealLuna(model=MODEL, reasoning_effort=EFFORT,
                         codex_command=config.codex_executable, work_root=root)
        supervisor = TaskSupervisor(config, store, brain, telegram, asyncio.Event(), transcriber=NoTranscriber())
        try:
            store.set_state("observer_health", {"available": False, "last_sample_time":
                            datetime.now(timezone.utc).isoformat(), "last_error_type": "SyntheticObserverDown"})
            store.set_state("telegram_health", {"ok": False})
            await supervisor.submit(IncomingMessage(
                update_id=1, sender_id=101, chat_id=101, message_id=1,
                text="В последнее время ответы задерживаются. Ты сейчас нормально работаешь? Что реально известно про связь?",
            ), Identity.PRIMARY)
            await asyncio.gather(*list(supervisor.tasks), return_exceptions=True)
            answer = telegram.sent[-1][1] if telegram.sent else ""
            checked = store.get_state("last_diagnostics")
            names = [call["name"] for call in brain.tool_calls]
            success = (names == ["service_diagnostics"] and len(brain.prompts) == 1 and isinstance(checked, dict)
                       and checked["brain"].get("error_code") == "synthetic_bridge_unavailable"
                       and checked["telegram"] == {"ok": False}
                       and not answer.startswith("Не получилось завершить ответ"))
            print(f"model={MODEL}; effort={EFFORT}; tools={names}; fake_telegram=1")
            print("model_selected_service_diagnostics=" + ("PASS" if success else "FAIL"))
            if not success:
                print("answer_excerpt=" + repr(answer[:320]))
                raise SystemExit(1)
        finally:
            await supervisor.shutdown()
            store.close()


if __name__ == "__main__":
    asyncio.run(main())
