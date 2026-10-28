"""Opt-in real Luna/medium browser-tool routing with synthetic pages and offline Telegram."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personal_agent.backends import CodexSubscriptionBackend, resolve_codex_command
from personal_agent.config import Config
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage
from personal_agent.supervisor import TaskSupervisor
from personal_agent.testing import block_host_desktop, traced_handler


MODEL, REASONING = "gpt-6-luna", "medium"
MARKER_FIRST = "СИНТЕТИЧЕСКИЙ_ЯНТАРЬ_481"
MARKER_SECOND = "СИНТЕТИЧЕСКИЙ_НЕФРИТ_927"


class OfflineTelegram:
    def __init__(self):
        self.sent = []

    async def send_text(self, chat_id, text):
        self.sent.append((chat_id, text))


class NoTranscriber:
    async def close(self):
        pass


class TraceLuna(CodexSubscriptionBackend):
    def __init__(self, **kwargs):
        if kwargs.get("model") != MODEL or kwargs.get("reasoning_effort") != REASONING:
            raise ValueError("Only real gpt-6-luna medium is permitted")
        super().__init__(**kwargs)
        self.calls = []

    async def generate(self, prompt, request_id=None, *, images=None, tools=None, tool_handler=None):
        trace = {
            "first_marker": MARKER_FIRST in prompt,
            "second_marker": MARKER_SECOND in prompt,
            "snapshot_metadata": '"current_browser_snapshot"' in prompt,
        }
        self.calls.append(trace)
        sink = []
        result = await super().generate(prompt, request_id=request_id, images=images, tools=tools,
                                        tool_handler=traced_handler(tool_handler, sink))
        trace["tools"] = [call["name"] for call in sink]
        trace["tool_first_marker"] = any(MARKER_FIRST in call["output"] for call in sink)
        trace["tool_second_marker"] = any(MARKER_SECOND in call["output"] for call in sink)
        return result


async def main():
    if "--real" not in sys.argv:
        raise SystemExit("Use --real to allow genuine gpt-6-luna medium calls")
    block_host_desktop()

    with tempfile.TemporaryDirectory(prefix="agent-browser-semantic-") as directory:
        root = Path(directory)
        config = Config(data_dir=root, primary_id=101, secondary_id=202,
                        backend="codex_subscription", model=MODEL, reasoning=REASONING,
                        brain_timeout=150, timezone_name="UTC")
        brain = TraceLuna(model=MODEL, reasoning_effort=REASONING,
                          codex_command=resolve_codex_command(config.codex_executable), work_root=root)
        telegram = OfflineTelegram()
        store = MemoryStore(root)
        supervisor = TaskSupervisor(config, store, brain, telegram, asyncio.Event(),
                                    transcriber=NoTranscriber())
        supervisor.context.knowledge = None
        page = {
            "browser": "Edge", "captured_at": datetime.now(timezone.utc).isoformat(),
            "url": "https://example.invalid/synthetic", "title": "Выдуманная лабораторная заметка",
            "page_text": f"{MARKER_FIRST}: макет исследования. Главный вывод: цвет янтарь сохраняет тепло.",
            "selected_text": "", "dom_summary": {},
        }
        supervisor.browser_context_provider = lambda: [page]
        try:
            async def ask(text, number):
                start = len(brain.calls)
                await supervisor.submit(IncomingMessage(
                    update_id=number, sender_id=101, chat_id=101, message_id=number, text=text,
                ), Identity.PRIMARY)
                await asyncio.gather(*list(supervisor.tasks))
                if not telegram.sent:
                    raise AssertionError("Fake Telegram received no reply")
                return brain.calls[start:], telegram.sent[-1][1]

            ordinary_calls, ordinary_answer = await ask("Сколько будет семь плюс пять?", 1)
            ordinary_ok = (len(ordinary_calls) == 1
                           and ordinary_calls[0]["tools"] == []
                           and ordinary_calls[0]["snapshot_metadata"]
                           and not ordinary_calls[0]["first_marker"]
                           and not ordinary_calls[0]["second_marker"]
                           and ("12" in ordinary_answer or "двенад" in ordinary_answer.casefold()))

            natural_calls, natural_answer = await ask("Что самое важное в открытой статье?", 2)
            natural_ok = (len(natural_calls) == 1
                          and "read_browser_page" in natural_calls[0]["tools"]
                          and not natural_calls[0]["first_marker"]
                          and natural_calls[0]["tool_first_marker"]
                          and "янтар" in natural_answer.casefold())

            indirect_calls, indirect_answer = await ask(
                "Я пытаюсь разобраться в том, что читаю. Какой там главный вывод?", 3)
            indirect_ok = (len(indirect_calls) == 1
                           and "read_browser_page" in indirect_calls[0]["tools"]
                           and not indirect_calls[0]["first_marker"]
                           and indirect_calls[0]["tool_first_marker"]
                           and "янтар" in indirect_answer.casefold())

            page["page_text"] = f"{MARKER_SECOND}: новая заметка. Главный вывод: нефрит сохраняет прохладу."
            page["captured_at"] = datetime.now(timezone.utc).isoformat()
            followup_calls, followup_answer = await ask("А теперь?", 4)
            followup_ok = (len(followup_calls) == 1
                           and "read_browser_page" in followup_calls[0]["tools"]
                           and not followup_calls[0]["second_marker"]
                           and followup_calls[0]["tool_second_marker"]
                           and "нефрит" in followup_answer.casefold())

            for name, ok, calls in (
                ("ordinary_one_call_no_page_text", ordinary_ok, ordinary_calls),
                ("natural_page_tool_and_answer", natural_ok, natural_calls),
                ("indirect_page_tool_and_answer", indirect_ok, indirect_calls),
                ("followup_refresh_and_answer", followup_ok, followup_calls),
            ):
                print(f"{name}={'PASS' if ok else 'FAIL'} calls={len(calls)} trace={calls}", flush=True)
            print(f"model={MODEL}; reasoning={REASONING}; fake_telegram=1; synthetic_only=1", flush=True)
            if not all((ordinary_ok, natural_ok, indirect_ok, followup_ok)):
                print("synthetic_answers=" + repr([ordinary_answer[:160], natural_answer[:160],
                                                    indirect_answer[:160], followup_answer[:160]]))
                raise SystemExit(1)
        finally:
            await supervisor.shutdown()
            store.close()


if __name__ == "__main__":
    asyncio.run(main())
