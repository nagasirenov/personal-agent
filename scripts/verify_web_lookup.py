"""Opt-in Luna/medium web lookup check with synthetic sources and offline Telegram."""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personal_agent.backends import CodexSubscriptionBackend, resolve_codex_command
from personal_agent.config import Config
from personal_agent.memory import MemoryStore
from personal_agent.models import Identity, IncomingMessage
from personal_agent.supervisor import TaskSupervisor
from personal_agent.testing import block_host_desktop, traced_handler
from personal_agent.web_lookup import WebLookupSession


MODEL = "gpt-6-luna"
EFFORT = "medium"
SOURCE = "https://mission-asterion.org/launch"
MARKER = "ASTERION_731"


class OfflineTelegram:
    def __init__(self):
        self.sent = []

    async def send_text(self, chat_id, text):
        self.sent.append((chat_id, text))


class NoTranscriber:
    async def close(self):
        pass


class CheckedLuna(CodexSubscriptionBackend):
    def __init__(self, **kwargs):
        if kwargs.get("model") != MODEL or kwargs.get("reasoning_effort") != EFFORT:
            raise ValueError("Synthetic real-model check requires only gpt-6-luna / medium")
        super().__init__(**kwargs)
        self.tool_calls = []
        self.evidence_trace = []

    @property
    def plans(self):
        return [call["name"] for call in self.tool_calls]

    async def generate(self, prompt, request_id=None, *, images=None, tools=None, tool_handler=None):
        start = len(self.tool_calls)
        answer = await super().generate(prompt, request_id=request_id, images=images, tools=tools,
                                        tool_handler=traced_handler(tool_handler, self.tool_calls))
        new = self.tool_calls[start:]
        self.evidence_trace.append({
            "length": len(prompt), "tools": [call["name"] for call in new],
            "search_result": any(call["name"] == "web_search" and SOURCE in call["output"] for call in new),
            "open_result": any(call["name"] == "web_open" and MARKER in call["output"] for call in new),
            "source": SOURCE in answer, "marker": MARKER in answer})
        return answer


async def main():
    if "--real" not in sys.argv:
        raise SystemExit("Use --real for synthetic gpt-6-luna / medium verification")
    block_host_desktop()

    def handler(request: httpx.Request):
        if request.url.path == "/search":
            rss = ("<rss version='2.0'><channel><item><title>Asterion observatory official launch timeline</title>"
                   f"<link>{SOURCE}</link><description>Asterion observatory launch schedule and project milestones</description>"
                   "</item></channel></rss>")
            return httpx.Response(200, content=rss.encode())
        if str(request.url) == SOURCE:
            html = ("<html><body><h1>Asterion observatory mission timeline</h1>"
                    "<p>Scheduled launch: April 17, 2032.</p>"
                    f"<p>Fixture verification code: {MARKER}.</p></body></html>")
            return httpx.Response(200, headers={"content-type": "text/html"}, content=html.encode())
        raise AssertionError("Unexpected network target in synthetic web test")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        with tempfile.TemporaryDirectory(prefix="personal-agent-web-lookup-") as directory:
            root = Path(directory)
            config = Config(data_dir=root, primary_id=101, secondary_id=202, backend="codex_subscription",
                            model=MODEL, reasoning=EFFORT, brain_timeout=150, timezone_name="UTC")
            brain = CheckedLuna(model=config.model, reasoning_effort=config.reasoning,
                                codex_command=resolve_codex_command(config.codex_executable), work_root=root)
            telegram = OfflineTelegram()
            store = MemoryStore(root)
            supervisor = TaskSupervisor(config, store, brain, telegram, asyncio.Event(), transcriber=NoTranscriber())
            # The model receives only synthetic state and public tool contracts.
            supervisor.context.knowledge = None
            try:
                with (patch("personal_agent.tools.WebLookupSession", side_effect=lambda: WebLookupSession(client)),
                      patch("personal_agent.web_lookup._public_url", new=AsyncMock(return_value=True))):
                    await supervisor.submit(IncomingMessage(
                        update_id=1, sender_id=101, chat_id=101, message_id=1,
                        text="Когда запланирован запуск обсерватории Asterion? Проверь дату через интернет и дай ссылку на источник.",
                    ), Identity.PRIMARY)
                    await asyncio.gather(*list(supervisor.tasks))
                if not telegram.sent:
                    raise RuntimeError("No offline Telegram answer")
                answer = telegram.sent[-1][1]
                success = ("web_search" in brain.plans and "web_open" in brain.plans
                           and "17" in answer and "2032" in answer and SOURCE in answer
                           and store.get_state("last_completed_turn") is not None)
                print(f"model={MODEL}; reasoning={EFFORT}; plans={brain.plans}; offline_telegram=1")
                print("model_inputs=" + str(brain.evidence_trace))
                print("model_selected_search_open_answer=" + ("PASS" if success else "FAIL"))
                if not success:
                    print("answer=" + answer[:450])
                    raise SystemExit(1)
            finally:
                await supervisor.shutdown()
                store.close()


if __name__ == "__main__":
    asyncio.run(main())
