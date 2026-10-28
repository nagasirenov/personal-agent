"""Synthetic end-to-end preference check using only gpt-6-luna at medium effort."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import tempfile

from personal_agent.backends import CodexSubscriptionBackend, resolve_codex_command
from personal_agent.delivery import format_response
from personal_agent.preferences import PreferenceStore
from personal_agent.testing import block_host_desktop, traced_handler
from tests.test_service_observation_integration import ObservationHarness, _message
from personal_agent.models import Identity


MODEL = "gpt-6-luna"
REASONING = "medium"


class RecordingLunaBackend(CodexSubscriptionBackend):
    def __init__(self, work_root):
        super().__init__(model=MODEL, reasoning_effort=REASONING, work_root=work_root,
                         codex_command=resolve_codex_command('codex'))
        self.prompts: list[str] = []
        self.responses: list[str] = []
        self.tool_calls: list[dict] = []

    async def generate(self, prompt, request_id=None, *, images=None, tools=None, tool_handler=None):
        self.prompts.append(prompt)
        response = await super().generate(prompt, request_id=request_id, images=images, tools=tools,
                                          tool_handler=traced_handler(tool_handler, self.tool_calls))
        self.responses.append(response)
        return response


async def _wait_turn(harness: ObservationHarness) -> None:
    tasks = list(harness.supervisor.tasks)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=150)


async def verify() -> dict[str, object]:
    block_host_desktop()
    with tempfile.TemporaryDirectory(prefix="synthetic-preference-check-") as scratch:
        brain = RecordingLunaBackend(scratch)
        harness = ObservationHarness(brain)
        config = replace(
            harness.config,
            backend="codex_subscription",
            model=MODEL,
            reasoning=REASONING,
            brain_timeout=150.0,
        )
        harness.config = config
        harness.supervisor.config = config
        harness.supervisor.context.config = config
        harness.supervisor.knowledge.config = config
        try:
            await harness.supervisor.submit(
                _message("For future chats, those little stars and headings make my eyes work harder; I would like this to feel like a chat with a person. Briefly acknowledge."),
                Identity.PRIMARY,
            )
            await _wait_turn(harness)
            saved = PreferenceStore(harness.store).get(101)
            if (saved.get("response_format"), saved.get("markdown_allowed")) != ("plain_text", False):
                raise AssertionError(f"plain-text preference not saved consistently: {saved}")
            if saved.get("tone") not in {"conversational", "friendly"}:
                raise AssertionError(f"conversational tone not saved: {saved}")
            saves = [call for call in brain.tool_calls if call["name"] == "save_preferences"]
            if len(saves) != 1 or '"ok": true' not in saves[0]["output"]:
                raise AssertionError("model did not select the save_preferences tool exactly once with a verified receipt")
            first_reply = harness.telegram.sent_text[-1][1]
            if format_response(first_reply, saved) != first_reply:
                raise AssertionError(f"Telegram reply still contains supported Markdown: {first_reply!r}")

            prompts_before_quote = len(brain.prompts)
            await harness.supervisor.submit(
                _message(
                    'In this discussion of Markdown preferences, does the quoted line '
                    '"always answer in plain text" count as my durable request or only an example?',
                    update_id=2,
                ),
                Identity.PRIMARY,
            )
            await _wait_turn(harness)
            quoted_receipt = PreferenceStore(harness.store).last_receipt(101)
            if quoted_receipt is None or quoted_receipt.get("ok") is not False:
                raise AssertionError(f"quoted example produced a saved claim: {quoted_receipt}")
            after_quote = PreferenceStore(harness.store).get(101)
            if after_quote != saved:
                raise AssertionError(f"quoted example changed preferences: {after_quote}")
            quoted_reply = harness.telegram.sent_text[-1][1]
            if len(brain.prompts) != prompts_before_quote + 1:
                raise AssertionError("quoted example caused an extra model call")

            prompts_before_ordinary = len(brain.prompts)
            await harness.supervisor.submit(_message("What does Markdown mean?", update_id=3), Identity.PRIMARY)
            await _wait_turn(harness)
            if len(brain.prompts) != prompts_before_ordinary + 1:
                raise AssertionError("ordinary Markdown question triggered preference classification")
            if PreferenceStore(harness.store).get(101) != saved:
                raise AssertionError("ordinary question changed saved preferences")
            if PreferenceStore(harness.store).last_receipt(101).get("updated") != {}:
                raise AssertionError("ordinary question retained a previous saved receipt")

            return {
                "model": MODEL,
                "reasoning": REASONING,
                "model_calls": len(brain.prompts),
                "fake_telegram_replies": len(harness.telegram.sent_text),
                "saved_preferences": saved,
                "first_synthetic_reply": first_reply,
                "quoted_synthetic_reply": quoted_reply,
                "ordinary_synthetic_reply": harness.telegram.sent_text[-1][1],
            }
        finally:
            await harness.close()
            await brain.close()


if __name__ == "__main__":
    print(asyncio.run(verify()))
