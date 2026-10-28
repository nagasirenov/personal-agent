"""Opt-in real check of native Responses function calling through the isolated gateway.

Synthetic data only; model is hard-pinned to gpt-6-luna / medium. Nothing touches Telegram.
Usage: python scripts/verify_native_tools_real.py --real [--debug-upstream]
       python scripts/verify_native_tools_real.py --real --slow [--slow-seconds 150]

--slow runs only the long-tool scenario: the tool handler sleeps while the gateway holds the CLI's HTTP
request open without response headers, which shows whether the Codex CLI waits that long.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from personal_agent.backends import CodexSubscriptionBackend, resolve_codex_command
from personal_agent.brain_gateway import BrainRequestGateway, ToolOutcome
from verify_conversation import EVALUATION_MODEL, EVALUATION_REASONING, red_png

SECRET_VALUE = "4721-ZETA"
SLOW_VALUE = "9083-SLOW"

TOOLS = [
    {"type": "function", "name": "lookup_fixture_value",
     "description": "Look up a synthetic fixture value by key. The values exist only in this tool.",
     "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                    "required": ["key"], "additionalProperties": False}, "strict": False},
    {"type": "function", "name": "slow_fixture_lookup",
     "description": "Look up the slow fixture value. It is slow on purpose; call it exactly once and wait.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}, "strict": False},
    {"type": "function", "name": "capture_fixture_image",
     "description": "Capture a synthetic fixture image and show it to you.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
     "strict": False},
]


class SpikeBackend(CodexSubscriptionBackend):
    debug_upstream = False

    def __init__(self, **kwargs):
        if (kwargs.get("model") != EVALUATION_MODEL
                or kwargs.get("reasoning_effort") != EVALUATION_REASONING):
            raise ValueError("Real native-tool checks require gpt-6-luna with medium reasoning")
        super().__init__(**kwargs)
        self.gateways: list[BrainRequestGateway] = []

    def _request_gateway(self, prompt, images, tools=(), tool_handler=None):
        backend = self

        class Gateway(BrainRequestGateway):
            async def __aenter__(self):
                await super().__aenter__()
                backend.gateways.append(self)
                if backend.debug_upstream:
                    async def show(response):
                        if response.status_code != 200:
                            body = await response.aread()
                            print(f"  [debug] upstream {response.status_code}: "
                                  f"{body[:600].decode('utf-8', 'replace')}", flush=True)
                    self._client.event_hooks["response"].append(show)
                return self

        return Gateway(instructions=self.APPLICATION_INSTRUCTIONS, prompt=prompt, model=self.model,
                       reasoning_effort=self.reasoning_effort, images=images,
                       upstream_url=self._upstream_url(), require_auth=self._gateway_auth_required(),
                       tools=tools, tool_handler=tool_handler)


async def main() -> int:
    if "--real" not in sys.argv:
        print("Refusing to run without --real (this makes real gpt-6-luna calls).")
        return 2
    results: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="personal-agent-native-tools-") as folder:
        root = Path(folder)
        image = root / "fixture-red.png"
        red_png(image)
        backend = SpikeBackend(model=EVALUATION_MODEL, reasoning_effort=EVALUATION_REASONING,
                               codex_command=resolve_codex_command("codex"), work_root=root)
        backend.debug_upstream = "--debug-upstream" in sys.argv
        print(f"evaluation_model={backend.model}; reasoning={backend.reasoning_effort}; "
              f"codex={backend.codex_command}", flush=True)
        calls: list[tuple[str, dict]] = []
        slow_seconds = float(sys.argv[sys.argv.index("--slow-seconds") + 1]) if "--slow-seconds" in sys.argv else 150.0

        async def handler(name: str, arguments: dict, call_id: str) -> ToolOutcome:
            calls.append((name, arguments))
            if name == "lookup_fixture_value":
                return ToolOutcome(json.dumps({"ok": True, "key": arguments.get("key"),
                                               "value": SECRET_VALUE}))
            if name == "slow_fixture_lookup":
                await asyncio.sleep(slow_seconds)
                return ToolOutcome(json.dumps({"ok": True, "value": SLOW_VALUE}))
            if name == "capture_fixture_image":
                return ToolOutcome(json.dumps({"ok": True, "note": "image attached"}), [image])
            return ToolOutcome(json.dumps({"ok": False, "error": {"code": "unknown_tool"}}))

        async def scenario(name: str, prompt: str, check):
            calls.clear()
            try:
                answer = await backend.generate(prompt, f"native-{name}", tools=TOOLS,
                                                tool_handler=handler)
                ok, detail = check(answer)
            except Exception as exc:  # report, keep going
                answer, ok, detail = "", False, f"{type(exc).__name__}: {getattr(exc, 'code', exc)}"
            gateway = backend.gateways[-1]
            requests = gateway.upstream_requests
            print(f"{name}={'PASS' if ok else 'FAIL'}; upstream_requests={requests}; "
                  f"tool_calls={[c[0] for c in calls]}; {detail}", flush=True)
            if not ok:
                print(f"  answer={answer[:300]!r}", flush=True)
            results.append((name, ok, detail))

        try:
            if "--slow" in sys.argv:
                started = time.monotonic()
                await scenario(
                    "e_slow_tool",
                    "Call slow_fixture_lookup once, wait for it, and tell me the exact value it returns. "
                    "Answer with the value only.",
                    lambda a: (SLOW_VALUE in a and [c[0] for c in calls] == ["slow_fixture_lookup"],
                               f"value_in_answer={SLOW_VALUE in a}; tool_sleep={slow_seconds:.0f}s; "
                               f"elapsed={time.monotonic() - started:.0f}s"))
                print(f"total_scenarios={len(results)}; passed={sum(ok for _, ok, _ in results)}; "
                      "telegram_network_calls=0", flush=True)
                return 0 if all(ok for _, ok, _ in results) else 1
            await scenario(
                "a_lookup_value",
                "Use the lookup_fixture_value tool with key 'alpha' and tell me the exact value it "
                "returns. Answer with the value only.",
                lambda a: (SECRET_VALUE in a and [c[0] for c in calls] == ["lookup_fixture_value"],
                           "value_in_answer=" + str(SECRET_VALUE in a)))
            await scenario(
                "b_tool_image",
                "Call capture_fixture_image, look at the image it returns, and tell me the dominant "
                "color of the image in one word (English).",
                lambda a: ("red" in a.casefold() and "capture_fixture_image" in [c[0] for c in calls],
                           "says_red=" + str("red" in a.casefold())))
            await scenario(
                "c_no_tool_needed",
                "What is 12 plus 30? Answer with just the number.",
                lambda a: ("42" in a and not calls, "no_tool_calls=" + str(not calls)))
            await scenario(
                "d_two_lookups",
                "Look up the fixture values for keys 'alpha' and 'beta' using the tool (call it once "
                "for each key), then tell me both values separated by a comma.",
                lambda a: (a.count(SECRET_VALUE) >= 2 and len(calls) == 2,
                           f"tool_calls={len(calls)}; values_in_answer={a.count(SECRET_VALUE)}"))
        finally:
            await backend.close()
    print(f"total_scenarios={len(results)}; passed={sum(ok for _, ok, _ in results)}; "
          "telegram_network_calls=0", flush=True)
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
