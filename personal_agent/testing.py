"""Offline test doubles for the native tool loop (no network, no model).

`ScriptedBrain.generate` plays the part of the model in one turn: it calls the tools that were
offered, in a scripted order, through the real `tool_handler` (so every real boundary runs), records
the outcomes and then returns the final text, optionally computed from those outcomes.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .brain_gateway import ToolOutcome


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    outcome: ToolOutcome

    @property
    def data(self) -> Any:
        try:
            return json.loads(self.outcome.output)
        except ValueError:
            return None

    @property
    def ok(self) -> bool:
        data = self.data
        return isinstance(data, dict) and data.get("ok") is True

    @property
    def error_code(self) -> str | None:
        data = self.data
        error = data.get("error") if isinstance(data, dict) else None
        return error.get("code") if isinstance(error, dict) else None


# A scripted call is (tool_name, arguments) or (tool_name, callable(records_so_far) -> arguments).
ScriptedCall = tuple[str, Any]


@dataclass
class ScriptedTurn:
    calls: Sequence[ScriptedCall] = ()
    reply: Any = "brain response"  # str or callable(records, prompt) -> str
    block: asyncio.Event | None = None  # wait on this before replying (cancellation tests)


def traced_handler(tool_handler, sink: list):
    """Wrap a native tool handler so verification scripts can record which tools a real model chose.

    Each call appends {"name", "arguments", "output", "images"} to `sink`.  None stays None.
    """
    if tool_handler is None:
        return None

    async def traced(name, arguments, call_id):
        outcome = await tool_handler(name, arguments, call_id)
        sink.append({"name": name, "arguments": arguments, "output": outcome.output,
                     "images": len(outcome.images)})
        return outcome

    return traced


class ScriptedBrain:
    """Model double with the same `generate(prompt, request_id, *, images, tools, tool_handler)` contract."""

    def __init__(self, turns: Sequence[ScriptedTurn | str] = (), *, response: str = "brain response",
                 image_input_supported: bool = False, error: Exception | None = None) -> None:
        self.turns = [turn if isinstance(turn, ScriptedTurn) else ScriptedTurn(reply=turn) for turn in turns]
        self.response = response
        self.image_input_supported = image_input_supported
        self.error = error
        self.generate_calls: list[dict] = []
        self.records: list[ToolCallRecord] = []
        self.cancel_calls: list[str | None] = []
        self.health_calls = 0
        self.started = asyncio.Event()

    @property
    def tool_names_called(self) -> list[str]:
        return [record.name for record in self.records]

    async def generate(self, prompt: str, request_id: str | None = None, *, images: Sequence[Path] | None = None,
                       tools: Sequence[dict] | None = None, tool_handler=None) -> str:
        call = {"prompt": prompt, "request_id": request_id, "images": list(images) if images else [],
                "tools": list(tools or []), "tool_names": [tool["name"] for tool in (tools or [])]}
        self.generate_calls.append(call)
        self.started.set()
        if self.error is not None:
            raise self.error
        index = len(self.generate_calls) - 1
        turn = self.turns[index] if index < len(self.turns) else ScriptedTurn(reply=self.response)
        records: list[ToolCallRecord] = []
        call["records"] = records
        for position, (name, arguments) in enumerate(turn.calls):
            if tool_handler is None:
                raise AssertionError("the supervisor offered no tool handler")
            if callable(arguments):
                arguments = arguments(records)
            outcome = await tool_handler(name, arguments, f"call-{index}-{position}")
            record = ToolCallRecord(name, arguments, outcome)
            records.append(record)
            self.records.append(record)
        if turn.block is not None:
            await turn.block.wait()
        return turn.reply(records, prompt) if callable(turn.reply) else turn.reply

    async def cancel(self, request_id: str | None = None) -> None:
        self.cancel_calls.append(request_id)

    async def health(self) -> dict:
        self.health_calls += 1
        return {"ok": True, "backend": "fixture"}

    def model_info(self) -> dict:
        return {"backend": "fixture", "model": "fixture-model",
                "image_input_supported": self.image_input_supported}

    async def close(self) -> None:
        return None


def block_host_desktop():
    """Keep real-model verification scripts away from the host desktop; returns a callable that undoes it.

    Real screen capture fails loudly and real UI Automation reports an unavailable desktop.  Scripts that
    need a synthetic screen patch `capture_screenshot` / `read_active_window` themselves on top of this.
    """
    from unittest.mock import AsyncMock, patch
    patches = [
        patch("personal_agent.tools.capture_screenshot",
              side_effect=AssertionError("Real screen capture is disabled in this verification.")),
        patch("personal_agent.windows_uia.read_active_window", new=AsyncMock(return_value={
            "ok": False, "kind": "read_active_window", "error": {"code": "synthetic_test_no_desktop"}})),
    ]
    for item in patches:
        item.start()

    def restore() -> None:
        for item in reversed(patches):
            item.stop()
    return restore


def make_synthetic_video(path: Path, *, segments=(("red", 1.0), ("green", 1.0), ("blue", 1.0)),
                         labels: Sequence[str] | None = None, size: str = "320x240", fps: int = 10,
                         audio: "Path | str | None" = "tone") -> Path:
    """Write a small H.264 test video: one solid color (optionally a big word) per segment, plus audio.

    `audio` is "tone" (a sine wave), a WAV path (for example synthesized speech) or None for a silent video.
    Uses the ffmpeg shipped by imageio-ffmpeg; offline and deterministic.
    """
    import subprocess
    from imageio_ffmpeg import get_ffmpeg_exe
    path = Path(path)
    total = sum(float(seconds) for _, seconds in segments)
    parts = []
    for index, (color, seconds) in enumerate(segments):
        chain = f"color=c={color}:s={size}:r={fps}:d={seconds}"
        if labels and labels[index]:
            chain += (f",drawtext=text='{labels[index]}':fontcolor=white:fontsize=h/4:x=(w-tw)/2:y=(h-th)/2:"
                      r"fontfile='C\:/Windows/Fonts/arialbd.ttf':borderw=4:bordercolor=black")
        parts.append(f"{chain}[v{index}]")
    joined = "".join(f"[v{i}]" for i in range(len(segments)))
    graph = ";".join(parts) + f";{joined}concat=n={len(segments)}:v=1:a=0,format=yuv420p[v]"
    argv = [get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if audio == "tone":
        argv += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={total}"]
    elif audio is not None:
        argv += ["-i", str(audio)]
    argv += ["-filter_complex", graph, "-map", "[v]"]
    if audio is not None:
        argv += ["-map", "0:a:0", "-c:a", "aac", "-shortest"]
    else:
        argv += ["-an"]
    argv += ["-c:v", "libx264", "-preset", "ultrafast", "-t", str(total), str(path)]
    subprocess.run(argv, check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    return path
