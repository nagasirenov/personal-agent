"""Authorization and execution boundary for model-proposed PC actions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .security import ActionGate


_MAX_REQUESTS = 3
_MAX_ACTION_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class ActionRequest:
    kind: str
    arguments: dict


class ActionDenied(Exception):
    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(public_message)


def _deny(code: str, message: str) -> None:
    raise ActionDenied(code, message)


def build_action_request(kind: object, arguments: object) -> ActionRequest:
    """Validate the arguments of one native write_file/run_command call."""
    if not isinstance(arguments, dict):
        _deny("malformed_plan", "Action arguments must be an object.")
    item = dict(arguments)
    if kind == "run_command":
        required, optional = {"command"}, {"cwd", "timeout_seconds"}
        command = item.get("command")
        timeout = item.get("timeout_seconds", 30)
        if (not isinstance(command, str) or not command.strip() or "\x00" in command
                or len(command.encode("utf-8")) > _MAX_ACTION_BYTES
                or ("cwd" in item and (not isinstance(item["cwd"], str) or not item["cwd"].strip()))
                or type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120):
            _deny("invalid_arguments", "run_command requires a bounded command and valid optional arguments.")
    elif kind == "write_file":
        required, optional = {"path", "text"}, {"append"}
        if (not isinstance(item.get("path"), str) or not item["path"].strip()
                or not isinstance(item.get("text"), str)
                or len(item["text"].encode("utf-8")) > _MAX_ACTION_BYTES
                or ("append" in item and type(item["append"]) is not bool)):
            _deny("invalid_arguments", "write_file requires a path, text and an optional boolean append.")
    else:
        _deny("unsupported_kind", "Only write_file and run_command actions are allowed.")
    if not required <= set(item) or set(item) - required - optional:
        _deny("malformed_plan", "Action has unknown or missing arguments.")
    return ActionRequest(kind, item)


class ActionBroker:
    """Keep action authorization and execution separate from read-only observation."""

    def __init__(self, gate: ActionGate, data_dir: Path) -> None:
        self._gate = gate
        self._data_dir = Path(data_dir)

    @property
    def gate(self) -> ActionGate:
        return self._gate

    @gate.setter
    def gate(self, value: ActionGate) -> None:
        self._gate = value

    def available_for(self, sender_id: int) -> bool:
        try:
            self._gate.authorize(sender_id, {"kind": "run_command"})
        except PermissionError:
            return False
        return True

    async def execute(self, sender_id: int, requests: list[ActionRequest]) -> list[dict]:
        if not isinstance(requests, list) or not requests or len(requests) > _MAX_REQUESTS:
            _deny("request_budget", "An action plan must contain one to three requests.")
        # Authorize every operation before executing any part of a batch.
        try:
            for request in requests:
                if not isinstance(request, ActionRequest):
                    _deny("malformed_plan", "Action request has an invalid type.")
                self._gate.authorize(sender_id, {"kind": request.kind})
        except PermissionError:
            _deny("action_denied", "PC actions require PRIMARY and pc_access=full.")

        from .pc_tools import execute_pc_request

        results = []
        for request in requests:
            results.append(await execute_pc_request(request.kind, request.arguments, data_dir=self._data_dir))
        return results
