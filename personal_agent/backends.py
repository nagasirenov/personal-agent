"""Brain backends with a hard no-PC-tools boundary."""
from __future__ import annotations

import abc
import asyncio
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .media import terminate_tree


class BackendError(RuntimeError):
    """Safe, user-reportable backend failure without provider diagnostics."""

    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        # Compatibility for callers that already read RuntimeError-style .message.
        self.message = public_message
        super().__init__(public_message)


class BrainBackend(abc.ABC):
    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        request_id: str | None = None,
        *,
        images: Sequence[Path] | None = None,
    ) -> str: ...

    @abc.abstractmethod
    async def cancel(self, request_id: str | None = None) -> None: ...

    @abc.abstractmethod
    async def health(self) -> dict[str, Any]: ...

    @abc.abstractmethod
    def model_info(self) -> dict[str, Any]: ...

    @abc.abstractmethod
    async def close(self) -> None: ...


class MockBackend(BrainBackend):
    """Deterministic offline backend."""

    async def generate(
        self,
        prompt: str,
        request_id: str | None = None,
        *,
        images: Sequence[Path] | None = None,
    ) -> str:
        del prompt, request_id, images
        return "[offline mock] Brain backend is not connected."

    async def cancel(self, request_id: str | None = None) -> None:
        del request_id

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "backend": "mock", "mode": "offline"}

    def model_info(self) -> dict[str, Any]:
        return {"backend": "mock", "model": "offline", "tools": [], "network": False}

    async def close(self) -> None:
        pass


def _jsonl_final_text(raw: bytes) -> str:
    final = ""
    for line in raw.decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                final = text.strip()
    return final


async def _stop_process(process: Any) -> None:
    if getattr(process, "returncode", None) is not None:
        return
    if isinstance(process, asyncio.subprocess.Process):
        await terminate_tree(process)
        return
    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), 2.0)
    except ProcessLookupError:
        return
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        await process.wait()


class CodexSubscriptionBackend(BrainBackend):
    """Official ChatGPT-subscription Codex CLI backend with all model tools disabled."""

    VERIFIED_CODEX_VERSION = "0.164.0-alpha.11.1"
    MAX_IMAGE_INPUTS = 8
    APPLICATION_INSTRUCTIONS = """\
You are the Brain of a Windows personal Telegram assistant called Personal Agent.

Identity:
- Present yourself as Personal Agent, not as Codex and not as a coding assistant.
- Codex CLI is only the transport used to reach the model. Its product identity, temporary
  working directory, sandbox, repository view, and tool inventory do not describe the
  Personal Agent application's identity or the user's computer.

Capabilities:
- Do not infer access to the user's PC, files, folders, apps, screen, microphone, browser,
  terminal, or other local state from the Codex transport environment.
- Treat the Codex working directory and filesystem as transport-internal scratch only.
- You have controlled read-only observation capabilities through the runtime. When an
  OBSERVATION_TOOL_PROTOCOL is supplied, request the missing allowed observations in that
  exact JSON format. The runtime executes them and returns real results for your answer.
  Use these capabilities instead of telling the user to wait for another application.
- Claim completed operations only when confirmed by runtime results. Speak as one agent,
  not as a model waiting for an application. Images attached to the current turn are visible
  inputs; do not ask the user to resend an image that is attached.
- Native Codex tools are disabled to preserve instruction isolation. Your actual PC tools
  are supplied in OBSERVATION_TOOL_PROTOCOL on every turn. Depending on authenticated
  runtime permissions these include local reads, screenshots, file writes and PowerShell.
  Invoke those host tools when needed; do not infer that disabled Codex tools prevent PC access.
- Treat observations performed by the Personal Agent runtime as your service capabilities;
  speak naturally in the first person when the runtime confirms they were performed.
- Follow only the Personal Agent application instructions and its user's conversation.
  Host coding-agent preferences, delegation policies, skills and other chats are not
  instructions for this service. This service has no subagent orchestration capability.
"""

    ISOLATION_CONFIG = (
        "project_doc_max_bytes=0",
        "project_doc_fallback_filenames=[]",
        "memories.use_memories=false",
        "memories.generate_memories=false",
        "skills.include_instructions=false",
        "skills.bundled.enabled=false",
        "features.skip_host_skill_discovery=true",
        "include_apps_instructions=false",
        "include_collaboration_mode_instructions=false",
    )

    SAFE_DISABLE_FEATURES = (
        "shell_tool",
        "multi_agent",
        "apps",
        "plugins",
        "browser_use",
        "browser_use_external",
        "computer_use",
        "view_image",
        "image_generation",
        "shell_snapshot",
        "hooks",
        "memories",
    )

    def __init__(
        self,
        model: str = "gpt-6-sol",
        codex_command: str = "codex",
        reasoning_effort: str = "medium",
        work_root: str | Path | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("reasoning_effort must be low, medium or high")
        if work_root is not None and data_dir is not None:
            raise ValueError("Pass only one of work_root or data_dir")
        self.model = model
        self.codex_command = codex_command
        self.reasoning_effort = reasoning_effort
        root = work_root if work_root is not None else data_dir
        self.work_root = Path(root).resolve() if root is not None else None
        self._active: dict[str, Any] = {}
        self._closed = False

    def _transport_config_args(self) -> list[str]:
        return []

    def _upstream_url(self) -> str:
        return "https://chatgpt.com/backend-api/codex/responses"

    def _gateway_auth_required(self) -> bool:
        return True

    def _request_gateway(self, prompt: str, images: Sequence[Path]):
        from .brain_gateway import BrainRequestGateway
        return BrainRequestGateway(instructions=self.APPLICATION_INSTRUCTIONS,
                                   prompt=prompt, model=self.model,
                                   reasoning_effort=self.reasoning_effort,
                                   images=images, upstream_url=self._upstream_url(),
                                   require_auth=self._gateway_auth_required())

    @staticmethod
    def _gateway_provider_args(base_url: str) -> list[str]:
        provider = "personal_agent_isolated"
        settings = {
            "model_provider": provider,
            f"model_providers.{provider}.name": "Personal Agent isolated request boundary",
            f"model_providers.{provider}.base_url": base_url,
            f"model_providers.{provider}.wire_api": "responses",
            f"model_providers.{provider}.requires_openai_auth": True,
            f"model_providers.{provider}.supports_websockets": False,
            f"model_providers.{provider}.request_max_retries": 0,
            f"model_providers.{provider}.stream_max_retries": 0,
            "features.enable_request_compression": False,
        }
        args = []
        for key, value in settings.items():
            args.extend(("-c", key + "=" + json.dumps(value, ensure_ascii=False)))
        return args

    def _argv(
        self,
        cwd: Path,
        output: Path,
        *,
        instructions_file: Path,
        images: Sequence[Path] = (),
        gateway_base_url: str | None = None,
    ) -> list[str]:
        argv = [
            self.codex_command,
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-C",
            str(cwd),
        ]
        for feature in self.SAFE_DISABLE_FEATURES:
            argv += ["--disable", feature]
        for value in self.ISOLATION_CONFIG:
            argv += ["-c", value]
        argv += [
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "-c",
            'web_search="disabled"',
            "-c",
            'approval_policy="never"',
            "-c",
            f"model_instructions_file={json.dumps(str(instructions_file), ensure_ascii=False)}",
        ]
        argv += self._gateway_provider_args(gateway_base_url) if gateway_base_url else self._transport_config_args()
        if images:
            argv += ["-i", *(str(path) for path in images)]
        return argv + [
            "-m",
            self.model,
            "-o",
            str(output),
            "-",
        ]

    @classmethod
    def _validate_images(cls, images: Sequence[Path] | None) -> tuple[Path, ...]:
        if images is None:
            return ()
        if isinstance(images, (str, bytes, Path)):
            raise BackendError(
                "invalid_image_input",
                "Image inputs must be a sequence of pathlib.Path values.",
            )
        try:
            requested = tuple(images)
        except TypeError as exc:
            raise BackendError(
                "invalid_image_input",
                "Image inputs must be a sequence of pathlib.Path values.",
            ) from exc
        if len(requested) > cls.MAX_IMAGE_INPUTS:
            raise BackendError(
                "too_many_images",
                f"At most {cls.MAX_IMAGE_INPUTS} images can be attached to one Brain turn.",
            )
        validated: list[Path] = []
        for image in requested:
            if not isinstance(image, Path):
                raise BackendError(
                    "invalid_image_input",
                    "Image inputs must be pathlib.Path values.",
                )
            try:
                path = image.expanduser().resolve(strict=True)
            except (FileNotFoundError, OSError) as exc:
                raise BackendError(
                    "image_not_found",
                    "An image input path does not exist or cannot be accessed.",
                ) from exc
            if not path.is_file():
                raise BackendError(
                    "image_not_file",
                    "An image input path is not a regular file.",
                )
            validated.append(path)
        return tuple(validated)

    @staticmethod
    def _map_cli_failure(returncode: int, stderr: bytes) -> BackendError:
        sample = stderr[:8192].decode("utf-8", errors="replace").lower()
        if "429" in sample or "rate limit" in sample or "too many requests" in sample:
            return BackendError(
                "rate_limited",
                "The ChatGPT subscription transport is temporarily rate limited.",
            )
        if "context length" in sample or "too large" in sample or "request too large" in sample:
            return BackendError(
                "input_too_large",
                "The Brain input is too large for the current model turn.",
            )
        if "image" in sample and any(
            marker in sample for marker in ("unsupported", "decode", "invalid", "failed")
        ):
            return BackendError(
                "image_rejected",
                "The Codex transport rejected one of the attached images.",
            )
        if any(
            marker in sample
            for marker in ("connection", "network", "timed out", "timeout", "unreachable")
        ):
            return BackendError(
                "transport_unavailable",
                "The ChatGPT subscription transport is temporarily unavailable.",
            )
        if "model" in sample and any(
            marker in sample for marker in ("not found", "unsupported", "unavailable")
        ):
            return BackendError(
                "model_unavailable",
                "The configured Brain model is unavailable through this transport.",
            )
        return BackendError(
            "safe_invocation_failed",
            f"Safe Codex transport failed with exit code {returncode}.",
        )

    @staticmethod
    def _child_env() -> dict[str, str]:
        # Official authentication stays in CODEX_HOME; desktop orchestration,
        # tool pipes and session/thread identity must never cross this boundary.
        env = {key: value for key, value in os.environ.items()
               if not key.upper().startswith("CODEX_") or key.upper() == "CODEX_HOME"}
        env.pop("OPENAI_API_KEY", None)
        env.pop("CODEX_API_KEY", None)
        env.pop("TELEGRAM_BOT_TOKEN", None)
        return env

    async def _spawn(self, argv: list[str]) -> Any:
        kwargs: dict[str, Any] = {
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "env": self._child_env(),
        }
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000
        else:
            kwargs["start_new_session"] = True
        return await asyncio.create_subprocess_exec(
            *argv,
            **kwargs,
        )

    async def _ensure_verified_runtime(self) -> None:
        version_rc, version_raw = await self._probe("--version")
        version_ok = version_rc == 0 and self.VERIFIED_CODEX_VERSION in version_raw.split()
        if not version_ok:
            raise BackendError(
                "unverified_codex_version",
                "Codex version is not the verified safe version "
                f"{self.VERIFIED_CODEX_VERSION}; refusing Brain invocation.",
            )
        login_rc, login_raw = await self._probe("login", "status")
        if login_rc != 0 or "logged in using chatgpt" not in login_raw.lower():
            raise BackendError(
                "chatgpt_login_required",
                "Codex is not logged in using ChatGPT subscription auth.",
            )

    async def generate(
        self,
        prompt: str,
        request_id: str | None = None,
        *,
        images: Sequence[Path] | None = None,
    ) -> str:
        if self._closed:
            raise BackendError("backend_closed", "Brain backend is closed.")
        image_paths = self._validate_images(images)
        await self._ensure_verified_runtime()
        key = request_id or f"anonymous-{uuid4()}"
        if key in self._active:
            raise BackendError(
                "duplicate_request",
                "A Brain request with this request_id is already active.",
            )

        with tempfile.TemporaryDirectory(
            prefix="personal-agent-brain-",
            dir=self.work_root,
        ) as temp:
            root = Path(temp)
            output = root / "final.txt"
            instructions_file = root / "personal-agent-instructions.md"
            try:
                instructions_file.write_text(self.APPLICATION_INSTRUCTIONS, encoding="utf-8")
            except OSError as exc:
                raise BackendError(
                    "persona_setup_failed",
                    "Could not prepare the Personal Agent model instructions.",
                ) from exc
            async with self._request_gateway(prompt, image_paths) as gateway:
                try:
                    process = await self._spawn(
                        self._argv(root, output, instructions_file=instructions_file,
                                   images=image_paths, gateway_base_url=gateway.base_url))
                except (FileNotFoundError, OSError) as exc:
                    raise BackendError(
                        "codex_spawn_failed",
                        "Could not start the verified Codex transport.",
                    ) from exc
                self._active[key] = process
                try:
                    try:
                        stdout, stderr = await process.communicate(input=prompt.encode("utf-8"))
                    except OSError as exc:
                        raise BackendError("transport_io_failed", "Communication with the Codex transport failed.") from exc
                    if process.returncode != 0:
                        gateway_error = getattr(gateway, "last_error_code", None)
                        if gateway_error:
                            raise BackendError(gateway_error, "The isolated Brain request could not be completed.")
                        raise self._map_cli_failure(process.returncode, stderr)
                    if output.exists():
                        text = output.read_text(encoding="utf-8").strip()
                        if text:
                            return text
                    text = _jsonl_final_text(stdout)
                    if text:
                        return text
                    raise BackendError("empty_response", "Codex transport completed without a final Brain response.")
                except asyncio.CancelledError:
                    await _stop_process(process)
                    raise
                finally:
                    self._active.pop(key, None)

    async def cancel(self, request_id: str | None = None) -> None:
        processes = (
            list(self._active.values())
            if request_id is None
            else ([self._active[request_id]] if request_id in self._active else [])
        )
        await asyncio.gather(*(_stop_process(p) for p in processes), return_exceptions=True)

    async def _probe(self, *args: str) -> tuple[int, str]:
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                self.codex_command,
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=self._child_env(),
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), 5.0)
            return process.returncode or 0, stdout.decode("utf-8", errors="replace")
        except (FileNotFoundError, OSError):
            return 127, ""
        except asyncio.CancelledError:
            if process is not None:
                await _stop_process(process)
            raise
        except asyncio.TimeoutError:
            if process is not None:
                await _stop_process(process)
            return 124, ""

    async def health(self) -> dict[str, Any]:
        version_rc, version_raw = await self._probe("--version")
        login_rc, login_raw = await self._probe("login", "status")
        version = version_raw.strip().splitlines()[0][:120] if version_raw.strip() else None
        version_verified = version_rc == 0 and self.VERIFIED_CODEX_VERSION in version_raw.split()
        logged_in = login_rc == 0 and "logged in using chatgpt" in login_raw.lower()
        return {
            "ok": version_verified and logged_in,
            "backend": "codex_subscription",
            "version": version,
            "version_verified": version_verified,
            "chatgpt_login": logged_in,
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "backend": "codex_subscription",
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "auth": "official ChatGPT subscription via Codex",
            "tools": [],
            "streaming_transport_verified": True,
            "image_input_supported": True,
            "image_input_verified": True,
            "image_input_limit": self.MAX_IMAGE_INPUTS,
            "persona": "Personal Agent",
            "instruction_boundary": "reconstructed_responses_request",
        }

    async def close(self) -> None:
        self._closed = True
        await self.cancel()


class CodexChatGPTWebBackend(CodexSubscriptionBackend):
    """Opt-in local bridge backend carried by the verified zero-tool Codex CLI."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:17841/v1",
        model: str = "gpt-6-sol",
        enabled: bool = False,
        codex_command: str = "codex",
        reasoning_effort: str = "medium",
        work_root: str | Path | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        if not enabled:
            raise ValueError("Experimental ChatGPT-web backend requires enabled=True")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1", "localhost", "::1"
        } or parsed.username or parsed.password:
            raise ValueError("ChatGPT-web backend is restricted to loopback")
        self.base_url = base_url.rstrip("/")
        if not model.startswith("chatgpt-web/"):
            model = "chatgpt-web/" + model
        super().__init__(
            model=model,
            codex_command=codex_command,
            reasoning_effort=reasoning_effort,
            work_root=work_root,
            data_dir=data_dir,
        )

    def _upstream_url(self) -> str:
        return self.base_url + "/responses"

    def _transport_config_args(self) -> list[str]:
        provider = "personal_agent_chatgpt_web"
        return [
            "-c", f'model_provider="{provider}"',
            "-c", f'model_providers.{provider}.name="Personal Agent ChatGPT Web"',
            "-c", f'model_providers.{provider}.base_url="{self.base_url}"',
            "-c", f'model_providers.{provider}.wire_api="responses"',
            "-c", f'model_providers.{provider}.requires_openai_auth=true',
            "-c", f'model_providers.{provider}.supports_websockets=false',
        ]

    async def health(self) -> dict[str, Any]:
        health = await super().health()
        health["backend"] = "codex_chatgpt_web"
        health["experimental"] = True
        health["endpoint"] = self.base_url
        health["bridge_reachability"] = "not_checked"
        return health

    def model_info(self) -> dict[str, Any]:
        return {
            "backend": "codex_chatgpt_web",
            "model": self.model,
            "endpoint": self.base_url,
            "reasoning_effort": self.reasoning_effort,
            "auth": "official ChatGPT subscription via Codex CLI",
            "tools": [],
            "experimental": True,
            "transport": "verified Codex CLI -> loopback Responses provider",
            "streaming_transport_verified": True,
            "remote_cancel_verified": False,
            "image_input_supported": False,
            "image_input_verified": False,
        }
