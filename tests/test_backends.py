from __future__ import annotations

import asyncio
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from personal_agent.backends import (
    BackendError,
    BrainBackend,
    CodexChatGPTWebBackend,
    CodexSubscriptionBackend,
    MockBackend,
    _jsonl_final_text,
)


class FakeProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        block: bool = False,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.final_returncode = returncode
        self.returncode = None if block else returncode
        self.event = asyncio.Event()
        if not block:
            self.event.set()
        self.input: bytes | None = None
        self.terminated = False
        self.killed = False

    async def communicate(self, input: bytes | None = None):
        self.input = input
        await self.event.wait()
        if self.returncode is None:
            self.returncode = self.final_returncode
        return self.stdout, self.stderr

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.event.set()

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.event.set()

    async def wait(self):
        await self.event.wait()
        return self.returncode


class FakeSubscriptionBackend(CodexSubscriptionBackend):
    def __init__(self, processes: list[FakeProcess], **kwargs):
        super().__init__(**kwargs)
        self.processes = list(processes)
        self.spawned: list[list[str]] = []
        self.instructions_seen: list[str] = []
        self.spawn_event = asyncio.Event()
        self.probe_version = self.VERIFIED_CODEX_VERSION
        self.login_ok = True
        self.probe_calls: list[tuple[str, ...]] = []
        self.wire_calls: list[str] = []
        self.wire_result = (False, "unit tests never start the real CLI")

    async def _wire_self_check(self, version: str):
        self.wire_calls.append(version)
        return self.wire_result

    async def _probe(self, *args: str):
        self.probe_calls.append(args)
        if args == ("--version",):
            return 0, f"codex-cli {self.probe_version}\n"
        if args == ("login", "status"):
            return (0, "Logged in using ChatGPT\n") if self.login_ok else (1, "not logged in\n")
        raise AssertionError(args)

    async def _spawn(self, argv: list[str]):
        self.spawned.append(argv)
        config = next(
            value for value in argv if value.startswith("model_instructions_file=")
        )
        instructions_path = Path(json.loads(config.split("=", 1)[1]))
        self.instructions_seen.append(instructions_path.read_text(encoding="utf-8"))
        process = self.processes.pop(0)
        self.spawn_event.set()
        return process


class FakeWebBackend(CodexChatGPTWebBackend):
    def __init__(self, processes: list[FakeProcess], **kwargs):
        super().__init__(**kwargs)
        self.processes = list(processes)
        self.spawned: list[list[str]] = []

    async def _wire_self_check(self, version: str):
        return False, "unit tests never start the real CLI"

    async def _probe(self, *args: str):
        if args == ("--version",):
            return 0, f"codex-cli {self.VERIFIED_CODEX_VERSION}\n"
        if args == ("login", "status"):
            return 0, "Logged in using ChatGPT\n"
        raise AssertionError(args)

    async def _spawn(self, argv: list[str]):
        self.spawned.append(argv)
        return self.processes.pop(0)


class BackendsTests(unittest.IsolatedAsyncioTestCase):
    def test_public_generate_signature(self):
        signature = inspect.signature(BrainBackend.generate)
        self.assertEqual(
            list(signature.parameters), ["self", "prompt", "request_id", "images", "tools", "tool_handler", "builtin_tools"]
        )
        self.assertIsNone(signature.parameters["request_id"].default)
        self.assertEqual(
            signature.parameters["images"].kind, inspect.Parameter.KEYWORD_ONLY
        )
        for name in ("tools", "tool_handler"):
            self.assertIsNone(signature.parameters[name].default)
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(signature.parameters["images"].default)

    async def test_mock_is_explicitly_offline(self):
        backend = MockBackend()
        self.assertEqual(await backend.generate("one"), await backend.generate("two"))
        self.assertEqual(backend.model_info()["tools"], [])
        self.assertEqual((await backend.health())["mode"], "offline")

    def test_subscription_argv_is_zero_tool_boundary_and_stdin_only(self):
        backend = CodexSubscriptionBackend(reasoning_effort="medium")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            instructions = root / "instructions.md"
            argv = backend._argv(
                root,
                root / "out.txt",
                instructions_file=instructions,
            )
        self.assertEqual(argv[:2], ["codex", "exec"])
        for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral", "--json"):
            self.assertIn(flag, argv)
        disabled = {
            argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--disable"
        }
        self.assertEqual(disabled, set(CodexSubscriptionBackend.SAFE_DISABLE_FEATURES))
        self.assertIn("shell_snapshot", disabled)
        self.assertIn('web_search="disabled"', argv)
        self.assertIn("read-only", argv)
        self.assertNotIn("--enable", argv)
        self.assertIn('model_reasoning_effort="medium"', argv)
        persona_arg = next(
            value for value in argv if value.startswith("model_instructions_file=")
        )
        self.assertEqual(
            Path(json.loads(persona_arg.split("=", 1)[1])),
            instructions,
        )
        self.assertNotIn("-i", argv)
        self.assertEqual(argv[-1], "-")
        self.assertNotIn("SECRET PROMPT", argv)

    def test_child_env_scrubs_api_keys(self):
        backend = CodexSubscriptionBackend()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret", "CODEX_API_KEY": "secret2", "TELEGRAM_BOT_TOKEN": "bot-secret", "KEEP_ME": "yes",
                                    "CODEX_HOME": "test-auth-home", "CODEX_THREAD_ID": "coordinator-thread", "CODEX_APP_TOOLS_PIPE_PATH": "coordinator-pipe", "CODEX_SESSION_ID": "coordinator-session"}):
            env = backend._child_env()
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("CODEX_API_KEY", env)
        self.assertNotIn("TELEGRAM_BOT_TOKEN", env)
        self.assertEqual(env["CODEX_HOME"], "test-auth-home")
        self.assertNotIn("CODEX_THREAD_ID", env)
        self.assertNotIn("CODEX_SESSION_ID", env)
        self.assertNotIn("CODEX_APP_TOOLS_PIPE_PATH", env)
        self.assertEqual(env["KEEP_ME"], "yes")

    def test_jsonl_parser(self):
        raw = b'{"type":"item.completed","item":{"type":"agent_message","text":"PONG"}}\n'
        self.assertEqual(_jsonl_final_text(raw), "PONG")

    async def test_generate_sends_prompt_via_stdin_and_uses_owned_work_root(self):
        process = FakeProcess(
            stdout=b'{"type":"item.completed","item":{"type":"agent_message","text":"PONG"}}\n'
        )
        with tempfile.TemporaryDirectory() as temp:
            work_root = Path(temp)
            backend = FakeSubscriptionBackend([process], work_root=work_root)
            self.assertEqual(await backend.generate("very private prompt", "req"), "PONG")
            argv = backend.spawned[0]
            cwd = Path(argv[argv.index("-C") + 1])
            self.assertTrue(cwd.is_relative_to(work_root.resolve()))
        self.assertEqual(process.input, b"very private prompt")
        self.assertNotIn("very private prompt", argv)
        persona = backend.instructions_seen[0]
        self.assertIn("Ты — Personal Agent", persona)
        self.assertIn("Codex CLI — только канал связи", persona)
        self.assertIn("рабочая папка, песочница и набор инструментов ничего не говорят", persona)

    async def test_generate_attaches_validated_images_via_cli_input(self):
        process = FakeProcess(
            stdout=b'{"type":"item.completed","item":{"type":"agent_message","text":"SEEN"}}\n'
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first.png"
            second = root / "second.jpg"
            from personal_agent.screenshot import _encode_png
            first.write_bytes(_encode_png(1, 1, bytes((0, 0, 255, 0))))
            second.write_bytes(b"\xff\xd8\xfffixture-two")
            backend = FakeSubscriptionBackend([process])
            self.assertEqual(
                await backend.generate(
                    "describe",
                    "images",
                    images=[first, second],
                ),
                "SEEN",
            )
            argv = backend.spawned[0]
            image_index = argv.index("-i")
            self.assertEqual(
                argv[image_index + 1 : image_index + 3],
                [str(first.resolve()), str(second.resolve())],
            )
            self.assertLess(image_index, argv.index("-m"))
        self.assertEqual(process.input, b"describe")

    async def test_image_validation_fails_before_spawn(self):
        backend = FakeSubscriptionBackend([FakeProcess()])
        missing = Path(tempfile.gettempdir()) / f"missing-{os.getpid()}.png"
        with self.assertRaises(BackendError) as caught:
            await backend.generate("describe", images=[missing])
        self.assertEqual(caught.exception.code, "image_not_found")
        self.assertEqual(backend.spawned, [])

        too_many = [Path("x.png")] * (CodexSubscriptionBackend.MAX_IMAGE_INPUTS + 1)
        with self.assertRaises(BackendError) as caught:
            await backend.generate("describe", images=too_many)
        self.assertEqual(caught.exception.code, "too_many_images")
        self.assertEqual(backend.spawned, [])

    async def test_unverified_version_fails_closed_before_spawn(self):
        backend = FakeSubscriptionBackend([FakeProcess()])
        backend.probe_version = "0.165.0"
        with self.assertRaises(BackendError) as caught:
            await backend.generate("ping", "bad-version")
        self.assertEqual(caught.exception.code, "unverified_codex_version")
        self.assertEqual(backend.spawned, [])

    async def test_cancelling_runtime_probe_reaps_its_process(self):
        process = FakeProcess(block=True)
        backend = CodexSubscriptionBackend()
        with patch("personal_agent.backends.asyncio.create_subprocess_exec", return_value=process):
            task = asyncio.create_task(backend._probe("--version"))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(process.terminated)

    async def test_nonzero_safe_cli_has_no_unsafe_fallback(self):
        backend = FakeSubscriptionBackend(
            [FakeProcess(returncode=2, stderr=b"429 rate limit secret-token-123")]
        )
        with self.assertRaises(BackendError) as caught:
            await backend.generate("ping", "bad")
        self.assertEqual(caught.exception.code, "rate_limited")
        self.assertEqual(
            caught.exception.public_message,
            "The ChatGPT subscription transport is temporarily rate limited.",
        )
        self.assertEqual(caught.exception.message, caught.exception.public_message)
        self.assertNotIn("secret-token-123", str(caught.exception))
        self.assertEqual(len(backend.spawned), 1)

    def test_subscription_model_info_reports_direct_image_capability(self):
        info = CodexSubscriptionBackend().model_info()
        self.assertTrue(info["image_input_supported"])
        self.assertTrue(info["image_input_verified"])
        self.assertEqual(info["image_input_limit"], CodexSubscriptionBackend.MAX_IMAGE_INPUTS)

    async def test_cancellation_is_request_scoped_and_cancel_all(self):
        p1, p2 = FakeProcess(block=True), FakeProcess(block=True)
        backend = FakeSubscriptionBackend([p1, p2])
        t1 = asyncio.create_task(backend.generate("one", "one"))
        await backend.spawn_event.wait()
        backend.spawn_event.clear()
        t2 = asyncio.create_task(backend.generate("two", "two"))
        await backend.spawn_event.wait()
        await backend.cancel("one")
        self.assertTrue(p1.terminated)
        self.assertFalse(p2.terminated)
        await backend.cancel()
        self.assertTrue(p2.terminated)
        await asyncio.gather(t1, t2, return_exceptions=True)

    def test_web_backend_requires_explicit_enable_and_loopback(self):
        with self.assertRaisesRegex(ValueError, "enabled=True"):
            CodexChatGPTWebBackend()
        with self.assertRaisesRegex(ValueError, "loopback"):
            CodexChatGPTWebBackend(enabled=True, base_url="https://example.com/v1")
        with self.assertRaisesRegex(ValueError, "loopback"):
            CodexChatGPTWebBackend(enabled=True, base_url="http://token@127.0.0.1:17841/v1")

    async def test_web_backend_uses_verified_cli_with_local_responses_provider(self):
        process = FakeProcess(
            stdout=b'{"type":"item.completed","item":{"type":"agent_message","text":"PONG"}}\n'
        )
        backend = FakeWebBackend(
            [process],
            enabled=True,
            base_url="http://127.0.0.1:17841/v1",
            model="gpt-6-sol",
        )
        self.assertEqual(await backend.generate("bridge prompt", "web"), "PONG")
        argv = backend.spawned[0]
        joined = "\n".join(argv)
        self.assertIn('model_provider="personal_agent_isolated"', argv)
        boundary_url = next(value for value in argv if value.startswith('model_providers.personal_agent_isolated.base_url='))
        self.assertIn('http://127.0.0.1:', boundary_url)
        self.assertNotIn('17841', boundary_url)
        self.assertEqual(backend._upstream_url(), 'http://127.0.0.1:17841/v1/responses')
        self.assertTrue(backend._gateway_auth_required())
        self.assertIn('model_providers.personal_agent_isolated.wire_api="responses"', argv)
        self.assertIn('model_providers.personal_agent_isolated.requires_openai_auth=true', argv)
        self.assertIn('model_providers.personal_agent_isolated.supports_websockets=false', argv)
        self.assertIn("--ignore-user-config", argv)
        self.assertEqual(argv[argv.index("-m") + 1], "chatgpt-web/gpt-6-sol")
        self.assertNotIn("bridge prompt", joined)
        self.assertEqual(process.input, b"bridge prompt")
        self.assertEqual(backend.model_info()["tools"], [])
        self.assertFalse(backend.model_info()["remote_cancel_verified"])
        self.assertFalse(backend.model_info()["image_input_supported"])
        self.assertFalse(backend.model_info()["image_input_verified"])


if __name__ == "__main__":
    unittest.main()
