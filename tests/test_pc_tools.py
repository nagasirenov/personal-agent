import asyncio
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from personal_agent.pc_tools import MAX_OUTPUT_BYTES, MAX_WRITE_BYTES, execute_pc_request


class DummyProcess:
    def __init__(self, stdout=b"", stderr=b"", returncode=0, blocked=False):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stdout.feed_eof()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()
        self.returncode = None if blocked else returncode
        self.blocked = blocked
        self.waiting = asyncio.Event()

    async def wait(self):
        self.waiting.set()
        if self.blocked:
            await asyncio.Event().wait()
        return self.returncode


class PcToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    async def request(self, kind, arguments):
        return await execute_pc_request(kind, arguments, data_dir=self.root)

    async def test_write_utf8_parents_append_and_overwrite(self):
        target = self.root / "new" / "result.txt"
        result = await self.request("write_file", {"path": str(target), "text": "Привет"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], str(target))
        self.assertEqual(result["bytes"], len("Привет".encode("utf-8")))
        result = await self.request("write_file", {"path": str(target), "text": "!", "append": True})
        self.assertEqual(result["bytes"], 1)
        self.assertEqual(target.read_text(encoding="utf-8"), "Привет!")
        await self.request("write_file", {"path": str(target), "text": "replacement"})
        self.assertEqual(target.read_bytes(), b"replacement")

    async def test_size_is_utf8_bytes_and_boundary_is_allowed(self):
        target = self.root / "result.txt"
        result = await self.request("write_file", {"path": str(target), "text": "я" * (MAX_WRITE_BYTES // 2)})
        self.assertEqual(result["bytes"], MAX_WRITE_BYTES)
        result = await self.request("write_file", {"path": str(target), "text": "я" * (MAX_WRITE_BYTES // 2 + 1)})
        self.assertEqual(result["error"]["code"], "content_too_large")
        self.assertEqual(target.stat().st_size, MAX_WRITE_BYTES)

    async def test_rejects_credentials_without_creating_parents(self):
        for name in (".env", ".env.local", "auth", "auth.json", "auth/token", ".ssh/config", "ssh/key", "bot_api.txt", "ID_RSA", "credentials.json"):
            with self.subTest(name=name):
                target = self.root / "uncreated" / name
                result = await self.request("write_file", {"path": str(target), "text": "x"})
                self.assertEqual(result["error"]["code"], "credential_path_denied")
                self.assertFalse(target.exists())
        self.assertFalse((self.root / "uncreated").exists())

    async def test_rejects_nonlocal_device_ads_and_relative_paths(self):
        names = ["relative.txt", "C:relative", r"\\server\share\file", "//server/share/file", r"\\?\C:\file", r"\\.\NUL", r"C:\temp\file:stream", r"C:\temp\NUL.txt", r"C:\temp\COM1", r"C:\temp\file. ", "bad\x00path"]
        for name in names:
            with self.subTest(name=name):
                result = await self.request("write_file", {"path": name, "text": "x"})
                self.assertEqual(result["error"]["code"], "invalid_path")

    async def test_symlink_to_credentials_is_denied(self):
        secret = self.root / ".env"
        secret.write_text("original", encoding="utf-8")
        alias = self.root / "alias.txt"
        try:
            alias.symlink_to(secret)
        except OSError:
            self.skipTest("Symlinks unavailable for current user")
        result = await self.request("write_file", {"path": str(alias), "text": "overwrite"})
        self.assertEqual(result["error"]["code"], "credential_path_denied")
        self.assertEqual(secret.read_text(encoding="utf-8"), "original")

    async def test_invalid_requests_and_sanitized_exceptions(self):
        for kind, args in [("unknown", {}), ("write_file", []), ("write_file", {"text": 1}), ("write_file", {"text": "x", "append": 1})]:
            self.assertFalse((await self.request(kind, args))["ok"])
        with patch("personal_agent.pc_tools.Path.open", side_effect=OSError("Bearer secret-diagnostic")):
            result = await self.request("write_file", {"path": str(self.root / "file.txt"), "text": "x"})
        self.assertEqual(result["error"]["code"], "execution_failed")
        self.assertNotIn("secret-diagnostic", str(result))

    async def test_invalid_command_arguments_never_spawn(self):
        with patch("personal_agent.pc_tools.asyncio.create_subprocess_exec", new_callable=AsyncMock) as spawn:
            for timeout in [0, -1, 121, True, "30", float("nan"), float("inf")]:
                result = await self.request("run_command", {"command": "echo test", "timeout_seconds": timeout})
                self.assertEqual(result["error"]["code"], "invalid_arguments")
            for command in [None, "", "  ", "a\x00b"]:
                self.assertFalse((await self.request("run_command", {"command": command}))["ok"])
            result = await self.request("run_command", {"command": "echo test", "cwd": "relative"})
            self.assertEqual(result["error"]["code"], "invalid_path")
            spawn.assert_not_awaited()

    @unittest.skipUnless(os.name == "nt", "Windows subprocess contract")
    async def test_command_argv_defaults_redaction_and_nonzero_exit(self):
        token = b"123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_123456789"
        process = DummyProcess(b"hello " + token, b"Authorization: Bearer secret-value", returncode=7)
        command = "'literal $value `text'; # user suffix"
        environment = {"TELEGRAM_BOT_TOKEN": "test-token", "OPENAI_API_KEY": "test-openai", "codex_api_key": "test-codex", "PATH": "test-path", "TEMP": str(self.root)}
        with patch.dict(os.environ, environment, clear=True), patch("personal_agent.pc_tools.asyncio.create_subprocess_exec", AsyncMock(return_value=process)) as spawn, patch("personal_agent.pc_tools.media.terminate_tree", new_callable=AsyncMock) as stop:
            result = await self.request("run_command", {"command": command})
        args, kwargs = spawn.call_args
        self.assertEqual(args, ("powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                               "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
                               "$OutputEncoding = [Console]::OutputEncoding;\n" + command))
        self.assertEqual(kwargs["env"], {"PATH": "test-path", "TEMP": str(self.root)})
        self.assertEqual(kwargs["creationflags"], subprocess.CREATE_NO_WINDOW)
        self.assertEqual(kwargs["cwd"], str(self.root))
        self.assertEqual(kwargs["stdin"], asyncio.subprocess.DEVNULL)
        self.assertEqual(result["error"]["code"], "command_failed")
        self.assertEqual(result["returncode"], 7)
        self.assertNotIn(token.decode(), str(result))
        self.assertNotIn("secret-value", str(result))
        stop.assert_awaited_once_with(process)

    @unittest.skipUnless(os.name == "nt", "Windows subprocess contract")
    async def test_output_bound_and_private_key_redaction(self):
        for process in [DummyProcess(b"x" * 100000, b"y" * 100000), DummyProcess(b"-----BEGIN RSA PRIVATE KEY-----\nsensitive-key\n")]:
            with patch("personal_agent.pc_tools.asyncio.create_subprocess_exec", AsyncMock(return_value=process)), patch("personal_agent.pc_tools.media.terminate_tree", new_callable=AsyncMock):
                result = await self.request("run_command", {"command": "dummy"})
            self.assertTrue(result["ok"])
            self.assertLessEqual(len(result["stdout"].encode()) + len(result["stderr"].encode()), MAX_OUTPUT_BYTES)
            self.assertNotIn("sensitive-key", str(result))
        self.assertIn("REDACTED", result["stdout"])

    @unittest.skipUnless(os.name == "nt", "Windows subprocess contract")
    async def test_timeout_and_cancellation_terminate_owned_tree(self):
        process = DummyProcess(blocked=True)
        with patch("personal_agent.pc_tools.asyncio.create_subprocess_exec", AsyncMock(return_value=process)), patch("personal_agent.pc_tools.media.terminate_tree", new_callable=AsyncMock) as stop:
            result = await self.request("run_command", {"command": "dummy", "timeout_seconds": 0.01})
            self.assertEqual(result["error"]["code"], "timeout")
            stop.assert_awaited_once_with(process)
        process = DummyProcess(blocked=True)
        with patch("personal_agent.pc_tools.asyncio.create_subprocess_exec", AsyncMock(return_value=process)), patch("personal_agent.pc_tools.media.terminate_tree", new_callable=AsyncMock) as stop:
            task = asyncio.create_task(self.request("run_command", {"command": "dummy"}))
            await process.waiting.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            stop.assert_awaited_once_with(process)

    @unittest.skipUnless(os.name == "nt", "Windows subprocess contract")
    async def test_cancel_during_spawn_still_cleans_up(self):
        entered, release = asyncio.Event(), asyncio.Event()
        process = DummyProcess(blocked=True)

        async def spawn(*args, **kwargs):
            entered.set()
            await release.wait()
            return process

        with patch("personal_agent.pc_tools.asyncio.create_subprocess_exec", spawn), patch("personal_agent.pc_tools.media.terminate_tree", new_callable=AsyncMock) as stop:
            task = asyncio.create_task(self.request("run_command", {"command": "dummy"}))
            await entered.wait()
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            stop.assert_awaited_once_with(process)

    @unittest.skipUnless(os.name == "nt", "Harmless Windows temp-file integration")
    async def test_live_powershell_only_writes_temporary_file(self):
        command = "[IO.File]::WriteAllText((Join-Path (Get-Location) 'harmless.txt'), 'temporary'); [Console]::Write('Готово'); [Console]::Error.Write('Проверка')"
        result = await self.request("run_command", {"command": command, "cwd": str(self.root)})
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["stdout"], "Готово")
        self.assertEqual(result["stderr"], "Проверка")
        self.assertEqual((self.root / "harmless.txt").read_text(), "temporary")


if __name__ == "__main__":
    unittest.main()
