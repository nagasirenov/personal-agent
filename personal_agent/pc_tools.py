"""Bounded host actions. Caller authorization belongs to the supervisor."""
from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess

from . import media

MAX_OUTPUT_BYTES = 16 * 1024
MAX_WRITE_BYTES = 256 * 1024
_POWERSHELL_PREAMBLE = (
    "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
    "$OutputEncoding = [Console]::OutputEncoding;\n"
)
_PRIVATE_ENV_NAMES = {"TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"}


def _error(kind: str, code: str, message: str) -> dict:
    return {"ok": False, "kind": kind, "error": {"code": code, "message": message}}


def _check_path_text(value: str) -> None:
    windows = PureWindowsPath(value)
    if value.startswith(("\\\\", "//")) or "\x00" in value:
        raise ValueError
    if os.name == "nt" or windows.drive:
        if not re.fullmatch(r"[A-Za-z]:", windows.drive) or not windows.is_absolute():
            raise ValueError
        for part in windows.parts[1:]:
            if ":" in part or part.endswith((" ", ".")) or re.search(r'[<>"|?*\x00-\x1f]', part):
                raise ValueError
            if re.fullmatch(r"(?:CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[0-9¹²³]|LPT[0-9¹²³])(?:\..*)?", part, re.I):
                raise ValueError
    elif not Path(value).is_absolute():
        raise ValueError


def _local_path(value: object) -> Path:
    if not isinstance(value, str) or not value or len(value) > 32767:
        raise ValueError
    _check_path_text(value)
    path = Path(value).resolve()
    _check_path_text(str(path))
    if os.name == "nt":
        import ctypes
        # Reject mapped network drives as well as UNC/reparse destinations.
        if ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor)) not in (2, 3, 5, 6):
            raise ValueError
    return path


def _credential_path(path: Path) -> bool:
    for part in path.parts:
        name = part.casefold()
        if (name in {"ssh", ".ssh", "auth", ".auth", "bot_api.txt", "credentials", ".aws", ".azure"}
                or name.startswith((".env", "auth.", "credentials.", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"))):
            return True
    return False


def _redact(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    # Withhold the entire stream if a key is present, including a truncated key.
    if re.search(r"-----BEGIN[^\r\n]*PRIVATE", text, re.I):
        return "[REDACTED PRIVATE KEY OUTPUT]"
    text = re.sub(r"\b\d{5,16}:[A-Za-z0-9_-]+", "[REDACTED TELEGRAM TOKEN]", text)
    text = re.sub(r"\bBearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", text, flags=re.I)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED API KEY]", text)
    return text.encode("utf-8")[:MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore")


async def _run_command(arguments: dict, data_dir: Path) -> dict:
    kind = "run_command"
    command = arguments.get("command")
    timeout = arguments.get("timeout_seconds", 30)
    if not isinstance(command, str) or not command.strip() or "\x00" in command or len(command.encode("utf-8")) > MAX_WRITE_BYTES:
        return _error(kind, "invalid_arguments", "A nonempty bounded command is required.")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        return _error(kind, "invalid_arguments", "timeout_seconds must be greater than zero and at most 120.")
    try:
        cwd = _local_path(arguments["cwd"]) if "cwd" in arguments else _local_path(str(data_dir))
        if not cwd.is_dir():
            raise ValueError
    except (OSError, ValueError, RuntimeError):
        return _error(kind, "invalid_path", "cwd must be an existing absolute local directory.")
    if os.name != "nt":
        return _error(kind, "unsupported_platform", "PowerShell host actions require Windows.")

    process = None
    readers = []
    buffers = [bytearray(), bytearray()]
    remaining = MAX_OUTPUT_BYTES
    truncated = False

    async def drain(stream, index):
        nonlocal remaining, truncated
        while chunk := await stream.read(4096):
            count = min(len(chunk), remaining)
            buffers[index].extend(chunk[:count])
            remaining -= count
            truncated |= count < len(chunk)

    try:
        # Shield startup so cancellation cannot lose ownership of a spawned child.
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _POWERSHELL_PREAMBLE + command,
            cwd=str(cwd), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env={key: value for key, value in os.environ.items() if key.upper() not in _PRIVATE_ENV_NAMES},
        ))
        try:
            process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            process = await spawn
            raise
        readers = [asyncio.create_task(drain(process.stdout, 0)), asyncio.create_task(drain(process.stderr, 1))]
        timed_out = False
        try:
            async with asyncio.timeout(timeout):
                await process.wait()
                await asyncio.gather(*readers)
        except TimeoutError:
            timed_out = True
        if timed_out:
            result = _error(kind, "timeout", "Command exceeded its timeout.")
        elif process.returncode:
            result = _error(kind, "command_failed", "Command exited with a nonzero status.")
        else:
            result = {"ok": True, "kind": kind}
        stdout, stderr = (_redact(bytes(buf)) for buf in buffers)
        # Redaction markers can expand short inputs; bound the combined result too.
        stdout_bytes = stdout.encode("utf-8")[:MAX_OUTPUT_BYTES]
        stderr_bytes = stderr.encode("utf-8")[:MAX_OUTPUT_BYTES - len(stdout_bytes)]
        result.update(stdout=stdout_bytes.decode("utf-8", errors="ignore"),
                      stderr=stderr_bytes.decode("utf-8", errors="ignore"),
                      returncode=process.returncode, truncated=truncated or len(stderr_bytes) < len(stderr.encode("utf-8")))
        return result
    finally:
        try:
            if process is not None:
                await asyncio.shield(media.terminate_tree(process))
        finally:
            for reader in readers:
                reader.cancel()
            if readers:
                await asyncio.gather(*readers, return_exceptions=True)


async def execute_pc_request(kind: str, arguments: dict, *, data_dir: Path) -> dict:
    """Execute an already-authorized action without returning raw exceptions.

    Cancellation propagates after owned-process cleanup. No identity checks or
    automatic credential discovery are performed here.
    """
    if not isinstance(kind, str):
        return _error("unknown", "invalid_arguments", "kind must be a string.")
    if not isinstance(arguments, dict):
        return _error(kind, "invalid_arguments", "arguments must be an object.")
    try:
        if kind == "run_command":
            return await _run_command(arguments, data_dir)
        if kind != "write_file":
            return _error(kind, "unsupported_kind", "Unsupported PC action.")
        text = arguments.get("text")
        append = arguments.get("append", False)
        if not isinstance(text, str) or type(append) is not bool:
            return _error(kind, "invalid_arguments", "text and a boolean append are required.")
        payload = text.encode("utf-8")
        if len(payload) > MAX_WRITE_BYTES:
            return _error(kind, "content_too_large", "UTF-8 content exceeds 256 KiB.")
        try:
            original = arguments.get("path")
            path = _local_path(original)
        except (OSError, ValueError, RuntimeError):
            return _error(kind, "invalid_path", "An absolute local file path is required.")
        if _credential_path(Path(original)) or _credential_path(path):
            return _error(kind, "credential_path_denied", "Credential-sensitive file paths are denied.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab" if append else "wb") as handle:
            count = handle.write(payload)
        return {"ok": True, "kind": kind, "path": str(path), "bytes": count, "append": append}
    except asyncio.CancelledError:
        raise
    except Exception:
        # Paths, command lines, OS/provider diagnostics can contain credentials.
        return _error(kind, "execution_failed", "PC action could not be completed.")
