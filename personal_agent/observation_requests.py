"""Explicit, bounded, read-only observation requests.

Parsing is intentionally deterministic.  Callers are responsible for invoking
it only for the current authorized user's Telegram message.
"""
from __future__ import annotations

import codecs
import fnmatch
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from itertools import islice
from pathlib import Path, PureWindowsPath
from typing import Literal


MAX_DIRECTORY_ENTRIES = 100
MAX_RESULT_BYTES = 32 * 1024
MAX_FILE_BYTES = 32 * 1024  # default chunk of one read
MAX_CHUNK_BYTES = 64 * 1024  # hard cap of one read
FIND_MAX_DEPTH = 8
FIND_MAX_ENTRIES = 20_000
FIND_MAX_SECONDS = 8.0
FIND_MAX_RESULTS = 200
_FIND_SKIPPED_DIRS = {"node_modules", ".git", "__pycache__", "$recycle.bin", "system volume information"}


@dataclass(frozen=True, slots=True)
class ObservationRequest:
    kind: Literal["list_directory", "read_file", "find_files", "screenshot", "read_active_window",
                  "read_browser_page"]
    path: str | None = None
    offset: int = 0  # read_file: byte offset
    max_bytes: int = MAX_FILE_BYTES  # read_file: chunk size
    pattern: str | None = None  # find_files: name glob (a bare word means "name contains")
    max_results: int = 50  # find_files


_SLASH = re.compile(r"^/(ls|read|screenshot)(?:\s+(.+?))?$", re.IGNORECASE)


def _clean_argument(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value or None


def parse_slash_command(text: str) -> ObservationRequest | None:
    """Recognize only the exact deterministic commands /ls <path>, /read <path> and /screenshot.

    Free text is never parsed here: choosing a tool for a natural-language request is the
    model's job (native function calling), not a phrase recognizer's.
    """
    if not isinstance(text, str):
        return None
    match = _SLASH.fullmatch(text.strip())
    if not match:
        return None
    command, argument = match.groups()
    command = command.casefold()
    if command == "screenshot":
        return ObservationRequest("screenshot") if argument is None else None
    return ObservationRequest("list_directory" if command == "ls" else "read_file", _clean_argument(argument))


def _desktop_path() -> Path:
    """Return the shell Desktop location, including OneDrive/redirection."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class GUID(ctypes.Structure):
                _fields_ = [
                    ("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8),
                ]

            desktop = GUID(
                0xB4BFCC3A,
                0xDB2C,
                0x424C,
                (ctypes.c_ubyte * 8)(0xB0, 0x29, 0x7F, 0xE9, 0x9A, 0x87, 0xC6, 0x41),
            )
            out = ctypes.c_wchar_p()
            shell32 = ctypes.WinDLL("shell32", use_last_error=True)
            ole32 = ctypes.WinDLL("ole32", use_last_error=True)
            shell32.SHGetKnownFolderPath.argtypes = [ctypes.POINTER(GUID), wintypes.DWORD, wintypes.HANDLE, ctypes.POINTER(ctypes.c_wchar_p)]
            shell32.SHGetKnownFolderPath.restype = ctypes.c_long
            result = shell32.SHGetKnownFolderPath(ctypes.byref(desktop), 0, None, ctypes.byref(out))
            if result == 0 and out.value:
                value = out.value
                ole32.CoTaskMemFree(out)
                return Path(value)
            if out:
                ole32.CoTaskMemFree(out)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return Path.home() / "Desktop"


def known_paths(data_dir: Path, project_roots: dict[str, Path] | None = None) -> dict[str, str]:
    """Build user-facing path aliases without reading their contents."""
    home = Path.home()
    aliases = {
        "desktop": str(_desktop_path()),
        "home": str(home),
        "documents": str(home / "Documents"),
        "downloads": str(home / "Downloads"),
        "memory": str(Path(data_dir).resolve() / "memory"),
    }
    agents = home / ".codex" / "AGENTS.md"
    if agents.is_file():
        aliases["agents"] = str(agents)
    if project_roots:
        aliases.update({str(name).casefold(): str(Path(path)) for name, path in project_roots.items()})
    return aliases


def _provenance(kind: str, path: str | None) -> dict[str, object]:
    return {
        "source": "local_filesystem",
        "operation": kind,
        "path": path,
        "read_only": True,
    }


def _result_error(request: ObservationRequest, code: str, message: str, *, denied: bool = False) -> dict:
    return {
        "ok": False,
        "kind": request.kind,
        "path": request.path,
        "error": {"code": code, "message": message},
        "denied": denied,
        "provenance": _provenance(request.kind, request.path),
    }


def _local_windows_path(raw: str) -> Path:
    value = os.path.expandvars(os.path.expanduser(raw.strip()))
    win = PureWindowsPath(value)
    if value.startswith(("\\\\", "//")) or win.drive.startswith("\\\\"):
        raise ValueError("UNC/network paths are not allowed")
    if value.startswith(("\\\\?\\", "\\\\.\\")):
        raise ValueError("Windows device paths are not allowed")
    if not re.fullmatch(r"[A-Za-z]:", win.drive) or not win.root:
        raise ValueError("An absolute local drive path is required")
    return Path(os.path.abspath(value))


_SENSITIVE_PARTS = {".ssh", ".aws", ".azure", "credentials", "credential", "cookies", "auth"}
_SENSITIVE_NAMES = {
    ".env",
    "bot_api.txt",
    "credentials.json",
    "credential.json",
    "cookies.sqlite",
    "cookies.txt",
    "id_rsa",
    "id_ed25519",
    "token.json",
    "tokens.json",
    "auth.json",
    "secrets.json",
}
_SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def _credential_sensitive(path: Path) -> bool:
    parts = [part.casefold() for part in path.parts]
    name = path.name.casefold()
    if any(part in _SENSITIVE_PARTS for part in parts):
        return True
    if name in _SENSITIVE_NAMES or name.startswith(".env."):
        return True
    return path.suffix.casefold() in _SENSITIVE_SUFFIXES


def _entry_payload(entry: os.DirEntry[str]) -> dict[str, str]:
    try:
        if entry.is_symlink():
            kind = "symlink"
        elif entry.is_dir(follow_symlinks=False):
            kind = "directory"
        elif entry.is_file(follow_symlinks=False):
            kind = "file"
        else:
            kind = "other"
    except OSError:
        kind = "unknown"
    return {"name": entry.name, "type": kind}


def _list_directory(request: ObservationRequest, path: Path) -> dict:
    entries: list[dict[str, str]] = []
    used = 0
    truncated = False
    try:
        with os.scandir(path) as scan:
            sampled = list(islice(scan, MAX_DIRECTORY_ENTRIES + 1))
    except OSError as exc:
        return _result_error(request, "os_error", f"{type(exc).__name__}: {exc}")

    if len(sampled) > MAX_DIRECTORY_ENTRIES:
        sampled = sampled[:MAX_DIRECTORY_ENTRIES]
        truncated = True
    for entry in sampled:
        item = _entry_payload(entry)
        encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if used + len(encoded) > MAX_RESULT_BYTES:
            truncated = True
            break
        entries.append(item)
        used += len(encoded)

    return {
        "ok": True,
        "kind": request.kind,
        "path": str(path),
        "data": {"entries": entries, "count": len(entries), "truncated": truncated, "bytes": used},
        "provenance": _provenance(request.kind, str(path)),
    }


class TextChunkError(Exception):
    def __init__(self, code: str, message: str, *, denied: bool = False) -> None:
        self.code, self.message, self.denied = code, message, denied
        super().__init__(code)


_PRIVATE_KEY = re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_CREDENTIAL_TEXT = re.compile(
    r"\b\d{5,16}:[A-Za-z0-9_-]{20,}\b|\bsk-[A-Za-z0-9_-]{20,}\b|[\"'](?:access_token|refresh_token)[\"']\s*:")
_HEAD_BYTES = MAX_CHUNK_BYTES


def _detect_text_encoding(head: bytes, head_is_whole_file: bool) -> tuple[str, str, int]:
    """(codec, reported label, BOM length) from the file head; raises TextChunkError for binary data."""
    if head.startswith(codecs.BOM_UTF8):
        return "utf-8", "utf-8-bom", 3
    if head.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le", "utf-16le", 2
    if head.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be", "utf-16be", 2
    if b"\x00" in head:
        raise TextChunkError("binary_file", "Binary files are not returned.", denied=True)
    try:
        codecs.getincrementaldecoder("utf-8")("strict").decode(head, final=head_is_whole_file)
        return "utf-8", "utf-8", 0
    except UnicodeDecodeError:
        pass
    try:
        head.decode("cp1251")
    except UnicodeDecodeError:
        raise TextChunkError("non_utf8_file", "The file is not UTF-8, UTF-16 or Windows-1251 text.",
                             denied=True) from None
    return "cp1251", "cp1251", 0


def read_text_chunk(path: Path, offset: int = 0, max_bytes: int = MAX_FILE_BYTES) -> dict:
    """One bounded chunk of a text file from a byte offset, with encoding detection.

    The encoding is detected from the file head on every call so all chunks of one file are decoded
    consistently.  `next_offset` is the first byte that was not consumed and never points inside a
    character, so a caller can continue exactly where the previous chunk ended.
    """
    offset = max(0, int(offset))
    max_bytes = min(MAX_CHUNK_BYTES, max(1024, int(max_bytes)))
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        head = handle.read(_HEAD_BYTES)
        codec, label, bom = _detect_text_encoding(head, size <= _HEAD_BYTES)
        start = max(offset, bom)
        if codec.startswith("utf-16") and (start - bom) % 2:
            start += 1
        if start >= size:
            return {"text": "", "bytes": 0, "offset": min(offset, size), "next_offset": size,
                    "total_bytes": size, "eof": True, "truncated": False, "encoding": label}
        if codec == "utf-8" and start > bom:
            handle.seek(start)
            for _ in range(3):  # an offset inside a multi-byte character: move to the next boundary
                byte = handle.read(1)
                if not byte or not 0x80 <= byte[0] <= 0xBF:
                    break
                start += 1
        scan_from = max(0, start - 80)  # overlap so a marker split by a chunk boundary is still seen
        handle.seek(scan_from)
        prefix = handle.read(start - scan_from)
        raw = handle.read(max_bytes)
        eof = not handle.read(1)
    if not codec.startswith("utf-16"):
        if b"\x00" in raw:
            raise TextChunkError("binary_file", "Binary files are not returned.", denied=True)
        if _PRIVATE_KEY.search(prefix + raw):
            raise TextChunkError("private_key_denied", "Private-key material was detected and withheld.",
                                 denied=True)
    replaced = False
    decoder = codecs.getincrementaldecoder(codec)("strict")
    try:
        text = decoder.decode(raw, final=eof)
    except UnicodeDecodeError:
        decoder = codecs.getincrementaldecoder(codec)("replace")
        text = decoder.decode(raw, final=eof)
        replaced = True
    consumed = len(raw) - len(decoder.getstate()[0])
    if "\x00" in text:
        raise TextChunkError("binary_file", "Binary files are not returned.", denied=True)
    if codec.startswith("utf-16") and _PRIVATE_KEY.search(text.encode("utf-8", "replace")):
        raise TextChunkError("private_key_denied", "Private-key material was detected and withheld.",
                             denied=True)
    if _CREDENTIAL_TEXT.search(text):
        raise TextChunkError("credential_content_denied", "Credential content was detected and withheld.",
                             denied=True)
    result = {"text": text, "bytes": consumed, "offset": start, "next_offset": start + consumed,
              "total_bytes": size, "eof": eof, "truncated": not eof, "encoding": label}
    if replaced:
        result["invalid_bytes_replaced"] = True
    return result


def _read_file(request: ObservationRequest, path: Path) -> dict:
    if _credential_sensitive(path):
        return _result_error(
            request,
            "credential_path_denied",
            "Credential-sensitive paths are denied and their contents are not returned.",
            denied=True,
        )
    try:
        data = read_text_chunk(path, request.offset, request.max_bytes)
    except TextChunkError as exc:
        return _result_error(request, exc.code, exc.message, denied=exc.denied)
    except OSError as exc:
        return _result_error(request, "os_error", f"{type(exc).__name__}: {exc}")
    return {
        "ok": True,
        "kind": request.kind,
        "path": str(path),
        "data": data,
        "provenance": _provenance(request.kind, str(path)),
    }


def _reparse_point(entry: os.DirEntry[str]) -> bool:
    try:
        if entry.is_symlink():
            return True
        return bool(getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0) & 0x400)
    except OSError:
        return True


def normalize_name_pattern(pattern: object) -> str:
    if (not isinstance(pattern, str) or not pattern.strip() or len(pattern) > 100
            or any(c in pattern for c in ("\x00", "\\", "/", ":"))):
        raise ValueError("invalid_pattern")
    pattern = pattern.strip()
    return pattern if any(c in pattern for c in "*?[") else f"*{pattern}*"


def _find_files(request: ObservationRequest, root: Path) -> dict:
    try:
        pattern = normalize_name_pattern(request.pattern)
    except ValueError:
        return _result_error(request, "invalid_pattern",
                             "A file name pattern without path separators is required.")
    limit = min(FIND_MAX_RESULTS, max(1, int(request.max_results)))
    if not root.is_dir():
        return _result_error(request, "not_a_directory", "The search root is not a directory.")
    folded = pattern.casefold()
    matches: list[dict] = []
    queue = deque([(root, 0)])
    scanned, reason = 0, None
    deadline = time.monotonic() + FIND_MAX_SECONDS
    while queue and reason is None:
        directory, depth = queue.popleft()
        try:
            with os.scandir(directory) as scan:
                for entry in scan:
                    scanned += 1
                    if scanned > FIND_MAX_ENTRIES:
                        reason = "entry_limit"
                        break
                    if scanned % 256 == 0 and time.monotonic() > deadline:
                        reason = "time_limit"
                        break
                    if _reparse_point(entry):
                        continue
                    child = Path(entry.path)
                    if _credential_sensitive(child):
                        continue
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if fnmatch.fnmatchcase(entry.name.casefold(), folded):
                        item = {"path": str(child), "type": "directory" if is_dir else "file"}
                        if not is_dir:
                            try:
                                item["size"] = entry.stat(follow_symlinks=False).st_size
                            except OSError:
                                pass
                        matches.append(item)
                        if len(matches) >= limit:
                            reason = "result_limit"
                            break
                    if is_dir and depth < FIND_MAX_DEPTH and entry.name.casefold() not in _FIND_SKIPPED_DIRS:
                        queue.append((child, depth + 1))
        except OSError:
            continue
    return {
        "ok": True,
        "kind": request.kind,
        "path": str(root),
        "data": {"matches": matches, "count": len(matches), "pattern": pattern, "truncated": reason is not None,
                 "truncated_reason": reason, "scanned_entries": scanned, "max_depth": FIND_MAX_DEPTH},
        "provenance": _provenance(request.kind, str(root)),
    }


def execute_read_request(request: ObservationRequest) -> dict:
    """Execute bounded directory/file reads; screenshots use capture_screenshot."""
    if not isinstance(request, ObservationRequest):
        raise TypeError("request must be ObservationRequest")
    if request.kind == "screenshot":
        return _result_error(request, "wrong_executor", "Use capture_screenshot(root) for screenshot requests.")
    if request.kind == "read_active_window":
        return _result_error(request, "wrong_executor", "Use the Windows UI Automation observer for active-window requests.")
    if request.path is None:
        return {
            "ok": False,
            "kind": request.kind,
            "path": None,
            "needs_path": True,
            "error": {"code": "needs_path", "message": "An absolute local path is required."},
            "provenance": _provenance(request.kind, None),
        }
    try:
        path = _local_windows_path(request.path)
    except ValueError as exc:
        return _result_error(request, "invalid_path", str(exc), denied=True)
    if request.kind == "list_directory":
        return _list_directory(request, path)
    if request.kind == "read_file":
        return _read_file(request, path)
    if request.kind == "find_files":
        return _find_files(request, path)
    raise ValueError(f"Unsupported observation request kind: {request.kind}")
