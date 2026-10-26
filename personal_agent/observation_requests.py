"""Explicit, bounded, read-only observation requests.

Parsing is intentionally deterministic.  Callers are responsible for invoking
it only for the current authorized user's Telegram message.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path, PureWindowsPath
from typing import Literal


MAX_DIRECTORY_ENTRIES = 100
MAX_RESULT_BYTES = 32 * 1024
MAX_FILE_BYTES = 32 * 1024


@dataclass(frozen=True, slots=True)
class ObservationRequest:
    kind: Literal["list_directory", "read_file", "screenshot"]
    path: str | None = None


_SLASH = re.compile(r"^\s*/(ls|read|screenshot)(?:\s+(.+?))?\s*$", re.IGNORECASE)
_LIST_RU = re.compile(
    r"^\s*(?:просмотри|покажи)(?:\s+мне)?(?:\s+содержимое)?\s+папк(?:и|у)(?:\s+(.+?))?[.!?]?\s*$",
    re.IGNORECASE,
)
_SHOT_TARGET = r"(?:скрин(?:шот)?(?: экрана)?|снимок(?: экрана)?|экран)"
_SCREENSHOT_RU = re.compile(
    r"^\s*(?:не\s+видно\s*[-—,:]?\s*)?(?:а\s+)?(?:пожалуйста[,\s]+)?(?:можешь\s+)?"
    rf"(?:(?:сделай|сделать|сними|снять)\s+(?:мне\s+)?{_SHOT_TARGET}"
    r"|(?:покажи|показать)\s+(?:мне\s+)?экран)[.!?]?\s*$",
    re.IGNORECASE,
)
_NEGATED = re.compile(
    r"\b(?:не\s+(?:делай|делать|сделай|сделать|надо|нужно|хочу)|без\s+скриншот|не\s+показывай|не\s+просматривай)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL = re.compile(
    r"\b(?:если\s+бы|если\s+я|допустим|представь|например|что\s+будет\s+если|как\s+(?:попросить|написать))\b",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"[`\"'«»]")

_DIRECT_START = r"(?:сделай|сделать|покажи|посмотри|посмотреть|просмотри|выведи|прочитай|прочесть|давай\s+про|/)"
_DIRECT_SPLIT = re.compile(
    rf"(?:\r?\n|[.!?]+\s+|\s*,\s*(?=(?:(?:и|а|но)\s+)?{_DIRECT_START})|\s+(?=(?:и|а|но)\s+{_DIRECT_START}))",
    re.IGNORECASE,
)
_DIRECT_LEAD = r"(?:и\s+|а\s+|но\s+)?(?:пожалуйста[,\s]+)?(?:можешь\s+)?"
_LIST_DIRECT = re.compile(
    rf"^\s*{_DIRECT_LEAD}(?:без\s+команд\s+)?(?:покажи|посмотри|посмотреть|просмотри|выведи)"
    r"(?:\s+мне)?\s+(?:(?:список|содержимое)\s+)?(?:файл(?:ы|ов)?|папк(?:у|и)|содержимое)\b(?P<tail>.*)$",
    re.IGNORECASE,
)
_READ_DIRECT = re.compile(
    rf"^\s*{_DIRECT_LEAD}(?:прочитай|прочесть|открой|покажи)(?:\s+мне)?(?:\s+(?:файл|документ))?\s+(?P<tail>.+?)\s*$",
    re.IGNORECASE,
)
_ABOUT_DIRECT = re.compile(
    rf"^\s*{_DIRECT_LEAD}давай\s+про\s+(?P<tail>.+?)\s*$",
    re.IGNORECASE,
)
_SCREEN_DIRECT = re.compile(
    rf"^\s*{_DIRECT_LEAD}(?:(?:сделай|сделать|сними|снять)\s+(?:мне\s+)?{_SHOT_TARGET}"
    r"|(?:покажи|показать)\s+(?:мне\s+)?экран)"
    r"(?:\s*(?:,\s*|\s+)(?:и\s+)?(?:скажи|опиши|объясни|расскажи)\b[^\r\n]*)?\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_WHAT_PREFIX = (
    r"^\s*(?:и\s+)?(?:пожалуйста[,\s]+)?(?:можешь\s+)?"
    r"(?:(?:покажи|посмотри|посмотреть|просмотри|скажи)(?:,\s*)?)?"
)
_WHAT_DESKTOP = re.compile(
    _WHAT_PREFIX
    + r"(?:что|какие\s+файлы)\s+(?:(?:у\s+(?:меня|тебя)|там|сейчас)\s+)*"
    r"(?:(?:находится|лежит|есть|видно)\s+)?(?:на\s+)?(?:рабочем\s+столе|desktop)\s*[?.!]*$",
    re.IGNORECASE,
)
_WHAT_FOLDER = re.compile(
    _WHAT_PREFIX
    + r"(?:что|какие\s+файлы)\s+(?:(?:у\s+(?:меня|тебя)|там|сейчас)\s+)*"
    r"(?:(?:находится|лежит|есть|видно)\s+)?"
    r"в\s+(?:папке|директории|каталоге)\s+(?P<tail>.+?)\s*[?.!]*$",
    re.IGNORECASE,
)
_WHAT_FILE = re.compile(
    _WHAT_PREFIX
    + r"(?:что|какие\s+данные)\s+(?:(?:у\s+(?:меня|тебя)|там|сейчас)\s+)*"
    r"(?:(?:написано|находится|лежит|есть|видно)\s+)?"
    r"в\s+(?:файле|документе)\s+(?P<tail>.+?)\s*[?.!]*$",
    re.IGNORECASE,
)
_ABSOLUTE_QUOTED_PATH = re.compile(r"(?P<quote>[\"'])(?P<path>[A-Za-z]:\\[^\"']+)(?P=quote)")
_ABSOLUTE_PATH = re.compile(r"(?P<path>[A-Za-z]:\\.+?)\s*$")

_ALIAS_TERMS: dict[str, tuple[str, ...]] = {
    "desktop": ("рабочем столе", "рабочий стол", "рабочего стола", "desktop"),
    "home": ("домашней папке", "домашнюю папку", "домашняя папка", "home"),
    "documents": ("документах", "документы", "documents"),
    "downloads": ("загрузках", "загрузки", "downloads"),
    "memory": ("памяти", "память", "memory"),
    "agents": ("agents.md", "agents"),
}


def _clean_argument(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value or None


def parse_observation_request(text: str) -> ObservationRequest | None:
    """Parse only direct, explicit observation requests.

    Mentioned/quoted commands, negations and common hypothetical forms are
    rejected conservatively so ordinary model/instruction text cannot look like
    an observation request accidentally.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    stripped = text.strip()
    if _NEGATED.search(stripped) or _HYPOTHETICAL.search(stripped):
        return None

    slash = _SLASH.fullmatch(stripped)
    if slash:
        command, argument = slash.groups()
        command = command.casefold()
        if command == "screenshot":
            return ObservationRequest("screenshot") if argument is None else None
        return ObservationRequest(
            "list_directory" if command == "ls" else "read_file",
            _clean_argument(argument),
        )

    # Quotation marks in natural-language text usually mean the user is
    # discussing a command rather than issuing it.  Slash commands above may
    # still quote a path containing spaces.
    if _QUOTED.search(stripped):
        return None

    if _SCREENSHOT_RU.fullmatch(stripped):
        return ObservationRequest("screenshot")

    listing = _LIST_RU.fullmatch(stripped)
    if listing:
        return ObservationRequest("list_directory", _clean_argument(listing.group(1)))
    return None


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


def _alias_path(text: str, aliases: dict[str, str]) -> str | None:
    folded = text.casefold()
    for key, terms in _ALIAS_TERMS.items():
        path = aliases.get(key)
        if path and any(term in folded for term in terms):
            return path
    for key, path in aliases.items():
        if key in _ALIAS_TERMS:
            continue
        if re.search(rf"(?<![\w.-]){re.escape(key.casefold())}(?![\w.-])", folded):
            return path
    return None


def _explicit_path(text: str) -> str | None:
    quoted = _ABSOLUTE_QUOTED_PATH.search(text)
    if quoted:
        return quoted.group("path").strip()
    match = _ABSOLUTE_PATH.search(text)
    if not match:
        return None
    return match.group("path").strip().rstrip(".!?") or None


def _split_direct_clauses(text: str) -> list[str]:
    # Locate separators on a same-length mask so quoted logs/examples cannot
    # become executable clauses, while the original text (including quoted
    # paths) remains untouched for parsing.
    mask = list(text)
    quote: str | None = None
    closing = {"\"": "\"", "'": "'", "`": "`", "«": "»"}
    for index, char in enumerate(text):
        if quote is not None:
            mask[index] = " "
            if char == quote:
                quote = None
        elif char in closing:
            quote = closing[char]
            mask[index] = " "
    masked = "".join(mask)
    clauses: list[str] = []
    start = 0
    for match in _DIRECT_SPLIT.finditer(masked):
        clauses.append(text[start:match.start()])
        start = match.end()
    clauses.append(text[start:])
    return clauses


def _parse_direct_clause(clause: str, aliases: dict[str, str]) -> ObservationRequest | None:
    stripped = clause.strip()
    if not stripped or _HYPOTHETICAL.search(stripped):
        return None

    legacy = parse_observation_request(stripped)
    if legacy is not None:
        if legacy.path:
            path = _explicit_path(legacy.path) or _alias_path(legacy.path, aliases)
            if path:
                return ObservationRequest(legacy.kind, path)
        return legacy

    # Negation is scoped to this clause. "без команд" is an execution-style
    # preference, while "без скриншота" suppresses capture only.
    folded = stripped.casefold()
    if re.search(r"\bне\s+(?:делай|делать|сделай|сделать|показывай|просматривай)\b", folded):
        return None

    if _SCREEN_DIRECT.fullmatch(stripped):
        return ObservationRequest("screenshot")

    about = _ABOUT_DIRECT.fullmatch(stripped)
    if about:
        target = about.group("tail")
        path = _explicit_path(target) or _alias_path(target, aliases)
        if path:
            return ObservationRequest("read_file", path)
        return None

    listing = _LIST_DIRECT.fullmatch(stripped)
    if listing:
        target = listing.group("tail")
        path = _explicit_path(target) or _alias_path(target, aliases)
        if path:
            return ObservationRequest("list_directory", path)
        return None

    reading = _READ_DIRECT.fullmatch(stripped)
    if reading:
        target = reading.group("tail")
        path = _explicit_path(target) or _alias_path(target, aliases)
        if path:
            if os.path.isdir(path) or path == aliases.get("memory"):
                return ObservationRequest("list_directory", path)
            return ObservationRequest("read_file", path)

    # Natural questions such as "что у меня на рабочем столе?" are still
    # direct requests for current local evidence, not commands found in it.
    if _WHAT_DESKTOP.fullmatch(stripped):
        path = aliases.get("desktop")
        if path:
            return ObservationRequest("list_directory", path)

    for pattern in (_WHAT_FOLDER, _WHAT_FILE):
        match = pattern.fullmatch(stripped)
        if not match:
            continue
        target = match.group("tail")
        path = _explicit_path(target) or _alias_path(target, aliases)
        if path:
            kind = "list_directory" if pattern is _WHAT_FOLDER else "read_file"
            if kind == "read_file" and os.path.isdir(path):
                kind = "list_directory"
            return ObservationRequest(kind, path)
    return None


def parse_observation_requests(
    text: str,
    *,
    aliases: dict[str, str] | None = None,
    last_path: str | None = None,
) -> list[ObservationRequest]:
    """Parse up to three direct read-only observation directives from a message."""
    if not isinstance(text, str) or not text.strip():
        return []

    resolved_aliases = {str(key).casefold(): str(value) for key, value in (aliases or {}).items()}
    requests: list[ObservationRequest] = []
    seen: set[ObservationRequest] = set()

    for clause in _split_direct_clauses(text.strip()):
        request = _parse_direct_clause(clause, resolved_aliases)
        if request is None:
            continue
        if request.kind == "list_directory" and request.path is None:
            # Missing arguments are not permission to guess a different folder.
            if last_path and re.search(r"эт[уаой]+\s+папк|там", clause, re.I):
                request = ObservationRequest("list_directory", last_path)
        if request not in seen:
            seen.add(request)
            requests.append(request)
        if len(requests) == 3:
            break
    return requests


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


def _read_file(request: ObservationRequest, path: Path) -> dict:
    if _credential_sensitive(path):
        return _result_error(
            request,
            "credential_path_denied",
            "Credential-sensitive paths are denied and their contents are not returned.",
            denied=True,
        )
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        return _result_error(request, "os_error", f"{type(exc).__name__}: {exc}")

    truncated = len(payload) > MAX_FILE_BYTES
    payload = payload[:MAX_FILE_BYTES]
    if b"\x00" in payload:
        return _result_error(request, "binary_file", "Binary files are not returned.", denied=True)
    try:
        # A byte limit may split the last UTF-8 character; withhold malformed
        # files but allow a valid truncated prefix.
        import codecs
        text = codecs.getincrementaldecoder("utf-8")("strict").decode(payload, final=not truncated)
    except UnicodeDecodeError:
        return _result_error(request, "non_utf8_file", "Only UTF-8 text files are returned.", denied=True)
    if re.search(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", payload):
        return _result_error(request, "private_key_denied", "Private-key material was detected and withheld.", denied=True)
    if re.search(r"\b\d{5,16}:[A-Za-z0-9_-]{20,}\b|\bsk-[A-Za-z0-9_-]{20,}\b|[\"'](?:access_token|refresh_token)[\"']\s*:", text):
        return _result_error(request, "credential_content_denied", "Credential content was detected and withheld.", denied=True)

    return {
        "ok": True,
        "kind": request.kind,
        "path": str(path),
        "data": {"text": text, "bytes": len(payload), "truncated": truncated},
        "provenance": _provenance(request.kind, str(path)),
    }


def execute_read_request(request: ObservationRequest) -> dict:
    """Execute bounded directory/file reads; screenshots use capture_screenshot."""
    if not isinstance(request, ObservationRequest):
        raise TypeError("request must be ObservationRequest")
    if request.kind == "screenshot":
        return _result_error(request, "wrong_executor", "Use capture_screenshot(root) for screenshot requests.")
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
    raise ValueError(f"Unsupported observation request kind: {request.kind}")
