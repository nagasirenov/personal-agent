"""Strict broker for model-proposed, host-executed read-only observations."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path, PureWindowsPath

from .observation_requests import ObservationRequest


_MAX_REQUESTS = 3
_KINDS = {"list_directory", "read_file", "find_files", "screenshot", "read_active_window", "read_browser_page"}
_SENSITIVE_PARTS = {".ssh", ".aws", ".azure", "credentials", "credential", "cookies", "auth"}
_SENSITIVE_NAMES = {
    ".env", "credentials.json", "credential.json", "token.json", "tokens.json",
    "auth.json", "secrets.json", "cookies.txt", "cookies.sqlite", "id_rsa", "id_ed25519", "bot_api.txt",
}
_SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


@dataclass(frozen=True, slots=True)
class ObservationScope:
    roots: tuple[Path, ...] = ()
    allow_screenshot: bool = False
    max_requests: int = 3
    allow_reparse: bool = False
    allow_active_window: bool = False


def local_drive_roots() -> tuple[Path, ...]:
    """Read capabilities cover local volumes accessible to the daemon's OS user."""
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
        mask = kernel.GetLogicalDrives()
        roots = tuple(Path(f"{chr(65 + i)}:\\") for i in range(26)
                      if mask & (1 << i) and kernel.GetDriveTypeW(f"{chr(65 + i)}:\\") in {2, 3, 5, 6})
        if roots:
            return roots
    return (Path(Path.home().anchor),)


class ObservationDenied(Exception):
    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(public_message)


def _deny(code: str, message: str) -> None:
    raise ObservationDenied(code, message)


def _local_windows_path(raw: str) -> Path:
    if any(token in raw for token in ("\0", "\r", "\n", "|", "&", ";", "<", ">", "`", "$(")):
        _deny("unsafe_path", "Observation path contains unsafe characters.")
    if raw.startswith(("\\\\", "//", "\\\\?\\", "\\\\.\\")):
        _deny("network_or_device_path", "Network and device paths are not allowed.")
    win = PureWindowsPath(raw)
    if not re.fullmatch(r"[A-Za-z]:", win.drive) or not win.root:
        _deny("relative_path", "Observation path must be an absolute local Windows path.")
    if ":" in raw[2:]:
        _deny("unsafe_path", "Alternate data streams are not allowed.")
    return Path(os.path.abspath(raw))


def _sensitive(path: Path) -> bool:
    parts = [part.casefold() for part in path.parts]
    name = path.name.casefold()
    return (
        any(part in _SENSITIVE_PARTS for part in parts)
        or name in _SENSITIVE_NAMES
        or name.startswith(".env.")
        or path.suffix.casefold() in _SENSITIVE_SUFFIXES
    )


def _normalized(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.realpath(path)))


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    target = _normalized(path)
    for root in roots:
        base = _normalized(root)
        try:
            if os.path.commonpath((target, base)) == base:
                return True
        except ValueError:
            continue
    return False


def _has_reparse(path: Path, roots: tuple[Path, ...]) -> bool:
    junction = getattr(os.path, "isjunction", lambda _: False)
    root_norms = {_normalized(root) for root in roots}
    current = path
    while True:
        if current.exists() and (current.is_symlink() or junction(current)):
            return True
        if _normalized(current) in root_norms or current.parent == current:
            return False
        current = current.parent


def validate_observation_plan(
    requests: list[ObservationRequest], scope: ObservationScope
) -> list[ObservationRequest]:
    """Validate a proposed plan against only the current authenticated-request scope."""
    if not isinstance(scope, ObservationScope) or not isinstance(requests, list):
        _deny("malformed_plan", "Observation plan or scope is invalid.")
    budget = min(_MAX_REQUESTS, max(0, scope.max_requests))
    if len(requests) > budget:
        _deny("request_budget", "Observation plan exceeds the current request budget.")
    seen: set[tuple[str, str | None]] = set()
    for request in requests:
        if not isinstance(request, ObservationRequest) or request.kind not in _KINDS:
            _deny("unsupported_kind", "Only typed read-only observations are allowed.")
        key = (request.kind, os.path.normcase(request.path or ""))
        if key in seen:
            _deny("duplicate_request", "Duplicate observations are not allowed.")
        seen.add(key)
    validated: list[ObservationRequest] = []
    for request in requests:
        if request.kind == "read_browser_page":
            if request.path is not None:
                _deny("malformed_plan", "Browser page observations do not accept paths.")
            validated.append(request)
            continue
        if request.kind == "screenshot":
            if request.path is not None:
                _deny("malformed_plan", "Screenshot observations do not accept paths.")
            if not scope.allow_screenshot:
                _deny("screenshot_denied", "Screenshot access is not authorized by the current request.")
            validated.append(request)
            continue
        if request.kind == "read_active_window":
            if request.path is not None:
                _deny("malformed_plan", "Active-window observations do not accept paths.")
            if not scope.allow_active_window:
                _deny("active_window_denied", "Active-window text is not authorized by the current scope.")
            validated.append(request)
            continue
        if request.path is None:
            _deny("malformed_plan", "File observations require a path.")
        path = _local_windows_path(request.path)
        if _sensitive(path):
            _deny("credential_path_denied", "Credential-sensitive paths are not available to the model.")
        if (not scope.allow_reparse and _has_reparse(path, scope.roots)) or not _inside(path, scope.roots):
            _deny("path_outside_scope", "Observation path is outside the current trusted roots.")
        validated.append(replace(request, path=str(path)))
    return validated
