"""Best-effort, read-only observation of local terminal process trees."""
from __future__ import annotations

import ctypes
import hashlib
import json
import re
import sys
from collections import defaultdict, deque
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable

from ..models import TimelineEvent
from ..observers import _safe_worker_path, _safe_worker_text

_MAX_PROCESSES = 8192
_MAX_SESSIONS = 64
_MAX_CHILDREN = 8
_MAX_CHILD_DETAILS = 3
_MAX_STALE = timedelta(seconds=15)
_HEARTBEAT = timedelta(seconds=60)

_SHELLS = {
    "powershell.exe", "pwsh.exe", "cmd.exe", "bash.exe", "wsl.exe", "ubuntu.exe",
    "zsh.exe", "fish.exe", "sh.exe", "git-bash.exe", "nu.exe",
}
_TERMINAL_APPS = {
    "windowsterminal.exe": "Windows Terminal",
    "wezterm-gui.exe": "WezTerm",
    "conemu64.exe": "ConEmu",
    "conemuc.exe": "ConEmu",
    "alacritty.exe": "Alacritty",
    "mintty.exe": "Mintty",
    "code.exe": "VS Code",
    "cursor.exe": "Cursor",
}
_TERMINAL_HOSTS = {"conhost.exe", "openconsole.exe"}
_SECRET_ARGUMENT = re.compile(
    r'(?i)(?<!\w)(--?(?:password|passwd|token|secret|api[-_]?key|access[-_]?key)\s+)(?:"[^"]*"|\'[^\']*\'|\S+)'
)


def _safe_command_line(value: str) -> str:
    """Bound and redact before a command line reaches persistent timeline storage."""
    return _safe_worker_text(_SECRET_ARGUMENT.sub(r"\1[redacted]", value), 512)


def _read_process_details(pid: int) -> dict[str, str]:
    """Read live native Windows process parameters; never guess cwd from an image path.

    Same-bitness only. Inaccessible, exiting, and cross-bitness targets are
    explicitly unavailable. The native process cwd may differ from shell pwd.
    """
    if sys.platform != "win32":
        return {"error": "unsupported_platform"}

    class ProcessBasicInformation(ctypes.Structure):
        _fields_ = [
            ("reserved1", ctypes.c_void_p), ("peb", ctypes.c_void_p),
            ("reserved2", ctypes.c_void_p * 2), ("pid", ctypes.c_void_p),
            ("reserved3", ctypes.c_void_p),
        ]

    class UnicodeString(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.USHORT), ("max_length", wintypes.USHORT),
            ("buffer", ctypes.c_void_p),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsWow64Process.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsWow64Process.restype = wintypes.BOOL
    kernel.ReadProcessMemory.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel.ReadProcessMemory.restype = wintypes.BOOL
    ntdll.NtQueryInformationProcess.argtypes = [
        wintypes.HANDLE, wintypes.ULONG, ctypes.c_void_p,
        wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
    ]
    ntdll.NtQueryInformationProcess.restype = wintypes.LONG

    handle = kernel.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED_INFORMATION | VM_READ
    if not handle:
        return {"error": "access_denied_or_exited"}
    try:
        current_wow64, target_wow64 = wintypes.BOOL(), wintypes.BOOL()
        if (not kernel.IsWow64Process(kernel.GetCurrentProcess(), ctypes.byref(current_wow64))
                or not kernel.IsWow64Process(handle, ctypes.byref(target_wow64))):
            return {"error": "architecture_unknown"}
        if bool(current_wow64.value) != bool(target_wow64.value):
            return {"error": "unsupported_architecture"}

        def read(address: int, size: int) -> bytes | None:
            if not address or size <= 0 or size > 8192:
                return None
            buffer = ctypes.create_string_buffer(size)
            done = ctypes.c_size_t()
            if not kernel.ReadProcessMemory(handle, ctypes.c_void_p(address), buffer,
                                            size, ctypes.byref(done)) or done.value != size:
                return None
            return buffer.raw

        basic = ProcessBasicInformation()
        if ntdll.NtQueryInformationProcess(handle, 0, ctypes.byref(basic),
                                           ctypes.sizeof(basic), None) != 0 or not basic.peb:
            return {"error": "process_query_failed"}
        is_64 = ctypes.sizeof(ctypes.c_void_p) == 8
        pointer_size = ctypes.sizeof(ctypes.c_void_p)
        pointer = read(basic.peb + (0x20 if is_64 else 0x10), pointer_size)
        parameters = int.from_bytes(pointer, "little") if pointer else 0
        if not parameters:
            return {"error": "process_parameters_unavailable"}

        def read_unicode(offset: int) -> str | None:
            raw = read(parameters + offset, ctypes.sizeof(UnicodeString))
            if raw is None:
                return None
            string = UnicodeString.from_buffer_copy(raw)
            if (not string.length or string.length % 2 or string.length > 8192
                    or string.length > string.max_length or not string.buffer):
                return None
            value = read(string.buffer, string.length)
            if value is None:
                return None
            try:
                return value.decode("utf-16-le", "strict")
            except UnicodeError:
                return None

        command_line = read_unicode(0x70 if is_64 else 0x40)
        process_cwd = read_unicode(0x38 if is_64 else 0x24)
        result = {}
        if command_line is not None:
            result["command_line"] = command_line
        if process_cwd is not None:
            result["process_cwd"] = process_cwd
        return result
    finally:
        kernel.CloseHandle(handle)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:24]


def _name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return Path(value.replace("\\", "/")).name[:100]


def _enumerate_processes() -> list[dict[str, Any]]:
    """Return PID, parent PID and executable basename using Toolhelp only."""
    if sys.platform != "win32":
        raise OSError("Windows process snapshots are unavailable on this platform")

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W)]
    kernel.Process32FirstW.restype = wintypes.BOOL
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W)]
    kernel.Process32NextW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL

    handle = kernel.CreateToolhelp32Snapshot(0x00000002, 0)
    if not handle or handle == ctypes.c_void_p(-1).value:
        raise OSError("Windows process snapshot could not be created")
    result: list[dict[str, Any]] = []
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        if not kernel.Process32FirstW(handle, ctypes.byref(entry)):
            raise OSError("Windows process snapshot could not be read")
        while len(result) < _MAX_PROCESSES:
            result.append({
                "pid": int(entry.th32ProcessID),
                "parent_pid": int(entry.th32ParentProcessID),
                "name": entry.szExeFile,
            })
            entry.dwSize = ctypes.sizeof(entry)
            if not kernel.Process32NextW(handle, ctypes.byref(entry)):
                break
        return result
    finally:
        kernel.CloseHandle(handle)


class TerminalObserver:
    """Emit bounded terminal/shell process metadata without reading command output."""

    def __init__(
        self,
        process_provider: Callable[[], Iterable[dict[str, Any]]] | None = None,
        *,
        details_provider: Callable[[int], dict[str, str]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._process_provider = process_provider or _enumerate_processes
        # Synthetic process trees must never trigger reads of unrelated live PIDs.
        self._details_provider = details_provider or (
            _read_process_details if process_provider is None else lambda _pid: {"error": "not_collected"}
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._known: dict[str, dict[str, Any]] = {}
        self._last_poll_at: datetime | None = None
        self._processes_scanned = 0
        self._events_emitted = 0
        self._last_error: str | None = None

    def poll(self) -> list[TimelineEvent]:
        now = self._clock()
        now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
        self._last_poll_at = now
        self._processes_scanned = 0
        self._events_emitted = 0
        self._last_error = None
        try:
            raw_processes = list(islice(self._process_provider(), _MAX_PROCESSES))
        except Exception as exc:
            self._last_error = type(exc).__name__
            return []

        processes: dict[int, dict[str, Any]] = {}
        for item in raw_processes:
            if not isinstance(item, dict):
                continue
            pid, parent_pid, name = item.get("pid"), item.get("parent_pid"), _name(item.get("name"))
            if type(pid) is int and pid > 0 and type(parent_pid) is int and parent_pid >= 0 and name:
                processes[pid] = {"pid": pid, "parent_pid": parent_pid, "name": name.casefold()}
        self._processes_scanned = len(processes)

        children: dict[int, list[int]] = defaultdict(list)
        for pid, process in processes.items():
            parent_pid = process["parent_pid"]
            if parent_pid in processes:
                children[parent_pid].append(pid)

        sessions: dict[str, dict[str, Any]] = {}
        shells = [process for process in processes.values() if process["name"] in _SHELLS]
        for shell in sorted(shells, key=lambda item: item["pid"])[:_MAX_SESSIONS]:
            app = self._terminal_app(shell["pid"], processes)
            pid = shell["pid"]
            session_id = _hash(f"{pid}:{shell['parent_pid']}:{shell['name']}")
            child_items = self._descendants(pid, processes, children)
            data = {
                "session_id": session_id,
                "process": shell["name"],
                "pid": pid,
                "parent_pid": shell["parent_pid"],
                "terminal_app": app,
                "child_processes": [item["name"] for item in child_items],
                "status": "active",
            }
            data.update(self._process_metadata(pid))
            data["child_details"] = [
                {"pid": item["pid"], "process": item["name"],
                 **self._process_metadata(item["pid"])}
                for item in child_items[:_MAX_CHILD_DETAILS]
            ]
            fingerprint = _hash(json.dumps(data, sort_keys=True, separators=(",", ":")))
            previous = self._known.get(session_id)
            emit = (
                previous is None
                or previous["fingerprint"] != fingerprint
                or now - previous["last_event_at"] >= _HEARTBEAT
            )
            sessions[session_id] = {
                "fingerprint": fingerprint,
                "last_seen": now,
                "last_event_at": now if emit else previous["last_event_at"],
                "should_emit": emit,
                "data": data,
            }

        events: list[TimelineEvent] = []
        for session_id, state in sessions.items():
            if state["should_emit"]:
                events.append(self._event(session_id, state["data"], now, "terminal.context_updated"))

        for session_id, previous in list(self._known.items()):
            if session_id in sessions:
                continue
            if now - previous["last_seen"] <= _MAX_STALE:
                expired = dict(previous)
                expired["missing_since"] = expired.get("missing_since", now)
                self._known[session_id] = expired
                continue
            data = dict(previous["data"])
            data["status"] = "closed"
            events.append(self._event(session_id, data, now, "terminal.session_closed"))

        self._known = sessions | {
            session_id: value for session_id, value in self._known.items()
            if session_id not in sessions and "missing_since" in value and now - value["last_seen"] <= _MAX_STALE
        }
        self._events_emitted = len(events)
        return events

    @staticmethod
    def _terminal_app(pid: int, processes: dict[int, dict[str, Any]]) -> str:
        current = processes.get(pid)
        visited: set[int] = {pid}
        for _ in range(24):
            if current is None:
                break
            parent_pid = current["parent_pid"]
            if parent_pid in visited:
                break
            visited.add(parent_pid)
            current = processes.get(parent_pid)
            if current is None:
                break
            app_name = _TERMINAL_APPS.get(current["name"])
            if app_name:
                return app_name
            if current["name"] in _TERMINAL_HOSTS:
                continue
        return ""

    @staticmethod
    def _descendants(
        pid: int,
        processes: dict[int, dict[str, Any]],
        children: dict[int, list[int]],
    ) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        queue = deque([pid])
        visited = {pid}
        while queue and len(visited) <= 256:
            parent = queue.popleft()
            for child_pid in children.get(parent, ()):
                if child_pid in visited:
                    continue
                visited.add(child_pid)
                queue.append(child_pid)
                child_name = processes[child_pid]["name"]
                if child_name in _TERMINAL_HOSTS or child_name in _SHELLS or child_name in _TERMINAL_APPS:
                    continue
                if child_name not in {item["name"] for item in found}:
                    found.append({"pid": child_pid, "name": child_name})
                    if len(found) >= _MAX_CHILDREN:
                        return found
        return found

    def _process_metadata(self, pid: int) -> dict[str, Any]:
        try:
            raw = self._details_provider(pid)
            if not isinstance(raw, dict):
                raw = {"error": "invalid_details"}
        except Exception:
            # A failed read of one short-lived process must not abort the whole poll.
            raw = {"error": "process_read_failed"}
        metadata: dict[str, Any] = {
            "command_line_status": "unavailable",
            "process_cwd_status": "unavailable",
        }
        command = raw.get("command_line")
        if isinstance(command, str) and command.strip():
            metadata["command_line"] = _safe_command_line(command)
            metadata["command_line_status"] = "known"
        cwd = raw.get("process_cwd")
        if isinstance(cwd, str) and cwd.strip():
            metadata["process_cwd"] = _safe_worker_path(cwd, 512)
            metadata["process_cwd_status"] = "known"
            metadata["process_cwd_source"] = "windows_process_parameters"
        if metadata["command_line_status"] != "known" or metadata["process_cwd_status"] != "known":
            reason = raw.get("error")
            if reason in {
                "unsupported_platform", "access_denied_or_exited", "architecture_unknown",
                "unsupported_architecture", "process_query_failed", "process_parameters_unavailable",
                "not_collected", "invalid_details", "process_read_failed",
            }:
                metadata["unavailable_reason"] = reason
        return metadata

    @staticmethod
    def _event(session_id: str, data: dict[str, Any], when: datetime, event_type: str) -> TimelineEvent:
        payload = dict(data)
        event_id = _hash(f"{session_id}:{event_type}:{when.isoformat()}:{json.dumps(payload, sort_keys=True)}")
        return TimelineEvent(
            source="terminal",
            type=event_type,
            data=payload,
            project=None,
            time=when,
            id=event_id,
        )

    def health(self) -> dict[str, Any]:
        return {
            "available": self._last_poll_at is not None and self._last_error is None,
            "last_poll_at": self._last_poll_at.isoformat() if self._last_poll_at else None,
            "processes_scanned": self._processes_scanned,
            "active_terminals": len(self._known),
            "events_emitted": self._events_emitted,
            "error_type": self._last_error,
        }
