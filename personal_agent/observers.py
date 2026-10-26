"""Deterministic, read-only Windows observation adapters."""
from __future__ import annotations

import ctypes
import os
import sys
from collections import Counter
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .models import TimelineEvent


@runtime_checkable
class ObserverAdapter(Protocol):
    def detect(self) -> bool: ...
    def snapshot(self) -> dict[str, Any]: ...
    def events(self) -> list[TimelineEvent]: ...


_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}
_EDITOR_MARKERS = (
    "code.exe",
    "codium",
    "pycharm",
    "idea64",
    "devenv.exe",
    "sublime_text",
    "notepad++",
)
_TERMINAL_MARKERS = (
    "windowsterminal",
    "powershell",
    "pwsh",
    "cmd.exe",
    "wezterm",
    "alacritty",
)
_GAME_MARKERS = (
    "dota2",
    "cs2.exe",
    "eldenring",
    "witcher",
    "game.exe",
)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("observer timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


if sys.platform == "win32":
    class _LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


    class _PROCESSENTRY32W(ctypes.Structure):
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


class _Win32Sampler:
    TH32CS_SNAPPROCESS = 0x00000002
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("Windows APIs are only available on Windows")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._bind()

    def _bind(self) -> None:
        self.user32.GetForegroundWindow.argtypes = []
        self.user32.GetForegroundWindow.restype = wintypes.HWND

        self.user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD

        self.user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self.user32.GetWindowTextLengthW.restype = ctypes.c_int
        self.user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.user32.GetWindowTextW.restype = ctypes.c_int

        self.user32.GetLastInputInfo.argtypes = [ctypes.POINTER(_LASTINPUTINFO)]
        self.user32.GetLastInputInfo.restype = wintypes.BOOL

        self.kernel32.GetTickCount64.argtypes = []
        self.kernel32.GetTickCount64.restype = ctypes.c_ulonglong

        self.kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel32.OpenProcess.restype = wintypes.HANDLE
        self.kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = wintypes.BOOL

        self.kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self.kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        self.kernel32.Process32FirstW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PROCESSENTRY32W),
        ]
        self.kernel32.Process32FirstW.restype = wintypes.BOOL
        self.kernel32.Process32NextW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PROCESSENTRY32W),
        ]
        self.kernel32.Process32NextW.restype = wintypes.BOOL

    def foreground(self) -> dict[str, Any] | None:
        hwnd = self.user32.GetForegroundWindow()
        if not hwnd:
            return None
        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        length = max(0, int(self.user32.GetWindowTextLengthW(hwnd)))
        buffer = ctypes.create_unicode_buffer(min(length + 1, 4096))
        if len(buffer) > 1:
            self.user32.GetWindowTextW(hwnd, buffer, len(buffer))
        exe = self._process_path(int(pid.value))
        return {
            "hwnd": int(hwnd),
            "pid": int(pid.value),
            "exe": exe,
            "process": Path(exe).name if exe else "",
            "title": buffer.value,
        }

    def _process_path(self, pid: int) -> str:
        handle = self.kernel32.OpenProcess(self.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not self.kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return ""
            return buffer.value
        finally:
            self.kernel32.CloseHandle(handle)

    def idle_seconds(self) -> float | None:
        info = _LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if not self.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        now_low32 = int(self.kernel32.GetTickCount64()) & 0xFFFFFFFF
        elapsed_ms = (now_low32 - int(info.dwTime)) & 0xFFFFFFFF
        return elapsed_ms / 1000.0

    def processes(self) -> dict[int, str]:
        handle = self.kernel32.CreateToolhelp32Snapshot(self.TH32CS_SNAPPROCESS, 0)
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, 0, invalid):
            return {}
        result: dict[int, str] = {}
        try:
            entry = _PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(entry)
            if not self.kernel32.Process32FirstW(handle, ctypes.byref(entry)):
                return result
            while True:
                result[int(entry.th32ProcessID)] = entry.szExeFile
                entry.dwSize = ctypes.sizeof(entry)
                if not self.kernel32.Process32NextW(handle, ctypes.byref(entry)):
                    break
            return result
        finally:
            self.kernel32.CloseHandle(handle)


class WindowsActivityObserver:
    def __init__(
        self,
        project_roots: dict[str, Path],
        interval_seconds: float = 5.0,
        *,
        snapshot_provider: Callable[[], dict[str, Any]] | None = None,
        clock: Callable[[], datetime] | None = None,
        max_files: int = 10_000,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.project_roots = {name: Path(path) for name, path in project_roots.items()}
        self.interval_seconds = float(interval_seconds)
        self.max_files = max(1, int(max_files))
        self._snapshot_provider = snapshot_provider
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._win32: _Win32Sampler | None = None
        self._previous: dict[str, Any] | None = None
        self._samples = 0
        self._last_sample_time: datetime | None = None
        self._last_error: str | None = None
        self._truncated_projects: list[str] = []

    def detect(self) -> bool:
        return self._snapshot_provider is not None or sys.platform == "win32"

    def _native(self) -> _Win32Sampler:
        if self._win32 is None:
            self._win32 = _Win32Sampler()
        return self._win32

    def snapshot(self) -> dict[str, Any]:
        if self._snapshot_provider is not None:
            return self._snapshot_provider()
        native = self._native()
        files, truncated = self._scan_projects()
        self._truncated_projects = truncated
        return {
            "foreground": native.foreground(),
            "idle_seconds": native.idle_seconds(),
            "processes": native.processes(),
            "files": files,
        }

    def _scan_projects(self) -> tuple[dict[str, dict[str, tuple[int, int]]], list[str]]:
        all_projects: dict[str, dict[str, tuple[int, int]]] = {}
        truncated: list[str] = []
        remaining = self.max_files
        entry_budget = self.max_files * 2
        for project, root in sorted(self.project_roots.items()):
            observed: dict[str, tuple[int, int]] = {}
            if remaining <= 0 or entry_budget <= 0:
                truncated.append(project)
                all_projects[project] = observed
                continue
            try:
                root_path = root.resolve()
            except OSError:
                all_projects[project] = observed
                continue
            stack = [root_path]
            while stack and remaining > 0 and entry_budget > 0:
                directory = stack.pop()
                try:
                    with os.scandir(directory) as iterator:
                        from itertools import islice
                        entries = list(islice(iterator, entry_budget + 1))
                        if len(entries) > entry_budget:
                            truncated.append(project)
                            entries = entries[:entry_budget]
                        entries.sort(key=lambda item: item.name.casefold(), reverse=True)
                except (OSError, PermissionError):
                    continue
                for entry in entries:
                    entry_budget -= 1
                    if entry.name.casefold() in _EXCLUDED_DIRS:
                        continue
                    try:
                        if os.path.isjunction(entry.path):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            stat = entry.stat(follow_symlinks=False)
                            relative = os.path.relpath(entry.path, root_path)
                            observed[relative] = (int(stat.st_mtime_ns), int(stat.st_size))
                            remaining -= 1
                            if remaining <= 0:
                                break
                    except (OSError, PermissionError):
                        continue
            if stack or remaining <= 0 or entry_budget <= 0:
                truncated.append(project)
            all_projects[project] = observed
        return all_projects, sorted(set(truncated))

    def _project_for_foreground(self, foreground: dict[str, Any] | None) -> str | None:
        if not foreground:
            return None
        title = str(foreground.get("title") or "").casefold()
        exe = str(foreground.get("exe") or "")
        for project, root in sorted(self.project_roots.items()):
            if project.casefold() in title or root.name.casefold() in title:
                return project
            if exe:
                try:
                    if os.path.commonpath([str(root.resolve()), str(Path(exe).resolve())]) == str(root.resolve()):
                        return project
                except (OSError, ValueError):
                    pass
        return None

    @staticmethod
    def _foreground_key(value: dict[str, Any] | None) -> tuple[Any, ...] | None:
        if not value:
            return None
        return tuple(value.get(key) for key in ("hwnd", "pid", "exe", "title"))

    @staticmethod
    def _idle_state(value: Any) -> str:
        if value is None:
            return "unknown"
        return "idle" if float(value) >= 60.0 else "active"

    @staticmethod
    def _normalize_processes(value: Any) -> dict[int, str]:
        if not isinstance(value, dict):
            return {}
        result: dict[int, str] = {}
        for pid, exe in value.items():
            try:
                result[int(pid)] = str(exe)
            except (TypeError, ValueError):
                continue
        return result

    def sample(
        self,
        *,
        now: datetime | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> list[TimelineEvent]:
        when = _aware_utc(now or self._clock())
        try:
            current = snapshot if snapshot is not None else self.snapshot()
            previous = self._previous
            result: list[TimelineEvent] = []

            foreground = current.get("foreground")
            previous_foreground = previous.get("foreground") if previous else None
            if self._foreground_key(foreground) != self._foreground_key(previous_foreground):
                data = dict(foreground or {})
                data["present"] = foreground is not None
                result.append(
                    TimelineEvent(
                        source="windows",
                        type="window.foreground",
                        data=data,
                        project=self._project_for_foreground(foreground),
                        time=when,
                    )
                )

            idle = current.get("idle_seconds")
            idle_state = self._idle_state(idle)
            previous_idle_state = self._idle_state(previous.get("idle_seconds")) if previous else None
            if previous is None or idle_state != previous_idle_state:
                result.append(
                    TimelineEvent(
                        source="windows",
                        type="input.idle",
                        data={"state": idle_state, "idle_seconds": idle},
                        time=when,
                    )
                )

            current_processes = self._normalize_processes(current.get("processes"))
            if previous is not None:
                old_processes = self._normalize_processes(previous.get("processes"))
                changed_pids = {
                    pid for pid in current_processes.keys() & old_processes.keys()
                    if current_processes[pid] != old_processes[pid]
                }
                for pid in sorted((old_processes.keys() - current_processes.keys()) | changed_pids):
                    result.append(
                        TimelineEvent(
                            source="windows",
                            type="process.stopped",
                            data={"pid": pid, "exe": old_processes[pid]},
                            time=when,
                        )
                    )
                for pid in sorted((current_processes.keys() - old_processes.keys()) | changed_pids):
                    result.append(
                        TimelineEvent(
                            source="windows",
                            type="process.started",
                            data={"pid": pid, "exe": current_processes[pid]},
                            time=when,
                        )
                    )

            current_files = current.get("files") if isinstance(current.get("files"), dict) else {}
            if previous is not None:
                previous_files = previous.get("files") if isinstance(previous.get("files"), dict) else {}
                for project in sorted(set(current_files) | set(previous_files)):
                    new = current_files.get(project, {}) or {}
                    old = previous_files.get(project, {}) or {}
                    for path in sorted(old.keys() - new.keys()):
                        result.append(
                            TimelineEvent(
                                source="filesystem",
                                type="file.deleted",
                                data={"path": path},
                                project=project,
                                time=when,
                            )
                        )
                    for path in sorted(new.keys() - old.keys()):
                        mtime_ns, size = new[path]
                        result.append(
                            TimelineEvent(
                                source="filesystem",
                                type="file.created",
                                data={"path": path, "mtime_ns": mtime_ns, "size": size},
                                project=project,
                                time=when,
                            )
                        )
                    for path in sorted(new.keys() & old.keys()):
                        if tuple(new[path]) != tuple(old[path]):
                            mtime_ns, size = new[path]
                            result.append(
                                TimelineEvent(
                                    source="filesystem",
                                    type="file.modified",
                                    data={"path": path, "mtime_ns": mtime_ns, "size": size},
                                    project=project,
                                    time=when,
                                )
                            )

            self._previous = current
            self._samples += 1
            self._last_sample_time = when
            self._last_error = None
            return result
        except Exception as exc:
            self._samples += 1
            self._last_sample_time = when
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def events(self) -> list[TimelineEvent]:
        return self.sample()

    def health(self) -> dict[str, Any]:
        previous = self._previous or {}
        files = previous.get("files") if isinstance(previous.get("files"), dict) else {}
        tracked_files = sum(len(items or {}) for items in files.values())
        return {
            "available": self.detect(),
            "samples": self._samples,
            "last_sample_time": self._last_sample_time.isoformat() if self._last_sample_time else None,
            "last_error": self._last_error,
            "tracked_processes": len(self._normalize_processes(previous.get("processes"))),
            "tracked_files": tracked_files,
            "truncated_projects": list(self._truncated_projects),
            "interval_seconds": self.interval_seconds,
        }


def _event_exe(event: TimelineEvent) -> str:
    return str(event.data.get("exe") or event.data.get("process") or "").casefold()


def _matches_any(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


def _infer_session(events: list[TimelineEvent]) -> tuple[str, str, list[str]]:
    editor_events: list[TimelineEvent] = []
    file_events: list[TimelineEvent] = []
    terminal_events: list[TimelineEvent] = []
    game_events: list[TimelineEvent] = []
    for event in events:
        exe = _event_exe(event)
        if event.type == "window.foreground" and _matches_any(exe, _EDITOR_MARKERS):
            editor_events.append(event)
        if event.type.startswith("file."):
            file_events.append(event)
        if event.type == "window.foreground" and _matches_any(exe, _TERMINAL_MARKERS):
            terminal_events.append(event)
        if event.type == "window.foreground" and _matches_any(exe, _GAME_MARKERS):
            game_events.append(event)

    for editor in editor_events:
        for file_event in file_events:
            if file_event.time <= editor.time:
                continue
            terminals = [event for event in terminal_events if event.time >= file_event.time]
            if terminals:
                transitions = ["gaming"] if any(
                    event.time > terminals[0].time for event in game_events
                ) else []
                return "development/debugging", "high", transitions
    if editor_events and terminal_events:
        return "development/debugging", "medium", ["gaming"] if game_events else []
    if file_events:
        return "project work", "medium", []
    if game_events:
        return "gaming", "high", []
    return "uncertain", "low", []


def sessionize(events: list[TimelineEvent]) -> list[dict[str, Any]]:
    ordered = sorted(events, key=lambda item: (item.time, item.id))
    if not ordered:
        return []
    groups: list[list[TimelineEvent]] = [[ordered[0]]]
    max_gap = timedelta(minutes=20)
    for event in ordered[1:]:
        known_project = next((prior.project for prior in reversed(groups[-1]) if prior.project), None)
        project_changed = bool(event.project and known_project and event.project != known_project)
        returned = (event.type == "input.idle" and event.data.get("state") == "active"
                    and any(prior.type == "input.idle" and prior.data.get("state") == "idle" for prior in groups[-1]))
        if event.time - groups[-1][-1].time > max_gap or project_changed or returned:
            groups.append([event])
        else:
            groups[-1].append(event)

    sessions: list[dict[str, Any]] = []
    for group in groups:
        projects = Counter(event.project for event in group if event.project)
        project: str | None = None
        project_confidence = "unknown"
        if projects:
            ranked = projects.most_common()
            project = ranked[0][0]
            project_confidence = "high" if len(ranked) == 1 or ranked[0][1] > ranked[1][1] else "uncertain"
        activity, confidence, transitions = _infer_session(group)
        sessions.append(
            {
                "start": group[0].time,
                "end": group[-1].time,
                "event_ids": [event.id for event in group],
                "project": project,
                "project_confidence": project_confidence,
                "activity": activity,
                "confidence": confidence,
                "transitions": transitions,
            }
        )
    return sessions


def summarize_activity(events: list[TimelineEvent]) -> str:
    sessions = sessionize(events)
    if not sessions:
        return "No activity evidence was recorded."
    lines: list[str] = []
    by_id = {event.id: event for event in events}
    for session in sessions:
        start = session["start"].astimezone(timezone.utc).strftime("%H:%M")
        end = session["end"].astimezone(timezone.utc).strftime("%H:%M")
        ids = session["event_ids"]
        selected_ids = ids if len(ids) <= 8 else ids[:4] + ids[-4:]
        evidence = ", ".join(selected_ids)
        if len(ids) > 8:
            evidence += f" ({len(ids)} events total; IDs abbreviated)"
        project = session["project"]
        activity = session["activity"]
        confidence = session["confidence"]
        if activity == "uncertain":
            statement = "Activity is uncertain from the available evidence"
        else:
            statement = f"Likely {activity}"
        if project:
            qualifier = "" if session["project_confidence"] == "high" else " (project inference uncertain)"
            statement += f" in project `{project}`{qualifier}"
        statement += f"; confidence {confidence}"
        if "gaming" in session["transitions"]:
            game_ids = [
                event_id for event_id in session["event_ids"]
                if _matches_any(_event_exe(by_id[event_id]), _GAME_MARKERS)
            ]
            statement += f". Later game foreground activity appears in evidence {', '.join(game_ids[-3:])}"
        group = [by_id[event_id] for event_id in ids]
        apps = list(dict.fromkeys(str(event.data.get("exe") or event.data.get("process") or "unknown") for event in group if event.type == "window.foreground"))
        if apps:
            statement += ". Applications observed: " + ", ".join(apps[:12])
        files = list(dict.fromkeys(str(event.data.get("path", "")) for event in group if event.type.startswith("file.")))
        if files:
            statement += ". Files changed: " + ", ".join(files[-8:])
        lines.append(f"{start}-{end} UTC: {statement}. Evidence: {evidence}.")
    return "\n".join(lines)
