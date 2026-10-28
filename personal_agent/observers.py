"""Deterministic, read-only Windows observation adapters."""
from __future__ import annotations

import ctypes
import os
import queue
import re
import sys
import threading
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
    ".state",
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


class _ProjectFileEventHandler:
    def __init__(self, owner: "WindowsActivityObserver", project: str, root: Path) -> None:
        self.owner = owner
        self.project = project
        self.root = root

    def dispatch(self, event: Any) -> None:
        callback = getattr(self, f"on_{getattr(event, 'event_type', '')}", None)
        if callback is not None:
            callback(event)

    def _record(self, event: Any, kind: str, attribute: str = "src_path") -> None:
        if bool(getattr(event, "is_directory", False)):
            return
        path = getattr(event, attribute, None)
        if path:
            self.owner._queue_path_change(self.project, self.root, str(path), kind)

    def on_created(self, event: Any) -> None:
        self._record(event, "created")

    def on_modified(self, event: Any) -> None:
        self._record(event, "modified")

    def on_deleted(self, event: Any) -> None:
        self._record(event, "deleted")

    def on_moved(self, event: Any) -> None:
        self._record(event, "deleted", "src_path")
        self._record(event, "created", "dest_path")


class WindowsActivityObserver:
    def __init__(
        self,
        project_roots: dict[str, Path],
        interval_seconds: float = 5.0,
        *,
        snapshot_provider: Callable[[], dict[str, Any]] | None = None,
        clock: Callable[[], datetime] | None = None,
        max_files: int = 10_000,
        event_queue_size: int | None = None,
        watchdog_loader: Callable[[], type[Any]] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.project_roots = {name: Path(path) for name, path in project_roots.items()}
        self.interval_seconds = float(interval_seconds)
        self.max_files = max(1, int(max_files))
        self.event_queue_size = max(
            1,
            int(self.max_files if event_queue_size is None else event_queue_size),
        )
        self._snapshot_provider = snapshot_provider
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._watchdog_loader = watchdog_loader
        self._win32: _Win32Sampler | None = None
        self._previous: dict[str, Any] | None = None
        self._samples = 0
        self._last_sample_time: datetime | None = None
        self._last_error: str | None = None
        self._truncated_projects: list[str] = []
        self._file_queue: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=self.event_queue_size)
        self._file_queue_lock = threading.Lock()
        self._queued_paths: set[tuple[str, str]] = set()
        self._pending_file_changes: dict[tuple[str, str], str] = {}
        self._dropped_file_events = 0
        self._coalesced_file_events = 0
        self._watchdog_observer: Any | None = None
        self._watched_projects: list[str] = []
        self._watch_errors: dict[str, str] = {}
        self._file_observer_started = False

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
        return {
            "foreground": native.foreground(),
            "idle_seconds": native.idle_seconds(),
            "processes": native.processes(),
        }

    @staticmethod
    def _load_watchdog_observer() -> type[Any]:
        try:
            from watchdog.observers import Observer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "watchdog is required for event-based project file observation"
            ) from exc
        return Observer

    def start(self) -> None:
        if self._file_observer_started:
            return
        if self._snapshot_provider is not None:
            self._file_observer_started = True
            return

        observer: Any | None = None
        try:
            observer_type = (
                self._watchdog_loader()
                if self._watchdog_loader is not None
                else self._load_watchdog_observer()
            )
            observer = observer_type()
            watched_projects: list[str] = []
            watch_errors: dict[str, str] = {}
            for project, root in sorted(self.project_roots.items()):
                try:
                    root_path = root.resolve()
                    if not root_path.is_dir():
                        raise FileNotFoundError(str(root_path))
                    handler = _ProjectFileEventHandler(self, project, root_path)
                    observer.schedule(handler, str(root_path), recursive=True)
                    watched_projects.append(project)
                except Exception as exc:
                    watch_errors[project] = f"{type(exc).__name__}: {exc}"
            observer.start()
        except Exception as exc:
            if observer is not None:
                try:
                    observer.stop()
                    observer.join(timeout=5.0)
                except Exception:
                    pass
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

        self._watchdog_observer = observer
        self._watched_projects = watched_projects
        self._watch_errors = watch_errors
        self._file_observer_started = True
        self._last_error = None

    def stop(self) -> None:
        observer = self._watchdog_observer
        self._file_observer_started = False
        self._watchdog_observer = None
        if observer is None:
            return
        try:
            observer.stop()
            observer.join(timeout=5.0)
            is_alive = getattr(observer, "is_alive", None)
            if callable(is_alive) and is_alive():
                self._last_error = "RuntimeError: watchdog observer did not stop within 5 seconds"
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _merge_file_change(previous: str | None, current: str) -> str | None:
        if previous is None:
            return current
        if previous == "created":
            return None if current == "deleted" else "created"
        if previous == "deleted":
            return "modified" if current == "created" else "deleted"
        if current == "deleted":
            return "deleted"
        return "modified"

    @staticmethod
    def _relative_file_path(root: Path, path: str) -> str | None:
        try:
            relative = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
        except (OSError, ValueError):
            return None
        if (
            relative == os.pardir
            or relative.startswith(os.pardir + os.sep)
            or os.path.isabs(relative)
        ):
            return None
        parts = Path(relative).parts
        if any(part.casefold() in _EXCLUDED_DIRS for part in parts[:-1]):
            return None
        return relative

    def _queue_path_change(self, project: str, root: Path, path: str, kind: str) -> None:
        relative = self._relative_file_path(root, path)
        if relative is None:
            return
        key = (project, relative)
        with self._file_queue_lock:
            previous = self._pending_file_changes.get(key)
            merged = self._merge_file_change(previous, kind)
            if key in self._queued_paths:
                self._coalesced_file_events += 1
                if merged is None:
                    self._pending_file_changes.pop(key, None)
                else:
                    self._pending_file_changes[key] = merged
                return
            if merged is None:
                return
            try:
                self._file_queue.put_nowait(key)
            except queue.Full:
                self._dropped_file_events += 1
                return
            self._queued_paths.add(key)
            self._pending_file_changes[key] = merged

    def _timeline_file_event(
        self,
        project: str,
        relative: str,
        kind: str,
        when: datetime,
    ) -> TimelineEvent | None:
        event_type = kind
        data: dict[str, Any] = {"path": relative}
        if kind != "deleted":
            root = self.project_roots.get(project)
            if root is None:
                return None
            try:
                stat = (root / relative).stat()
            except FileNotFoundError:
                event_type = "deleted"
            except OSError:
                return None
            else:
                data["mtime_ns"] = int(stat.st_mtime_ns)
                data["size"] = int(stat.st_size)
        return TimelineEvent(
            source="filesystem",
            type=f"file.{event_type}",
            data=data,
            project=project,
            time=when,
        )

    def _drain_file_events(self, when: datetime) -> list[TimelineEvent]:
        result: list[TimelineEvent] = []
        for _ in range(self._file_queue.qsize()):
            try:
                key = self._file_queue.get_nowait()
            except queue.Empty:
                break
            try:
                with self._file_queue_lock:
                    self._queued_paths.discard(key)
                    kind = self._pending_file_changes.pop(key, None)
                if kind is None:
                    continue
                event = self._timeline_file_event(key[0], key[1], kind, when)
                if event is not None:
                    result.append(event)
            finally:
                self._file_queue.task_done()
        return result

    @staticmethod
    def _snapshot_file_events(
        current_files: dict[str, Any],
        previous_files: dict[str, Any],
        when: datetime,
    ) -> list[TimelineEvent]:
        result: list[TimelineEvent] = []
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
        return result

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
            if previous is not None and (
                "files" in current or isinstance(previous.get("files"), dict)
            ):
                previous_files = previous.get("files") if isinstance(previous.get("files"), dict) else {}
                result.extend(self._snapshot_file_events(current_files, previous_files, when))
            elif "files" not in current:
                result.extend(self._drain_file_events(when))

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
        with self._file_queue_lock:
            pending_file_events = len(self._pending_file_changes)
            dropped_file_events = self._dropped_file_events
            coalesced_file_events = self._coalesced_file_events
        observer_alive: bool | None = None
        if self._watchdog_observer is not None:
            is_alive = getattr(self._watchdog_observer, "is_alive", None)
            if callable(is_alive):
                observer_alive = bool(is_alive())
        last_error = self._last_error
        if self._file_observer_started and observer_alive is False and last_error is None:
            last_error = "RuntimeError: watchdog observer thread is not running"
        return {
            "available": self.detect(),
            "samples": self._samples,
            "last_sample_time": self._last_sample_time.isoformat() if self._last_sample_time else None,
            "last_error": last_error,
            "tracked_processes": len(self._normalize_processes(previous.get("processes"))),
            "tracked_files": tracked_files,
            "truncated_projects": list(self._truncated_projects),
            "interval_seconds": self.interval_seconds,
            "file_observer_backend": "watchdog",
            "file_observer_started": self._file_observer_started,
            "file_observer_alive": observer_alive,
            "watched_projects": list(self._watched_projects),
            "watch_errors": dict(self._watch_errors),
            "pending_file_events": pending_file_events,
            "file_event_queue_capacity": self.event_queue_size,
            "dropped_file_events": dropped_file_events,
            "coalesced_file_events": coalesced_file_events,
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
    worker_sources = {"codex", "opencode", "vscode", "terminal", "browser"}
    worker_events = [event for event in events if event.source in worker_sources]
    desktop_events = [event for event in events if event.source not in worker_sources]
    sessions = sessionize(desktop_events)
    if not sessions and not worker_events:
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
    worker_groups: dict[tuple[str, str], list[TimelineEvent]] = {}
    for event in sorted(worker_events, key=lambda item: (item.time, item.id)):
        session_id = str(event.data.get("session_id") or event.project or "unknown")[:100]
        worker_groups.setdefault((event.source, session_id), []).append(event)
    for (source, _session_id), group in sorted(
        worker_groups.items(), key=lambda item: item[1][-1].time
    )[-12:]:
        label = {"codex": "Codex", "opencode": "OpenCode", "vscode": "VS Code", "terminal": "Terminal", "browser": "Browser"}[source]
        first, last = group[0], group[-1]
        task = next((str(event.data.get("task", "")).strip() for event in group
                     if isinstance(event.data.get("task"), str) and event.data["task"].strip()), "")
        task = _safe_worker_text(task, 240)
        project = next((event.project for event in reversed(group) if event.project), None)
        kinds = Counter(event.type.rsplit(".", 1)[-1] for event in group)
        status = next((str(event.data.get("status", "")).strip().casefold()
                       for event in reversed(group) if event.data.get("status")), "")
        details: list[str] = []
        if task:
            details.append(task)
        if project:
            details.append(f"project {Path(project).name or project}")
        counts = []
        commands = sum(count for kind, count in kinds.items() if "command" in kind or "tool" in kind)
        changes = sum(count for kind, count in kinds.items() if "file" in kind or "change" in kind)
        if commands:
            counts.append(f"{commands} command/tool events")
        if changes:
            counts.append(f"{changes} file/change events")
        if counts:
            details.append(", ".join(counts))
        if status:
            details.append(f"latest status {status[:40]}")
        if source == "vscode":
            active_file = _safe_worker_path(str(last.data.get("active_file", "")), 512)
            branch = _safe_worker_text(str(last.data.get("git_branch", "")), 100)
            if active_file:
                details.append(f"active file {active_file}")
            if isinstance(last.data.get("open_file_count"), int):
                details.append(f"{max(0, last.data['open_file_count'])} open files")
            diagnostics = last.data.get("diagnostics")
            if isinstance(diagnostics, dict):
                errors = diagnostics.get("error", 0)
                warnings = diagnostics.get("warning", 0)
                if isinstance(errors, int) and isinstance(warnings, int) and (errors or warnings):
                    details.append(f"diagnostics {errors} errors, {warnings} warnings")
            if branch:
                details.append(f"Git branch {branch}")
            if isinstance(last.data.get("terminal_count"), int) and last.data["terminal_count"]:
                details.append(f"{last.data['terminal_count']} terminals open")
        elif source == "terminal":
            process = _safe_worker_text(str(last.data.get("process", "")), 80)
            cwd = _safe_worker_path(str(last.data.get("process_cwd", "")), 512)
            command = _safe_worker_text(str(last.data.get("command_line", "")), 300)
            children = last.data.get("child_processes")
            if process:
                details.append(f"process {process}")
            if children:
                child_names = children if isinstance(children, list) else [children]
                names = [_safe_worker_text(str(item), 60) for item in child_names[:6]]
                if names:
                    details.append("children " + ", ".join(name for name in names if name))
            if cwd:
                details.append(f"native process cwd {cwd} (shell location may differ)")
            elif last.data.get("process_cwd_status") == "unavailable":
                details.append("native process cwd unavailable")
            if command:
                details.append(f"process launch command line {command}")
            elif last.data.get("command_line_status") == "unavailable":
                details.append("process command line unavailable")
            child_details = last.data.get("child_details")
            if isinstance(child_details, list):
                for child in child_details[:3]:
                    if not isinstance(child, dict):
                        continue
                    child_command = child.get("command_line")
                    name = _safe_worker_text(str(child.get("process", "")), 80)
                    if isinstance(child_command, str) and child_command:
                        details.append(f"running child {name} command line {_safe_worker_text(child_command, 300)}")
                    child_cwd = child.get("process_cwd")
                    if isinstance(child_cwd, str) and child_cwd:
                        details.append(f"child {name} native process cwd {_safe_worker_path(child_cwd, 512)}")
        elif source == "browser":
            title = _safe_worker_text(str(last.data.get("title", "")), 200)
            url = _safe_worker_text(str(last.data.get("url", "")), 500)
            if title:
                details.append(f"page {title}")
            if url:
                details.append(f"URL {url}")
            if isinstance(last.data.get("page_text_length"), int):
                details.append(f"readable text {max(0, last.data['page_text_length'])} chars")
            if isinstance(last.data.get("selected_text_length"), int) and last.data["selected_text_length"]:
                details.append(f"selection {max(0, last.data['selected_text_length'])} chars")
        start = first.time.astimezone(timezone.utc).strftime("%H:%M")
        end = last.time.astimezone(timezone.utc).strftime("%H:%M")
        description = "; ".join(details) if details else f"{len(group)} activity events"
        lines.append(f"{start}-{end} UTC: {label} activity: {description}.")
    return "\n".join(lines)


_WORKER_SECRET_PATTERNS = (
    re.compile(r"(?i)\b\d{5,}:[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b"),
    re.compile(r"(?i)\b(?:token|password|secret|api[_ -]?key)\s*[:=]\s*\S+"),
)


def _safe_worker_text(value: str, limit: int) -> str:
    value = " ".join(value.replace("\x00", " ").split())
    for pattern in _WORKER_SECRET_PATTERNS:
        value = pattern.sub("[redacted]", value)
    return value[:limit]


def _safe_worker_path(value: str, limit: int) -> str:
    value = value.replace("\x00", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    for pattern in _WORKER_SECRET_PATTERNS:
        value = pattern.sub("[redacted]", value)
    return value[:limit]
