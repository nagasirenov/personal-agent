# Observation design

The first milestone observes Windows activity without taking screenshots or modifying observed applications/files. Sampling is deterministic at the adapter boundary so tests can inject snapshots and timestamps.

## Win32 sources

- Foreground activity: `GetForegroundWindow` yields the current foreground `HWND`; `GetWindowThreadProcessId` maps it to a PID; `GetWindowTextW` reads the window caption. Microsoft documents that `GetForegroundWindow` may return `NULL` transiently, so a missing foreground window is a valid sample rather than an error.
- Executable path: open the PID with `PROCESS_QUERY_LIMITED_INFORMATION`, then call `QueryFullProcessImageNameW`. Access failures are tolerated because protected/system processes may not be queryable from a normal user process.
- Idle time: `GetLastInputInfo` returns a 32-bit tick count. The observer compares it with `GetTickCount64` using low 32 bits so wraparound is handled modulo `2**32`.
- Process lifecycle: `CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)` plus `Process32FirstW`/`Process32NextW` enumerates a read-only process snapshot. `PROCESSENTRY32W.dwSize` is set before enumeration and the snapshot handle is always closed.

Official references:

- https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-getforegroundwindow
- https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-getwindowthreadprocessid
- https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-getwindowtextw
- https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-getlastinputinfo
- https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-queryfullprocessimagenamew
- https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-openprocess
- https://learn.microsoft.com/windows/win32/api/tlhelp32/nf-tlhelp32-createtoolhelp32snapshot
- https://learn.microsoft.com/windows/win32/api/tlhelp32/nf-tlhelp32-process32firstw
- https://learn.microsoft.com/windows/win32/api/tlhelp32/nf-tlhelp32-process32nextw

## ctypes correctness

All Win32 functions receive explicit `argtypes` and `restype`. Handle-returning functions use `wintypes.HANDLE`/`wintypes.HWND`, never a 32-bit integer, so 64-bit handles are not truncated. PID and tick-count fields use fixed-width Win32 types. `PROCESSENTRY32W` follows the Windows layout, including `ULONG_PTR` for `th32DefaultHeapID`.

## File observation

Known project roots are watched by bounded recursive stat polling. Each scan ignores `.git`, caches, virtual environments, build outputs, and dependency trees such as `node_modules`. The first scan establishes a baseline and emits no events. Later scans emit create/modify/delete events from path, size, and nanosecond mtime differences. Scans never write to watched roots and retain only the bounded metadata needed to compare consecutive snapshots.

## Event and inference model

Samples can emit foreground changes, idle changes, process starts/stops, and project-file changes. Project inference prefers an explicit file-root match and otherwise uses recent project evidence. `sessionize` groups nearby events deterministically; `summarize_activity` uses only rules over event types/data. It may label a development/debugging pattern when editor activity is followed by project-file and terminal evidence, with later game activity reported separately. Uncertain classifications are stated as uncertain.
