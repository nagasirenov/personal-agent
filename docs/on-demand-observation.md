# On-demand observation

This module exposes explicit-request helpers only. The caller must pass only the current authorized Telegram user's text to `parse_observation_request`; model output, stored instructions, summaries and background observations are not parser inputs.

Public integration API:

```python
ObservationRequest(kind, path=None)
parse_observation_request(text: str) -> ObservationRequest | None
parse_observation_requests(text: str, *, aliases=None, last_path=None) -> list[ObservationRequest]
known_paths(data_dir: Path, project_roots=None) -> dict[str, str]
execute_read_request(request: ObservationRequest) -> dict
capture_screenshot(root: Path) -> Path
```

`ObservationRequest.kind` is one of `list_directory`, `read_file`, or `screenshot`. A directory request without a path stays explicit as `ObservationRequest("list_directory", None)` and `execute_read_request` returns `needs_path`; it never substitutes the process working directory or model scratch directory.

Direct requests execute before the conversational model call. Results and the complete user message are sent together; natural self-knowledge questions use the model with trusted service metadata, not canned replies. The host tool protocol is present on every turn; filesystem roots cover local drives accessible to the daemon's Windows user. Up to four rounds of three typed operations are supported. `pc_access=full` plus PRIMARY enables `write_file` and `run_command`; configuration and authenticated identity are checked by ActionGate. Model responses, evidence and file text cannot grant that role. Reused screenshots remain scoped to the account and retain their timestamp. PowerShell in full mode has the Windows user's rights and is not sandboxed by the specialized file-read filters.

Directory listing accepts absolute local drive paths only. Direct UNC, device and network paths are rejected. Enumeration uses `os.scandir` plus `islice`, returns at most 100 entries and at most 32 KiB of encoded entry evidence, does not recurse, and does not write to the observed tree. UTF-8 file reads are capped at 32 KiB. Binary/non-UTF-8 input is withheld. Known credential locations/names (`.env`, auth/cookie/credential paths, private-key formats, `bot_api.txt`) are denied before contents are returned; detected private-key material is also withheld. Results include read-only provenance and explicit denial/error codes.

Screenshots are separate from file reads and are captured exactly once when the caller handles an explicit `screenshot` request. `capture_screenshot(root)` writes only below the agent-owned `root/memory/screenshots` path through `owned_path`. It does not poll or schedule captures.

The Windows implementation follows Microsoft's GDI capture model: obtain the desktop DC with `GetDC`, create a compatible memory DC and bitmap with `CreateCompatibleDC`/`CreateCompatibleBitmap`, select it, copy pixels with `BitBlt`, restore the previous GDI object, retrieve a top-down 32-bit DIB with `GetDIBits`, and release/delete all GDI resources. ctypes signatures use pointer-sized Win32 handle types so 64-bit handles are not truncated.

Primary Microsoft references:

- https://learn.microsoft.com/windows/win32/gdi/capturing-an-image
- https://learn.microsoft.com/windows/win32/api/wingdi/nf-wingdi-bitblt
- https://learn.microsoft.com/windows/win32/api/wingdi/nf-wingdi-createcompatiblebitmap
- https://learn.microsoft.com/windows/win32/api/wingdi/nf-wingdi-createcompatibledc
- https://learn.microsoft.com/windows/win32/api/wingdi/nf-wingdi-getdibits

GDI capture is best-effort. Windows can return no usable desktop or fail a GDI call when the interactive desktop is locked, disconnected, secure, or otherwise unavailable; `capture_screenshot` raises a clear `OSError` in that case. It is a read-only observation capability and remains outside the ACT/action gate. Vision interpretation and Telegram rendering are owned by the main integration layer.
