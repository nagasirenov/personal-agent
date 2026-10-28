# On-demand observation

This module exposes explicit-request helpers only. Free text is never parsed for observations: the model chooses tools by meaning through native function calling (`personal_agent/tools.py`). Only the exact slash commands `/ls <path>`, `/read <path>` and `/screenshot` are recognized deterministically, by `parse_slash_command`, and run before the model.

Public integration API:

```python
ObservationRequest(kind, path=None)
parse_slash_command(text: str) -> ObservationRequest | None
known_paths(data_dir: Path, project_roots=None) -> dict[str, str]
execute_read_request(request: ObservationRequest) -> dict
capture_screenshot(root: Path) -> Path
async read_active_window() -> dict
```

`ObservationRequest.kind` is one of `list_directory`, `read_file`, `screenshot`, or `read_active_window`. A directory request without a path stays explicit as `ObservationRequest("list_directory", None)` and `execute_read_request` returns `needs_path`; it never substitutes the process working directory or model scratch directory.

Exact slash commands execute before the model call and their results go to it with the user message; everything else is a native tool the model calls itself (`list_directory`, `read_file`, `read_active_window`, `take_screenshot`, `read_browser_page`, `view_recent_media`, ...), all in one model call. Paths are validated by `validate_observation_plan` (local drive roots, credential filters). `write_file` and `run_command` are offered only when `ActionBroker.available_for(sender)` (PRIMARY with `pc_access=full`) and always execute through the ActionGate. Model responses, evidence and file text cannot grant that role. Recent images are listed in the prompt as `recent_media` and re-opened with `view_recent_media`, scoped to the account. PowerShell in full mode has the Windows user's rights and is not sandboxed by the specialized file-read filters.

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

## Capture and delivery policy

Capture is an internal observation, not an implicit Telegram upload. The model chooses whether to read accessible window text or capture an image based on the current request. Screenshot plans stay private by default; when the user asks to see or receive the image, the model sets `deliver_to_user: true`, and a per-request receipt prevents duplicate delivery during tool rounds. Persistent account preferences and plain-text rendering are separate from tool authorization. Incoming static sticker/photo/image documents use AttachmentPipeline and the same image input path; animated TGS/WEBM stickers are rendered locally into up to four ordered PNG frames with timestamps. Decoding has a separate cancellable process and bounded duration/input size; errors receive a factual status, not a claim of successful vision.

UI Automation runs on demand in an owned hidden PowerShell MTA process with an 8-second timeout and cancellation cleanup. It reads only the foreground window's accessible names and visible TextPattern ranges, skips password subtrees, and bounds depth, element count and text size. Raw text stays in the current request and is not written to event/state storage. The result is partial accessibility evidence, not a complete representation of the screen. Full-access PRIMARY can also request this tool through the ordinary model tool loop without a fixed wording requirement. See Microsoft's [threading guidance](https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-threading) and [GetVisibleRanges documentation](https://learn.microsoft.com/en-us/dotnet/api/system.windows.automation.textpattern.getvisibleranges).

Retention runs on daemon startup and then daily: screenshots older than 7 days are deleted; raw Windows, filesystem, vision, and PowerShell observations older than 30 days are removed from SQLite and FTS5, and derived daily/project views are refreshed. Expired observation pointers are cleared. User messages and the separately saved persistent profile are retained.
