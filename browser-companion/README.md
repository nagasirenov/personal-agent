# Personal Agent Browser Context

Read-only Chrome/Edge Manifest V3 companion. It reports the selected tab's URL, title, visible page text, selected text, headings, and small link summary to the local Personal Agent. It does not click, navigate, fill forms, or capture screenshots.

## Local setup

1. Start the Personal Agent daemon.
2. Open PowerShell in the project root and run `.\.venv\Scripts\python.exe -m personal_agent browser-token --config config.local.toml`. Copy the printed token locally.
3. Open `chrome://extensions` or `edge://extensions`, enable Developer mode, and load this folder as an unpacked extension.
4. Open the extension's **Options**, paste the token, and save.

The extension needs permission to read HTTP/HTTPS pages so it can capture the active tab automatically. It reads page text only after the background worker requests a snapshot for the last focused window. The model receives bounded availability metadata and can request the page text and selection through its browser tool when needed; timeline events retain URL/title and counts only. The local receiver listens on `127.0.0.1:17842`, requires the saved bearer token, accepts extension-origin requests, and deletes active snapshot files when the daemon stops.

The page snapshot is capped and common token/password patterns are scrubbed. Internal browser pages are not accessible to the extension. When a page is closed or its snapshot expires, the local copy is removed.

Live capture: the service worker keeps a WebSocket to `ws://127.0.0.1:17842/v1/ws` (same loopback port; the token is sent as the `pa.<token>` subprotocol and the extension Origin and loopback Host are checked). When the agent's `read_browser_page` tool runs, the daemon asks for a capture, the extension answers with a fresh snapshot tagged with the request id, and the tool falls back to the last stored snapshot after about 3 seconds. A ping every 20 seconds keeps the worker alive; it reconnects with backoff, and on every alarm, startup and install. The one-minute alarm push remains as a fallback. Snapshots stay usable for 30 minutes. If the page text cannot be read (background, frozen or discarded tab, scripting error) an `active` snapshot with `content_available: false` (URL and title from `chrome.tabs`) is sent instead of `closed`; `closed` is sent only when there is no http(s) active tab.

After changing the extension files, open `edge://extensions` (or `chrome://extensions`) and press **Reload** on the unpacked extension.

If the active tab is restricted (not http/https), the companion invalidates its previous page. It checks the active tab again before sending and serializes updates so a delayed older tab cannot replace the newer one. The agent reads the latest snapshot when answering rather than waiting for the next background poll.

On Windows with Edge and Node.js installed, `.\.venv\Scripts\python.exe scripts\verify_browser_live.py` checks the unpacked extension in a temporary headless Edge profile against a synthetic local page and a separate local receiver. It does not use the normal browser profile, Telegram, or an LLM.
