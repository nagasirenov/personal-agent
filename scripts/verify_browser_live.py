"""Verify the companion with a temporary headless Edge profile and synthetic pages."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from personal_agent.adapters.browser import BrowserContextObserver
from personal_agent.browser_bridge import BrowserContextReceiver


class FixturePage(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        label = "FIRST_PAGE_MARKER" if self.path == "/first" else "SECOND_PAGE_MARKER"
        body = (f"<!doctype html><title>{label}</title><h1>{label}</h1>"
                "<p id='fixture-selection'>SELECTED_PASSAGE_MARKER</p>").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FixtureServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], ConnectionResetError):
            return
        super().handle_error(request, client_address)


def wait_for(predicate, seconds=12):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.1)
    return None


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    state = (root / ".state").resolve()
    if not state.is_relative_to(root):
        raise RuntimeError("Temporary directory must stay inside the project")
    state.mkdir(exist_ok=True)
    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    node = shutil.which("node")
    if sys.platform != "win32" or not edge.is_file() or not node:
        raise RuntimeError("This check requires Windows, Edge and Node.js")
    server = FixtureServer(("127.0.0.1", 0), FixturePage)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    with tempfile.TemporaryDirectory(prefix="browser-live-", dir=state) as folder:
        temp = Path(folder).resolve()
        if not temp.is_relative_to(state):
            raise RuntimeError("Temporary directory escaped the project")
        receiver = BrowserContextReceiver(temp / "agent", port=0, token="t" * 48)
        process = None
        try:
            if not receiver.start():
                raise RuntimeError("Synthetic receiver failed to start")
            extension = temp / "extension"
            shutil.copytree(root / "browser-companion", extension)
            worker = extension / "service_worker.js"
            source = worker.read_text(encoding="utf-8")
            source = source.replace("127.0.0.1:17842", f"127.0.0.1:{receiver.port}")
            worker.write_text(source, encoding="utf-8")
            profile = temp / "profile"
            fixture_url = f"http://127.0.0.1:{server.server_port}"
            process = subprocess.Popen([
                str(edge), "--headless=new", f"--user-data-dir={profile}",
                f"--disable-extensions-except={extension}", f"--load-extension={extension}",
                "--remote-debugging-port=0", "--remote-allow-origins=*", "--no-first-run",
                "--no-default-browser-check", "--disable-background-networking", fixture_url + "/first",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            port_file = profile / "DevToolsActivePort"
            if not wait_for(lambda: port_file.is_file() or process.poll() is not None, 20) or not port_file.is_file():
                raise RuntimeError("Isolated Edge failed to start")
            devtools_port = int(port_file.read_text().splitlines()[0])
            endpoint = f"http://127.0.0.1:{devtools_port}/json/list"
            def has_worker():
                try:
                    targets = httpx.get(endpoint, timeout=2, trust_env=False).json()
                    return any(item.get("type") == "service_worker" and item.get("url", "").endswith("/service_worker.js") for item in targets)
                except (httpx.HTTPError, ValueError):
                    return False
            if not wait_for(has_worker):
                raise RuntimeError("Companion worker did not start")
            helper = root / "scripts" / "browser_probe_cdp.js"
            def cdp(action, value=""):
                subprocess.run([node, str(helper), str(devtools_port), action, value], check=True,
                               timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
            observer = BrowserContextObserver(receiver.root, receiver=receiver)
            cdp("setup", "t" * 48)
            first = wait_for(lambda: next((item for item in observer.current_contexts()
                                         if "FIRST_PAGE_MARKER" in item["page_text"]), None))
            if not first:
                raise RuntimeError("Synthetic first page did not reach the agent")
            cdp("select")
            selected = wait_for(lambda: next((item for item in observer.current_contexts()
                                            if "SELECTED_PASSAGE_MARKER" in item["selected_text"]), None))
            if not selected:
                raise RuntimeError("Selection did not reach the agent")
            cdp("navigate", fixture_url + "/second")
            second = wait_for(lambda: next((item for item in observer.current_contexts()
                                          if "SECOND_PAGE_MARKER" in item["page_text"]), None))
            if not second:
                raise RuntimeError("Changed page did not reach the agent")
            if not wait_for(lambda: receiver.websocket_clients() == 1):
                raise RuntimeError("Extension did not open its live-capture WebSocket")
            cdp("navigate", fixture_url + "/first")
            time.sleep(0.5)
            reply = receiver.request_capture(3.0)
            if reply.get("responded") != 1:
                raise RuntimeError(f"Live capture request was not answered: {reply}")
            live = next((item for item in observer.current_contexts() if "FIRST_PAGE_MARKER" in item["page_text"]), None)
            if not live:
                raise RuntimeError("Live capture did not deliver the current page")
            cdp("navigate", "edge://version")
            def closed():
                paths = list(receiver.root.glob("*.json"))
                try:
                    return bool(paths and json.loads(paths[0].read_text(encoding="utf-8")).get("status") == "closed"
                                and not observer.current_contexts())
                except (OSError, ValueError):
                    return False
            if not wait_for(closed):
                raise RuntimeError("Restricted page did not invalidate the previous page")
            print("PASS: real Edge extension -> authenticated receiver -> observer; selection, navigation, on-demand WebSocket capture and restricted page")
            print("LLM calls: 0; Telegram calls: 0; user browser profile: untouched")
            return 0
        finally:
            if process is not None and process.poll() is None:
                subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW, timeout=10)
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            receiver.stop()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
