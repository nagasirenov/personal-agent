"use strict";

// CDP helper for verify_browser_live.py. The browser/profile and token are synthetic.
const [port, action, value] = process.argv.slice(2);
const endpoint = `http://127.0.0.1:${port}/json/list`;

async function command(target, method, params) {
  return await new Promise((resolve, reject) => {
    const ws = new WebSocket(target.webSocketDebuggerUrl);
    const timer = setTimeout(() => { ws.close(); reject(new Error("CDP timeout")); }, 5000);
    ws.addEventListener("open", () => ws.send(JSON.stringify({ id: 1, method, params })));
    ws.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id !== 1) return;
      clearTimeout(timer);
      ws.close();
      if (message.error || message.result?.exceptionDetails) reject(new Error("CDP command failed"));
      else resolve(message.result);
    });
    ws.addEventListener("error", () => { clearTimeout(timer); reject(new Error("CDP connection failed")); });
  });
}

async function main() {
  const targets = await (await fetch(endpoint)).json();
  if (action === "setup") {
    for (const target of targets.filter((item) => item.type === "service_worker")) {
      try {
        const result = await command(target, "Runtime.evaluate", { expression: "typeof scheduleCapture", returnByValue: true });
        if (result.result.value !== "function") continue;
        const expression = `chrome.storage.local.set({token:${JSON.stringify(value)}}).then(() => { scheduleCapture(true); return true; })`;
        await command(target, "Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
        return;
      } catch (_) { /* Other built-in service workers are unrelated. */ }
    }
    throw new Error("Companion service worker unavailable");
  }
  const page = targets.find((item) => item.type === "page" && item.url.startsWith("http://127.0.0.1:"));
  if (!page) throw new Error("Synthetic page unavailable");
  if (action === "navigate") {
    await command(page, "Page.navigate", { url: value });
  } else if (action === "select") {
    await command(page, "Runtime.evaluate", {
      expression: "window.getSelection().selectAllChildren(document.getElementById('fixture-selection')); true",
      returnByValue: true,
    });
  } else {
    throw new Error("Unknown action");
  }
}

main().catch((error) => { console.error(error.message); process.exitCode = 1; });
