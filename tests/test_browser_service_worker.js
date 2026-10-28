"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function event() {
  const listeners = [];
  return { addListener: (callback) => listeners.push(callback), emit: (...args) => listeners.forEach((fn) => fn(...args)) };
}

function fixture() {
  const local = { token: "t".repeat(40) };
  const session = {};
  const makeStorage = (data) => ({
    get: async (key) => ({ [key]: data[key] }),
    set: async (values) => Object.assign(data, values),
  });
  const state = { active: null, pages: new Map(), sent: [], fetch: null, sockets: [], timers: [] };
  class FakeWebSocket {
    constructor(url, protocols) {
      this.url = url; this.protocols = protocols; this.readyState = 0; this.out = [];
      state.sockets.push(this);
    }
    open() { this.readyState = 1; this.onopen?.(); }
    send(text) { this.out.push(JSON.parse(text)); }
    close() { this.readyState = 3; this.onclose?.(); }
    receive(message) { return this.onmessage?.({ data: JSON.stringify(message) }); }
  }
  const chrome = {
    storage: { local: makeStorage(local), session: makeStorage(session), onChanged: event() },
    runtime: { onMessage: event(), onInstalled: event(), onStartup: event() },
    tabs: {
      onActivated: event(), onUpdated: event(), onRemoved: event(),
      query: async () => state.active ? [{ ...state.active }] : [],
      sendMessage: async (id) => {
        const page = state.pages.get(id);
        if (page instanceof Error) throw page;
        return typeof page === "function" ? page() : page;
      },
    },
    scripting: { executeScript: async () => { throw new Error("unavailable"); } },
    windows: { onFocusChanged: event(), onRemoved: event(), WINDOW_ID_NONE: -1 },
    alarms: { onAlarm: event(), create: async () => {} },
  };
  const fetch = async (_url, options) => {
    const snapshot = JSON.parse(options.body);
    state.sent.push(snapshot);
    if (state.fetch) await state.fetch(snapshot);
    return { status: 204 };
  };
  const source = fs.readFileSync(path.join(__dirname, "../browser-companion/service_worker.js"), "utf8");
  vm.runInNewContext(source, {
    chrome, fetch, navigator: { userAgent: "Chrome" },
    crypto: { randomUUID: () => "f27bc9ba-01aa-4f7d-98d1-374f7ef97b0c" },
    AbortController, setTimeout, clearTimeout, Date, WebSocket: FakeWebSocket,
    setInterval: (fn, ms) => { state.timers.push({ fn, ms }); return state.timers.length; },
    clearInterval: () => {},
  });
  return { state, chrome };
}

async function until(condition) {
  for (let i = 0; i < 100; i++) {
    if (condition()) return;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.fail("timed out waiting for browser companion");
}

test("a delayed old tab cannot replace the newly active page", async () => {
  const { state, chrome } = fixture();
  let release;
  let firstCaptureStarted = false;
  const delayed = new Promise((resolve) => { release = resolve; });
  state.active = { id: 1, windowId: 1, active: true, url: "https://a.test/" };
  state.pages.set(1, () => { firstCaptureStarted = true; return delayed; });
  chrome.tabs.onActivated.emit({ tabId: 1, windowId: 1 });
  await until(() => firstCaptureStarted);
  state.active = { id: 2, windowId: 1, active: true, url: "https://b.test/" };
  state.pages.set(2, { url: "https://b.test/", page_text: "new" });
  chrome.tabs.onActivated.emit({ tabId: 2, windowId: 1 });
  await until(() => state.sent.length === 1);
  release({ url: "https://a.test/", page_text: "old" });
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(state.sent.length, 1);
  assert.equal(state.sent[0].url, "https://b.test/");
});

test("restricted pages close the snapshot but unreadable http(s) tabs stay active as metadata only", async () => {
  const { state, chrome } = fixture();
  state.active = { id: 1, windowId: 1, active: true, url: "https://a.test/" };
  state.pages.set(1, { url: "https://a.test/", page_text: "old" });
  chrome.tabs.onActivated.emit({ tabId: 1, windowId: 1 });
  await until(() => state.sent.length === 1);
  assert.equal(state.sent[0].content_available, true);
  state.active = { id: 2, windowId: 1, active: true, url: "chrome://settings/" };
  chrome.tabs.onActivated.emit({ tabId: 2, windowId: 1 });
  await until(() => state.sent.length === 2);
  assert.equal(state.sent[1].status, "closed");
  state.active = { id: 3, windowId: 1, active: true, url: "https://unavailable.test/", title: "Frozen tab" };
  state.pages.set(3, new Error("unavailable"));
  chrome.tabs.onActivated.emit({ tabId: 3, windowId: 1 });
  await until(() => state.sent.length === 3);
  assert.equal(state.sent[2].status, "active");
  assert.equal(state.sent[2].content_available, false);
  assert.equal(state.sent[2].url, "https://unavailable.test/");
  assert.equal(state.sent[2].title, "Frozen tab");
  assert.equal(state.sent[2].page_text, "");
  state.active = null;
  chrome.windows.onRemoved.emit(1);
  await until(() => state.sent.length === 4);
  assert.equal(state.sent[3].status, "closed");
});

test("a live capture request over the WebSocket answers with a fresh tagged snapshot", async () => {
  const { state } = fixture();
  await until(() => state.sockets.length === 1);
  const ws = state.sockets[0];
  assert.equal(ws.url, "ws://127.0.0.1:17842/v1/ws");
  assert.equal(JSON.stringify(ws.protocols), JSON.stringify(["pa." + "t".repeat(40)]));
  ws.open();
  assert.equal(state.timers[0].ms, 20000);
  state.timers[0].fn();
  assert.equal(JSON.stringify(ws.out.pop()), JSON.stringify({ type: "ping" }));
  state.active = { id: 7, windowId: 1, active: true, url: "https://live.test/" };
  state.pages.set(7, { url: "https://live.test/", page_text: "LIVE", title: "Live" });
  await ws.receive({ type: "capture", request_id: "abc123" });
  await until(() => ws.out.length === 1);
  assert.equal(ws.out[0].type, "snapshot");
  assert.equal(ws.out[0].request_id, "abc123");
  assert.equal(ws.out[0].snapshot.page_text, "LIVE");
  assert.equal(ws.out[0].snapshot.content_available, true);
  // A background tab that cannot be read is still answered, as metadata only.
  state.pages.set(7, new Error("frozen"));
  await ws.receive({ type: "capture", request_id: "def456" });
  await until(() => ws.out.length === 2);
  assert.equal(ws.out[1].snapshot.content_available, false);
  assert.equal(ws.out[1].snapshot.status, "active");
  // Malformed requests are ignored.
  await ws.receive({ type: "capture", request_id: 5 });
  await ws.receive({ type: "other" });
  assert.equal(ws.out.length, 2);
});

test("the WebSocket reconnects with backoff after it closes", async () => {
  const { state } = fixture();
  await until(() => state.sockets.length === 1);
  state.sockets[0].open();
  state.sockets[0].close();
  await until(() => state.sockets.length === 2);  // after ~1 s
});

test("slow posts are serialized so newer snapshots arrive last", async () => {
  const { state, chrome } = fixture();
  let release;
  const slow = new Promise((resolve) => { release = resolve; });
  state.fetch = (snapshot) => snapshot.tab_id === 1 ? slow : Promise.resolve();
  state.active = { id: 1, windowId: 1, active: true, url: "https://a.test/" };
  state.pages.set(1, { url: "https://a.test/" });
  chrome.tabs.onActivated.emit({ tabId: 1, windowId: 1 });
  await until(() => state.sent.length === 1);
  state.active = { id: 2, windowId: 1, active: true, url: "https://b.test/" };
  state.pages.set(2, { url: "https://b.test/" });
  chrome.tabs.onActivated.emit({ tabId: 2, windowId: 1 });
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(state.sent.length, 1);
  release();
  await until(() => state.sent.length === 2);
  assert.equal(state.sent[1].url, "https://b.test/");
});
