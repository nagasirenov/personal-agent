"use strict";

const RECEIVER_URL = "http://127.0.0.1:17842/v1/context";
const TAB_REFRESH_ALARM = "personal-agent-active-tab-refresh";
let instanceIdPromise;
let captureVersion = 0;
let postQueue = Promise.resolve();

function instanceId() {
  if (!instanceIdPromise) {
    instanceIdPromise = (async () => {
      const saved = await chrome.storage.local.get("instance_id");
      if (typeof saved.instance_id === "string" && /^[0-9a-f-]{36}$/i.test(saved.instance_id)) {
        return saved.instance_id;
      }
      const value = crypto.randomUUID();
      await chrome.storage.local.set({ instance_id: value });
      return value;
    })().catch((error) => {
      instanceIdPromise = undefined;
      throw error;
    });
  }
  return instanceIdPromise;
}

async function interactionTimestamp(recordInteraction) {
  let saved = await chrome.storage.session.get("last_interaction_at");
  if (recordInteraction || typeof saved.last_interaction_at !== "string") {
    saved = { last_interaction_at: new Date().toISOString() };
    await chrome.storage.session.set(saved);
  }
  return saved.last_interaction_at;
}

function browserName() {
  return /Edg\//.test(navigator.userAgent) ? "Edge" : "Chrome";
}

async function postSnapshot(snapshot) {
  const saved = await chrome.storage.local.get("token");
  if (typeof saved.token !== "string" || saved.token.length < 32) return false;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2500);
  try {
    const response = await fetch(RECEIVER_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${saved.token}`,
      },
      body: JSON.stringify(snapshot),
      signal: controller.signal,
    });
    return response.status === 204;
  } catch (_) {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

async function activeTab() {
  try {
    const tabs = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    return tabs[0] || null;
  } catch (_) {
    return null;
  }
}

async function stillCurrent(tab) {
  const current = await activeTab();
  return tab === null ? current === null : Boolean(
    current && current.id === tab.id && current.windowId === tab.windowId && current.url === tab.url
  );
}

async function queueSnapshot(version, snapshot, tab) {
  const send = postQueue.then(async () => {
    if (version !== captureVersion || !(await stillCurrent(tab)) || version !== captureVersion) return;
    await postSnapshot(snapshot);
  });
  postQueue = send.catch(() => {});
  await send;
}

const CAPTURE_TIMEOUT_MS = 2500;

function withTimeout(promise, ms) {
  let timer;
  const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("timeout")), ms); });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function readPage(tab) {
  try {
    return await chrome.tabs.sendMessage(tab.id, { type: "personal_agent_capture" });
  } catch (_) {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
    return await chrome.tabs.sendMessage(tab.id, { type: "personal_agent_capture" });
  }
}

// Builds the snapshot for the active tab.  "closed" only when there is no http(s) active tab; when the page
// text cannot be read (background, frozen or discarded tab, scripting error) the page is still reported as
// active with metadata only, so the agent knows which page is open.
async function buildSnapshot(recordInteraction) {
  const tab = await activeTab();
  const id = await instanceId();
  const windowId = Number.isInteger(tab?.windowId) && tab.windowId >= 0 ? tab.windowId : 0;
  const base = { schema_version: 1, instance_id: id, browser: browserName(), window_id: windowId };
  if (!tab || !Number.isInteger(tab.id) || !tab.active || !/^https?:\/\//i.test(tab.url || "")) {
    return { tab, snapshot: { ...base, tab_id: -1, status: "closed" } };
  }
  const interaction_at = await interactionTimestamp(recordInteraction);
  let page = null;
  try {
    page = await withTimeout(readPage(tab), CAPTURE_TIMEOUT_MS);
  } catch (_) {
    page = null;
  }
  if (page && typeof page.url === "string" && page.url === tab.url) {
    return { tab, snapshot: { ...page, ...base, tab_id: tab.id, interaction_at, status: "active", content_available: true } };
  }
  return {
    tab,
    snapshot: {
      ...base, tab_id: tab.id, interaction_at, status: "active", content_available: false,
      url: tab.url, title: typeof tab.title === "string" ? tab.title : "",
      page_text: "", selected_text: "", dom_summary: {},
    },
  };
}

async function captureActive(version, recordInteraction) {
  try {
    const built = await buildSnapshot(recordInteraction);
    if (version !== captureVersion) return;
    await queueSnapshot(version, built.snapshot, built.tab);
  } catch (_) {
    // Browser windows and tabs can disappear while events are handled.
  }
}

// -- live capture on request over a local WebSocket ------------------------------------------
const SOCKET_URL = "ws://127.0.0.1:17842/v1/ws";
const PING_INTERVAL_MS = 20_000;
const MAX_RECONNECT_MS = 30_000;
let socket = null;
let connecting = false;
let reconnectDelay = 1000;
let reconnectTimer = null;
let pingTimer = null;

function stopPing() {
  if (pingTimer) clearInterval(pingTimer);
  pingTimer = null;
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    void connectSocket();
  }, reconnectDelay);
  reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_MS);
}

async function connectSocket() {
  if (connecting || typeof WebSocket !== "function") return;
  if (socket && (socket.readyState === 0 || socket.readyState === 1)) return;
  connecting = true;
  try {
    const saved = await chrome.storage.local.get("token");
    if (typeof saved.token !== "string" || saved.token.length < 32) return;
    const ws = new WebSocket(SOCKET_URL, ["pa." + saved.token]);
    socket = ws;
    ws.onopen = () => {
      reconnectDelay = 1000;
      stopPing();
      // Traffic every ~20 s also keeps the MV3 service worker alive while the socket is open.
      pingTimer = setInterval(() => {
        try { ws.send(JSON.stringify({ type: "ping" })); } catch (_) { /* closed */ }
      }, PING_INTERVAL_MS);
    };
    ws.onmessage = (event) => { void handleSocketMessage(ws, event.data); };
    ws.onerror = () => {};
    ws.onclose = () => {
      if (socket === ws) {
        socket = null;
        stopPing();
        scheduleReconnect();
      }
    };
  } catch (_) {
    socket = null;
    scheduleReconnect();
  } finally {
    connecting = false;
  }
}

function reconnectSocket() {
  const old = socket;
  socket = null;
  stopPing();
  reconnectDelay = 1000;
  try { old?.close(); } catch (_) { /* already closed */ }
  void connectSocket();
}

async function handleSocketMessage(ws, data) {
  let message;
  try { message = JSON.parse(data); } catch (_) { return; }
  if (message?.type !== "capture" || typeof message.request_id !== "string" || message.request_id.length > 64) return;
  captureVersion++;  // an older periodic capture must not overwrite this fresher one
  let snapshot;
  try {
    snapshot = (await buildSnapshot(false)).snapshot;
  } catch (_) {
    return;
  }
  try { ws.send(JSON.stringify({ type: "snapshot", request_id: message.request_id, snapshot })); } catch (_) { /* closed */ }
}

function scheduleCapture(recordInteraction = false) {
  const version = ++captureVersion;
  void captureActive(version, recordInteraction);
}

chrome.runtime.onMessage.addListener((message, sender) => {
  if (message?.type === "personal_agent_configuration_saved") {
    reconnectSocket();
    scheduleCapture(true);
    return;
  }
  if (!message || !["personal_agent_selection_changed", "personal_agent_dom_changed"].includes(message.type)) return;
  if (!sender.tab?.active) return;
  scheduleCapture(message.type === "personal_agent_selection_changed");
});

chrome.storage.onChanged?.addListener((changes, area) => {
  if (area === "local" && changes.token) reconnectSocket();
});

chrome.tabs.onActivated.addListener(() => scheduleCapture(true));
chrome.tabs.onUpdated.addListener((_tabId, changeInfo, tab) => {
  if (tab.active && (changeInfo.status === "complete" || changeInfo.url)) scheduleCapture(true);
});
chrome.windows.onFocusChanged.addListener((windowId) => {
  if (windowId !== chrome.windows.WINDOW_ID_NONE) scheduleCapture(true);
});
chrome.tabs.onRemoved.addListener(() => scheduleCapture(true));
chrome.windows.onRemoved.addListener(() => scheduleCapture(true));
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name !== TAB_REFRESH_ALARM) return;
  void connectSocket();
  scheduleCapture();
});

async function initialize() {
  await chrome.alarms.create(TAB_REFRESH_ALARM, { periodInMinutes: 1 });
  void connectSocket();
  scheduleCapture(true);
}

chrome.runtime.onInstalled.addListener(() => { void initialize(); });
chrome.runtime.onStartup.addListener(() => { void initialize(); });

void connectSocket();  // also runs whenever the service worker is restarted by any event
