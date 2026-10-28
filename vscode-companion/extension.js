"use strict";

const vscode = require("vscode");
const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");

const SCHEMA_VERSION = 1;
const SNAPSHOT_INTERVAL_MS = 30_000;
const DEBOUNCE_MS = 400;
const MAX_FILES = 50;
const MAX_TERMINALS = 12;
const MAX_FOLDERS = 16;
const SENSITIVE_BASENAME = /^(?:\.env(?:\..*)?|credentials?(?:\..*)?|secrets?(?:\..*)?|id_(?:rsa|ed25519)|authorized_keys|known_hosts)$/i;
const SENSITIVE_EXTENSION = /\.(?:pem|key|p12|pfx|kdbx)$/i;
const SECRET_PATTERNS = [
  /\b\d{5,}:[A-Za-z0-9_-]{20,}\b/gi,
  /\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b/gi,
  /\b(?:token|password|secret|api[_ -]?key)\s*[:=]\s*\S+/gi,
];

let snapshotPath;
let windowId;
let interval;
let debounceTimer;
let writeQueue = Promise.resolve();
let output;

function clean(value, limit = 300) {
  if (typeof value !== "string") return "";
  let result = value.replace(/[\0\r\n\t]/g, " ").replace(/\s+/g, " ").trim();
  for (const pattern of SECRET_PATTERNS) result = result.replace(pattern, "[redacted]");
  return result.slice(0, limit);
}

function cleanPath(value, limit = 1024) {
  if (typeof value !== "string") return "";
  let result = value.replace(/[\0\r\n\t]/g, " ");
  for (const pattern of SECRET_PATTERNS) result = result.replace(pattern, "[redacted]");
  return result.slice(0, limit);
}

function localPath(uri) {
  if (!uri || uri.scheme !== "file" || uri.authority) return "";
  const value = cleanPath(uri.fsPath);
  if (!value) return "";
  const base = path.basename(value);
  return SENSITIVE_BASENAME.test(base) || SENSITIVE_EXTENSION.test(base) ? "" : value;
}

function uniquePaths(values) {
  return [...new Set(values.filter(Boolean))];
}

function timestamp() {
  return new Date().toISOString();
}

async function terminalMetadata(terminals) {
  const result = [];
  for (const terminal of terminals.slice(0, MAX_TERMINALS)) {
    const item = { name: clean(terminal.name, 80) };
    try {
      const pid = await terminal.processId;
      if (Number.isInteger(pid) && pid > 0) item.process_id = pid;
    } catch (_) {
      // A terminal can close while its process id is being queried.
    }
    try {
      const cwd = localPath(terminal.shellIntegration?.cwd);
      if (cwd) item.cwd = cwd;
    } catch (_) {
      // Shell integration is optional and may not be ready yet.
    }
    result.push(item);
  }
  return result;
}

async function gitBranch() {
  try {
    const extension = vscode.extensions.getExtension("vscode.git");
    if (!extension) return "";
    const exports = extension.isActive ? extension.exports : await extension.activate();
    const api = exports.getAPI(1);
    const repo = api.repositories.find((item) => item.rootUri?.scheme === "file" && !item.rootUri.authority);
    return clean(repo?.state?.HEAD?.name, 120);
  } catch (_) {
    return "";
  }
}

function openFilePaths() {
  const tabGroups = vscode.window.tabGroups?.all || [];
  const fromTabs = tabGroups.flatMap((group) => group.tabs || []).map((tab) => {
    const input = tab.input;
    return input && input.uri ? localPath(input.uri) : "";
  });
  const fromDocuments = vscode.workspace.textDocuments.map((document) => localPath(document.uri));
  const all = uniquePaths([...fromTabs, ...fromDocuments]);
  return { items: all.slice(0, MAX_FILES), count: all.length };
}

function diagnosticCounts() {
  const counts = { error: 0, warning: 0, information: 0, hint: 0 };
  const keys = ["error", "warning", "information", "hint"];
  let total = 0;
  for (const [, diagnostics] of vscode.languages.getDiagnostics()) {
    for (const diagnostic of diagnostics) {
      if (++total > 50_000) return counts;
      const key = keys[diagnostic.severity];
      if (key) counts[key] += 1;
    }
  }
  return counts;
}

async function collectSnapshot() {
  const editor = vscode.window.activeTextEditor;
  const activeFile = editor ? localPath(editor.document.uri) : "";
  let selection = { length: 0, lines: 0 };
  if (editor && !editor.selection.isEmpty) {
    selection = {
      length: Math.min(
        editor.document.offsetAt(editor.selection.end) - editor.document.offsetAt(editor.selection.start),
        1_000_000,
      ),
      lines: Math.min(Math.abs(editor.selection.end.line - editor.selection.start.line) + 1, 1_000_000),
    };
  }
  const folders = (vscode.workspace.workspaceFolders || [])
    .slice(0, MAX_FOLDERS)
    .map((folder) => localPath(folder.uri))
    .filter(Boolean);
  const terminals = await terminalMetadata(vscode.window.terminals || []);
  const openFiles = openFilePaths();
  return {
    schema_version: SCHEMA_VERSION,
    captured_at: timestamp(),
    workspace: folders[0] || "",
    workspace_folders: folders,
    active_file: activeFile,
    open_files: openFiles.items,
    open_file_count: openFiles.count,
    selection,
    diagnostics: diagnosticCounts(),
    git_branch: await gitBranch(),
    active_editor_language: clean(editor?.document?.languageId, 60),
    terminals,
    terminal_count: (vscode.window.terminals || []).length,
  };
}

async function writeSnapshot() {
  if (!snapshotPath) return;
  const snapshot = await collectSnapshot();
  const tempPath = `${snapshotPath}.${windowId}.tmp`;
  const payload = JSON.stringify(snapshot);
  try {
    await fs.promises.mkdir(path.dirname(snapshotPath), { recursive: true });
    await fs.promises.writeFile(tempPath, payload, { encoding: "utf8" });
    await fs.promises.rename(tempPath, snapshotPath);
  } catch (error) {
    try { await fs.promises.unlink(tempPath); } catch (_) { /* no partial file */ }
    output?.appendLine(`Snapshot write failed: ${error.code || error.name || "Error"}`);
  }
}

function enqueueSnapshot() {
  writeQueue = writeQueue.then(writeSnapshot).catch((error) => {
    output?.appendLine(`Snapshot capture failed: ${error.code || error.name || "Error"}`);
  });
  return writeQueue;
}

function scheduleSnapshot() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => { void enqueueSnapshot(); }, DEBOUNCE_MS);
}

function activate(context) {
  output = vscode.window.createOutputChannel("Personal Agent Context Bridge");
  context.subscriptions.push(output);
  const appData = process.env.APPDATA || path.join(os.homedir(), "AppData", "Roaming");
  snapshotPath = path.join(appData, "PersonalAgent", "vscode-context", `${process.pid}-${crypto.randomBytes(8).toString("hex")}.json`);
  windowId = path.basename(snapshotPath, ".json");

  const events = [
    vscode.window.onDidChangeActiveTextEditor,
    vscode.workspace.onDidChangeWorkspaceFolders,
    vscode.workspace.onDidOpenTextDocument,
    vscode.workspace.onDidCloseTextDocument,
    vscode.workspace.onDidChangeTextDocument,
    vscode.languages.onDidChangeDiagnostics,
    vscode.window.onDidOpenTerminal,
    vscode.window.onDidCloseTerminal,
    vscode.window.onDidChangeWindowState,
  ];
  for (const event of events) {
    if (event) context.subscriptions.push(event(scheduleSnapshot));
  }

  interval = setInterval(() => { void enqueueSnapshot(); }, SNAPSHOT_INTERVAL_MS);
  context.subscriptions.push({ dispose: () => clearInterval(interval) });
  scheduleSnapshot();
  void enqueueSnapshot();
}

async function deactivate() {
  clearInterval(interval);
  clearTimeout(debounceTimer);
  await writeQueue;
  if (snapshotPath) {
    try { await fs.promises.unlink(snapshotPath); } catch (_) { /* observer will age out stale snapshots */ }
  }
}

module.exports = { activate, deactivate };
