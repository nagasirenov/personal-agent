"""On-demand, read-only text extraction from the foreground Windows window."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from .media import terminate_tree


_MAX_TEXT = 12_000
_MAX_OUTPUT = 128_000
_PRIVATE_ENV_NAMES = {"TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"}
_REDACTIONS = (
    re.compile(r"\b\d{5,16}:[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bBearer\s+[^\s\"'<>]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b((?:api[_ -]?key|secret|token|password|пароль)\s*[:=]\s*)\S+", re.IGNORECASE),
)

_POWERSHELL_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class PersonalAgentForegroundWindow {
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();
}
"@
$handle = [PersonalAgentForegroundWindow]::GetForegroundWindow()
if ($handle -eq [IntPtr]::Zero) {
    [ordered]@{ ok = $false; code = 'no_foreground_window' } | ConvertTo-Json -Compress
    exit 0
}
$root = [System.Windows.Automation.AutomationElement]::FromHandle($handle)
$rootInfo = $root.Current
$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
$queue = New-Object System.Collections.Queue
$queue.Enqueue(@($root, 0))
$parts = New-Object 'System.Collections.Generic.List[string]'
$seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
$characters = 0
$elements = 0
$truncated = $false
while ($queue.Count -gt 0 -and $elements -lt 180 -and $characters -lt 12000) {
    $entry = $queue.Dequeue()
    $element = $entry[0]
    $depth = [int]$entry[1]
    try { $current = $element.Current } catch { continue }
    $elements++
    if ($current.IsPassword) { continue }
    if (-not $current.IsOffscreen -and -not $current.IsPassword) {
        $values = New-Object 'System.Collections.Generic.List[string]'
        if ($depth -gt 0 -and $current.Name) { $values.Add([string]$current.Name) }
        try {
            $textPattern = $element.GetCurrentPattern([System.Windows.Automation.TextPattern]::Pattern)
            $ranges = $textPattern.GetVisibleRanges()
            for ($rangeIndex = 0; $rangeIndex -lt [Math]::Min(8, $ranges.Length); $rangeIndex++) {
                $visibleText = $ranges[$rangeIndex].GetText(1801)
                if ($visibleText.Length -gt 1800) { $visibleText = $visibleText.Substring(0, 1800); $truncated = $true }
                if ($visibleText) { $values.Add([string]$visibleText) }
            }
            if ($ranges.Length -gt 8) { $truncated = $true }
        } catch { }
        foreach ($value in $values) {
            if ($value.Length -gt 1800) { $value = $value.Substring(0, 1800); $truncated = $true }
            $value = ([string]$value -replace '[\r\n\t]+', ' ').Trim()
            if (-not $value -or -not $seen.Add($value)) { continue }
            $remaining = 12000 - $characters
            if ($value.Length -gt $remaining) {
                if ($remaining -gt 0) { $parts.Add($value.Substring(0, $remaining)); $characters += $remaining }
                $truncated = $true
                break
            }
            $parts.Add($value)
            $characters += $value.Length + 1
        }
    }
    if ($depth -ge 12) { $truncated = $true; continue }
    $child = $null
    try { $child = $walker.GetFirstChild($element) } catch { }
    $siblings = 0
    while ($child -and $siblings -lt 40 -and $elements + $queue.Count -lt 180) {
        $queue.Enqueue(@($child, $depth + 1))
        $siblings++
        try { $child = $walker.GetNextSibling($child) } catch { $child = $null }
    }
    if ($child) { $truncated = $true }
}
if ($queue.Count -gt 0) { $truncated = $true }
$text = [string]::Join("`n", $parts)
$title = [string]$rootInfo.Name
if ($title.Length -gt 500) { $title = $title.Substring(0, 500) }
[ordered]@{
    ok = $true
    kind = 'read_active_window'
    window_title = $title
    process_id = [int]$rootInfo.ProcessId
    text = $text
    element_count = $elements
    truncated = $truncated
} | ConvertTo-Json -Compress -Depth 3
'''


def _error(code: str) -> dict:
    return {"ok": False, "kind": "read_active_window", "error": {"code": code},
            "provenance": {"source": "windows_uia", "read_only": True}}


def _clean_text(value: str) -> tuple[str, int]:
    text = value
    count = 0
    for pattern in _REDACTIONS:
        text, replaced = pattern.subn(
            lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", text
        )
        count += replaced
    return text[:_MAX_TEXT], count


async def read_active_window() -> dict:
    """Read bounded accessible text from only the current foreground window.

    UI Automation runs in a separate MTA PowerShell process so a hung provider
    cannot block the daemon's event loop or leave its COM apartment initialized.
    """
    if sys.platform != "win32":
        return _error("unsupported_platform")
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        return _error("powershell_unavailable")
    env = {key: value for key, value in os.environ.items() if key.upper() not in _PRIVATE_ENV_NAMES}
    encoded = base64.b64encode(_POWERSHELL_SCRIPT.encode("utf-16-le")).decode("ascii")
    process = None
    try:
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-MTA", "-EncodedCommand", encoded,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=env,
        ))
        try:
            process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            process = await spawn
            raise
        output, _ = await asyncio.wait_for(process.communicate(), 8)
        if process.returncode != 0:
            return _error("uia_unavailable")
        return _normalize_response(output)
    except TimeoutError:
        return _error("uia_timeout")
    except OSError:
        return _error("uia_unavailable")
    finally:
        if process is not None:
            await asyncio.shield(terminate_tree(process))


def _normalize_response(output: bytes) -> dict:
    if len(output) > _MAX_OUTPUT:
        return _error("uia_response_too_large")
    try:
        payload = json.loads(output.decode("utf-8-sig"))
    except (json.JSONDecodeError, UnicodeError):
        return _error("uia_invalid_response")
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        code = payload.get("code") if isinstance(payload, dict) else None
        return _error(code if code == "no_foreground_window" else "uia_unavailable")
    title = payload.get("window_title", "")
    text = payload.get("text", "")
    if not isinstance(title, str) or not isinstance(text, str):
        return _error("uia_invalid_response")
    title, title_redactions = _clean_text(title)
    text, redactions = _clean_text(text)
    process_id = payload.get("process_id")
    element_count = payload.get("element_count")
    return {
        "ok": True,
        "kind": "read_active_window",
        "window_title": title[:500],
        "process_id": process_id if type(process_id) is int and process_id > 0 else None,
        "text": text,
        "character_count": len(text),
        "element_count": element_count if type(element_count) is int and element_count >= 0 else 0,
        "truncated": payload.get("truncated") is True or len(payload.get("text", "")) > _MAX_TEXT,
        "redactions_applied": redactions + title_redactions,
        "sufficient_text": bool(text.strip()),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {"source": "windows_uia", "read_only": True, "scope": "foreground_window", "coverage": "partial_accessibility_text"},
    }
