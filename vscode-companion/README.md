# Personal Agent Context Bridge

This local VS Code extension writes small, short-lived JSON snapshots for the Personal Agent observer. It has no runtime dependencies and runs in the local UI extension host.

## Snapshot data

Snapshots contain the workspace and open file paths, active language, selection length and line count, diagnostic counts, the Git branch, and terminal names with process IDs and working directories when VS Code exposes them. They retain the full open-file count while storing at most 50 paths and 12 terminal entries. Selected text, document contents, terminal output, and shell history are never read or stored. Common credential filenames and key files are omitted, and common token/password patterns are redacted from metadata.

Files are written atomically under `%APPDATA%\PersonalAgent\vscode-context`. The Python observer accepts schema version 1, ignores stale snapshots, and caps its reads. Deactivating the extension removes its snapshot; if VS Code stops unexpectedly, the observer expires it after its stale window.

## Local installation

Open this folder in VS Code and use **Developer: Install Extension from Location...**, or package it as a VSIX with the VS Code CLI and install that file. No marketplace publication is required. After installation, `/diagnostics` in the Personal Agent should report the `vscode` observer as available when VS Code is open.

For a reproducible PowerShell installation, run these commands from this directory with Node.js and the VS Code CLI installed:

```powershell
npm.cmd exec --yes --package @vscode/vsce -- vsce package --out "$env:TEMP\personal-agent-context-bridge-0.1.0.vsix"
code --install-extension "$env:TEMP\personal-agent-context-bridge-0.1.0.vsix"
code --list-extensions --show-versions | Select-String 'personal-agent.personal-agent-context-bridge'
```

The extension activates on startup in supported VS Code windows. If a folder or file is opened with restricted Workspace Trust, VS Code may prevent extension activation until you explicitly trust that workspace. Use the normal VS Code trust dialog if appropriate; do not disable Workspace Trust to enable the bridge. Newly opened trusted windows load the installed extension; already-open windows may need **Developer: Reload Window** when convenient.
