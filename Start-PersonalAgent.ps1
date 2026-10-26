param(
    [string]$CredentialsPath = 'C:\Users\nagas\Desktop\bot_api.txt',
    [string]$ConfigPath = (Join-Path $PSScriptRoot 'config.local.toml')
)

$ErrorActionPreference = 'Stop'
$workspacePath = $PSScriptRoot
$pythonPath = Join-Path $workspacePath '.venv\Scripts\python.exe'
$statePath = Join-Path $workspacePath '.state'
$pidPath = Join-Path $statePath 'daemon.pid'
$ConfigPath = (Resolve-Path -LiteralPath $ConfigPath).Path
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'Project .venv is missing; follow README setup instructions.'
}

if (Test-Path -LiteralPath $pidPath) {
    $savedProcessId = 0
    if ([int]::TryParse((Get-Content -LiteralPath $pidPath -Raw).Trim(), [ref]$savedProcessId)) {
        $existingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $savedProcessId"
        if ($existingProcess -and $existingProcess.CommandLine -like '*-m personal_agent run*' -and $existingProcess.CommandLine.Contains($ConfigPath)) {
            Write-Output "Personal agent already running. PID: $savedProcessId"
            return
        }
    }
}

$credentialText = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $CredentialsPath).Path)
$botTokens = @([regex]::Matches($credentialText, '\b\d{5,16}:[A-Za-z0-9_-]{20,}\b') | ForEach-Object { $_.Value } | Select-Object -Unique)
if ($botTokens.Count -ne 1) {
    throw 'Expected exactly one Telegram bot token in the credentials file.'
}
$logPath = Join-Path $statePath 'logs'
[System.IO.Directory]::CreateDirectory($logPath) | Out-Null
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$stdoutPath = Join-Path $logPath "daemon-$runStamp.stdout.log"
$stderrPath = Join-Path $logPath "daemon-$runStamp.stderr.log"
$previousToken = [Environment]::GetEnvironmentVariable('TELEGRAM_BOT_TOKEN', 'Process')
try {
    [Environment]::SetEnvironmentVariable('TELEGRAM_BOT_TOKEN', $botTokens[0], 'Process')
    $daemonProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'personal_agent', 'run', '--config', ('"' + $ConfigPath + '"')) -WorkingDirectory $workspacePath -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
} finally {
    [Environment]::SetEnvironmentVariable('TELEGRAM_BOT_TOKEN', $previousToken, 'Process')
    $credentialText = $null
    $botTokens = $null
}
$daemonProcess.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
@{ pid = $daemonProcess.Id; config = $ConfigPath; stdout = $stdoutPath; stderr = $stderrPath; started_at_utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $statePath 'daemon-process.json') -Encoding UTF8
Write-Output "Personal agent started. PID: $($daemonProcess.Id)"
Write-Output "Logs: $logPath"
