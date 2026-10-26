@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-PersonalAgent.ps1" %*
if errorlevel 1 (
  echo.
  echo Startup failed. The error is shown above.
  pause
)
endlocal
