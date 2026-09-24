@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ^
  "%~dp0scripts\manager.ps1"
set "exit_code=%ERRORLEVEL%"
echo.
pause
exit /b %exit_code%
