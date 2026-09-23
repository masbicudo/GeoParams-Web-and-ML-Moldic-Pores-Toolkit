@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ^
  "%~dp0remove-docker-app.ps1"
set "exit_code=%ERRORLEVEL%"
echo.
pause
exit /b %exit_code%
