@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_win11_demo.ps1" %*
endlocal
