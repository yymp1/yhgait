@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_win11_training.ps1" %*
endlocal
