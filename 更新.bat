@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 檢查並下載最新版（紀錄和設定不會動）...
".venv\Scripts\python.exe" tools\updater.py
pause
