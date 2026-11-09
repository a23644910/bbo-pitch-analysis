@echo off
chcp 65001 >nul
cd /d "%~dp0"
if "%1"=="--after-update" timeout /t 3 /nobreak >nul
if not exist ".venv\Scripts\python.exe" (
  echo 還沒安裝，請先執行「1_第一次安裝.bat」
  pause
  exit /b 1
)
".venv\Scripts\python.exe" main.py --port 8000
pause
