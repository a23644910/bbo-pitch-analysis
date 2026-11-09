@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 請先開遊戲並停在「打擊畫面」（看得到好球帶），再按任意鍵開始診斷...
pause >nul
".venv\Scripts\python.exe" tools\diagnose.py
echo.
echo 請把資料夾裡的「診斷結果.zip」傳給開發者
pause
