@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 先開遊戲（視窗模式、放在主螢幕），再按任意鍵開始錄影...
pause >nul
".venv\Scripts\python.exe" tools\record_game.py
echo.
echo 錄好的影片在 data\trajectory 資料夾
pause
