@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 從軌跡錄影切出「全部」的球（已經切過的會跳過），約需 10～15 分鐘...
".venv\Scripts\python.exe" tools\cut_pitch_clips.py --all
echo.
echo 片段在 data\trajectory\clips 資料夾
pause
