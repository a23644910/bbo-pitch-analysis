@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 從軌跡錄影切出每一球的小片段（預設 120 球，各球種平均挑）...
echo 需要幾分鐘，請稍候。
".venv\Scripts\python.exe" tools\cut_pitch_clips.py %*
echo.
echo 片段在 data\trajectory\clips 資料夾
pause
