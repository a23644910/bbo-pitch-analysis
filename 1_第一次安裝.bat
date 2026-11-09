@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo   BBO 投球分析 - 第一次安裝（約 5~15 分鐘）
echo ==========================================
echo.
set PY=
py -3.13 -c "import sys" >nul 2>&1 && set PY=py -3.13
if defined PY goto havepy
python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,13) else 1)" >nul 2>&1 && set PY=python
if defined PY goto havepy

echo 找不到 Python 3.13，正在自動安裝（會跳出安裝視窗，請按允許）...
winget install -e --id Python.Python.3.13 --scope user --accept-source-agreements --accept-package-agreements
if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
  set PY="%LocalAppData%\Programs\Python\Python313\python.exe"
  goto havepy
)
echo.
echo [失敗] 無法自動安裝 Python。
echo 請到 https://www.python.org/downloads/ 下載 Python 3.13 安裝（勾選 Add python.exe to PATH），再重新執行本檔。
pause
exit /b 1

:havepy
echo [1/3] 建立獨立的 Python 環境...
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo [失敗] 無法建立 Python 環境。
  pause
  exit /b 1
)
echo [2/3] 下載安裝需要的套件（檔案較大，請耐心等候）...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo [失敗] 套件安裝失敗，請確認網路後重新執行本檔。
  pause
  exit /b 1
)
echo [3/3] 預先下載文字辨識模型...
".venv\Scripts\python.exe" -c "from paddleocr import PaddleOCR; PaddleOCR(lang='chinese_cht', enable_mkldnn=False, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)"
echo.
echo ==========================================
echo   安裝完成！之後每次都用「2_啟動.bat」開啟
echo ==========================================
pause
