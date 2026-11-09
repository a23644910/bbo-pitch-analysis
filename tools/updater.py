"""
自動更新：從 GitHub 下載最新版程式，只換掉程式檔。
不會動到：data\\（紀錄）、.venv\\（Python 環境）、config\\calibration.json（你的 ID 與設定）。

單獨執行：  .venv\\Scripts\\python.exe tools\\updater.py
"""
from __future__ import annotations
import io
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, Optional

REPO = "a23644910/bbo-pitch-analysis"
BRANCH = "main"
ROOT = Path(__file__).resolve().parent.parent
VERSION_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/version.json"
ZIP_URL = f"https://codeload.github.com/{REPO}/zip/refs/heads/{BRANCH}"

# 這些路徑永遠不覆蓋
KEEP_PREFIXES = ("data/", ".venv/", "診斷結果/", "backup")
KEEP_FILES = {"config/calibration.json"}


def _get(url: str, timeout: float = 10.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "BBO-updater", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def local_version() -> str:
    try:
        return json.loads((ROOT / "version.json").read_text(encoding="utf-8")).get("version", "")
    except Exception:
        return ""


def remote_info(timeout: float = 5.0) -> Optional[Dict]:
    try:
        return json.loads(_get(VERSION_URL, timeout).decode("utf-8"))
    except Exception:
        return None


def check() -> Dict:
    loc = local_version()
    rem = remote_info()
    return {
        "local": loc,
        "remote": (rem or {}).get("version", ""),
        "notes": (rem or {}).get("notes", ""),
        "available": bool(rem and rem.get("version") and rem.get("version") != loc),
        "reachable": rem is not None,
    }


def apply_update(log: Callable[[str], None] = print) -> bool:
    log("下載最新版程式…")
    data = _get(ZIP_URL, timeout=60)
    old_req = (ROOT / "requirements.txt").read_text(encoding="utf-8") if (ROOT / "requirements.txt").exists() else ""
    n = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        top = names[0].split("/")[0] + "/"
        for name in names:
            if name.endswith("/") or not name.startswith(top):
                continue
            rel = name[len(top):]
            if not rel or rel in KEEP_FILES or rel.startswith(KEEP_PREFIXES):
                continue
            dest = ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    log(f"已更新 {n} 個程式檔")
    new_req = (ROOT / "requirements.txt").read_text(encoding="utf-8") if (ROOT / "requirements.txt").exists() else ""
    if new_req != old_req and new_req:
        log("新版需要新的套件，安裝中…")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
        if r.returncode != 0:
            log("套件安裝失敗，請確認網路後再執行一次「更新.bat」")
            return False
    log(f"更新完成：{local_version()}")
    return True


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    info = check()
    if not info["reachable"]:
        print("連不到更新伺服器（GitHub），請確認網路")
        sys.exit(1)
    print(f"目前版本：{info['local'] or '未知'}　最新版本：{info['remote']}")
    if not info["available"] and "--force" not in sys.argv:
        print("已經是最新版")
        sys.exit(0)
    sys.exit(0 if apply_update() else 1)
