"""
診斷工具：遊戲開著、停在打擊畫面時執行，把「程式看到什麼」整理成 診斷結果.zip，傳給開發者看。
不會讀取或上傳任何其他檔案，只有：視窗清單（標題）、螢幕資訊、程式截到的遊戲畫面、辨識結果。
"""
from __future__ import annotations
import ctypes
import json
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "診斷結果"
OUT.mkdir(exist_ok=True)
report = []


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    report.append(line)


def section(t):
    log("")
    log("=== " + t + " ===")


try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402

section("系統")
log("Windows:", platform.platform())
log("Python:", sys.version.split()[0])
try:
    log("系統管理員權限:", bool(ctypes.windll.shell32.IsUserAnAdmin()))
except Exception:
    pass
try:
    import mss
    with mss.mss() as m:
        for i, mon in enumerate(m.monitors):
            log(f"螢幕 {i}:", mon)
except Exception as e:
    log("mss 錯誤:", e)
try:
    log("主螢幕縮放 DPI:", ctypes.windll.user32.GetDpiForSystem())
except Exception:
    pass
for name in ("dxcam", "mss", "win32gui", "windows_capture", "paddleocr"):
    try:
        __import__(name)
        log(f"套件 {name}: OK")
    except Exception as e:
        log(f"套件 {name}: 失敗 {e}")

section("視窗清單（含標題的可見視窗）")
try:
    import win32gui
    wins = []

    def cb(h, _):
        if win32gui.IsWindowVisible(h):
            t = win32gui.GetWindowText(h)
            if t.strip():
                l, tp, r, b = win32gui.GetClientRect(h)
                wins.append((t, r - l, b - tp, bool(win32gui.IsIconic(h))))
    win32gui.EnumWindows(cb, None)
    for t, w, h, ic in wins[:60]:
        log(f"  [{w}x{h}{' 最小化' if ic else ''}] {t}")
except Exception as e:
    log("列視窗失敗:", e)

section("找遊戲視窗")
from capture.window import find_game_window, is_foreground  # noqa: E402
win = find_game_window()
if win is None:
    log("找不到遊戲視窗！標題要含「全民打棒球」。請確認遊戲已開、不是最小化。")
else:
    log("找到:", win)
    log(f"畫面大小 {win.width}x{win.height}，比例 {win.width / win.height:.3f}（16:9 = 1.778）")
    log("在最上層:", is_foreground(win.hwnd))

frames = {}
if win is not None:
    section("截圖測試")
    from capture.grabber import ScreenGrabber
    g = ScreenGrabber()
    log("截圖方式:", g.backend_name)
    try:
        img = g.grab_bbox(win.bbox)
        frames["screen"] = img
        log("螢幕截圖:", None if img is None else (img.shape, round(float(img.mean()), 1)))
    except Exception as e:
        log("螢幕截圖失敗:", e)
    try:
        img = g.grab_window_background(win.hwnd, win.width, win.height)
        frames["printwindow"] = img
        log("PrintWindow:", None if img is None else (img.shape, round(float(img.mean()), 1)))
    except Exception as e:
        log("PrintWindow 失敗:", e)
    try:
        from capture.wgc import BackgroundCapture
        bc = BackgroundCapture()
        img = bc.grab_client(win.hwnd, win.left, win.top, win.width, win.height)
        frames["wgc"] = img
        log("背景擷取(WGC):", None if img is None else (img.shape, round(float(img.mean()), 1)), bc.last_error)
        bc.stop()
    except Exception as e:
        log("背景擷取失敗:", e)
    for k, img in frames.items():
        if img is not None and img.size:
            cv2.imencode(".png", img)[1].tofile(str(OUT / f"截圖_{k}.png"))

    section("辨識測試（用螢幕截圖）")
    img = frames.get("screen")
    if img is None or not img.size:
        img = next((v for v in frames.values() if v is not None and v.size), None)
    if img is None:
        log("沒有任何截圖可以測試")
    else:
        try:
            from capture.calibration import CalibrationConfig
            from detect.scoreboard import read_scoreboard
            from detect.cursor import is_batting_camera
            from detect.markers import detect_pitch_marker
            cfg = CalibrationConfig.load()
            h, w = img.shape[:2]
            norm = cv2.resize(img, (1680, int(round(h * 1680 / w))))
            log("記分板:", read_scoreboard(norm, cfg))
            log("打擊畫面:", is_batting_camera(norm, cfg))
            log("落點標記:", detect_pitch_marker(norm, cfg))
            # 畫上校正框
            vis = norm.copy()
            H, W = vis.shape[:2]
            for name in ("strike_zone", "scoreboard", "info_panel", "marker_search_zone"):
                r = getattr(cfg, name, None)
                if r is None:
                    continue
                x1, y1, x2, y2 = (int(r.x1 * W), int(r.y1 * H), int(r.x2 * W), int(r.y2 * H))
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.imencode(".jpg", vis)[1].tofile(str(OUT / "校正框.jpg"))
        except Exception:
            log("辨識測試錯誤:\n" + traceback.format_exc())

(OUT / "報告.txt").write_text("\n".join(report), encoding="utf-8")
zp = ROOT / "診斷結果.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in OUT.iterdir():
        z.write(f, f.name)
log("")
log(f"完成！請把這個檔案傳給開發者：{zp}")
