"""
軌跡辨識用錄影：只錄遊戲畫面（不含標題列、其他視窗、桌面），固定每秒 60 張。

規格（給軌跡分析用）：
- 1920x1080（遊戲畫面等比縮放）、60 fps 固定幀率、H.264 高畫質（有 NVIDIA 顯卡用 NVENC，不吃 CPU；沒有就用 CPU 編碼）
- 同名 .csv：每一張畫面實際被截到的時間（毫秒），用來精準算球速
- 存在 data\\trajectory\\<日期時間>.mp4

用法：雙擊「4_軌跡錄影.bat」。看到比賽結算畫面（確認BOX）會自動停止；也可以在黑色視窗按 Enter 結束；最長錄 2 小時。
可以同時開著分析程式；如果遊戲明顯變卡，錄影時先把分析程式關掉。
"""
from __future__ import annotations
import ctypes
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402

OUT_W, OUT_H, FPS = 1920, 1080, 60
MAX_SEC = 2 * 60 * 60
GAMEOVER_CHECK_EVERY = 30   # 每 0.5 秒看一次是不是結算畫面
GAMEOVER_NEED = 4           # 連續 4 次（約 2 秒）才算比賽結束
MIN_SEC_BEFORE_AUTOSTOP = 120


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def pick_encoder(ff: str):
    """有 NVIDIA 顯卡就用 NVENC（幾乎不吃 CPU），否則用 CPU 的 x264。先實際試編一小段確認能用。"""
    nvenc = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
    x264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    try:
        test = subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x144:r=60",
             "-frames:v", "5", *nvenc, "-f", "null", "-"],
            capture_output=True, timeout=20,
        )
        if test.returncode == 0:
            return nvenc, "NVIDIA 顯卡編碼（NVENC）"
    except Exception:
        pass
    return x264, "CPU 編碼（x264）"


def main() -> int:
    from capture.window import find_game_window
    win = find_game_window()
    if win is None:
        print("找不到遊戲視窗：請先開遊戲（視窗模式，不要縮到最小）")
        return 1
    print(f"遊戲畫面：{win.width}x{win.height}（位置 {win.left},{win.top}）")
    try:
        import dxcam
    except Exception:
        print("缺少 dxcam 套件，請先執行「1_第一次安裝.bat」")
        return 1
    sw, sh = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    if win.left < 0 or win.top < 0 or win.right > sw or win.bottom > sh:
        print("遊戲視窗要完整放在「主螢幕」上才能用高速錄影，請把視窗移回主螢幕再試一次")
        return 1

    ff = ffmpeg_exe()
    enc, enc_name = pick_encoder(ff)
    out_dir = ROOT / "data" / "trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4 = out_dir / f"{stamp}.mp4"
    csv = out_dir / f"{stamp}.csv"

    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-",
           *enc, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    cam = dxcam.create(output_color="BGR")
    cam.start(region=(win.left, win.top, win.right, win.bottom), target_fps=FPS, video_mode=True)

    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    print(f"開始錄影（{enc_name}）→ {mp4.name}")
    print("錄影中…比賽結束（結算畫面）會自動停止，也可以按 Enter 結束")
    from detect.gameover import has_confirm_box_button
    over_streak = 0

    t0 = time.perf_counter()
    n = 0
    slow = 0
    with open(csv, "w", encoding="utf-8") as fcsv:
        fcsv.write("frame,ms\n")
        try:
            while not stop.is_set() and time.perf_counter() - t0 < MAX_SEC:
                frame = cam.get_latest_frame()
                if frame is None:
                    continue
                ts = (time.perf_counter() - t0) * 1000.0
                if frame.shape[1] != OUT_W or frame.shape[0] != OUT_H:
                    frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
                try:
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                except (BrokenPipeError, OSError):
                    print("編碼程式停止了，結束錄影")
                    break
                fcsv.write(f"{n},{ts:.1f}\n")
                n += 1
                if n % GAMEOVER_CHECK_EVERY == 0 and ts / 1000.0 > MIN_SEC_BEFORE_AUTOSTOP:
                    try:
                        over_streak = over_streak + 1 if has_confirm_box_button(frame) else 0
                    except Exception:
                        over_streak = 0
                    if over_streak >= GAMEOVER_NEED:
                        print("偵測到比賽結算畫面，自動停止錄影")
                        break
                if n % (FPS * 10) == 0:
                    real = n / max(0.001, time.perf_counter() - t0)
                    if real < FPS * 0.9:
                        slow += 1
                    print(f"  已錄 {n // FPS} 秒（實際每秒 {real:.1f} 張）", flush=True)
        except KeyboardInterrupt:
            pass
    cam.stop()
    try:
        proc.stdin.close()
    except Exception:
        pass
    proc.wait()
    dur = time.perf_counter() - t0
    size = mp4.stat().st_size / 1024 / 1024 if mp4.exists() else 0
    print(f"完成：{mp4}（{dur:.0f} 秒、{n} 張、{size:.0f} MB）")
    avg = n / max(0.001, dur)
    print(f"平均每秒 {avg:.1f} 張")
    if avg < 50:
        print("注意：整段平均低於每秒 50 張，電腦可能太忙；遊戲明顯變卡時再關掉其他程式")
    return 0


if __name__ == "__main__":
    sys.exit(main())
