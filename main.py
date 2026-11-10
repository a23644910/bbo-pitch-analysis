"""
Main entry point (`main.py`) for 《全民打棒球2》 Real-Time Pitch Recognition & Analysis System.
Usage:
  1. Live capture + Web UI server (default):
     python main.py --port 8000 --batting-row top
  2. Replay an existing video file into the tracker + Web UI:
     python main.py --video path/to/video.mp4 --batting-row top
  3. Show calibration overlay on screen:
     python main.py --calibrate
"""

from __future__ import annotations
import argparse
import os
import socket
import sys
import threading
import time
from pathlib import Path

# Ensure working directory and module path are anchored to the project root
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 輕量化：限制數學運算函式庫的執行緒數（要在載入 numpy / OpenCV / Paddle 之前設定），並把程式設成「低於一般」優先權，讓遊戲優先
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "3")
try:
    import ctypes as _ct
    _ct.windll.kernel32.SetPriorityClass(_ct.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
except Exception:
    pass

# 讓 Windows 回報真實像素座標（不同縮放比例 125%/150%、多螢幕時，截圖位置才不會偏掉）
try:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

import webbrowser
import cv2

cv2.setNumThreads(2)
import uvicorn

from capture.calibration import CalibrationConfig
from capture.grabber import ScreenGrabber
from capture.wgc import BackgroundCapture
from capture.recorder import FlightRecorder, SessionVideo
from capture.overlay import CalibrationOverlay
from capture.window import detect_client_area_in_frame, find_game_window, is_foreground
from ocr.engine import BBOOCREngine
from state.tracker import GameSessionTracker
from store.db import PitchStore
from web.app import WebUIBridge, create_app


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def run_live_capture_loop(
    tracker: GameSessionTracker,
    bridge: WebUIBridge,
    target_fps: float = 30.0,
    stop_event: threading.Event | None = None,
) -> None:
    grabber = ScreenGrabber()
    recorder = bridge.recorder
    frame_interval = 1.0 / max(1.0, target_fps)
    t0 = time.perf_counter()
    fps_counter = 0
    fps_t0 = time.perf_counter()
    last_log_t = 0.0
    last_status_sig = None
    paused_by_focus = False
    bg_capture = BackgroundCapture()
    warned_aspect = False
    idle_since = None
    cpu_t0 = time.process_time()
    last_win_log = 0.0

    while stop_event is None or not stop_event.is_set():
        loop_start = time.perf_counter()
        win = find_game_window()
        client_bgr = None

        if win is not None:
            bridge.connected_window = True
            # 遊戲在最上層：直接截螢幕（最快、顏色和校正時一致）。
            # 遊戲不在最上層（Alt+Tab、被其他視窗蓋住）：改向遊戲視窗本身要畫面，
            # 不會截到蓋在上面的 YouTube 等視窗。
            if is_foreground(win.hwnd):
                if paused_by_focus:
                    print("  [前景] 遊戲回到最上層，改用螢幕擷取", flush=True)
                    paused_by_focus = False
                client_bgr = grabber.grab_bbox((win.left, win.top, win.right, win.bottom))
                tracker.capture_mode = "前景"
            else:
                tracker.capture_mode = "背景"
                client_bgr = bg_capture.grab_client(win.hwnd, win.left, win.top, win.width, win.height)
                if client_bgr is None:
                    client_bgr = grabber.grab_window_background(win.hwnd, win.width, win.height)
                if not paused_by_focus:
                    if client_bgr is None:
                        hint = (
                            "請先執行 install_background_capture.bat 安裝背景擷取元件"
                            if not BackgroundCapture.available
                            else f"背景擷取失敗：{bg_capture.last_error or '拿不到畫面'}"
                        )
                        print(f"  [背景] 遊戲不在最上層，暫停辨識（{hint}）", flush=True)
                    else:
                        print("  [背景] 遊戲不在最上層，改用背景擷取繼續辨識", flush=True)
                    paused_by_focus = True
                if client_bgr is None:
                    time.sleep(0.2)
                    continue
        # 找不到遊戲視窗時不再改截整個螢幕（之前會把 YouTube 等畫面誤認成比賽）

        if client_bgr is None or client_bgr.size == 0:
            if time.perf_counter() - last_win_log > 10:
                last_win_log = time.perf_counter()
                print("  [等待] 找不到遊戲視窗（標題要含「全民打棒球」，且不能縮到最小／不能是全螢幕獨佔模式）", flush=True)
            bridge.connected_window = False
            tracker.fps = 0.0
            time.sleep(0.5)
            continue

        # 一律縮放成 1680 寬：辨識參數都是在 1680x945 調的，視窗比較小（例如 1280x720）時也要放大到同樣大小
        ch, cw = client_bgr.shape[:2]
        if cw != 1680:
            new_h = int(round(ch * (1680.0 / cw)))
            interp = cv2.INTER_AREA if cw > 1680 else cv2.INTER_LINEAR
            client_bgr = cv2.resize(client_bgr, (1680, new_h), interpolation=interp)
        if not warned_aspect and abs(cw / max(1, ch) - 16 / 9) > 0.03:
            warned_aspect = True
            print(f"  [警告] 遊戲畫面比例是 {cw}x{ch}，不是 16:9，辨識位置可能會偏。請把遊戲視窗調成 16:9（例如 1600x900、1920x1080）", flush=True)

        ts_sec = time.perf_counter() - t0
        bridge.last_frame_bgr = client_bgr
        rec = tracker.process_frame(client_bgr, ts_sec)
        # 飛行記錄器：保留最近畫面與判斷過程；可疑狀況自動存檔
        bridge.last_ts = ts_sec
        try:
            dbg = tracker.frame_debug()
            recorder.add_frame(client_bgr, ts_sec, dbg, tracker.game_id)
            bridge.video.add(client_bgr, ts_sec, dbg, tracker.game_id)
            for ev in tracker.events:
                recorder.request(ev["kind"], ts_sec, note=ev.get("note", ""), pitch=ev, auto=True)
        except Exception as e:
            print(f"  [記錄器錯誤] {e}", flush=True)
        tracker.events.clear()
        if tracker.corrections:
            tracker.corrections.clear()
            bridge.notify_from_thread("update")
        if rec is not None:
            print(
                f"  [投球記錄] #{rec['id']} | {rec['inning']}局{rec['half']} | "
                f"落點:{rec['zone']} | 結果:{rec['result']} | "
                f"球種:{rec.get('pitch_raw') or '無標籤'} | 球速:{rec.get('speed_kmh') or '-'} km/h",
                flush=True,
            )
            bridge.notify_from_thread("pitch", rec)

        # Immediately push status via WebSocket whenever any game state field changes
        st_cur = tracker.get_status(connected=bridge.connected_window)
        cur_sig = (
            st_cur.inning,
            st_cur.half,
            st_cur.outs,
            st_cur.balls,
            st_cur.strikes,
            st_cur.batter,
            st_cur.pitcher,
            st_cur.is_my_batting,
            st_cur.my_batting_row,
        )
        if cur_sig != last_status_sig:
            last_status_sig = cur_sig
            bridge.notify_from_thread("status")

        fps_counter += 1
        now = time.perf_counter()
        if now - fps_t0 >= 1.0:
            tracker.fps = fps_counter / (now - fps_t0)
            # 程式佔整台電腦 CPU 的比例（所有執行緒加總，含文字辨識）
            cpu_now = time.process_time()
            tracker.cpu_pct = round((cpu_now - cpu_t0) / (now - fps_t0) / (os.cpu_count() or 1) * 100.0, 1)
            cpu_t0 = cpu_now
            fps_counter = 0
            fps_t0 = now
            bridge.notify_from_thread("status")
            if now - last_log_t >= 5.0:
                last_log_t = now
                st = tracker.get_status(connected=bridge.connected_window)
                mode_str = "視窗直連" if win is not None else "全螢幕擷取"
                print(
                    f"  [狀態] {mode_str} ({st.fps:.1f} FPS) | {st.inning}局{st.half} {st.outs}出局 "
                    f"| 球數:{st.balls}B-{st.strikes}S | 我方進攻:{'是' if st.is_my_batting else '否'} "
                    f"| 打擊游標:{'有' if st.cursor_active else '無'} | 投手:{st.pitcher or '-'}",
                    flush=True,
                )
                if tracker.reject_counts:
                    detail = "、".join(f"{k}×{v}" for k, v in tracker.reject_counts.items())
                    print(f"  [略過] 看到標記但未記錄（幀數）：{detail}", flush=True)
                    tracker.reject_counts = {}

        # 輕量化：不是打擊畫面（守備、換局動畫、結算）超過 2 秒、也沒有正在處理的球 → 降到每秒約 10 張
        interval = frame_interval
        idle = (not getattr(tracker, "_last_batting_cam", True)) and tracker._pending is None
        if idle:
            idle_since = idle_since or time.perf_counter()
            if time.perf_counter() - idle_since > 2.0:
                interval = max(frame_interval, 0.1)
        else:
            idle_since = None
        elapsed = time.perf_counter() - loop_start
        if elapsed < interval:
            time.sleep(interval - elapsed)


def run_video_replay_loop(
    video_path: str | Path,
    tracker: GameSessionTracker,
    bridge: WebUIBridge,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    client_rect = None
    frame_idx = 0
    bridge.connected_window = True
    tracker.fps = fps

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if client_rect is None and frame_idx == 10:
            client_rect = detect_client_area_in_frame(frame)
        if client_rect is not None:
            cx1, cy1, cx2, cy2 = client_rect
            client_bgr = frame[cy1:cy2, cx1:cx2]
        else:
            client_bgr = frame

        ts_sec = frame_idx / fps
        rec = tracker.process_frame(client_bgr, ts_sec)
        if rec is not None:
            bridge.notify_from_thread("pitch", rec)
        frame_idx += 1

    rec = tracker.flush_pending()
    if rec is not None:
        bridge.notify_from_thread("pitch", rec)
    cap.release()
    bridge.connected_window = False
    bridge.notify_from_thread("status")


def main() -> None:
    parser = argparse.ArgumentParser(description="《全民打棒球2》AI 投球辨識與配球分析系統")
    parser.add_argument("--host", default="0.0.0.0", help="Web 伺服器綁定位置 (預設 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Web 伺服器連接埠 (預設 8000)")
    parser.add_argument(
        "--batting-row",
        choices=["top", "bottom"],
        default="top",
        help="我方在計分板上的位置：top=上行(先攻), bottom=下行(後攻)",
    )
    parser.add_argument("--opponent", default="對手1", help="預設對手標籤名稱")
    parser.add_argument("--fps", type=float, default=30.0, help="畫面擷取目標 FPS (預設 30)")
    parser.add_argument("--video", default=None, help="指定測試影片路徑進行離線重播辨識")
    parser.add_argument("--calibrate", action="store_true", help="顯示半透明校正框視窗")
    parser.add_argument("--no-browser", action="store_true", help="啟動時不自動開啟瀏覽器")
    parser.add_argument("--ocr-inprocess", action="store_true", help="文字辨識不開獨立程序（舊做法，除錯用）")
    args = parser.parse_args()

    config = CalibrationConfig.load()
    if args.calibrate:
        overlay = CalibrationOverlay(config)
        overlay.show()
        return

    store = PitchStore()
    # 下一球預測 AI：啟動時在背景用全部資料重訓一次（不到 1 秒）
    threading.Thread(target=lambda: store.retrain_model(), daemon=True).start()
    # 文字辨識放在獨立程序（2026-11-10k）：PaddleOCR 推論會鎖住 Python，放在執行緒裡主程式每球卡 2～3 秒
    ocr_engine = None
    if not args.ocr_inprocess:
        try:
            from ocr.worker import RemoteOCREngine
            ocr_engine = RemoteOCREngine(db_path=store.db_path)
            print("  [文字辨識] 已在獨立程序啟動", flush=True)
        except Exception as e:
            print(f"  [文字辨識] 獨立程序啟動失敗，改用舊做法：{e}", flush=True)
            ocr_engine = None
    if ocr_engine is None:
        ocr_engine = BBOOCREngine()
    tracker = GameSessionTracker(
        store=store,
        ocr_engine=ocr_engine,
        config=config,
        my_batting_row=args.batting_row,
        opponent_tag=args.opponent,
    )
    bridge = WebUIBridge(store=store, tracker=tracker)
    bridge.recorder = FlightRecorder()
    bridge.video = SessionVideo()
    bridge.video.set_enabled(config.record_video)
    app = create_app(store=store, tracker=tracker, bridge=bridge)

    lan_ip = get_lan_ip()
    local_url = f"http://127.0.0.1:{args.port}"
    print("=" * 64)
    print("  《全民打棒球2》AI 投球落點與配球分析系統 已啟動")
    print(f"  本機瀏覽器開啟: {local_url}")
    print(f"  手機 Wi-Fi 開啟: http://{lan_ip}:{args.port}")
    print("=" * 64, flush=True)

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(local_url)).start()

    stop_event = threading.Event()
    if args.video:
        worker = threading.Thread(
            target=run_video_replay_loop,
            args=(args.video, tracker, bridge),
            daemon=True,
        )
    else:
        worker = threading.Thread(
            target=run_live_capture_loop,
            args=(tracker, bridge, args.fps, stop_event),
            daemon=True,
        )
    worker.start()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        stop_event.set()
        try:
            bridge.video.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
