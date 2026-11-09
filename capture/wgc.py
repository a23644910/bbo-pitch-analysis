"""
背景擷取：用 Windows Graphics Capture（windows-capture 套件）直接錄遊戲視窗。
遊戲被其他視窗蓋住、或 Alt+Tab 到別的視窗時，仍拿得到遊戲畫面（最小化時拿不到）。

需要先安裝：.venv\\Scripts\\python -m pip install windows-capture
（可直接雙擊「install_background_capture.bat」）
"""

from __future__ import annotations
import ctypes
import threading
import time
from typing import Optional

import numpy as np

try:
    from windows_capture import WindowsCapture  # type: ignore
except Exception:  # 套件未安裝
    WindowsCapture = None


def _extended_frame_origin(hwnd: int):
    """視窗實際畫出來的外框左上角（不含看不見的縮放邊框）。WGC 擷取的影像就是從這裡開始。"""
    try:
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
        rect = RECT()
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        res = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            ctypes.c_void_p(hwnd), ctypes.c_uint(DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect), ctypes.sizeof(rect),
        )
        if res == 0:
            return rect.left, rect.top
    except Exception:
        pass
    return None


class BackgroundCapture:
    available = WindowsCapture is not None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self._hwnd: Optional[int] = None
        self._control = None
        self._closed = False
        self.last_error: str = ""

    def _start(self, hwnd: int) -> bool:
        if WindowsCapture is None:
            self.last_error = "未安裝 windows-capture"
            return False
        self.stop()
        try:
            cap = None
            errors = []
            # 這台電腦不支援切換黃色外框／游標（Windows 10 常見），所以只用預設值；依序嘗試
            for opts in ({"cursor_capture": False}, {}):
                try:
                    cap = self._try_create(hwnd, opts)
                    break
                except Exception as e:
                    errors.append(f"{type(e).__name__}: {e}")
            if cap is None:
                raise RuntimeError(" / ".join(errors))

            self._closed = False
            self._control = cap._bbo_control
            self._hwnd = hwnd
            return True
        except Exception as e:  # 例如 Windows 版本太舊、不支援
            self.last_error = f"{type(e).__name__}: {e}"
            self._control = None
            self._hwnd = None
            return False

    def _try_create(self, hwnd: int, opts: dict):
        """建立並啟動擷取；不支援的選項會在啟動時丟出例外，交給上一層換下一組設定。"""
        cap = WindowsCapture(window_hwnd=int(hwnd), **opts)

        @cap.event
        def on_frame_arrived(frame, capture_control):
            buf = np.array(frame.frame_buffer[:, :, :3], copy=True)
            with self._lock:
                self._frame = buf
                self._frame_ts = time.time()

        @cap.event
        def on_closed():
            self._closed = True

        cap._bbo_control = cap.start_free_threaded()
        return cap

    def stop(self) -> None:
        if self._control is not None:
            try:
                self._control.stop()
            except Exception:
                pass
        self._control = None
        self._hwnd = None
        with self._lock:
            self._frame = None

    def grab_client(self, hwnd: int, client_left: int, client_top: int,
                    width: int, height: int) -> Optional[np.ndarray]:
        """回傳遊戲畫面（只含遊戲內容，不含標題列），拿不到時回傳 None。"""
        if self._hwnd != hwnd or self._closed:
            if not self._start(hwnd):
                return None
            # 剛開始錄，等第一張畫面
            t0 = time.time()
            while time.time() - t0 < 1.0:
                with self._lock:
                    if self._frame is not None:
                        break
                time.sleep(0.02)
        with self._lock:
            frame = self._frame
            ts = self._frame_ts
        if frame is None or time.time() - ts > 3.0:
            return None
        origin = _extended_frame_origin(hwnd)
        ox, oy = (client_left - origin[0], client_top - origin[1]) if origin else (0, 0)
        fh, fw = frame.shape[:2]
        if ox < 0 or oy < 0 or ox + width > fw or oy + height > fh:
            # 尺寸對不上時，退而求其次：貼齊右下角裁切（標題列在上方、邊框很細）
            ox = max(0, (fw - width) // 2)
            oy = max(0, fh - height - ox)
            if ox + width > fw or oy + height > fh:
                return None
        out = frame[oy:oy + height, ox:ox + width]
        if float(out.mean()) < 3.0:
            return None
        return out
