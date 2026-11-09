"""
High-speed screen grabber for 《全民打棒球2》.
Prioritizes `dxcam` for 30+ fps low-latency capture and falls back to `mss`.
Only captures the game window's client area (or requested ROI sub-regions)
and never reads game memory, network packets, or simulates input.
"""

from __future__ import annotations
import time
from typing import Optional, Tuple, Dict
import cv2
import numpy as np

from .window import ClientRect, find_game_window, detect_client_area_in_frame
from .calibration import CalibrationConfig, BoxRatio

try:
    import dxcam
except Exception:
    dxcam = None

try:
    import mss
except Exception:
    mss = None


class ScreenGrabber:
    """
    Captures the game window's client area or specific sub-regions at 30+ fps.
    Automatically handles window movement via `find_game_window()`.
    """

    def __init__(self, title_keyword: str = "全民打棒球", prefer_dxcam: bool = True):
        self.title_keyword = title_keyword
        self.prefer_dxcam = prefer_dxcam
        self._dxcam_camera = None
        self._mss_sct = None
        self.backend_name = "none"
        self.last_client_rect: Optional[ClientRect] = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.prefer_dxcam and dxcam is not None:
            try:
                self._dxcam_camera = dxcam.create(output_color="BGR")
                if self._dxcam_camera is not None:
                    self.backend_name = "dxcam"
                    return
            except Exception:
                self._dxcam_camera = None

        if mss is not None:
            try:
                self._mss_sct = mss.mss()
                self.backend_name = "mss"
            except Exception:
                self._mss_sct = None

    def refresh_window(self) -> Optional[ClientRect]:
        """Locate the game window and update its client area screen coordinates."""
        rect = find_game_window(self.title_keyword)
        self.last_client_rect = rect
        return rect

    def grab_bbox(self, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Grab a specific screen bounding box (left, top, right, bottom) as BGR."""
        left, top, right, bottom = bbox
        if right <= left or bottom <= top:
            return None

        # dxcam 只能截主螢幕；遊戲在副螢幕（或超出主螢幕範圍）時改用 mss
        on_primary = False
        if self._dxcam_camera is not None:
            try:
                import ctypes
                sw = ctypes.windll.user32.GetSystemMetrics(0)
                sh = ctypes.windll.user32.GetSystemMetrics(1)
                on_primary = left >= 0 and top >= 0 and right <= sw and bottom <= sh
            except Exception:
                on_primary = False
        if self._dxcam_camera is not None and on_primary:
            try:
                frame = self._dxcam_camera.grab(region=(left, top, right, bottom))
                if frame is not None and frame.size > 0:
                    return frame
            except Exception:
                pass

        if self._mss_sct is None and mss is not None:
            try:
                self._mss_sct = mss.mss()
            except Exception:
                return None

        if self._mss_sct is not None:
            try:
                monitor = {
                    "left": int(left),
                    "top": int(top),
                    "width": int(right - left),
                    "height": int(bottom - top),
                }
                raw = self._mss_sct.grab(monitor)
                img = np.asarray(raw)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            except Exception:
                return None
        return None

    def grab_window_background(self, hwnd: int, width: int, height: int) -> Optional[np.ndarray]:
        """
        直接向遊戲視窗要畫面（PrintWindow），遊戲被其他視窗蓋住、或你 Alt+Tab 到別的視窗時也截得到。
        視窗最小化時截不到。回傳 BGR；失敗或整張全黑時回傳 None。
        """
        try:
            import win32gui
            import win32ui
            from ctypes import windll
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        hwnd_dc = mfc_dc = save_dc = bmp = None
        try:
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bmp)
            # 1 = PW_CLIENTONLY（只要遊戲畫面，不含標題列）, 2 = PW_RENDERFULLCONTENT（支援 DirectX 畫面）
            ok = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
            if not ok:
                return None
            bits = bmp.GetBitmapBits(True)
            img = np.frombuffer(bits, dtype=np.uint8).reshape(height, width, 4)
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if float(frame.mean()) < 3.0:
                return None
            return frame
        except Exception:
            return None
        finally:
            try:
                if bmp is not None:
                    win32gui.DeleteObject(bmp.GetHandle())
                if save_dc is not None:
                    save_dc.DeleteDC()
                if mfc_dc is not None:
                    mfc_dc.DeleteDC()
                if hwnd_dc is not None:
                    win32gui.ReleaseDC(hwnd, hwnd_dc)
            except Exception:
                pass

    def grab_client_area(self) -> Tuple[Optional[np.ndarray], Optional[ClientRect]]:
        """Grab the entire client area of the game window."""
        rect = self.refresh_window()
        if rect is None:
            return None, None
        frame = self.grab_bbox(rect.bbox)
        return frame, rect

    def grab_regions(
        self, config: CalibrationConfig
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[ClientRect]]:
        """
        Grab only the required small sub-regions (marker_search_zone, scoreboard,
        info_panel, tab_panel) from the game window's client area for maximum FPS.
        """
        rect = self.refresh_window()
        if rect is None:
            return None, None

        regions = {}
        for name, box in [
            ("marker_search", config.marker_search_zone),
            ("strike_zone", config.strike_zone),
            ("scoreboard", config.scoreboard),
            ("info_panel", config.info_panel),
            ("tab_panel", config.tab_panel),
        ]:
            sub = rect.sub_bbox(box.x1, box.y1, box.x2, box.y2)
            crop = self.grab_bbox(sub)
            if crop is None:
                return None, rect
            regions[name] = crop
        return regions, rect

    def close(self) -> None:
        if self._dxcam_camera is not None:
            try:
                del self._dxcam_camera
            except Exception:
                pass
            self._dxcam_camera = None
        if self._mss_sct is not None:
            try:
                self._mss_sct.close()
            except Exception:
                pass
            self._mss_sct = None


def extract_client_area(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Given a raw screenshot or video frame, return the game window's client area image.
    Works seamlessly for both pure client-area screenshots and desktop window recordings.
    """
    x1, y1, x2, y2 = detect_client_area_in_frame(frame_bgr)
    return frame_bgr[y1:y2, x1:x2]
