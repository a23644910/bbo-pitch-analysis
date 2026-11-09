"""
Window locator for 《全民打棒球2》 (window title: 「全民打棒球」).
All coordinates are anchored to the window's client area so moving the window
never requires recalibration.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    import win32gui
except ImportError:
    win32gui = None


@dataclass(frozen=True)
class ClientRect:
    """Screen coordinates of the game window's client area."""
    hwnd: int
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Returns (left, top, right, bottom)."""
        return (self.left, self.top, self.right, self.bottom)

    def sub_bbox(self, x1_norm: float, y1_norm: float, x2_norm: float, y2_norm: float) -> Tuple[int, int, int, int]:
        """Convert normalized client-area coordinates (0..1) into screen (left, top, right, bottom)."""
        x1 = int(round(self.left + max(0.0, min(1.0, x1_norm)) * self.width))
        y1 = int(round(self.top + max(0.0, min(1.0, y1_norm)) * self.height))
        x2 = int(round(self.left + max(0.0, min(1.0, x2_norm)) * self.width))
        y2 = int(round(self.top + max(0.0, min(1.0, y2_norm)) * self.height))
        return (min(x1, x2), min(y1, y2), max(x1 + 1, x2), max(y1 + 1, y2))


def find_game_window(title_keyword: str = "全民打棒球") -> Optional[ClientRect]:
    """
    Locate the game window by title using pywin32 and return its client area in screen coordinates.
    Returns None if the window is not found or minimized.
    """
    if win32gui is None:
        return None

    hwnd = win32gui.FindWindow(None, title_keyword)
    if not hwnd:
        # Fallback: enumerate top-level windows whose title contains title_keyword or Ma9
        matched = []

        def _enum_cb(h, _):
            if win32gui.IsWindowVisible(h) and not win32gui.IsIconic(h):
                text = win32gui.GetWindowText(h) or ""
                if "投球落點分析" in text:
                    return
                if (
                    title_keyword in text
                    or "全民打棒球" in text
                    or text.upper().startswith("MA9")
                    or "BBO" in text.upper()
                ):
                    r = win32gui.GetClientRect(h)
                    if (r[2] - r[0]) >= 640 and (r[3] - r[1]) >= 400:
                        matched.append(h)

        win32gui.EnumWindows(_enum_cb, None)
        if matched:
            hwnd = matched[0]

    if not hwnd or not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
        return None

    try:
        cl_left, cl_top, cl_right, cl_bottom = win32gui.GetClientRect(hwnd)
        width = cl_right - cl_left
        height = cl_bottom - cl_top
        if width <= 100 or height <= 100:
            return None
        scr_x, scr_y = win32gui.ClientToScreen(hwnd, (0, 0))
        return ClientRect(hwnd=hwnd, left=scr_x, top=scr_y, width=width, height=height)
    except Exception:
        return None


def is_foreground(hwnd: int) -> bool:
    """遊戲視窗是否為目前的前景（最上層、有焦點）視窗。無法判斷時回傳 True，不阻擋擷取。"""
    if win32gui is None or not hwnd:
        return True
    try:
        fg = win32gui.GetForegroundWindow()
        if not fg:
            return True
        if fg == hwnd:
            return True
        # 前景是遊戲的子視窗也算
        return win32gui.GetAncestor(fg, 2) == hwnd  # GA_ROOT = 2
    except Exception:
        return True


def detect_client_area_in_frame(frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    For offline testing with desktop screen recordings (where the game runs in a window
    inside a larger desktop frame), automatically detect the game client area's
    (x1, y1, x2, y2) within frame_bgr.
    If frame_bgr is already a pure game client area screenshot, returns (0, 0, W, H).
    """
    h, w = frame_bgr.shape[:2]

    # Check if there is a Windows title bar near the top (y in 2%..4% of height)
    # and a taskbar near the bottom, with vertical window borders at left/right.
    y_probe1 = int(h * 0.038)
    y_probe2 = int(h * 0.068)
    if h >= 720 and w >= 1280:
        # Check vertical edges at x ~ 6.25% (e.g. 120 in 1920, 240 in 3840)
        x_left_expected = int(round(w * 120 / 1920))
        x_right_expected = int(round(w * 1800 / 1920))
        y_top_expected = int(round(h * 68 / 1080))
        y_bot_expected = int(round(h * 1013 / 1080))

        # Verify if there is a sharp vertical border at x_left_expected and x_right_expected
        # and a uniform title bar strip just above y_top_expected
        title_strip = frame_bgr[int(h * 0.040):y_top_expected - 1, x_left_expected + 80:x_right_expected - 80]
        if title_strip.size > 0:
            # Title bar is neutral grey/white with very low horizontal variance in the middle
            b_ch, g_ch, r_ch = (
                title_strip[:, :, 0].astype(float),
                title_strip[:, :, 1].astype(float),
                title_strip[:, :, 2].astype(float),
            )
            is_neutral_bar = (
                np.mean(np.abs(r_ch - b_ch)) < 8.0
                and np.mean(r_ch) > 140.0
                and np.std(r_ch[:, int(title_strip.shape[1] * 0.2):int(title_strip.shape[1] * 0.8)]) < 5.0
            )
            if is_neutral_bar:
                return (x_left_expected, y_top_expected, x_right_expected, y_bot_expected)

    return (0, 0, w, h)


# Compatibility aliases
WindowClientRect = ClientRect
find_bbo_window = find_game_window

