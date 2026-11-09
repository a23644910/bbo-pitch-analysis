"""
Detects:
1. The light-cyan semi-transparent batting cursor box (淡青色半透明方框) to confirm
   that our team is batting before a pitch arrives.
2. Whether the camera is currently on the batting view (好球帶外框可見) vs
   having cut to the fielding view (鏡頭切到守備畫面).
"""

from __future__ import annotations
import cv2
import numpy as np
from capture.calibration import CalibrationConfig, BoxRatio


def has_batting_cursor(client_bgr: np.ndarray, config: CalibrationConfig) -> bool:
    """
    Detect the light-cyan batting cursor rectangle (淡青色打擊游標) inside/near the strike zone.
    The cursor has thin horizontal and vertical cyan border lines (G > R, B > R, |B - G| small).
    """
    h, w = client_bgr.shape[:2]
    sz = config.strike_zone
    # Expand strike zone slightly to cover cursor movement around the zone edges
    pad_x = sz.width * 0.35
    pad_y = sz.height * 0.35
    roi_box = BoxRatio(
        x1=max(0.0, sz.x1 - pad_x),
        y1=max(0.0, sz.y1 - pad_y),
        x2=min(1.0, sz.x2 + pad_x),
        y2=min(1.0, sz.y2 + pad_y),
    )
    roi = roi_box.crop(client_bgr)
    if roi.size == 0:
        return False

    b, g, r = (
        roi[:, :, 0].astype(np.int16),
        roi[:, :, 1].astype(np.int16),
        roi[:, :, 2].astype(np.int16),
    )
    c = config.colors
    # Cyan edge/box pixels: both B and G are higher than R, and B ~ G
    bg_diff = max(14, int(c.cyan_bg_minus_r) - 5)
    v_min = max(90, int(c.cyan_v_min) - 25)
    cyan_mask = (
        (g - r >= bg_diff)
        & (b - r >= bg_diff)
        & (np.abs(b - g) <= 35)
        & (g >= v_min)
        & (b >= v_min)
    )

    total_cyan = int(cyan_mask.sum())
    min_line_px = max(12, int(round(sz.width * w * 0.16)))
    row_counts = cyan_mask.sum(axis=1)
    col_counts = cyan_mask.sum(axis=0)

    n_horiz = int(np.sum(row_counts >= min_line_px))
    n_vert = int(np.sum(col_counts >= min_line_px))
    return bool(total_cyan >= 100 and ((n_horiz >= 1 and n_vert >= 1) or (n_horiz + n_vert >= 2)))


def is_batting_camera(client_bgr: np.ndarray, config: CalibrationConfig) -> bool:
    """
    Determine whether the camera is currently in the batting/pitching view
    (where the grey strike zone outer frame is visible) vs fielding view.
    """
    h, w = client_bgr.shape[:2]
    sz_r = config.strike_zone
    sz_l = BoxRatio(
        x1=sz_r.x1 + config.lh_x_offset,
        y1=sz_r.y1,
        x2=sz_r.x2 + config.lh_x_offset,
        y2=sz_r.y2,
    )

    gray = client_bgr.astype(np.float32).mean(axis=2)
    scale = max(1.0, h / 945.0)
    nb1 = max(4, int(round(5 * scale)))
    nb2 = max(7, int(round(9 * scale)))

    def _edge(x: int, ym1: int, ym2: int) -> float:
        # 框線是半透明白線，疊在紅土上會偏粉紅，所以只比亮度：線要比左右兩側都亮。
        # 在 ±2 px 內找最亮的位置，容忍校正誤差與壓縮位移。
        best_v = -999.0
        for dx in range(-2, 3):
            xc = x + dx
            line = gray[ym1:ym2, xc - 1:xc + 2].mean()
            side = 0.5 * (
                gray[ym1:ym2, xc - nb2:xc - nb1].mean()
                + gray[ym1:ym2, xc + nb1:xc + nb2].mean()
            )
            best_v = max(best_v, float(line - side))
        return best_v

    def _frame_score(box: BoxRatio) -> float:
        x1, y1, x2, y2 = box.to_px(w, h)
        ym1 = y1 + int((y2 - y1) * 0.18)
        ym2 = y2 - int((y2 - y1) * 0.18)
        if ym2 <= ym1 or x1 < nb2 + 3 or x2 >= w - nb2 - 3:
            return 0.0
        # 左右兩條框線都要清楚才算（取較弱的一邊）
        return min(_edge(x1, ym1, ym2), _edge(x2, ym1, ym2))

    best = max(_frame_score(sz_r), _frame_score(sz_l))
    return bool(best >= 20.0)
