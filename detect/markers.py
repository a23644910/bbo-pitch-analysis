"""
Detects pitch landing markers in 《全民打棒球2》:
- Blue bullseye ◎ (好球／界外／擊出的最終落點; ignores moving hollow ring ○ in flight)
- Red ⊗ (壞球的最終落點)
Also extracts the adjacent result label crop (STRIKE / BALL + pitch type + speed)
and computes normalized strike-zone coordinates (x_norm, y_norm) and 9-grid / out-of-zone code.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np

from capture.calibration import CalibrationConfig, BoxRatio


def imread_unicode(path: str | Path) -> Optional[np.ndarray]:
    """Read an image from a path that may contain Traditional Chinese / Unicode characters on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: str | Path, img: np.ndarray) -> bool:
    """Write an image to a path that may contain Traditional Chinese / Unicode characters on Windows."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix or ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(p))
        return True
    except Exception:
        return False


@dataclass
class MarkerDetection:
    kind: str  # "bullseye" (◎) or "red_x" (⊗)
    cx: int
    cy: int
    score: float
    x_norm: float
    y_norm: float
    zone: str  # "1".."9" or "H", "L", "IN", "OUT", "H-IN", "H-OUT", "L-IN", "L-OUT"
    batter_hand: str  # "R" or "L"
    has_label: bool
    label_shape: str  # "square" (STRIKE), "circle" (BALL), or "none"
    label_crop: Optional[np.ndarray] = None
    pitch_text_crop: Optional[np.ndarray] = None


def _make_ring_kernel(r_in: float, r_out: float, size: int) -> np.ndarray:
    yy, xx = np.ogrid[-size:size + 1, -size:size + 1]
    d2 = xx * xx + yy * yy
    k = ((d2 >= r_in ** 2) & (d2 <= r_out ** 2)).astype(np.float32)
    s = float(k.sum())
    return k / s if s > 0 else k


def compute_zone(
    x_norm: float, y_norm: float, batter_hand: str = "R"
) -> str:
    """
    把好球帶內的相對座標（0..1）換成九宮格 1..9（捕手視角：1 左上 … 9 右下）。
    好球帶外的壞球沿九宮格的格線往外延伸，歸到最近的那一格：
    例如 1 號位左上方的壞球仍算 1 號位、正上方算 2 號位、左側算 4 號位。
    （原始座標仍保存在 x_norm / y_norm，需要時可分辨是否在框外）
    """
    x = min(max(float(x_norm), 0.0), 0.9999)
    y = min(max(float(y_norm), 0.0), 0.9999)
    col = 0 if x < (1.0 / 3.0) else (1 if x < (2.0 / 3.0) else 2)
    row = 0 if y < (1.0 / 3.0) else (1 if y < (2.0 / 3.0) else 2)
    return str(row * 3 + col + 1)


def extract_label_near_marker(
    client_bgr: np.ndarray, cx: int, cy: int, marker_kind: str
) -> Tuple[bool, str, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Inspect the region immediately to the bottom-right of (cx, cy) for a pitch result label.
    Returns (has_label, label_shape, label_crop, pitch_text_crop).
    """
    h, w = client_bgr.shape[:2]
    scale = h / 1080.0

    x1 = max(0, int(round(cx + 3 * scale)))
    y1 = max(0, int(round(cy + 3 * scale)))
    x2 = min(w, int(round(cx + 128 * scale)))
    y2 = min(h, int(round(cy + 102 * scale)))

    if x2 - x1 < int(36 * scale) or y2 - y1 < int(26 * scale):
        return False, "none", None, None

    # 給整張標籤 OCR 用的大範圍：原本 128x102 會切掉 BALL 標籤最下面的球速那行
    bx2 = min(w, int(round(cx + 165 * scale)))
    by2 = min(h, int(round(cy + 135 * scale)))
    big_crop = client_bgr[y1:by2, x1:bx2].copy()

    # 下面的判斷與球種裁切維持原本的範圍，pHash 快取才會一致
    crop = client_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]

    # In BBO2, every red_x (⊗) is a BALL and always has a circular BALL label attached,
    # even when landing on the bright white home plate where dark_ratio drops under HDR.
    if marker_kind == "red_x":
        py1, py2 = int(ch * 0.50), int(ch * 0.80)
        px1, px2 = int(cw * 0.08), int(cw * 0.92)
        pitch_text_crop = crop[py1:py2, px1:px2].copy()
        return True, "circle", big_crop, pitch_text_crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    b, g, r = (
        crop[:, :, 0].astype(np.int16),
        crop[:, :, 1].astype(np.int16),
        crop[:, :, 2].astype(np.int16),
    )

    # A valid label has a dark semi-transparent body with low saturation,
    # PLUS bright text pixels (white speed/title and sky-blue pitch name)
    dark_neutral = (gray < 105) & (np.abs(r - g) < 26) & (np.abs(g - b) < 32)
    dark_ratio = float(np.mean(dark_neutral))
    bright_text = (gray > 140) | ((b - r > 20) & (b > 105))
    bright_ratio = float(np.mean(bright_text))

    if dark_ratio < 0.25 or bright_ratio < 0.022:
        return False, "none", None, None

    # Determine whether label is square (STRIKE, dark blue header at top) or circle (BALL)
    top_band = crop[: int(ch * 0.35), :]
    tb_b, tb_r = top_band[:, :, 0].astype(np.int16), top_band[:, :, 2].astype(np.int16)
    blue_header_ratio = float(np.mean((tb_b - tb_r > 15) & (gray[: int(ch * 0.35), :] < 115)))

    if blue_header_ratio > 0.12 or marker_kind == "bullseye":
        label_shape = "square"
        # Pitch type line in square STRIKE label (or 2-line HR derby label) is in middle band
        py1, py2 = int(ch * 0.36), int(ch * 0.70)
    else:
        label_shape = "circle"
        # Pitch type line in circular BALL label is in lower-middle band
        py1, py2 = int(ch * 0.50), int(ch * 0.80)

    px1, px2 = int(cw * 0.08), int(cw * 0.92)
    pitch_text_crop = crop[py1:py2, px1:px2].copy()
    return True, label_shape, big_crop, pitch_text_crop


def detect_pitch_marker(
    client_bgr: np.ndarray, config: CalibrationConfig
) -> Optional[MarkerDetection]:
    """
    Detect a stationary pitch landing marker (◎ bullseye or ⊗ red_x) in client_bgr.
    Only triggers when the blue circle has its solid center dot (◎), ignoring hollow in-flight rings (○).
    """
    h, w = client_bgr.shape[:2]
    scale = h / 1080.0

    # Determine active strike zone (auto-handling right-handed vs left-handed batter camera offset)
    active_sz, batter_hand = config.detect_active_strike_zone(client_bgr)
    sz_x1, sz_y1, sz_x2, sz_y2 = active_sz.to_px(w, h)
    sz_w = max(1.0, float(sz_x2 - sz_x1))
    sz_h = max(1.0, float(sz_y2 - sz_y1))

    rx1, ry1, rx2, ry2 = config.marker_search_zone.to_px(w, h)
    # Wide search bounds to capture high/low/inside/outside out-of-zone balls (⊗ and ◎)
    rx1 = max(rx1, int(round(w * 0.33)))
    rx2 = min(rx2, int(round(w * 0.67)))
    ry1 = max(ry1, int(round(h * 0.54)))
    ry2 = min(ry2, int(round(h * 0.97)))

    roi_bgr = client_bgr[ry1:ry2, rx1:rx2]
    if roi_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(roi_bgr)
    c = config.colors

    blue_mask = (
        (hsv[:, :, 0] >= c.blue_h_min)
        & (hsv[:, :, 0] <= c.blue_h_max)
        & (hsv[:, :, 1] >= c.blue_s_min)
        & (hsv[:, :, 2] >= c.blue_v_min)
        & (b.astype(np.int16) - r.astype(np.int16) >= c.blue_b_minus_r)
    ).astype(np.float32)
    # Mask out outer player helmet/glove/cap regions for blue bullseye while keeping wide strike/foul range
    b_xmin = max(0, int(round(w * 0.38)) - rx1)
    b_xmax = min(roi_bgr.shape[1], int(round(w * 0.62)) - rx1)
    b_ymin = max(0, int(round(h * 0.59)) - ry1)
    b_ymax = min(roi_bgr.shape[0], int(round(h * 0.93)) - ry1)
    if b_xmin > 0:
        blue_mask[:, :b_xmin] = 0.0
    if b_xmax < roi_bgr.shape[1]:
        blue_mask[:, b_xmax:] = 0.0
    if b_ymin > 0:
        blue_mask[:b_ymin, :] = 0.0
    if b_ymax < roi_bgr.shape[0]:
        blue_mask[b_ymax:, :] = 0.0

    red_diff = r.astype(np.float32) - np.maximum(g.astype(np.float32), b.astype(np.float32))
    red_mask = (
        ((hsv[:, :, 0] <= c.red_h_low_max) | (hsv[:, :, 0] >= c.red_h_high_min))
        & (hsv[:, :, 1] >= c.red_s_min)
        & (hsv[:, :, 2] >= c.red_v_min)
        & (red_diff >= c.red_r_minus_gb)
    ).astype(np.float32)

    best_kind: Optional[str] = None
    best_cx: int = 0
    best_cy: int = 0
    best_score: float = 0.0

    # 1. Check Blue Bullseye ◎
    # ◎ 在不同畫面下大小略有差異（實測約為預設的 1.2 倍），依序試 3 種尺寸
    for size_mult in (1.0, 1.2, 1.4):
      if best_kind is not None:
        break
      s2 = scale * size_mult
      b_sz = max(8, int(round(22 * s2)))
      if blue_mask.sum() > 80 and roi_bgr.shape[0] > 2 * b_sz and roi_bgr.shape[1] > 2 * b_sz:
        k_b_inner = _make_ring_kernel(0.0, 4.0 * s2, b_sz)
        k_b_gap = _make_ring_kernel(7.5 * s2, 10.5 * s2, b_sz)
        k_b_outer = _make_ring_kernel(13.0 * s2, 18.5 * s2, b_sz)

        b_in = cv2.filter2D(blue_mask, -1, k_b_inner, borderType=cv2.BORDER_CONSTANT)
        b_gap = cv2.filter2D(blue_mask, -1, k_b_gap, borderType=cv2.BORDER_CONSTANT)
        b_out = cv2.filter2D(blue_mask, -1, k_b_outer, borderType=cv2.BORDER_CONSTANT)
        b_in[:b_sz, :] = b_in[-b_sz:, :] = b_in[:, :b_sz] = b_in[:, -b_sz:] = 0.0

        cond_b = (b_in > 0.85) & (b_gap < 0.28) & (b_out > 0.55)
        if np.any(cond_b):
            score_map = b_in * (1.0 - b_gap) * b_out
            by, bx = np.unravel_index(np.argmax(score_map * cond_b), score_map.shape)
            best_kind = "bullseye"
            best_cx = rx1 + int(bx)
            best_cy = ry1 + int(by)
            best_score = float(score_map[by, bx])

    # 2. Check Red ⊗
    r_sz = max(10, int(round(28 * scale)))
    if (
        best_kind is None
        and red_mask.sum() > 220
        and roi_bgr.shape[0] > 2 * r_sz
        and roi_bgr.shape[1] > 2 * r_sz
    ):
        k_r_center = _make_ring_kernel(0.0, 5.0 * scale, r_sz)
        k_r_outer = _make_ring_kernel(16.0 * scale, 22.5 * scale, r_sz)
        k_r_bg = _make_ring_kernel(24.5 * scale, 28.0 * scale, r_sz)

        yy, xx = np.ogrid[-r_sz:r_sz + 1, -r_sz:r_sz + 1]
        d2 = xx * xx + yy * yy
        in_mid = (d2 >= (7.0 * scale) ** 2) & (d2 <= (14.5 * scale) ** 2)
        angle = np.abs(np.arctan2(np.abs(yy), np.abs(xx)))
        k_r_diag = (in_mid & (np.abs(angle - np.pi / 4) < 0.30)).astype(np.float32)
        k_r_diag /= max(1.0, float(k_r_diag.sum()))
        k_r_card = (in_mid & (np.abs(angle - np.pi / 4) > 0.48)).astype(np.float32)
        k_r_card /= max(1.0, float(k_r_card.sum()))

        r_cen = cv2.filter2D(red_mask, -1, k_r_center, borderType=cv2.BORDER_CONSTANT)
        r_out = cv2.filter2D(red_mask, -1, k_r_outer, borderType=cv2.BORDER_CONSTANT)
        r_bg = cv2.filter2D(red_mask, -1, k_r_bg, borderType=cv2.BORDER_CONSTANT)
        diag_red = cv2.filter2D(red_diff, -1, k_r_diag, borderType=cv2.BORDER_CONSTANT)
        card_red = cv2.filter2D(red_diff, -1, k_r_card, borderType=cv2.BORDER_CONSTANT)
        r_cen[:r_sz, :] = r_cen[-r_sz:, :] = r_cen[:, :r_sz] = r_cen[:, -r_sz:] = 0.0

        cond_r = (
            (r_cen > 0.85)
            & (r_out > 0.75)
            & (r_bg < 0.25)
            & ((diag_red - card_red) > 12.0)
        )
        if np.any(cond_r):
            score_map = r_cen * r_out * (1.0 - r_bg)
            ry, rx = np.unravel_index(np.argmax(score_map * cond_r), score_map.shape)
            best_kind = "red_x"
            best_cx = rx1 + int(rx)
            best_cy = ry1 + int(ry)
            best_score = float(score_map[ry, rx])

    if best_kind is None:
        return None

    x_norm = (best_cx - sz_x1) / sz_w
    y_norm = (best_cy - sz_y1) / sz_h
    zone = compute_zone(x_norm, y_norm, batter_hand=batter_hand)
    has_label, label_shape, label_crop, pitch_text_crop = extract_label_near_marker(
        client_bgr, best_cx, best_cy, best_kind
    )

    return MarkerDetection(
        kind=best_kind,
        cx=best_cx,
        cy=best_cy,
        score=round(best_score, 4),
        x_norm=round(float(x_norm), 4),
        y_norm=round(float(y_norm), 4),
        zone=zone,
        batter_hand=batter_hand,
        has_label=has_label,
        label_shape=label_shape,
        label_crop=label_crop,
        pitch_text_crop=pitch_text_crop,
    )


def detect_label_only(
    client_bgr: np.ndarray, config: CalibrationConfig
) -> Optional[MarkerDetection]:
    """
    低球補救：標記被結果標籤蓋住時，改用標籤位置推回落點。
    BALL（圓形標籤）當作 ⊗、STRIKE（方形標籤）當作 ◎，結果仍以 OCR 讀到的字為準。
    """
    from detect.lowball import (
        find_result_label, estimate_marker_from_label, label_crops, find_label_text, label_from_text,
    )

    H, W = client_bgr.shape[:2]
    txt = find_label_text(client_bgr)
    if txt is not None:
        # 先找標籤上的「BALL / STRIKE」字樣（標籤背後有影子時也找得到）
        word, tx, ty = txt
        (cx, cy), label, text = label_from_text(client_bgr, word, tx, ty)
        shape = "circle" if word == "BALL" else "rect"
    else:
        found = find_result_label(client_bgr)
        if found is None:
            return None
        x, y, w, h, shape = found
        cx, cy = estimate_marker_from_label(x, y, w, h, H / 945.0)
        label, text = label_crops(client_bgr, x, y, w, h, shape)
    active_sz, batter_hand = config.detect_active_strike_zone(client_bgr)
    sx1, sy1, sx2, sy2 = active_sz.to_px(W, H)
    x_norm = (cx - sx1) / max(1.0, float(sx2 - sx1))
    y_norm = (cy - sy1) / max(1.0, float(sy2 - sy1))
    return MarkerDetection(
        kind="red_x" if shape == "circle" else "bullseye",
        cx=int(cx),
        cy=int(cy),
        score=0.5,
        x_norm=round(float(x_norm), 4),
        y_norm=round(float(y_norm), 4),
        zone=compute_zone(x_norm, y_norm, batter_hand=batter_hand),
        batter_hand=batter_hand,
        has_label=True,
        label_shape=shape,
        label_crop=label,
        pitch_text_crop=text,
    )

