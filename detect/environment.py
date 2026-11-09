"""
比賽環境：球場天氣、風向風速、投手／打者卡片的天氣屬性。
天氣會影響球路（屬性相剋），之後軌跡辨識要用；先每一球記下來。

位置（正規化座標，1920x1080 截圖量的，2 對 2 與單人相同）：
- 右上角天氣框：x 0.958–0.995、y 0.009–0.074（上半是天氣圖示，下半是風向箭頭＋「8 m」）
- 右下角卡片天氣圓圖示：打者 (0.892, 0.944)、投手 (0.984, 0.944)

天氣圖示：雨傘＝雨（藍）、太陽＝晴（橘紅）、雲＝陰（大片白灰）、雪花＝雪（細白線）。
"""
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

WEATHER_BOX = (0.958, 0.009, 0.995, 0.074)
CARD_ICON = {"batter": (0.892, 0.944), "pitcher": (0.984, 0.944)}


def _crop(img: np.ndarray, box) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    return img[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)]


def classify_weather_icon(icon_bgr: np.ndarray) -> str:
    """右上角天氣圖示 → 晴／雨／陰／雪／''（看不出來）。"""
    if icon_bgr is None or icon_bgr.size == 0:
        return ""
    hsv = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    n = float(H.size)
    blue = ((H >= 95) & (H <= 115) & (S > 100) & (V > 120)).sum() / n
    orange = (((H <= 16) | (H >= 165)) & (S > 120) & (V > 150)).sum() / n  # 太陽是橘紅；視窗邊緣的黃色光暈（H 19~25）不算
    white = ((S < 50) & (V > 190)).sum() / n
    if blue > 0.02:
        return "雨"
    if orange > 0.04:
        return "晴"
    if white > 0.08:
        return "陰"
    if white > 0.015:
        return "雪"
    return ""


_ARROW_TPL = None


def _square(mask: np.ndarray, size: int = 32) -> np.ndarray:
    h, w = mask.shape[:2]
    side = max(h, w)
    pad = np.zeros((side, side), np.uint8)
    pad[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = mask
    return cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA) > 127


def _arrow_templates():
    """朝上的箭頭（模板）每 15 度轉一次，供比對。"""
    global _ARROW_TPL
    if _ARROW_TPL is None:
        from pathlib import Path
        t = cv2.imread(str(Path(__file__).resolve().parent / "templates" / "wind_arrow_up.png"), cv2.IMREAD_GRAYSCALE)
        _ARROW_TPL = []
        if t is not None:
            side = int(max(t.shape) * 1.6)
            base = np.zeros((side, side), np.uint8)
            y0, x0 = (side - t.shape[0]) // 2, (side - t.shape[1]) // 2
            base[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
            for ang in range(0, 360, 15):
                # 畫面角度：0＝上、順時針；cv2 的角度是逆時針
                M = cv2.getRotationMatrix2D((side / 2, side / 2), -ang, 1.0)
                r = cv2.warpAffine(base, M, (side, side), flags=cv2.INTER_NEAREST)
                ys, xs = np.nonzero(r)
                if len(xs) == 0:
                    continue
                _ARROW_TPL.append((ang, _square(r[ys.min():ys.max() + 1, xs.min():xs.max() + 1])))
    return _ARROW_TPL


def read_wind_arrow(arrow_bgr: np.ndarray) -> Tuple[Optional[int], str]:
    """
    風向箭頭（黃綠色）→ (角度, 方向字)。角度以畫面「往上」為 0 度、順時針。
    做法：把「朝上的箭頭」模板轉 24 個角度，跟畫面上的箭頭形狀比對，最像的就是方向。
    """
    if arrow_bgr is None or arrow_bgr.size == 0:
        return None, ""
    hsv = cv2.cvtColor(arrow_bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 28) & (hsv[..., 0] <= 75) & (hsv[..., 1] > 120) & (hsv[..., 2] > 120)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None, ""
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = st[k]
    if a < 15:
        return None, ""
    obs = _square(((lab[y:y + h, x:x + w] == k).astype(np.uint8)) * 255)
    best, best_iou = None, -1.0
    for ang, tpl in _arrow_templates():
        inter = np.logical_and(obs, tpl).sum()
        union = np.logical_or(obs, tpl).sum()
        iou = inter / union if union else 0.0
        if iou > best_iou:
            best, best_iou = ang, iou
    if best is None:
        return None, ""
    names = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]  # 畫面上箭頭的方向（實際對打擊的影響之後用資料分析）
    return int(best), names[int(((best + 22.5) % 360) // 45)]


def classify_card_icon(img: np.ndarray, center) -> str:
    """右下角卡片的圓形天氣圖示（白底圓）→ 晴／雨／陰／雪／''。"""
    H, W = img.shape[:2]
    cx, cy = int(center[0] * W), int(center[1] * H)
    r = max(10, int(0.014 * W))
    win = img[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    if win.size == 0:
        return ""
    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 60) & (hsv[..., 2] > 200)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(white)
    if n <= 1:
        return ""
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = st[k]
    if w < 8 or h < 8 or a < 30:
        return ""
    # 圓圈內部（白底上的圖案）
    ox, oy = x + w // 2, y + h // 2
    rr = max(3, int(min(w, h) * 0.32))
    inner = hsv[max(0, oy - rr):oy + rr, max(0, ox - rr):ox + rr]
    if inner.size == 0:
        return ""
    colored = inner[..., 1] > 60
    frac = float(colored.mean())
    grey = float(((inner[..., 1] < 40) & (inner[..., 2] > 90) & (inner[..., 2] < 200)).mean())
    if frac >= 0.25:
        h = float(np.median(inner[..., 0][colored]))
        v = float(np.median(inner[..., 2][colored]))
        if h <= 20 or h >= 165:
            return "晴"      # 紅橘色太陽
        if 90 <= h < 112 and v > 200:
            return "雪"      # 亮藍色雪花
        if 105 <= h <= 135:
            return "雨"      # 深藍色雨傘
        return ""
    if grey > 0.3:
        return "陰"          # 灰色雲
    return ""


def read_environment(img: np.ndarray) -> Dict[str, object]:
    """一次讀齊：球場天氣、風向（角度＋方向字）、投手／打者卡片天氣。風速數字交給 OCR（見 wind_speed_crop）。"""
    box = _crop(img, WEATHER_BOX)
    out: Dict[str, object] = {"weather": "", "wind_angle": None, "wind_dir": "",
                              "pitcher_attr": "", "batter_attr": ""}
    if box.size:
        h, w = box.shape[:2]
        out["weather"] = classify_weather_icon(box[: int(h * 0.62), int(w * 0.35):])
        ang, name = read_wind_arrow(box[int(h * 0.55):, : int(w * 0.45)])
        out["wind_angle"], out["wind_dir"] = ang, name
    out["pitcher_attr"] = classify_card_icon(img, CARD_ICON["pitcher"])
    out["batter_attr"] = classify_card_icon(img, CARD_ICON["batter"])
    return out


def wind_speed_crop(img: np.ndarray) -> np.ndarray:
    """「8 m」那塊，給 OCR 讀數字。"""
    box = _crop(img, WEATHER_BOX)
    if box.size == 0:
        return box
    h, w = box.shape[:2]
    return box[int(h * 0.55):, int(w * 0.40):]
