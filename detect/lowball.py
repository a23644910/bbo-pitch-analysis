"""
低球（尤其壞球）的補救偵測。

球投得很低時，遊戲會把結果標籤（BALL 圓形 / STRIKE 方形）往上擺，
剛好蓋住落點標記 ⊗ / ◎，所以一般的標記偵測找不到。
這裡改成直接找「結果標籤」本身：一塊深灰色、帶白字的圓形或圓角方塊，
再依標籤位置推回落點：實測落點約在標籤左緣、標籤上緣往下約 1/3 高的位置。
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

_TPL_DIR = Path(__file__).resolve().parent / "templates"
_TPL_CACHE: dict = {}


def _templates(scale: float):
    """BALL / STRIKE 字樣的灰階樣板（以 1680x945 畫面截取），依畫面大小縮放，比對時再縮小一半加速。"""
    key = round(scale, 3)
    if key not in _TPL_CACHE:
        out = []
        for name, fn in (("BALL", "label_ball.png"), ("STRIKE", "label_strike.png")):
            t = cv2.imread(str(_TPL_DIR / fn), cv2.IMREAD_GRAYSCALE)
            if t is None:
                continue
            f = scale * 0.5
            t = cv2.resize(t, (max(8, int(round(t.shape[1] * f))), max(6, int(round(t.shape[0] * f)))),
                           interpolation=cv2.INTER_AREA)
            out.append((name, t))
        _TPL_CACHE[key] = out
    return _TPL_CACHE[key]


def find_label_text(client_bgr: np.ndarray, threshold: float = 0.8) -> Optional[Tuple[str, int, int]]:
    """
    直接找標籤上的白色粗體字「BALL」或「STRIKE」，回傳 (字, 字中心 x, 字中心 y)。
    標籤是半透明的，背後有本壘板、打者影子時，用顏色找整塊標籤會黏在一起失敗；
    字樣本身則很固定，比對很穩。
    """
    H, W = client_bgr.shape[:2]
    s = H / 945.0
    x0, y0 = int(W * 0.33), int(H * 0.50)
    roi = client_bgr[y0:int(H * 0.99), x0:int(W * 0.70)]
    if roi.size == 0:
        return None
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    best = None
    for name, t in _templates(s):
        if g.shape[0] < t.shape[0] or g.shape[1] < t.shape[1]:
            continue
        r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, (x, y) = cv2.minMaxLoc(r)
        if mx >= threshold and (best is None or mx > best[0]):
            best = (mx, name, x0 + (x + t.shape[1] / 2.0) * 2, y0 + (y + t.shape[0] / 2.0) * 2)
    if best is None:
        return None
    return best[1], int(round(best[2])), int(round(best[3]))


def label_from_text(client_bgr: np.ndarray, word: str, tx: int, ty: int):
    """
    由字的位置推回：落點座標、整張標籤、球種那一行。
    低球時標籤會往上蓋住落點，實測落點在字中心左邊約 55 px、同一高度（1680x945 畫面）。
    """
    H, W = client_bgr.shape[:2]
    s = H / 945.0
    if word == "BALL":
        mx, my = tx - 55 * s, ty + 4 * s
        bx1, by1, bx2, by2 = tx - 50 * s, ty - 25 * s, tx + 50 * s, ty + 60 * s
    else:
        mx, my = tx - 60 * s, ty + 2 * s
        bx1, by1, bx2, by2 = tx - 58 * s, ty - 22 * s, tx + 58 * s, ty + 60 * s
    clip = lambda v, hi: int(max(0, min(hi, round(v))))
    label = client_bgr[clip(by1, H):clip(by2, H), clip(bx1, W):clip(bx2, W)].copy()
    text = client_bgr[clip(ty + 18 * s, H):clip(ty + 38 * s, H), clip(tx - 40 * s, W):clip(tx + 40 * s, W)].copy()
    return (int(round(mx)), int(round(my))), label, text


def find_result_label(client_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int, str]]:
    """回傳 (x, y, w, h, shape)，shape 為 "circle"（BALL）或 "rect"（STRIKE）；找不到回傳 None。"""
    H, W = client_bgr.shape[:2]
    s = H / 945.0
    x0, y0 = int(W * 0.33), int(H * 0.54)
    roi = client_bgr[y0:int(H * 0.99), x0:int(W * 0.67)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    dark = ((hsv[:, :, 2] < 85) & (hsv[:, :, 1] < 90)).astype(np.uint8)
    k = max(3, int(round(7 * s)) | 1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (70 * s <= w <= 150 * s and 55 * s <= h <= 130 * s):
            continue
        fill = area / float(w * h)
        if fill < 0.5:
            continue
        sub = roi[y:y + h, x:x + w]
        white = float((sub.min(axis=2) > 185).mean())  # 標籤上的白字
        if white < 0.02:
            continue
        aspect = w / float(h)
        shape = "circle" if (0.85 <= aspect <= 1.18 and fill < 0.85) else "rect"
        cand = (x0 + int(x), y0 + int(y), int(w), int(h), shape)
        if best is None or area > best[0]:
            best = (area, cand)
    return best[1] if best else None


def estimate_marker_from_label(x: int, y: int, w: int, h: int, scale: float) -> Tuple[int, int]:
    """落點約在標籤左緣外一點、標籤上緣往下約 1/3 高。"""
    return int(round(x - 10 * scale)), int(round(y + h * 0.35))


def label_crops(client_bgr: np.ndarray, x: int, y: int, w: int, h: int, shape: str):
    """給 OCR 用的整張標籤，以及球種那一行（做 pHash 快取）。"""
    H, W = client_bgr.shape[:2]
    pad = 6
    lx1, ly1 = max(0, x - pad), max(0, y - pad)
    lx2, ly2 = min(W, x + w + pad), min(H, y + h + pad)
    label = client_bgr[ly1:ly2, lx1:lx2].copy()
    if shape == "circle":
        ty1, ty2 = int(h * 0.45), int(h * 0.72)
    else:
        ty1, ty2 = int(h * 0.36), int(h * 0.70)
    text = client_bgr[y + ty1:y + ty2, x + int(w * 0.08):x + int(w * 0.92)].copy()
    return label, text
