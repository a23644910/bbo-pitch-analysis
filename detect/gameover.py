"""
比賽結束畫面偵測：結算畫面記分板右上方有一顆藍色「確認BOX」按鈕。
純像素判斷（不用 OCR）：在畫面右上區域找一塊飽和亮藍、長寬比約 3.5 : 1 的長方形。
"""

from __future__ import annotations
import cv2
import numpy as np


def has_confirm_box_button(client_bgr: np.ndarray) -> bool:
    h, w = client_bgr.shape[:2]
    x0, y0 = int(w * 0.55), int(h * 0.02)
    crop = client_bgr[y0:int(h * 0.25), x0:int(w * 0.85)]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 115)
        & (hsv[:, :, 1] >= 150) & (hsv[:, :, 2] >= 170)
    ).astype(np.uint8)
    if int(mask.sum()) < 200:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        _, _, bw, bh, area = stats[i]
        if bh < 5:
            continue
        aspect = bw / bh
        fill = area / float(bw * bh)
        rel_w = bw / float(w)
        # 實測 1680 寬畫面：按鈕 114x32，長寬比 3.56，填滿率 0.75，寬度約畫面 6.8%
        if 2.3 <= aspect <= 5.0 and fill >= 0.55 and 0.045 <= rel_w <= 0.11:
            return True
    return False
