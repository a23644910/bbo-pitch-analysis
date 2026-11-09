"""
Scoreboard detector for 《全民打棒球2》 (top-right scoreboard).
Extracts:
- Batting row ("top" vs "bottom") via blue background highlight
- Top/Bottom half of inning ("上" vs "下")
- S (Strikes: 0..2), B (Balls: 0..3), O (Outs: 0..2) via fixed normalized dot coordinates
- Inning number (1..12)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from capture.calibration import CalibrationConfig


# Normalized (x, y) coordinates of S/B/O indicator dots within the scoreboard crop
S_DOTS = [(0.7320, 0.2485), (0.7710, 0.2485)]
B_DOTS = [(0.7320, 0.5150), (0.7710, 0.5150), (0.8100, 0.5150)]
O_DOTS = [(0.7320, 0.7880), (0.7710, 0.7880)]


@dataclass(frozen=True)
class ScoreboardState:
    valid: bool
    batting_row: str  # "top", "bottom", or "unknown"
    half: str  # "上" or "下"
    strikes: int  # 0..2
    balls: int  # 0..3
    outs: int  # 0..2
    top_blue_score: float = 0.0
    bottom_blue_score: float = 0.0
    inning: int = 1

    def is_my_batting(self, my_batting_row: str) -> bool:
        """Check whether the scoreboard blue highlight matches the user's selected row."""
        if not self.valid or self.batting_row not in ("top", "bottom"):
            return False
        return self.batting_row == my_batting_row


def _sample_patch_bgr(img: np.ndarray, x_rel: float, y_rel: float, radius: int = 2) -> Tuple[float, float, float]:
    h, w = img.shape[:2]
    cx = max(radius, min(w - 1 - radius, int(round(x_rel * w))))
    cy = max(radius, min(h - 1 - radius, int(round(y_rel * h))))
    patch = img[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1].astype(float)
    b = float(patch[:, :, 0].mean())
    g = float(patch[:, :, 1].mean())
    r = float(patch[:, :, 2].mean())
    return b, g, r


def read_scoreboard(
    client_bgr: np.ndarray, config: CalibrationConfig, current_inning: int = 1
) -> ScoreboardState:
    """
    Read the top-right scoreboard from a client-area BGR image.
    """
    sb = config.scoreboard.crop(client_bgr)
    if sb.size == 0 or sb.shape[0] < 20 or sb.shape[1] < 80:
        return ScoreboardState(
            valid=False,
            batting_row="unknown",
            half="上",
            strikes=0,
            balls=0,
            outs=0,
            inning=current_inning,
        )

    sh, sw = sb.shape[:2]
    # 0. Verify scoreboard frame is actually visible (left inning column has dark background, not bright sky/stadium)
    left_box = sb[int(sh * 0.15):int(sh * 0.85), int(sw * 0.02):int(sw * 0.11)]
    if left_box.size == 0 or float(np.mean(left_box)) > 118.0:
        return ScoreboardState(
            valid=False,
            batting_row="unknown",
            half="上",
            strikes=0,
            balls=0,
            outs=0,
            inning=current_inning,
        )

    # 1. Check batting row blue highlight in the team name bar (x: 0.22..0.55)
    top_bar = sb[int(sh * 0.12):int(sh * 0.45), int(sw * 0.22):int(sw * 0.55)].astype(float)
    bot_bar = sb[int(sh * 0.55):int(sh * 0.88), int(sw * 0.22):int(sw * 0.55)].astype(float)

    top_blue = float(np.mean(top_bar[:, :, 0] - top_bar[:, :, 2]))
    bot_blue = float(np.mean(bot_bar[:, :, 0] - bot_bar[:, :, 2]))
    top_luma = float(np.mean(top_bar))
    bot_luma = float(np.mean(bot_bar))

    min_blue_diff = float(config.colors.sb_blue_b_minus_r)
    if top_blue >= min_blue_diff and top_blue > bot_blue + 20.0 and bot_luma < 110.0:
        batting_row = "top"
        half = "上"
        valid = True
    elif bot_blue >= min_blue_diff and bot_blue > top_blue + 20.0 and top_luma < 110.0:
        batting_row = "bottom"
        half = "下"
        valid = True
    else:
        return ScoreboardState(
            valid=False,
            batting_row="unknown",
            half="上",
            strikes=0,
            balls=0,
            outs=0,
            top_blue_score=round(top_blue, 1),
            bottom_blue_score=round(bot_blue, 1),
            inning=current_inning,
        )

    # 2. Sample S (yellow), B (green), O (red) indicator dots
    rad = max(1, int(round(sh * 0.045)))

    strikes = 0
    for x_rel, y_rel in S_DOTS:
        b, g, r = _sample_patch_bgr(sb, x_rel, y_rel, radius=rad)
        # Lit yellow dot: R and G both bright and significantly above B
        if r > 110.0 and g > 100.0 and (r - b) > 40.0 and (g - b) > 30.0:
            strikes += 1

    balls = 0
    for x_rel, y_rel in B_DOTS:
        b, g, r = _sample_patch_bgr(sb, x_rel, y_rel, radius=rad)
        # Lit green dot: G significantly above R and B
        if g > 90.0 and (g - r) > 25.0 and (g - b) > 20.0:
            balls += 1

    outs = 0
    for x_rel, y_rel in O_DOTS:
        b, g, r = _sample_patch_bgr(sb, x_rel, y_rel, radius=rad)
        # Lit red dot: R significantly above G and B
        if r > 105.0 and (r - g) > 38.0 and (r - b) > 38.0:
            outs += 1

    return ScoreboardState(
        valid=valid,
        batting_row=batting_row,
        half=half,
        strikes=strikes,
        balls=balls,
        outs=outs,
        top_blue_score=round(top_blue, 1),
        bottom_blue_score=round(bot_blue, 1),
        inning=current_inning,
    )
