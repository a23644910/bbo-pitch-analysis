"""
Calibration and HDR/SDR color sampling module.
All element locations are stored as normalized client-area ratios (0.0 to 1.0).
Color thresholds can be dynamically sampled from a calibration frame so Windows HDR
(which appears dimmer/greyer) works without hardcoded pixel or color limits.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np


DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "config" / "calibration.json"


@dataclass
class BoxRatio:
    """Normalized rectangle (x1, y1, x2, y2) in [0, 1] relative to client area."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(1e-6, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1e-6, self.y2 - self.y1)

    def to_px(self, client_w: int, client_h: int) -> Tuple[int, int, int, int]:
        """Convert to pixel coordinates (x1, y1, x2, y2) within client area."""
        px1 = max(0, min(client_w - 1, int(round(self.x1 * client_w))))
        py1 = max(0, min(client_h - 1, int(round(self.y1 * client_h))))
        px2 = max(px1 + 1, min(client_w, int(round(self.x2 * client_w))))
        py2 = max(py1 + 1, min(client_h, int(round(self.y2 * client_h))))
        return (px1, py1, px2, py2)

    def crop(self, client_bgr: np.ndarray) -> np.ndarray:
        """Crop this normalized region from a client-area BGR image."""
        h, w = client_bgr.shape[:2]
        x1, y1, x2, y2 = self.to_px(w, h)
        return client_bgr[y1:y2, x1:x2]


@dataclass
class ColorThresholdProfile:
    """Adaptive HSV and BGR-difference thresholds for SDR and HDR modes."""
    # Blue bullseye ◎ (deep royal blue, distinct from pale cyan batting cursor)
    blue_h_min: int = 100
    blue_h_max: int = 135
    blue_s_min: int = 110
    blue_v_min: int = 80
    blue_b_minus_r: int = 45

    # Red ⊗
    red_h_low_max: int = 12
    red_h_high_min: int = 166
    red_s_min: int = 95
    red_v_min: int = 65
    red_r_minus_gb: int = 38

    # Cyan batting cursor (淡青色半透明方框)
    cyan_h_min: int = 80
    cyan_h_max: int = 108
    cyan_s_min: int = 35
    cyan_v_min: int = 120
    cyan_bg_minus_r: int = 20

    # Scoreboard blue row highlight (打擊方藍底)
    sb_blue_b_minus_r: int = 30

    # HDR tone compensation flag
    hdr_mode: bool = False
    brightness_scale: float = 1.0


@dataclass
class CalibrationConfig:
    """Complete calibration configuration anchored to the strike zone outer frame."""
    # Default strike zone outer frame (Right-handed batter camera, default)
    strike_zone: BoxRatio = field(
        default_factory=lambda: BoxRatio(x1=0.4516, y1=0.6935, x2=0.5381, y2=0.8426)
    )
    # Left-handed batter horizontal shift offset (auto-detected per frame, ~+0.0155)
    lh_x_offset: float = 0.0155

    # Expanded search region around strike zone for markers + labels (including wide out-of-zone balls)
    marker_search_zone: BoxRatio = field(
        default_factory=lambda: BoxRatio(x1=0.3300, y1=0.5400, x2=0.6700, y2=0.9700)
    )

    # Top-right scoreboard (右上記分板)
    scoreboard: BoxRatio = field(
        default_factory=lambda: BoxRatio(x1=0.8180, y1=0.0110, x2=0.9920, y2=0.0680)
    )

    # Bottom-right info panel (右下資訊小欄: 打者名、投手名)
    info_panel: BoxRatio = field(
        default_factory=lambda: BoxRatio(x1=0.8060, y1=0.8980, x2=0.9920, y2=0.9880)
    )

    # Right-side TAB panel (TAB 大面板: Pitcher 完整球種清單, covers both 16:9 and 16:10)
    tab_panel: BoxRatio = field(
        default_factory=lambda: BoxRatio(x1=0.8000, y1=0.6550, x2=0.9940, y2=0.9880)
    )

    # Batting side preference ("top" = 上行/先攻, "bottom" = 下行/後攻)
    my_batting_row: str = "top"

    # 玩家 ID：用來自動判斷我方是先攻（上行）還是後攻（下行）
    my_id: str = ""

    # True：不分先攻後攻，雙方投球都記錄（自動比賽測試用）
    record_both: bool = False

    # 整場錄影（測試用）
    record_video: bool = False

    # 電腦／真人標記（2026-11-10m）：電腦投的球不拿來訓練預測、不算命中率
    cpu_mode: bool = False   # True：電腦對戰（雙方都是電腦）
    my_auto: bool = True     # True：1 對 1 時我方是自動投球（2 對 2 一律當真人）

    # Color threshold profile
    colors: ColorThresholdProfile = field(default_factory=ColorThresholdProfile)

    # Save snapshots on each pitch for debugging
    save_snapshots: bool = True

    def update_from_strike_zone(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Update strike zone and recalculate relative marker search zone."""
        self.strike_zone = BoxRatio(x1=x1, y1=y1, x2=x2, y2=y2)
        w = self.strike_zone.width
        h = self.strike_zone.height
        self.marker_search_zone = BoxRatio(
            x1=max(0.0, x1 - 0.75 * w),
            y1=max(0.0, y1 - 0.55 * h),
            x2=min(1.0, x2 + 1.40 * w),
            y2=min(1.0, y2 + 0.55 * h),
        )

    def detect_active_strike_zone(self, client_bgr: np.ndarray) -> Tuple[BoxRatio, str]:
        """
        In 《全民打棒球2》, the strike zone outer frame shifts slightly right (+0.0155 in X)
        when a left-handed batter stands in the right batter's box.
        This method inspects the vertical grey frame lines and returns the active
        BoxRatio along with batter handedness ("R" or "L").
        """
        h, w = client_bgr.shape[:2]
        sz_r = self.strike_zone
        sz_l = BoxRatio(
            x1=sz_r.x1 + self.lh_x_offset,
            y1=sz_r.y1,
            x2=sz_r.x2 + self.lh_x_offset,
            y2=sz_r.y2,
        )

        def _vertical_edge_strength(box: BoxRatio) -> float:
            x1, y1, x2, y2 = box.to_px(w, h)
            scale = max(1.0, w / 1680.0)
            r_line = max(1, int(round(2 * scale)))
            r_bg1 = max(4, int(round(6 * scale)))
            r_bg2 = max(7, int(round(12 * scale)))
            y_mid1 = y1 + int((y2 - y1) * 0.15)
            y_mid2 = y2 - int((y2 - y1) * 0.15)
            if y_mid2 <= y_mid1 or x1 < r_bg2 or x2 >= w - r_bg2:
                return 0.0
            left_line = client_bgr[y_mid1:y_mid2, x1 - r_line:x1 + r_line + 1].astype(float).mean()
            left_bg = 0.5 * (
                client_bgr[y_mid1:y_mid2, x1 - r_bg2:x1 - r_bg1].astype(float).mean()
                + client_bgr[y_mid1:y_mid2, x1 + r_bg1:x1 + r_bg2].astype(float).mean()
            )
            right_line = client_bgr[y_mid1:y_mid2, x2 - r_line:x2 + r_line + 1].astype(float).mean()
            right_bg = 0.5 * (
                client_bgr[y_mid1:y_mid2, x2 - r_bg2:x2 - r_bg1].astype(float).mean()
                + client_bgr[y_mid1:y_mid2, x2 + r_bg1:x2 + r_bg2].astype(float).mean()
            )
            return float((left_line - left_bg) + (right_line - right_bg))

        score_r = _vertical_edge_strength(sz_r)
        score_l = _vertical_edge_strength(sz_l)
        if score_l > score_r + 4.0:
            return sz_l, "L"
        return sz_r, "R"

    def sample_colors_from_frame(self, client_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Inspect brightness/contrast of a captured frame (e.g., under Windows HDR)
        and adapt color thresholds automatically.
        """
        sz_crop = self.strike_zone.crop(client_bgr)
        mean_luma = float(sz_crop.mean()) if sz_crop.size > 0 else 128.0
        # Under Windows HDR raw capture without SDR tonemapping, luma is often dimmer/flatter
        if mean_luma < 95.0:
            self.colors.hdr_mode = True
            self.colors.brightness_scale = max(0.65, mean_luma / 125.0)
            self.colors.blue_s_min = 75
            self.colors.blue_v_min = 55
            self.colors.blue_b_minus_r = 28
            self.colors.red_s_min = 75
            self.colors.red_v_min = 50
            self.colors.red_r_minus_gb = 28
            self.colors.cyan_s_min = 25
            self.colors.cyan_v_min = 90
            self.colors.cyan_bg_minus_r = 14
        else:
            self.colors.hdr_mode = False
            self.colors.brightness_scale = 1.0
            self.colors.blue_s_min = 95
            self.colors.blue_v_min = 70
            self.colors.blue_b_minus_r = 38
            self.colors.red_s_min = 95
            self.colors.red_v_min = 65
            self.colors.red_r_minus_gb = 38
            self.colors.cyan_s_min = 35
            self.colors.cyan_v_min = 120
            self.colors.cyan_bg_minus_r = 20

        return {
            "mean_luma": round(mean_luma, 1),
            "hdr_mode": self.colors.hdr_mode,
            "brightness_scale": round(self.colors.brightness_scale, 3),
        }

    def save(self, path: Path = DEFAULT_CALIBRATION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "strike_zone": asdict(self.strike_zone),
            "lh_x_offset": self.lh_x_offset,
            "marker_search_zone": asdict(self.marker_search_zone),
            "scoreboard": asdict(self.scoreboard),
            "info_panel": asdict(self.info_panel),
            "tab_panel": asdict(self.tab_panel),
            "my_batting_row": self.my_batting_row,
            "my_id": self.my_id,
            "record_both": self.record_both,
            "record_video": self.record_video,
            "cpu_mode": self.cpu_mode,
            "my_auto": self.my_auto,
            "colors": asdict(self.colors),
            "save_snapshots": self.save_snapshots,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_CALIBRATION_PATH) -> "CalibrationConfig":
        if not path.exists():
            # 第一次執行：用隨程式附的預設校正值（calibration.default.json），之後的修改只存在自己的 calibration.json，更新不會覆蓋
            default = path.with_name("calibration.default.json")
            if default.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(default.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                cfg = cls()
                cfg.save(path)
                return cfg
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            cfg = cls(
                strike_zone=BoxRatio(**raw["strike_zone"]),
                lh_x_offset=float(raw.get("lh_x_offset", 0.0155)),
                marker_search_zone=BoxRatio(**raw["marker_search_zone"]),
                scoreboard=BoxRatio(**raw["scoreboard"]),
                info_panel=BoxRatio(**raw["info_panel"]),
                tab_panel=BoxRatio(**raw["tab_panel"]),
                my_batting_row=raw.get("my_batting_row", "top"),
                my_id=str(raw.get("my_id", "")),
                record_both=bool(raw.get("record_both", False)),
                record_video=bool(raw.get("record_video", False)),
                cpu_mode=bool(raw.get("cpu_mode", False)),
                my_auto=bool(raw.get("my_auto", True)),
                colors=ColorThresholdProfile(**raw.get("colors", {})),
                save_snapshots=bool(raw.get("save_snapshots", True)),
            )
            return cfg
        except Exception:
            return cls()
