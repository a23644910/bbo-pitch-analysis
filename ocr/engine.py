"""
PaddleOCR wrapper with perceptual hash (pHash) caching for:
1. Pitch result labels (STRIKE / BALL, pitch type, speed km/h)
2. Bottom-right info panel (batter name, pitcher year + name; only re-runs OCR when image changes)
3. Right-side TAB panel (Pitcher full repertoire list; Stage 7)
"""

from __future__ import annotations
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

# Ensure PaddlePaddle 3.x disables MKLDNN/PIR before importing paddleocr
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

from capture.calibration import CalibrationConfig
from detect.markers import imwrite_unicode
from ocr.phash import (
    KNOWN_BBO_PITCHES,
    classify_pitch_group,
    compute_phash,
    hamming_distance,
    match_pitch_to_repertoire,
    normalize_ocr_pitch_text,
)
from store.db import PitchStore


SPEED_RE = re.compile(r"(\d{2,3}(?:\.\d)?)\s*(?:[Kk][Mm]|[Kk]m/h|km/h)?")
PLAYER_NAME_RE = re.compile(r"(\d{2})\s*([A-Za-z\u4e00-\u9fff\.．\s\-]+)")


SIMP_TO_TRAD_NAMES = str.maketrans(
    {
        "维": "緯",
        "来": "來",
        "马": "馬",
        "兹": "茲",
        "尔": "爾",
        "锋": "鋒",
        "陈": "陳",
        "伟": "偉",
        "伦": "倫",
        "约": "約",
        "罗": "羅",
        "萨": "薩",
        "亚": "亞",
        "杰": "傑",
        "麦": "麥",
        "变": "變",
        "缝": "縫",
        "线": "線",
        "鸟": "鳥",
        "双": "雙",
    }
)


@dataclass
class LabelOCRResult:
    label_word: Optional[str]  # "STRIKE", "BALL", or None
    pitch_raw: Optional[str]
    pitch_group: Optional[str]
    speed_kmh: Optional[float]
    phash: str
    cache_hit: bool
    raw_lines: List[str]


class BBOOCREngine:
    _shared_ocr = None
    _ocr_lock = threading.Lock()

    def __init__(self, samples_dir: Optional[Path] = None):
        self._ocr = None
        self.samples_dir = samples_dir or (
            Path(__file__).resolve().parent.parent / "data" / "label_samples"
        )
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        # Cache for bottom-right info panel so OCR only runs when name strip changes
        self._last_batter_hash: str = ""
        self._last_pitcher_hash: str = ""
        self._last_batter_name: str = ""
        self._last_pitcher_name: str = ""

        # Cache for TAB panel so we only read once per TAB press
        self._last_tab_hash: str = ""
        self._last_tab_try: float = 0.0
        self._tab_misses: int = 0

        # Cache for scoreboard inning number box
        self._last_inning_hash: str = ""
        self._last_inning_num: Optional[int] = None

    def _get_ocr(self):
        if BBOOCREngine._shared_ocr is None:
          with BBOOCREngine._ocr_lock:
           if BBOOCREngine._shared_ocr is None:
            from paddleocr import PaddleOCR

            kw = dict(
                lang="chinese_cht",
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            try:
                # 輕量化：文字辨識最多用 3 個 CPU 執行緒，把其他核心留給遊戲
                BBOOCREngine._shared_ocr = PaddleOCR(cpu_threads=3, **kw)
            except TypeError:
                BBOOCREngine._shared_ocr = PaddleOCR(**kw)
        return BBOOCREngine._shared_ocr

    def run_ocr_lines(self, img_bgr: np.ndarray, upscale: float = 2.0) -> List[str]:
        if img_bgr is None or img_bgr.size == 0:
            return []
        if upscale != 1.0:
            work = cv2.resize(
                img_bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
            )
        else:
            work = img_bgr
        ocr = self._get_ocr()
        # PaddleOCR 不能多執行緒同時呼叫：背景工作與標籤辨識共用一把鎖
        with BBOOCREngine._ocr_lock:
            res = list(ocr.predict(work))
        if not res:
            return []
        texts = res[0].get("rec_texts", [])
        return [str(t).strip() for t in texts if str(t).strip()]

    # ------------------------------------------------------------------
    # 1. Pitch Result Label OCR + pHash Cache
    # ------------------------------------------------------------------
    def recognize_pitch_label(
        self,
        label_crop: np.ndarray,
        pitch_text_crop: Optional[np.ndarray],
        store: Optional[PitchStore] = None,
        pitcher: str = "",
    ) -> LabelOCRResult:
        phash = compute_phash(pitch_text_crop if pitch_text_crop is not None else label_crop)
        cached_raw: Optional[str] = None
        cached_group: Optional[str] = None
        cache_hit = False

        if store is not None and phash != "0000000000000000":
            hit = store.lookup_phash(phash, max_hamming=3)
            if hit is not None:
                cached_raw, cached_group = hit
                cache_hit = True

        lines = self.run_ocr_lines(label_crop, upscale=2.0)
        label_word: Optional[str] = None
        speed_kmh: Optional[float] = None
        candidate_pitch_lines: List[str] = []

        for line in lines:
            u = line.upper()
            if "STRIKE" in u or "STR1KE" in u or u == "STRIK":
                label_word = "STRIKE"
                continue
            if "BALL" in u or u == "BAL":
                label_word = "BALL"
                continue

            # Check speed line (e.g. "158.8 Km/h", "147.0Km/h")
            if "KM" in u or ("." in line and any(ch.isdigit() for ch in line)):
                m = SPEED_RE.search(line)
                if m:
                    try:
                        val = float(m.group(1))
                        if 70.0 <= val <= 200.0:
                            speed_kmh = val
                            continue
                    except ValueError:
                        pass

            # Ignore single-character noise (e.g. 'X' from corner of ⊗ marker)
            cleaned_cand = line.strip(" .+×xX*·-—_")
            if len(cleaned_cand) >= 2 and any("\u4e00" <= ch <= "\u9fff" for ch in cleaned_cand):
                candidate_pitch_lines.insert(0, cleaned_cand)
            elif len(cleaned_cand) >= 2:
                candidate_pitch_lines.append(cleaned_cand)

        if cache_hit and cached_raw:
            pitch_raw = cached_raw
            pitch_group = cached_group or classify_pitch_group(pitch_raw)
        else:
            raw_joined = candidate_pitch_lines[0] if candidate_pitch_lines else ""
            repertoire = store.get_pitcher_repertoire(pitcher) if (store and pitcher) else None
            pitch_raw = match_pitch_to_repertoire(raw_joined, pitcher_repertoire=repertoire)
            if not pitch_raw:
                pitch_raw = None
            pitch_group = classify_pitch_group(pitch_raw)

            if store is not None and pitch_raw and phash != "0000000000000000":
                sample_file = self.samples_dir / f"{phash}.png"
                if not sample_file.exists():
                    imwrite_unicode(
                        sample_file,
                        pitch_text_crop if pitch_text_crop is not None else label_crop,
                    )
                store.save_phash(
                    phash=phash,
                    pitch_raw=pitch_raw,
                    pitch_group=pitch_group,
                    sample_path=str(sample_file),
                )

        return LabelOCRResult(
            label_word=label_word,
            pitch_raw=pitch_raw,
            pitch_group=pitch_group,
            speed_kmh=speed_kmh,
            phash=phash,
            cache_hit=cache_hit,
            raw_lines=lines,
        )

    # ------------------------------------------------------------------
    # 2. Bottom-Right Info Panel (Batter & Pitcher Names)
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_player_text(lines: List[str], is_pitcher: bool = False) -> str:
        """
        Extract clean 'Year + Name' (e.g. '13 A.緯來特', '24 L. Webb', '23 Keynan Mid')
        from OCR lines of the info panel's red name bar.
        """
        full = " ".join(lines).translate(SIMP_TO_TRAD_NAMES).strip()
        if not full:
            return ""
        # Remove stats/roles that may appear near the right edge of the red strip
        cleaned = re.sub(
            r"\b(?:SP|RP|CP|DH|SS|LF|RF|CF|1B|2B|3B|ERA|AVG|HR|RBI|0W|0L|OW|OL|OO|O0|0O|OLERA)\b",
            " ",
            full,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\b\d+\s*[WL]\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\d+\.\d+", " ", cleaned)
        # Remove leading single-digit batting order if followed by 2-digit year (e.g. "1 11E.波尼法西")
        cleaned = re.sub(r"^\s*[1-9]\s+(?=\d{2})", "", cleaned)
        cleaned = re.sub(r"(\d{2})(?=[A-Za-z\u4e00-\u9fff])", r"\1 ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        m = PLAYER_NAME_RE.search(cleaned)
        if m:
            year = m.group(1)
            name = m.group(2).strip()
            name = re.sub(
                r"\s+(?:SS|DH|LF|RF|CF|1B|2B|3B|C|D|S|O|0|OLERA|ERA)+$",
                "",
                name,
                flags=re.IGNORECASE,
            ).strip()
            name = re.sub(r"(?<=[\u4e00-\u9fff])\s+[A-Za-z0-9\s]{1,12}$", "", name).strip()
            if len(name) >= 2:
                return f"{year} {name}"
        return ""

    @staticmethod
    def _has_name_bar(panel: np.ndarray) -> bool:
        ph, pw = panel.shape[:2]
        hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
        colored = (hsv[..., 1] > 90) & (hsv[..., 2] > 70)
        for x1, x2 in ((0.0, 0.48), (0.52, 1.0)):
            half = colored[:, int(pw * x1):int(pw * x2)]
            rows = np.where(half.mean(axis=1) > 0.5)[0]
            in_band = ((rows >= ph * 0.35) & (rows <= ph * 0.68)).sum()   # 名字色條的位置
            above = (rows < ph * 0.33).sum()                               # 上方被 TAB 大頭像蓋住時會很多
            if in_band >= max(4, int(ph * 0.10)) and above <= int(ph * 0.15):
                return True
        return False

    def read_info_panel(
        self, client_bgr: np.ndarray, config: CalibrationConfig, force: bool = False
    ) -> Tuple[str, str]:
        """
        Read (batter_name, pitcher_name) from the bottom-right info panel.
        Dynamically isolates the crimson-red horizontal name bar and uses pHash gating
        so OCR only runs when a player actually changes.
        """
        panel = config.info_panel.crop(client_bgr)
        if panel.size == 0 or panel.shape[0] < 20 or panel.shape[1] < 80:
            return self._last_batter_name, self._last_pitcher_name

        ph, pw = panel.shape[:2]
        # 確認名字色條有出現（沒被 TAB 的大頭像蓋住）。
        # 色條顏色依卡片而不同（紅、藍、紫、金），而且打者、投手兩條左右並排、顏色可能不同；
        # 以前把整列平均算「紅減藍」，打者藍條＋投手紅條會互相抵消 → 後攻那半局永遠讀不到名字。
        # 改成左右兩半各自看「高彩度」的列，任一邊有色條就算。
        if not self._has_name_bar(panel):
            return self._last_batter_name, self._last_pitcher_name

        # Use fixed normalized vertical bounds (0.39..0.63) so 1px video compression jitter
        # never shifts the crop or invalidates the pHash cache
        ry1 = int(ph * 0.39)
        ry2 = int(ph * 0.63)
        batter_strip = panel[ry1:ry2, int(pw * 0.07):int(pw * 0.36)]
        pitcher_strip = panel[ry1:ry2, int(pw * 0.56):int(pw * 0.82)]

        b_hash = compute_phash(batter_strip)
        p_hash = compute_phash(pitcher_strip)

        if (
            force
            or not self._last_batter_hash
            or hamming_distance(b_hash, self._last_batter_hash) > 6
        ):
            self._last_batter_hash = b_hash
            b_lines = self.run_ocr_lines(batter_strip, upscale=2.5)
            b_name = self._clean_player_text(b_lines, is_pitcher=False)
            if b_name:
                self._last_batter_name = b_name

        if (
            force
            or not self._last_pitcher_hash
            or hamming_distance(p_hash, self._last_pitcher_hash) > 6
        ):
            self._last_pitcher_hash = p_hash
            p_lines = self.run_ocr_lines(pitcher_strip, upscale=2.5)
            p_name = self._clean_player_text(p_lines, is_pitcher=True)
            if p_name:
                self._last_pitcher_name = p_name

        return self._last_batter_name, self._last_pitcher_name

    def read_scores(
        self, client_bgr: np.ndarray, config: CalibrationConfig
    ) -> Tuple[Optional[int], Optional[int]]:
        """讀右上記分板兩隊比分（上行、下行），讀不到回傳 None。"""
        sb = config.scoreboard.crop(client_bgr)
        if sb.size == 0 or sb.shape[0] < 20:
            return None, None
        sh, sw = sb.shape[:2]
        out: List[Optional[int]] = []
        for y1, y2 in ((0.08, 0.50), (0.50, 0.92)):
            crop = sb[int(sh * y1):int(sh * y2), int(sw * 0.115):int(sw * 0.215)]
            up = cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
            pad = cv2.copyMakeBorder(up, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            txt = "".join(self.run_ocr_lines(pad, upscale=1.0)).upper()
            txt = txt.replace("O", "0").replace("D", "0").replace("I", "1").replace("L", "1").replace("S", "5")
            m = re.search(r"(\d{1,2})", txt)
            out.append(int(m.group(1)) if m else None)
        return out[0], out[1]

    def read_inning_number(
        self, client_bgr: np.ndarray, config: CalibrationConfig, force: bool = False
    ) -> Optional[int]:
        """
        Read the inning number (1..12) from the top-left corner of the top-right scoreboard
        (e.g. '1局', '5局', '6局') with pHash caching so OCR only runs when the box changes.
        """
        sb = config.scoreboard.crop(client_bgr)
        if sb.size == 0 or sb.shape[0] < 20 or sb.shape[1] < 80:
            return self._last_inning_num
        sh, sw = sb.shape[:2]
        crop = sb[int(sh * 0.15):int(sh * 0.52), int(sw * 0.02):int(sw * 0.12)]
        if crop.size == 0:
            return self._last_inning_num

        i_hash = compute_phash(crop)
        if (
            not force
            and self._last_inning_hash
            and hamming_distance(i_hash, self._last_inning_hash) <= 4
            and self._last_inning_num is not None
        ):
            return self._last_inning_num

        self._last_inning_hash = i_hash
        up = cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        pad = cv2.copyMakeBorder(
            up, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(20, 20, 20)
        )
        lines = self.run_ocr_lines(pad, upscale=1.0)
        joined = "".join(lines).upper()
        joined = (
            joined.replace("S", "5")
            .replace("I", "1")
            .replace("L", "1")
            .replace("|", "1")
            .replace("B", "8")
            .replace("O", "0")
        )
        m = re.search(r"(\d{1,2})", joined)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 12:
                self._last_inning_num = val
                return val
        return self._last_inning_num

    # ------------------------------------------------------------------
    # 2b. 隊伍／玩家名稱（自動判斷我方先攻或後攻）
    # ------------------------------------------------------------------
    # 2 對 2 時左上 HOME、右上 AWAY 面板列出雙方玩家（比例座標，相對遊戲畫面）
    HOME_PANEL = (0.008, 0.095, 0.098, 0.160)
    AWAY_PANEL = (0.902, 0.095, 0.992, 0.160)

    def read_team_names(
        self, client_bgr: np.ndarray, config: CalibrationConfig
    ) -> dict:
        """
        回傳 {"top": 記分板上行名稱, "bottom": 下行名稱, "home": HOME 面板文字, "away": AWAY 面板文字}。
        單人模式記分板直接顯示玩家 ID；2 對 2 時記分板是 Away Team / Home Team，要看 HOME/AWAY 面板。
        """
        out = {"top": "", "bottom": "", "home": "", "away": ""}
        h, w = client_bgr.shape[:2]
        sb = config.scoreboard.crop(client_bgr)
        if sb.size > 0 and sb.shape[0] >= 20:
            sh, sw = sb.shape[:2]
            # 名字區：避開左邊比分數字（約 0.13–0.21）與右邊「全民打棒球2」浮水印（約 0.50 起）
            top_bar = sb[int(sh * 0.08):int(sh * 0.50), int(sw * 0.225):int(sw * 0.50)]
            bot_bar = sb[int(sh * 0.50):int(sh * 0.92), int(sw * 0.225):int(sw * 0.50)]
            out["top"] = " ".join(self.run_ocr_lines(top_bar, upscale=3.0))
            out["bottom"] = " ".join(self.run_ocr_lines(bot_bar, upscale=3.0))
        for key, (x1, y1, x2, y2) in (("home", self.HOME_PANEL), ("away", self.AWAY_PANEL)):
            crop = client_bgr[int(h * y1):int(h * y2), int(w * x1):int(w * x2)]
            if crop.size > 0:
                out[key] = " ".join(self.run_ocr_lines(crop, upscale=2.5))
        return out

    # ------------------------------------------------------------------
    # 3. TAB Panel Repertoire Reader (Stage 7)
    # ------------------------------------------------------------------
    def read_tab_panel_if_visible(
        self,
        client_bgr: np.ndarray,
        config: CalibrationConfig,
        store: Optional[PitchStore] = None,
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Check if the right-side 'Pitcher' TAB card is visible.
        When detected, OCR the pitcher's name and full pitch repertoire once and save to DB.
        """
        tab_crop = config.tab_panel.crop(client_bgr)
        if tab_crop.size == 0 or tab_crop.shape[0] < 60 or tab_crop.shape[1] < 80:
            return None

        th, tw = tab_crop.shape[:2]
        pitcher_card = tab_crop[: int(th * 0.56), :]
        # Fast pixel check: when TAB panel is open, the right side of pitcher_card is dark navy/grey
        # (whereas when TAB is closed, that region is green grass or brown dirt with high G or R-B)
        right_half = pitcher_card[:, int(tw * 0.35):]
        mean_g = float(np.mean(right_half[:, :, 1]))
        mean_b = float(np.mean(right_half[:, :, 0]))
        mean_r = float(np.mean(right_half[:, :, 2]))
        if mean_g > 95.0 or (mean_g - mean_b) > 12.0 or (mean_r - mean_b) > 15.0:
            return None

        card_hash = compute_phash(pitcher_card)
        if self._last_tab_hash and hamming_distance(card_hash, self._last_tab_hash) <= 5:
            return None
        # 節流：右下角有跑壘小地圖（CAM 1壘跑者）時畫面一直變，會讓這裡每幾幀就跑一次 OCR，
        # 拖慢整體辨識（FPS 掉到 1）。改成最多每 4 秒試一次，連續沒看到 Pitcher 就再放慢。
        import time as _t
        now = _t.time()
        wait = 4.0 if self._tab_misses < 3 else 15.0
        if now - self._last_tab_try < wait:
            return None
        self._last_tab_try = now

        self._last_tab_hash = card_hash
        raw_lines = self.run_ocr_lines(pitcher_card, upscale=2.0)
        lines = [ln.translate(SIMP_TO_TRAD_NAMES) for ln in raw_lines]
        joined_upper = " ".join(lines).upper()
        if "PITCHER" not in joined_upper:
            self._tab_misses += 1
            return None
        self._tab_misses = 0
        # Only parse lines starting from 'Pitcher' header so roster rows above it are ignored
        start_idx = 0
        for idx, ln in enumerate(lines):
            if "PITCHER" in ln.upper():
                start_idx = idx + 1
                break
        card_lines = lines[start_idx:]

        pitcher_name = ""
        repertoire: List[str] = []

        ignore_tokens = {
            "PITCHER",
            "體力",
            "控球",
            "最上",
            "超上",
            "上",
            "中",
            "下",
            "右投",
            "左投",
            "右打",
            "左打",
            "直球壓制者",
            "直球屋制者",
            "變化球大師",
            "F大師",
        }
        for ln in card_lines:
            u = ln.upper()
            if "PITCHER" in u or "/" in ln or "右投" in ln or "左投" in ln:
                continue
            m = PLAYER_NAME_RE.search(ln)
            if m and not pitcher_name:
                pitcher_name = self._clean_player_text([ln], is_pitcher=True)
                continue
            cleaned = ln
            for tok in (
                "體力",
                "雅力",
                "控球",
                "玻球",
                "最上",
                "超上",
                "直球壓制者",
                "直球屋制者",
                "F大師",
                "SP",
                "RP",
                "CP",
            ):
                cleaned = cleaned.replace(tok, " ")
            for part in cleaned.split():
                part_norm = normalize_ocr_pitch_text(part)
                # Strip leading or trailing single rating char ('上', '中', '下', '生') glued to pitch name
                while len(part_norm) >= 3 and part_norm[0] in ("上", "中", "下", "生"):
                    part_norm = part_norm[1:]
                while len(part_norm) >= 3 and part_norm[-1] in ("上", "中", "下", "生"):
                    part_norm = part_norm[:-1]
                if not part_norm or part_norm in ignore_tokens or len(part_norm) < 2:
                    continue
                matched = match_pitch_to_repertoire(part_norm, max_distance=1)
                if matched in KNOWN_BBO_PITCHES:
                    if matched not in repertoire:
                        repertoire.append(matched)
                elif (
                    len(part_norm) >= 3
                    and classify_pitch_group(part_norm) not in (None, "其他")
                    and part_norm not in repertoire
                ):
                    # 卡片專屬球種（例如「凱爾曲球」「孔恩滑球」）不在內建清單裡，保留原文
                    repertoire.append(part_norm)

        if not pitcher_name or (
            self._last_pitcher_name
            and pitcher_name[:2] == self._last_pitcher_name[:2]
            and len(self._last_pitcher_name) > len(pitcher_name)
        ):
            pitcher_name = self._last_pitcher_name or pitcher_name

        if pitcher_name and repertoire and store is not None:
            store.save_pitcher_repertoire(pitcher_name, repertoire)
            if self._last_pitcher_name and self._last_pitcher_name != pitcher_name:
                store.save_pitcher_repertoire(self._last_pitcher_name, repertoire)

        if repertoire:
            return (pitcher_name, repertoire)
        return None
