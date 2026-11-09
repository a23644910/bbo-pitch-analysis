"""
Perceptual hash (64-bit DCT pHash) and pitch-type classification / fuzzy matching
for 《全民打棒球2》 pitch labels.
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import cv2
import numpy as np


# Known BBO2 pitch names (preserving prefixes like V-, S-, LW-, C-, RA, etc.)
KNOWN_BBO_PITCHES: List[str] = [
    "四縫線",
    "二縫線",
    "雙縫線快速球",
    "V-二縫",
    "H-二縫",
    "伸卡球",
    "S-伸卡",
    "V-滑球",
    "H-滑球",
    "S-滑球",
    "RA滑球",
    "火車滑球",
    "小雞滑球",
    "滑球",
    "S-曲球",
    "V-曲球",
    "慢速曲球",
    "不死鳥曲",
    "滑曲球",
    "曲球",
    "卡特球",
    "切球",
    "S-卡特",
    "變速球",
    "LW-變速球",
    "C-變速球",
    "C-變速",
    "V-變速球",
    "V-變速",
    "圈指變速",
    "指叉球",
    "S-指叉",
    "V-指叉",
    "快速指叉球",
    "快速指叉",
    "掌心球",
    "蝴蝶球",
    "螺旋球",
]

# Automatic mapping from keywords to major pitch groups (大類)
PITCH_GROUP_RULES: List[Tuple[Tuple[str, ...], str]] = [
    (("二縫", "雙縫", "伸卡"), "二縫"),
    (("滑曲",), "曲球"),
    (("滑球", "滑"), "滑球"),
    (("曲球", "不死鳥曲", "曲"), "曲球"),
    (("變速", "掌心"), "變速"),
    (("指叉",), "指叉"),
    (("卡特", "切球"), "卡特"),
    (("四縫", "直球", "快速球", "速球"), "直球"),
    (("蝴蝶", "螺旋"), "其他"),
]


def classify_pitch_group(pitch_raw: Optional[str], custom_map: Optional[dict] = None) -> Optional[str]:
    """
    Map a detailed pitch name (e.g. 'LW-變速球', '火車滑球', 'V-二縫') to its major group
    ('直球', '二縫', '滑球', '曲球', '變速', '指叉', '卡特', '其他').
    """
    if not pitch_raw:
        return None
    text = pitch_raw.strip()
    if not text or text == "未知":
        return None
    if custom_map and text in custom_map:
        return custom_map[text]
    for keywords, group in PITCH_GROUP_RULES:
        for kw in keywords:
            if kw in text:
                return group
    return "其他"


def compute_phash(img_bgr: np.ndarray) -> str:
    """
    Compute a 64-bit DCT perceptual hash (returned as a 16-char hex string)
    for a pitch label text crop.
    Isolates the sky-blue / bright text strokes before computing the DCT so minor
    background variations do not affect the hash.
    """
    if img_bgr is None or img_bgr.size == 0:
        return "0000000000000000"

    if img_bgr.ndim == 3:
        b, g, r = (
            img_bgr[:, :, 0].astype(np.float32),
            img_bgr[:, :, 1].astype(np.float32),
            img_bgr[:, :, 2].astype(np.float32),
        )
        # Sky-blue pitch text has high B and G relative to background
        signal = np.clip(0.6 * b + 0.4 * g - 0.3 * r, 0, 255).astype(np.uint8)
    else:
        signal = img_bgr.astype(np.uint8)

    resized = cv2.resize(signal, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low_freq = dct[:8, :8].flatten()
    # Exclude DC component (index 0) when computing median
    med = float(np.median(low_freq[1:]))
    bits = (low_freq > med).astype(np.uint8)
    val = 0
    for bit in bits:
        val = (val << 1) | int(bit)
    return f"{val:016x}"


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute the bit-level Hamming distance between two 16-char hex pHashes."""
    try:
        v1 = int(hash1, 16)
        v2 = int(hash2, 16)
        return (v1 ^ v2).bit_count()
    except Exception:
        return 64


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the character-level Levenshtein edit distance between s1 and s2."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def normalize_ocr_pitch_text(raw_text: str) -> str:
    """Clean up common OCR punctuation/prefix artifacts while preserving V-, S-, LW-, C-."""
    t = raw_text.strip()
    t = t.replace("—", "-").replace("–", "-").replace("_", "-").replace(" ", "")
    # Common OCR confusions in Traditional Chinese baseball terms
    replacements = {
        "四维": "四縫",
        "四緯": "四縫",
        "缝": "縫",
        "综": "線",
        "綜": "線",
        "綠": "線",
        "綿": "線",
        "线": "線",
        "毬": "球",
        "变速": "變速",
        "隧速": "變速",
        "变": "變",
        "双": "雙",
        "鸟": "鳥",
        "曲求": "曲球",
        "滑求": "滑球",
    }
    for k, v in replacements.items():
        if k in t:
            t = t.replace(k, v)
    return t


def match_pitch_to_repertoire(
    raw_text: str,
    pitcher_repertoire: Optional[Iterable[str]] = None,
    max_distance: int = 2,
) -> str:
    """
    Fuzzy-match OCR output against the pitcher's known repertoire (from TAB panel) first,
    then against KNOWN_BBO_PITCHES using edit distance.
    If pitcher_repertoire is provided and has a match within distance <= 2, the repertoire wins.
    """
    cleaned = normalize_ocr_pitch_text(raw_text)
    if not cleaned:
        return ""

    if pitcher_repertoire:
        rep_list = [p.strip() for p in pitcher_repertoire if p and p.strip()]
        if cleaned in rep_list:
            return cleaned
        best_rep = None
        best_d = 999
        for cand in rep_list:
            d = levenshtein_distance(cleaned.upper(), cand.upper())
            if d < best_d:
                best_d = d
                best_rep = cand
        if best_rep is not None and best_d <= max_distance:
            return best_rep

    if cleaned in KNOWN_BBO_PITCHES:
        return cleaned

    best_known = None
    best_kd = 999
    for cand in KNOWN_BBO_PITCHES:
        d = levenshtein_distance(cleaned.upper(), cand.upper())
        if d < best_kd:
            best_kd = d
            best_known = cand
    if best_known is not None and best_kd <= 1:
        return best_known

    return cleaned
