"""
Stage 4 & Stage 5 State Machine (`GameSessionTracker`):
- Tracks scoreboard (S/B/O, half-inning, blue batting row highlight)
- Tracks bottom-right player info panel (batter, pitcher) with pHash change gating
- Tracks right-side TAB panel (Pitcher repertoire)
- Tracks light-cyan batting cursor (`has_batting_cursor`) to confirm our team is batting
- Implements the 4-step pitch decision tree:
  1. Is our team batting? (scoreboard row == my_batting_row AND cyan cursor seen)
  2. Red ⊗ -> ball (with label OCR; cross-checks label word)
  3. Blue ◎ with label -> strike (with label OCR; cross-checks label word)
  4. Blue ◎ without label -> observe post-marker frames (~0.80s):
     camera cut to fielding / batter or outs change -> in_play (擊出);
     otherwise -> foul (界外)
- Implements Stage 5 pitch count tracking (waits 5 consecutive identical S/B frames
  after each pitch to record post-pitch count as next pitch's pre-pitch count,
  avoiding the BBO2 pause/TAB 0-0 bug).
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from capture.calibration import CalibrationConfig
from detect.cursor import has_batting_cursor, is_batting_camera
from detect.markers import MarkerDetection, detect_pitch_marker, detect_label_only, imwrite_unicode
from detect.scoreboard import ScoreboardState, read_scoreboard
from detect.environment import read_environment, read_energy, wind_speed_crop
from detect.gameover import has_confirm_box_button
from detect.lowball import find_label_text
from ocr.engine import BBOOCREngine
from ocr.phash import classify_pitch_group, levenshtein_distance
import re
import threading
from store.db import PitchStore


@dataclass
class PendingPitch:
    first_seen_ts: float
    last_seen_ts: float
    confirm_frames: int
    best_detection: MarkerDetection
    best_frame_bgr: np.ndarray
    inning: int
    half: str
    outs_before: int
    batter: str
    pitcher: str
    balls_before: Optional[int]
    strikes_before: Optional[int]
    missing_frames: int = 0
    fielding_cut_frames: int = 0
    post_marker_frames: int = 0
    marker_gone: bool = False
    # 標記最後一次出現的位置（用來確認標記已經定住，不是飛行中的藍圈）
    last_cx: int = 0
    last_cy: int = 0
    # 標籤需經 OCR 讀到 STRIKE / BALL 才算數（暗色球場會把背景誤認成標籤）
    label_checked: bool = False
    label_ok: bool = False
    ocr_res: Optional[Any] = None
    # 投球方（記分板上行 top／下行 bottom）
    pitching_side: str = ""
    # 標籤 OCR 改在背景跑，不卡住畫面擷取
    label_job: Optional[str] = None
    label_submit_ts: float = 0.0
    # 標籤字樣（BALL / STRIKE）用圖形比對先確認，不用等 OCR
    tpl_word: Optional[str] = None


# 網頁會比對這個版本號，後台沒重開時提醒使用者
APP_VERSION = "2026-11-10i"


@dataclass
class TrackerStatus:
    connected: bool = False
    fps: float = 0.0
    game_id: str = "default"
    my_batting_row: str = "top"  # "top" (先攻/上行) or "bottom" (後攻/下行)
    opponent_tag: str = ""
    inning: int = 1
    half: str = "上"
    outs: int = 0
    balls: Optional[int] = 0
    strikes: Optional[int] = 0
    batter: str = ""
    pitcher: str = ""
    is_my_batting: bool = False
    cursor_active: bool = False
    batting_camera: bool = False
    last_pitch: Optional[Dict[str, Any]] = None
    pitcher_repertoire: List[str] = field(default_factory=list)
    my_id: str = ""
    row_mode: str = "auto"  # "auto" 依 ID 自動判斷, "manual" 使用者手動選
    row_status: str = "searching"  # searching / found / not_found / manual
    row_source: str = ""  # 例如「記分板上行」「AWAY 面板」
    game_over: bool = False  # 看到結算畫面後為 True，停止記錄直到新比賽開始
    record_both: bool = False  # 雙方投球都記錄
    app_version: str = APP_VERSION
    capture_mode: str = ""  # 前景（截螢幕）／背景（PrintWindow）
    team_top: str = ""  # 記分板上行（先攻）玩家 ID
    team_bottom: str = ""  # 記分板下行（後攻）玩家 ID
    pitching_players: str = ""  # 目前投球方的玩家 ID
    pitching_slot: str = ""  # 例如「主隊1・單數局」
    players: List[Dict[str, Any]] = field(default_factory=list)  # 四個（或兩個）玩家欄位
    audit: Dict[str, int] = field(default_factory=dict)  # 記分板審核：核對／改正／補漏記
    env: Dict[str, Any] = field(default_factory=dict)  # 天氣、風、卡片天氣屬性
    cpu_pct: float = 0.0  # 程式佔整台電腦 CPU 的比例


def _norm_id(text: str) -> str:
    t = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
    # OCR 常見混淆：0/O、8/B、1/I/L、5/S
    return t.translate(str.maketrans({"0": "O", "8": "B", "1": "I", "L": "I", "5": "S"}))


def id_in_text(my_id: str, text: str) -> bool:
    """文字中是否出現玩家 ID（容許 OCR 錯 1 個字）。"""
    target = _norm_id(my_id)
    hay = _norm_id(text)
    if not target or not hay:
        return False
    if target in hay:
        return True
    n = len(target)
    tol = 1 if n >= 4 else 0
    for L in (n - 1, n, n + 1):
        if L <= 0:
            continue
        for i in range(0, max(1, len(hay) - L + 1)):
            if levenshtein_distance(hay[i:i + L], target) <= tol:
                return True
    return False


def clean_player_id(text: str) -> str:
    """單人模式記分板名稱：去掉 OCR 雜訊，只留 ID。"""
    t = re.sub(r"\s+", " ", text or "")
    t = re.sub(r"全民打?棒?球?2?", " ", t)       # 記分板浮水印
    t = re.sub(r"^\s*\d{1,2}\s+", "", t)       # 開頭的比分數字
    t = re.sub(r"\s+\d{1,2}\s*$", "", t)
    t = re.sub(r"\s+", " ", t).strip(" .:-_|")
    return t[:24]


def parse_panel_players(text: str) -> List[str]:
    """2 對 2 的 HOME / AWAY 面板，例如「1P 嘉威好棒哦 2C 2P 阿羅拉多 2C」→ ['嘉威好棒哦', '阿羅拉多']。"""
    t = re.sub(r"\b\d{1,2}\s*C\b", " ", text or "", flags=re.IGNORECASE)  # 右側等級徽章
    t = re.sub(r"\b(HOME|AWAY)\b", " ", t, flags=re.IGNORECASE)
    parts = re.split(r"(?:^|\s)\d{1,2}\s*P\s*", " " + t)
    out = []
    for part in parts:
        name = re.sub(r"\s+", " ", part).strip(" .:-_|")
        if len(name) >= 2 and name not in out:
            out.append(name[:24])
    return out


class GameSessionTracker:
    def __init__(
        self,
        store: PitchStore,
        ocr_engine: BBOOCREngine,
        config: Optional[CalibrationConfig] = None,
        game_id: Optional[str] = None,
        my_batting_row: str = "top",
        opponent_tag: str = "",
        require_cursor: bool = False,
        mid_game_start: bool = False,
        snapshots_dir: Optional[Path] = None,
        on_pitch_recorded: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.store = store
        self.ocr_engine = ocr_engine
        self.config = config or CalibrationConfig.load()
        self.game_id = game_id or datetime.now().strftime("game_%Y%m%d_%H%M%S")
        self.my_batting_row = my_batting_row  # "top" or "bottom"
        self.opponent_tag = opponent_tag
        self.require_cursor = require_cursor
        self.on_pitch_recorded = on_pitch_recorded

        # 自動判斷先攻／後攻
        self.my_id: str = (self.config.my_id or "").strip()
        self.row_mode: str = "auto" if self.my_id else "manual"
        self.row_status: str = "searching" if self.my_id else "manual"
        self.row_source: str = ""
        self._row_tries: int = 0

        # 比賽結束／新比賽自動判斷
        self.game_over: bool = False
        self._newgame_streak: int = 0

        # 雙方玩家 ID（用來標記每一球是誰投的）
        self.record_both: bool = bool(getattr(self.config, "record_both", False))
        self.team_names: Dict[str, List[str]] = {"top": [], "bottom": []}
        self._team_change_cand = None
        self._team_change_streak = 0
        self._pitches_this_game = 0
        self.id_changed = False
        self.audit: Dict[str, int] = {"checked": 0, "fixed": 0, "missed": 0}
        self._missed_cand = None
        self.capture_mode: str = ""
        self.reject_counts: Dict[str, int] = {}
        # 給記錄器用：每幀的判斷結果、可疑事件
        self._frame_reject: str = ""
        self._last_det: Optional[MarkerDetection] = None
        self.events: List[Dict[str, Any]] = []

        # 所有 OCR 都丟到背景執行緒，主迴圈不再被卡住
        # （實測一場比賽有 40% 的時間卡在 OCR，期間投的球全部漏掉）
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-bg")
        self._label_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-label")
        self._jobs: Dict[str, Any] = {}
        self._job_seq: int = 0
        # 背景 OCR 排程（2026-11-10i）：一次只跑一件、投球前後先暫停，避免開場一次排一堆把 CPU 吃滿、主程式卡住好幾秒漏球
        self._wants: Dict[str, Tuple[Any, tuple, dict]] = {}
        self._bg_running: bool = False
        self._bg_free_since: float = -999.0
        # 名字讀不到時的重試間隔（連續失敗就放慢，不要每秒重讀）
        self._info_fail: int = 0
        self._info_next_ts: float = 0.0
        # 先記錄、晚點補上球種球速：標籤 OCR 工作 → 已記錄的那一球
        self._label_fill: Dict[str, Tuple[int, float]] = {}
        self._seen_unrecorded: int = 0

        self.snapshots_dir = snapshots_dir or (
            Path(__file__).resolve().parent.parent / "data" / "snapshots"
        )
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Inning & Scoreboard state
        self.inning: int = 1
        self.half: str = "上"
        self.active_batting_row: str = "unknown"
        # 穩定的打擊方：只採用打擊畫面中連續 10 幀相同的結果，
        # 避免擊球後鏡頭切換、記分板一閃而過時誤判成換邊
        self.stable_batting_row: str = "unknown"
        self._row_cand: str = "unknown"
        self._row_cand_streak: int = 0
        # 兩邊各自目前的投手（換邊時先切回那一邊上次的投手，再等 OCR 確認）
        self._pitcher_by_side: Dict[str, str] = {}
        self._pitch_inning: int = 1
        # 比賽環境：球場天氣、風、投手／打者卡片天氣屬性（每一球一起記）
        self.env: Dict[str, Any] = {"weather": "", "wind_dir": "", "wind_angle": None, "wind_speed": None,
                                    "pitcher_attr": "", "batter_attr": ""}
        self._wind_hash = None
        self._energy: Optional[Tuple[float, float]] = None  # (能量 0~1, 讀到的時間)
        self._name_counts: Optional[Dict[str, int]] = None
        self._force_teams: bool = True
        self._force_names: bool = False
        self.outs: int = 0
        self.batter: str = ""
        self.pitcher: str = ""
        self._half_candidate: str = "上"
        self._half_streak: int = 0
        self._outs_candidate: int = 0
        self._outs_streak: int = 0
        self._live_sb_target: Optional[Tuple[int, int]] = None
        self._live_sb_streak: int = 0
        self._invalid_sb_streak: int = 0

        # Stage 5: Tracked pre-pitch count (None if started mid-at-bat until next batter)
        self.tracked_balls: Optional[int] = None if mid_game_start else 0
        self.tracked_strikes: Optional[int] = None if mid_game_start else 0

        # Post-pitch 5-frame stabilization buffer
        self._awaiting_post_count: bool = False
        self._last_recorded_pitch_id: Optional[int] = None
        self._last_recorded_expected_count: Optional[Tuple[int, int]] = None
        self._last_recorded_note: str = ""
        self._stable_sb_target: Optional[Tuple[int, int]] = None
        self._stable_sb_count: int = 0

        # Cursor & Marker deduplication state
        self._last_cursor_ts: float = -999.0
        self._cursor_streak: int = 0
        self._cursor_memory_sec: float = 15.0
        self._last_marker_gone_ts: float = -999.0
        self._cooldown_sec: float = 1.0
        # 防止「play 結束鏡頭切回打擊畫面時，又顯示上一顆球的落點」被當成新的一球
        self._cam_on_since: float = -999.0  # 打擊畫面從什麼時候開始連續出現
        self._cam_off_start: float = -999.0
        self._last_cam_true_ts: float = -999.0
        self._prev_det_present: bool = False  # 上一幀是否已經有標記
        self._replay_block_until_gone: bool = False
        self._last_rec_pos: Optional[Tuple[int, int, str, float]] = None  # (cx, cy, kind, ts)
        self._min_confirm_frames: int = 2
        # 沒有標籤的球（界外或擊出）先暫記，等記分板球數出來再確認，必要時自動改正
        self._hit_check: Optional[Dict[str, Any]] = None
        self._prev_frame_ts: Optional[float] = None
        self.corrections: List[Dict[str, Any]] = []
        self._pending: Optional[PendingPitch] = None
        self._latched_until_gone: bool = False
        self._latched_missing_frames: int = 0

        # Frame counter for periodic tasks (info panel OCR, TAB panel check)
        self._frame_idx: int = 0
        self.last_recorded_pitch: Optional[Dict[str, Any]] = None
        self._last_sb: Optional[ScoreboardState] = None
        self._last_batting_cam: bool = False
        self._last_cursor_now: bool = False
        self.fps: float = 0.0

    def set_my_batting_row(self, row: str) -> None:
        """使用者手動選擇先攻／後攻（會關閉自動判斷）。"""
        if row in ("top", "bottom"):
            self.my_batting_row = row
            self.row_mode = "manual"
            self.row_status = "manual"
            self.row_source = ""

    def set_row_auto(self) -> None:
        """重新開始依 ID 自動判斷。"""
        self.row_mode = "auto" if self.my_id else "manual"
        self.row_status = "searching" if self.my_id else "manual"
        self.row_source = ""
        self._row_tries = 0

    def set_my_id(self, my_id: str) -> None:
        self.my_id = (my_id or "").strip()
        self.config.my_id = self.my_id
        try:
            self.config.save()
        except Exception:
            pass
        self.set_row_auto()

    def _row_for_logic(self) -> str:
        return self.stable_batting_row if self.stable_batting_row in ("top", "bottom") else self.active_batting_row

    @staticmethod
    def _other(row: str) -> str:
        return "bottom" if row == "top" else ("top" if row == "bottom" else "")

    def _on_side_change(self, new_row: str) -> None:
        """換邊（攻守交換）：投手換成另一隊的，先用那一隊上次的投手，並強制重讀右下角名字。"""
        old_row = self.stable_batting_row
        if old_row in ("top", "bottom") and self.pitcher:
            self._pitcher_by_side[self._pkey(self._other(old_row), self._pitch_inning)] = self.pitcher
        self.stable_batting_row = new_row
        self._emit("攻守交換", row=new_row, old=old_row)
        # 換到下一個半局：下半局→上半局時局數會 +1（局數稍後才更新，這裡先推算）
        nxt = self.inning + (1 if new_row == "top" and old_row == "bottom" else 0)
        self.pitcher = self._pitcher_by_side.get(self._pkey(self._other(new_row), nxt), "")
        self._pitch_inning = nxt
        self.batter = ""
        self._force_names = True
        self._force_teams = True
        self._info_fail = 0
        self._info_next_ts = 0.0

    _RESULT_NAME = {"foul": "界外", "in_play": "擊出", "strike": "好球", "ball": "壞球"}

    def _retrain_model(self) -> None:
        try:
            self.store.retrain_model()
        except Exception as e:
            print(f"  [預測 AI 重訓失敗] {e}", flush=True)

    def _canon_pitcher(self, name: str) -> str:
        """
        投手名錯字／截斷校正：跟以前讀過的名字比對（同背號優先）。
        例：「89 J.史」→「89 J.史莫滋」（被截斷）、「97 賀錦能」→「97 賀錦龍」（錯字，以出現較多次的為準）。
        """
        import difflib
        name = re.sub(r"\s+", " ", (name or "").strip())
        if not name:
            return name
        if self._name_counts is None:
            try:
                self._name_counts = dict(self.store.pitcher_name_counts())
            except Exception:
                self._name_counts = {}
        counts = self._name_counts

        def split(n):
            m = re.match(r"^(\d{1,3})\s*(.*)$", n)
            return (m.group(1), m.group(2)) if m else ("", n)

        num, rest = split(name)
        best, best_key = None, None
        for cand, cnt in counts.items():
            if cand == name or cnt < 2:
                continue
            cnum, crest = split(cand)
            if num and cnum != num:
                continue
            if not rest or not crest:
                continue
            if crest.startswith(rest) and len(crest) > len(rest):
                score = 1.0  # 被截斷
            else:
                score = difflib.SequenceMatcher(None, rest, crest).ratio()
            if score < 0.6 or (len(rest) >= 3 and score < 0.66):
                continue
            key = (score, cnt)
            if best_key is None or key > best_key:
                best, best_key = cand, key
        out = name
        if best and counts.get(best, 0) >= counts.get(name, 0):
            out = best
        counts[out] = counts.get(out, 0) + 1
        return out

    def _read_env(self, img: np.ndarray) -> None:
        """讀天氣、風向、卡片天氣（純顏色判斷，很快）；風速數字有變才丟給 OCR。"""
        try:
            e = read_environment(img)
        except Exception:
            return
        for k in ("weather", "wind_dir", "pitcher_attr", "batter_attr"):
            if e.get(k):
                self.env[k] = e[k]
        if e.get("wind_angle") is not None:
            self.env["wind_angle"] = e["wind_angle"]
        crop = wind_speed_crop(img)
        if crop.size:
            small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (16, 8), interpolation=cv2.INTER_AREA)
            h = (small > small.mean()).tobytes()
            if h != self._wind_hash:
                if self._submit("wind", self._ocr_wind_speed, crop.copy()):
                    self._wind_hash = h

    def _ocr_wind_speed(self, crop: np.ndarray) -> Optional[int]:
        try:
            lines = self.ocr_engine.run_ocr_lines(crop, upscale=3.0)
        except Exception:
            return None
        m = re.search(r"(\d{1,2})", " ".join(lines or []))
        return int(m.group(1)) if m else None

    def _check_missed(self, cur: Tuple[int, int], ts: float, det_seen: bool) -> bool:
        """
        記分板的球數比目前多走了一顆（壞球或好球），但程式沒有記到那一球：
        持續 3 秒、期間沒有看到落點標記 → 補一筆「漏記」。觀察期間回傳 True（暫不同步球數）。
        """
        tb, ts_ = self.tracked_balls, self.tracked_strikes
        if tb is None or ts_ is None or self._pitches_this_game == 0:
            self._missed_cand = None
            return False
        kind = None
        if cur == (tb + 1, ts_) and tb + 1 <= 3:
            kind = "ball"
        elif cur == (tb, ts_ + 1) and ts_ + 1 <= 2:
            kind = "strike"
        allowed = self.record_both or self._row_for_logic() == self.my_batting_row
        if kind is None or not allowed:
            self._missed_cand = None
            return False
        if det_seen:
            # 看得到標記代表這球正在處理中，交給正常流程
            self._missed_cand = None
            return True
        if not self._missed_cand or self._missed_cand[0] != cur:
            self._missed_cand = (cur, ts)
            return True
        if ts - self._missed_cand[1] < 3.0:
            return True
        self._missed_cand = None
        side = self._other(self._row_for_logic())
        record = {
            "game_id": self.game_id, "ts": datetime.now().isoformat(timespec="seconds"),
            "inning": self.inning, "half": self.half, "outs": self.outs,
            "pitcher": self.pitcher or "未知投手",
            "opponent_tag": self._players_for(side, self.inning) or self.opponent_tag,
            "batter": self.batter or "未知打者",
            "balls_before": tb, "strikes_before": ts_,
            "zone": "0", "x_norm": 0.5, "y_norm": 0.5, "result": kind,
            "note": f"漏記：記分板球數 {tb}-{ts_} → {cur[0]}-{cur[1]}，落點不明",
        }
        try:
            pid = self.store.insert_pitch(record)
            self.audit["missed"] += 1
            self.corrections.append({"id": pid, "result": kind, "note": record["note"]})
            print(f"  [記分板審核] 補一筆漏記的{self._RESULT_NAME[kind]}（#{pid}）", flush=True)
        except Exception as e:
            print(f"  [補漏記失敗] {e}", flush=True)
        return False

    def _audit_fix(self, exp: Tuple[int, int], after: Tuple[int, int]) -> bool:
        """
        記分板逐球審核：上一球記完後，記分板的球數跟推算的不一樣時，看得出原因就自動改：
        1. 好壞球判反：記「好球」但記分板多一顆壞球（或反過來）→ 改結果。
        2. 漏記一球：記分板比推算多走了一球 → 補一筆「漏記」（落點不知道，不算進九宮格）。
        改不了的（例如 0-0 暫停 bug）回傳 False，交給原本的處理。
        """
        rec = self.last_recorded_pitch
        if not rec or rec.get("id") != self._last_recorded_pitch_id:
            return False
        b0, s0, r = rec.get("balls_before"), rec.get("strikes_before"), rec.get("result")
        if b0 is None or s0 is None or after == (0, 0):
            return False
        flip = None
        if r == "strike" and after == (b0 + 1, s0) and b0 + 1 <= 3:
            flip = "ball"
        elif r == "ball" and after == (b0, s0 + 1) and s0 + 1 <= 2:
            flip = "strike"
        if flip:
            note_add = f"記分板審核：{self._RESULT_NAME[r]}→{self._RESULT_NAME[flip]}（球數 {after[0]}-{after[1]}）"
            note = f"{rec.get('note')}; {note_add}" if rec.get("note") else note_add
            try:
                self.store.update_pitch(rec["id"], result=flip, note=note)
            except Exception as e:
                print(f"  [審核修正失敗] {e}", flush=True)
                return False
            rec["result"], rec["note"] = flip, note
            self.tracked_balls, self.tracked_strikes = after
            self.audit["fixed"] += 1
            self.corrections.append({"id": rec["id"], "result": flip, "note": note_add})
            print(f"  [記分板審核] #{rec['id']} {note_add}", flush=True)
            return True
        missed = None
        if after == (exp[0] + 1, exp[1]) and exp[0] + 1 <= 3:
            missed = "ball"
        elif after == (exp[0], exp[1] + 1) and exp[1] + 1 <= 2:
            missed = "strike"
        if missed:
            record = {
                "game_id": self.game_id, "ts": datetime.now().isoformat(timespec="seconds"),
                "inning": rec.get("inning", self.inning), "half": rec.get("half", self.half), "outs": self.outs,
                "pitcher": rec.get("pitcher", ""), "opponent_tag": rec.get("opponent_tag", ""), "batter": rec.get("batter", ""),
                "balls_before": exp[0], "strikes_before": exp[1],
                "zone": "0", "x_norm": 0.5, "y_norm": 0.5, "result": missed,
                "note": f"漏記：記分板球數 {exp[0]}-{exp[1]} → {after[0]}-{after[1]}，落點不明",
            }
            try:
                pid = self.store.insert_pitch(record)
            except Exception as e:
                print(f"  [補漏記失敗] {e}", flush=True)
                return False
            self.tracked_balls, self.tracked_strikes = after
            self.audit["missed"] += 1
            self.corrections.append({"id": pid, "result": missed, "note": record["note"]})
            print(f"  [記分板審核] 補一筆漏記的{self._RESULT_NAME[missed]}（#{pid}）", flush=True)
            return True
        return False

    def _resolve_hit(self, new_result: str, why: str) -> None:
        """確認暫記的界外／擊出；和當初記的不同就改資料庫，並記一筆修正。"""
        hc = self._hit_check
        self._hit_check = None
        if hc is None or new_result == hc["result"]:
            return
        name = self._RESULT_NAME
        note_add = f"自動修正：{name.get(hc['result'])}→{name.get(new_result)}（{why}）"
        note = f"{hc['note']}; {note_add}" if hc.get("note") else note_add
        try:
            self.store.update_pitch(hc["id"], result=new_result, note=note)
        except Exception as e:
            print(f"  [修正失敗] {e}", flush=True)
            return
        if self.last_recorded_pitch and self.last_recorded_pitch.get("id") == hc["id"]:
            self.last_recorded_pitch["result"] = new_result
            self.last_recorded_pitch["note"] = note
        self.corrections.append({"id": hc["id"], "result": new_result, "note": note_add})
        print(f"  [自動修正] #{hc['id']} {note_add}", flush=True)

    _BG_ORDER = ("newgame", "teams", "info", "inning", "wind", "tab")
    _BG_GAP = 0.4  # 兩件背景 OCR 之間至少空這麼久，讓主程式喘口氣

    def _submit(self, key: str, fn, *args, label: bool = False, **kwargs) -> bool:
        """
        背景執行一件 OCR 工作；同名工作還在跑就不重複送。
        標籤（球種球速）馬上跑；其他（ID、名字、局數…）先排隊，由 _dispatch_bg 一次放一件、投球時暫停。
        """
        f = self._jobs.get(key)
        if f is not None and not f.done():
            return False
        if label:
            self._jobs[key] = self._label_pool.submit(fn, *args, **kwargs)
            return True
        self._wants[key] = (fn, args, kwargs)  # 同一件排隊中就換成最新的畫面
        return True

    def _pitch_quiet(self, ts: float) -> bool:
        """投球前後（能量條剛出現、畫面上有落點、正在判定、標籤 OCR 在跑）先不跑別的 OCR。"""
        if self._pending is not None or self._last_det is not None:
            return True
        if any(k.startswith("label") and not f.done() for k, f in self._jobs.items()):
            return True
        if self._energy is not None and ts - self._energy[1] < 1.5:
            return True
        return False

    def _dispatch_bg(self, ts: float) -> None:
        running = any(not f.done() for k, f in self._jobs.items() if not k.startswith("label"))
        if running:
            self._bg_running = True
            return
        if self._bg_running:
            self._bg_running = False
            self._bg_free_since = ts
        if not self._wants or ts - self._bg_free_since < self._BG_GAP or self._pitch_quiet(ts):
            return
        order = list(self._BG_ORDER) + [k for k in self._wants if k not in self._BG_ORDER]
        for key in order:
            if key in self._wants:
                fn, args, kwargs = self._wants.pop(key)
                self._jobs[key] = self._pool.submit(fn, *args, **kwargs)
                self._bg_running = True
                if key == "teams" and self.row_status in ("searching", "not_found"):
                    self._row_tries += 1
                return

    def _poll_label_fill(self, ts: float) -> None:
        """先記錄的球：標籤 OCR 讀完後補上球種、球速。"""
        for job, (pid, t0) in list(self._label_fill.items()):
            done, res = self._take(job)
            if not done:
                if ts - t0 > 10.0:
                    self._label_fill.pop(job, None)
                continue
            self._label_fill.pop(job, None)
            if res is None or not (res.pitch_raw or res.speed_kmh):
                continue
            grp = res.pitch_group or classify_pitch_group(res.pitch_raw)
            try:
                self.store.update_pitch(pid, pitch_raw=res.pitch_raw, pitch_group=grp, speed_kmh=res.speed_kmh)
            except Exception as e:
                print(f"  [補球種失敗] {e}", flush=True)
                continue
            lr = self.last_recorded_pitch
            if lr and lr.get("id") == pid:
                lr["pitch_raw"], lr["pitch_group"], lr["speed_kmh"] = res.pitch_raw, grp, res.speed_kmh
            self.corrections.append({"id": pid, "result": None, "note": "補上球種球速"})
            print(f"  [補上] #{pid} 球種:{res.pitch_raw or '-'} 球速:{res.speed_kmh or '-'}", flush=True)

    def _take(self, key: str):
        """取回背景工作結果：(是否完成, 結果)。"""
        f = self._jobs.get(key)
        if f is None or not f.done():
            return False, None
        del self._jobs[key]
        try:
            return True, f.result()
        except Exception:
            return True, None

    def _busy(self, key: str) -> bool:
        f = self._jobs.get(key)
        return f is not None and not f.done()

    def _reject(self, reason: str) -> None:
        """記錄「看到標記但沒記錄」的原因，命令列每 5 秒會印出來，方便找出漏抓原因。"""
        self.reject_counts[reason] = self.reject_counts.get(reason, 0) + 1
        self._frame_reject = reason

    def _emit(self, kind: str, **info: Any) -> None:
        """可疑狀況 → 交給記錄器把前後畫面存下來（給之後分析、修正用）。"""
        self.events.append({"kind": kind, **info})

    def frame_debug(self) -> Dict[str, Any]:
        """這一幀的判斷過程（記錄器每幀存一行）。"""
        sb = self._last_sb
        d = self._last_det
        p = self._pending
        return {
            "sb": [sb.valid, sb.batting_row, sb.balls, sb.strikes, sb.outs] if sb else None,
            "cam": bool(self._last_batting_cam),
            "row": self._row_for_logic(),
            "det": [d.kind, d.cx, d.cy, d.zone, bool(d.has_label), d.label_shape] if d else None,
            "pend": [p.confirm_frames, p.missing_frames, p.fielding_cut_frames, p.label_checked, p.label_ok] if p else None,
            "rej": self._frame_reject,
            "cnt": [self.tracked_balls, self.tracked_strikes],
            "inn": [self.inning, self.half, self.outs],
            "pit": self.pitcher,
            "bat": self.batter,
            "over": self.game_over,
        }

    def set_record_both(self, on: bool) -> None:
        self.record_both = bool(on)
        self.config.record_both = self.record_both
        try:
            self.config.save()
        except Exception:
            pass

    def _players_for(self, side: str, inning: Optional[int] = None) -> str:
        """投球方的玩家 ID。2 對 2：玩家1 投單數局、玩家2 投雙數局。"""
        names = self.team_names.get(side, []) if side else []
        if len(names) >= 2 and inning:
            return names[(int(inning) - 1) % 2]
        return " / ".join(names)

    def _pkey(self, side: str, inning: Optional[int] = None) -> str:
        """投手記憶的鍵：2 對 2 時同一隊兩位玩家分開記（單／雙數局）。"""
        if len(self.team_names.get(side, [])) >= 2:
            return f"{side}{(int(inning or self.inning) - 1) % 2}"
        return side

    @staticmethod
    def _slot_name(side: str, idx: int, two: bool) -> str:
        team = "客隊" if side == "top" else "主隊"
        return f"{team}{idx + 1}" if two else team

    def player_slots(self) -> List[Dict[str, Any]]:
        two = len(self.team_names.get("top", [])) >= 2 or len(self.team_names.get("bottom", [])) >= 2
        out = []
        for side in ("top", "bottom"):
            names = self.team_names.get(side, [])
            for i in range(2 if two else 1):
                pid = names[i] if i < len(names) else ""
                out.append({
                    "slot": self._slot_name(side, i, two), "id": pid, "side": side,
                    "me": bool(pid and self.my_id and id_in_text(self.my_id, pid)),
                })
        return out

    @staticmethod
    def _same_id(a: str, b: str) -> bool:
        """兩個 ID 是否為同一人（容許文字辨識少量錯字）。"""
        import difflib
        na = re.sub(r"\s+", "", (a or "").lower())
        nb = re.sub(r"\s+", "", (b or "").lower())
        if not na or not nb:
            return False
        return na == nb or na in nb or nb in na or difflib.SequenceMatcher(None, na, nb).ratio() >= 0.6

    def _same_team(self, a: List[str], b: List[str]) -> bool:
        return bool(a) and bool(b) and len(a) == len(b) and all(self._same_id(x, y) for x, y in zip(a, b))

    def _update_team_names(self, names: Dict[str, str]) -> None:
        """
        從記分板（單人）或 HOME / AWAY 面板（2 對 2）整理雙方玩家 ID。
        已經有 ID 時：連續兩次讀到不一樣的人才換（避免文字辨識偶發錯字）；
        任何一隊的人全部換掉＝已經是新的一場（沒看到結算畫面也一樣）→ 自動開新場次。
        """
        top, bot = self._parse_team_names(names)
        cur_top, cur_bot = self.team_names.get("top", []), self.team_names.get("bottom", [])
        # 讀到的人數比現在少（面板被擋住、只讀到一半）不算換人
        new_top = top if top and len(top) >= len(cur_top) and not self._same_team(top, cur_top) else None
        new_bot = bot if bot and len(bot) >= len(cur_bot) and not self._same_team(bot, cur_bot) else None
        if new_top is None and new_bot is None:
            self._team_change_cand = None
            self._team_change_streak = 0
            # 第一次讀到、或補上原本缺的那一隊
            if top and not cur_top:
                self.team_names["top"] = top
            if bot and not cur_bot:
                self.team_names["bottom"] = bot
            return
        if not cur_top and not cur_bot:
            if top:
                self.team_names["top"] = top
            if bot:
                self.team_names["bottom"] = bot
            return
        cand = (tuple(new_top or []), tuple(new_bot or []))
        if self._team_change_cand == cand:
            self._team_change_streak += 1
        else:
            self._team_change_cand = cand
            self._team_change_streak = 1
        if self._team_change_streak < 2:
            return
        self._team_change_streak = 0
        self._team_change_cand = None
        def all_new(new, cur):
            return bool(new and cur) and not any(self._same_id(x, y) for x in new for y in cur)
        # 比賽中途不會換人：任何一隊的人全部不一樣＝已經是新的一場（沒看到結算畫面也一樣）
        team_replaced = all_new(new_top, cur_top) or all_new(new_bot, cur_bot)
        old = f"{' / '.join(cur_top)} vs {' / '.join(cur_bot)}"
        if team_replaced and self._pitches_this_game > 0:
            print(f"  [對手變了] 有一隊的玩家全部換人（原本 {old}），自動開新的一場", flush=True)
            self.start_new_game()
            if top:
                self.team_names["top"] = list(top)
            if bot:
                self.team_names["bottom"] = list(bot)
        if new_top:
            self.team_names["top"] = list(new_top)
        if new_bot:
            self.team_names["bottom"] = list(new_bot)
        now = f"{' / '.join(self.team_names['top'])} vs {' / '.join(self.team_names['bottom'])}"
        print(f"  [玩家 ID 更新] {now}", flush=True)
        self._emit("玩家ID更新", note=f"{old} → {now}")
        self.id_changed = True

    def _parse_team_names(self, names: Dict[str, str]):
        """回傳 (上行玩家列表, 下行玩家列表)。"""
        top_txt = (names.get("top") or "").strip()
        bot_txt = (names.get("bottom") or "").strip()
        if "TEAM" in (top_txt + bot_txt).upper():
            # 2 對 2：AWAY 隊是上行（先攻），HOME 隊是下行
            top = parse_panel_players(names.get("away", ""))
            bot = parse_panel_players(names.get("home", ""))
        else:
            top = [clean_player_id(top_txt)] if clean_player_id(top_txt) else []
            bot = [clean_player_id(bot_txt)] if clean_player_id(bot_txt) else []
        return top, bot

    def start_new_game(self) -> None:
        """開始新的一場：新場次編號、局數球數歸零、清掉進行中的投球，並重新判斷先後攻。"""
        self.game_id = datetime.now().strftime("game_%Y%m%d_%H%M%S")
        self.inning = 1
        self.half = "上"
        self._half_candidate = "上"
        self._half_streak = 0
        self.outs = 0
        self._outs_candidate = 0
        self._outs_streak = 0
        self.reset_count(known=True)
        self._pending = None
        self._latched_until_gone = False
        self._last_recorded_pitch_id = None
        self._last_recorded_expected_count = None
        self.last_recorded_pitch = None
        self._hit_check = None
        self.batter = ""
        self.pitcher = ""
        self.game_over = False
        self._newgame_streak = 0
        # 上一場結束：在背景重訓下一球預測 AI，讓新資料馬上派上用場
        if getattr(self, "_pitches_this_game", 0) > 0:
            threading.Thread(target=self._retrain_model, daemon=True).start()
        self.team_names = {"top": [], "bottom": []}
        self._team_change_cand = None
        self._team_change_streak = 0
        self._pitches_this_game = 0
        self.audit = {"checked": 0, "fixed": 0, "missed": 0}
        self.stable_batting_row = "unknown"
        self._row_cand = "unknown"
        self._row_cand_streak = 0
        self._pitcher_by_side = {}
        self._pitch_inning = 1
        self._force_names = True
        # 換一場可能換邊，有設定 ID 就重新自動判斷
        if self.my_id:
            self.set_row_auto()
        print(f"  [新的一場] {self.game_id}", flush=True)

    def _try_detect_row(self, client_bgr: np.ndarray) -> None:
        """送出背景工作讀雙方名字；結果在 _apply_team_names 處理。"""
        self._submit("teams", self.ocr_engine.read_team_names, client_bgr.copy(), self.config)

    def _apply_team_names(self, names: Optional[Dict[str, str]]) -> None:
        if not names:
            return
        self._update_team_names(names)
        if not (self.row_mode == "auto" and self.row_status in ("searching", "not_found") and self.my_id):
            return
        found = None
        if id_in_text(self.my_id, names.get("top", "")):
            found = ("top", "記分板上行")
        elif id_in_text(self.my_id, names.get("bottom", "")):
            found = ("bottom", "記分板下行")
        elif id_in_text(self.my_id, names.get("away", "")):
            # 2 對 2：AWAY 隊是記分板上行（先攻）
            found = ("top", "AWAY 面板")
        elif id_in_text(self.my_id, names.get("home", "")):
            found = ("bottom", "HOME 面板")
        if found:
            self.my_batting_row, self.row_source = found
            self.row_status = "found"
            print(f"  [自動] 找到 {self.my_id}（{self.row_source}）→ 我方{'先攻' if found[0] == 'top' else '後攻'}", flush=True)
        elif self._row_tries >= 15:
            self.row_status = "not_found"

    def set_opponent_tag(self, tag: str) -> None:
        self.opponent_tag = tag.strip()

    def reset_count(self, known: bool = True) -> None:
        self.tracked_balls = 0 if known else None
        self.tracked_strikes = 0 if known else None
        self._awaiting_post_count = False
        self._stable_sb_target = None
        self._stable_sb_count = 0
        self._live_sb_target = None
        self._live_sb_streak = 0

    def get_status(self, connected: bool = True) -> TrackerStatus:
        sb_my = self._row_for_logic() == self.my_batting_row
        row = self._row_for_logic()
        pside = "bottom" if row == "top" else ("top" if row == "bottom" else "")
        pslot = ""
        if pside:
            two = len(self.team_names.get(pside, [])) >= 2
            idx = (int(self.inning) - 1) % 2 if two else 0
            pslot = self._slot_name(pside, idx, two) + ("・單數局" if two and idx == 0 else ("・雙數局" if two else ""))
        rep = self.store.get_pitcher_repertoire(self.pitcher) if self.pitcher else []
        return TrackerStatus(
            connected=bool(connected),
            fps=round(float(self.fps), 1),
            game_id=str(self.game_id),
            my_batting_row=str(self.my_batting_row),
            opponent_tag=str(self.opponent_tag),
            inning=int(self.inning),
            half=str(self.half),
            outs=int(self.outs),
            balls=int(self.tracked_balls) if self.tracked_balls is not None else None,
            strikes=int(self.tracked_strikes) if self.tracked_strikes is not None else None,
            batter=str(self.batter),
            pitcher=str(self.pitcher),
            is_my_batting=bool(sb_my),
            cursor_active=bool(self._last_cursor_now),
            batting_camera=bool(self._last_batting_cam),
            last_pitch=self.last_recorded_pitch,
            pitcher_repertoire=list(rep),
            my_id=str(self.my_id),
            row_mode=str(self.row_mode),
            row_status=str(self.row_status),
            row_source=str(self.row_source),
            game_over=bool(self.game_over),
            record_both=bool(self.record_both),
            capture_mode=str(self.capture_mode),
            team_top=" / ".join(self.team_names.get("top", [])),
            team_bottom=" / ".join(self.team_names.get("bottom", [])),
            pitching_players=self._players_for(pside, self.inning),
            pitching_slot=pslot,
            players=self.player_slots(),
            audit=dict(self.audit),
            env=dict(self.env),
            cpu_pct=getattr(self, "cpu_pct", 0.0),
        )

    @staticmethod
    def _expected_count_after(
        balls_before: Optional[int], strikes_before: Optional[int], result: str
    ) -> Optional[Tuple[int, int]]:
        if balls_before is None or strikes_before is None:
            return None
        if result == "ball":
            if balls_before + 1 >= 4:
                return (0, 0)
            return (balls_before + 1, strikes_before)
        if result == "strike":
            if strikes_before + 1 >= 3:
                return (0, 0)
            return (balls_before, strikes_before + 1)
        if result == "foul":
            return (balls_before, min(2, strikes_before + 1))
        if result == "in_play":
            return (0, 0)
        return None

    def process_frame(
        self, client_bgr: np.ndarray, ts_sec: float
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single client-area BGR frame at timestamp `ts_sec` (in seconds).
        Returns the newly recorded pitch dict if a pitch was finalized on this frame, else None.
        """
        self._frame_idx += 1
        self._frame_reject = ""
        self._dispatch_bg(ts_sec)
        if self._label_fill:
            self._poll_label_fill(ts_sec)

        # 1. Read scoreboard (fast pixel patch sampling)
        sb = read_scoreboard(client_bgr, self.config, current_inning=self.inning)
        self._last_sb = sb

        batting_cam = is_batting_camera(client_bgr, self.config)
        # 鏡頭真的離開打擊畫面 0.5 秒以上才算「切回來」；
        # 擊球瞬間的閃光會讓打擊畫面判斷短暫失效一兩幀，不能當成換鏡頭
        if batting_cam:
            self._last_cam_true_ts = ts_sec
        # 結果標籤或 ◎ 剛好蓋住好球帶框線時，「打擊畫面」判斷會失效；
        # 1 秒內還是打擊畫面，就當作仍在打擊畫面
        cam_ok = batting_cam or (ts_sec - self._last_cam_true_ts) < 1.0
        if not batting_cam and self._last_batting_cam:
            self._cam_off_start = ts_sec
        if batting_cam and not self._last_batting_cam and (ts_sec - self._cam_off_start) >= 0.5:
            self._cam_on_since = ts_sec
        self._last_batting_cam = batting_cam
        # 暫記中的擊球：累計鏡頭離開打擊畫面多久（擊出去通常 4 秒以上，界外 3 秒內）
        dt = 0.0 if self._prev_frame_ts is None else max(0.0, min(3.0, ts_sec - self._prev_frame_ts))
        self._prev_frame_ts = ts_sec
        if self._hit_check is not None and not batting_cam:
            self._hit_check["off"] += dt

        # 0. 比賽結束／新比賽
        #   看到結算畫面（右上藍色「確認BOX」按鈕）→ 停止記錄；
        #   之後要同時看到「1 局上」且「比分 0:0」連續兩次，才自動開新的一場。
        if not self.game_over:
            if not sb.valid and not batting_cam and self._frame_idx % 15 == 7:
                if has_confirm_box_button(client_bgr):
                    self.game_over = True
                    self._pending = None
                    self._latched_until_gone = False
                    self._awaiting_post_count = False
                    self._newgame_streak = 0
                    print("  [比賽結束] 偵測到結算畫面（確認BOX），停止記錄，等待新比賽", flush=True)
        else:
            self._last_det = None
            done, res = self._take("newgame")
            if sb.valid and self._frame_idx % 15 == 7 and not self._busy("newgame"):
                cb = client_bgr.copy()
                self._submit(
                    "newgame",
                    lambda img: (
                        self.ocr_engine.read_inning_number(img, self.config, force=True),
                        self.ocr_engine.read_scores(img, self.config),
                    ),
                    cb,
                )
            if done and res is not None:
                inn, (top_score, bot_score) = res
                if inn == 1 and sb.half == "上" and top_score == 0 and bot_score == 0:
                    self._newgame_streak += 1
                    if self._newgame_streak >= 2:
                        self.start_new_game()
                        self.half = sb.half
                        self.active_batting_row = sb.batting_row
                        return None
                else:
                    self._newgame_streak = 0
            return None

        if sb.valid:
            self._invalid_sb_streak = 0
            self.active_batting_row = sb.batting_row
            if batting_cam and sb.batting_row in ("top", "bottom"):
                if sb.batting_row == self._row_cand:
                    self._row_cand_streak += 1
                else:
                    self._row_cand = sb.batting_row
                    self._row_cand_streak = 1
                if self._row_cand_streak >= 10 and self.stable_batting_row != sb.batting_row:
                    self._on_side_change(sb.batting_row)
            # 讀雙方玩家 ID：還沒讀到時約每 5 秒一次；讀到之後約每 15 秒再核對一次（換人、換場才抓得到）；攻守交換時馬上讀
            missing = not self.team_names["top"] or not self.team_names["bottom"]
            if (
                not (self.row_mode == "auto" and self.row_status == "searching")
                and (
                    self._force_teams
                    or (missing and self._frame_idx % 150 == 20)
                    or (self._frame_idx % 450 == 20)
                )
            ):
                if self._submit("teams", self.ocr_engine.read_team_names, client_bgr.copy(), self.config):
                    self._force_teams = False
            # 依 ID 自動判斷先攻／後攻：還沒找到時約每秒試一次
            if (
                self.row_mode == "auto"
                and self.row_status in ("searching", "not_found")
                and self.my_id
                and self._frame_idx % 30 == 5
                and (self.row_status == "searching" or self._frame_idx % 300 == 5)
            ):
                self._try_detect_row(client_bgr)
            # Debounce half-inning transitions (require 10 consecutive frames ~0.33s)
            if sb.half != self.half:
                if sb.half == self._half_candidate:
                    self._half_streak += 1
                else:
                    self._half_candidate = sb.half
                    self._half_streak = 1

                if self._half_streak >= 10:
                    self._resolve_hit("in_play", "攻守交換")
                    if self.half == "下" and sb.half == "上":
                        self.inning += 1
                    self.half = sb.half
                    self._half_streak = 0
                    self.outs = sb.outs
                    self._outs_candidate = sb.outs
                    self._outs_streak = 0
                    self._last_cursor_ts = -999.0
                    self.reset_count(known=True)
                    self._submit("inning", self.ocr_engine.read_inning_number, client_bgr.copy(), self.config, force=True)
            else:
                self._half_candidate = self.half
                self._half_streak = 0
                if sb.outs != self.outs:
                    if sb.outs == self._outs_candidate:
                        self._outs_streak += 1
                    else:
                        self._outs_candidate = sb.outs
                        self._outs_streak = 1
                    if self._outs_streak >= 2:
                        if sb.outs > self.outs:
                            self._resolve_hit("in_play", "出局數增加")
                        self.outs = sb.outs
                        self._outs_streak = 0
                        self.reset_count(known=True)
                else:
                    self._outs_candidate = self.outs
                    self._outs_streak = 0

            # Real-time Ball / Strike count sync whenever scoreboard is stable for >= 2 frames
            if batting_cam and not self._awaiting_post_count and self._pending is None:
                cur_sb = (sb.balls, sb.strikes)
                if cur_sb == self._live_sb_target:
                    self._live_sb_streak += 1
                else:
                    self._live_sb_target = cur_sb
                    self._live_sb_streak = 1
                if self._live_sb_streak >= 2:
                    # Guard against BBO2 0-0 pause/TAB bug mid-at-bat: only ignore sudden 0-0 if we already had >0 balls/strikes
                    if cur_sb == (0, 0) and (
                        (self.tracked_balls or 0) > 0 or (self.tracked_strikes or 0) > 0
                    ):
                        pass
                    elif self._check_missed(cur_sb, ts_sec, det_seen=self._last_det is not None):
                        pass  # 還在觀察是不是漏記，先不同步
                    else:
                        self.tracked_balls, self.tracked_strikes = cur_sb
        else:
            self._invalid_sb_streak += 1
            if self._invalid_sb_streak >= 15:
                self.active_batting_row = "unknown"

        # 2. Detect stationary pitch marker (◎ bullseye or ⊗ red_x)
        # 輕量化：不是打擊畫面（超過 1 秒）又沒有正在處理的球時，不找落點標記
        if cam_ok or self._pending is not None:
            det = detect_pitch_marker(client_bgr, self.config)
        else:
            det = None
        if det is None and sb.valid and cam_ok:
            # 低球補救：落點標記被結果標籤蓋住時，改用標籤位置推回落點
            det = detect_label_only(client_bgr, self.config)
        self._last_det = det

        # 3. Check cyan batting cursor when no pitch marker is currently covering the zone
        cursor_now = False
        if det is None and batting_cam:
            cursor_now = has_batting_cursor(client_bgr, self.config)
            if cursor_now:
                self._last_cursor_ts = ts_sec
                self._cursor_streak += 1
            else:
                self._cursor_streak = 0
        else:
            self._cursor_streak = 0
        self._last_cursor_now = cursor_now

        # 4. 背景 OCR：右下角打者／投手、局數、TAB 面板、雙方名字
        #    全部在背景執行緒跑，這裡只負責「送出工作」和「套用已完成的結果」。
        done, names = self._take("teams")
        if done:
            self._apply_team_names(names)

        done, res = self._take("info")
        if done and res:
            new_batter, new_pitcher = res
            if new_pitcher:
                new_pitcher = self._canon_pitcher(new_pitcher)
            # 剛攻守交換時右下角有時還是上半局的名字：讀到的投手是「正在打擊那一隊」的投手 → 舊畫面，不採用
            bat_row = self.stable_batting_row
            if (
                new_pitcher
                and bat_row in ("top", "bottom")
                and new_pitcher in (self._pitcher_by_side.get(bat_row), self._pitcher_by_side.get(bat_row + "0"), self._pitcher_by_side.get(bat_row + "1"))
                and new_pitcher != self._pitcher_by_side.get(self._pkey(self._other(bat_row)))
            ):
                new_batter, new_pitcher = None, None
            if self._force_names and new_pitcher:
                self._force_names = False
                self._info_fail = 0
                self._info_next_ts = 0.0
            elif self._force_names and not new_pitcher:
                # 投手名字一直讀不到（例如名字有特殊小字）：放慢重讀，連續 3 次失敗後改成「畫面有變才讀」
                self._info_fail += 1
                if self._info_fail >= 3:
                    self._force_names = False
                else:
                    self._info_next_ts = ts_sec + (1.0, 2.0)[self._info_fail - 1]
            if new_batter and new_batter != self.batter:
                if self.batter:
                    # Batter changed -> reset count to 0-0
                    self.reset_count(known=True)
                self.batter = new_batter
            if new_pitcher and new_pitcher != self.pitcher:
                self.pitcher = new_pitcher
            if self.pitcher and self.stable_batting_row in ("top", "bottom"):
                self._pitcher_by_side[self._pkey(self._other(self.stable_batting_row))] = self.pitcher

        done, ocr_in = self._take("inning")
        if done and ocr_in is not None:
            self.inning = ocr_in

        self._take("tab")

        # 只在「打擊畫面＋記分板正常」時讀：守備畫面、過場時右下角是別的東西，
        # 以前會對著紅土或小地圖一直跑 OCR，每次卡住好幾秒
        if sb.valid and batting_cam and self._frame_idx % 15 == 7:
            self._read_env(client_bgr)
        # 投手頭上的能量條只在投球前短暫出現：打擊畫面、還沒投出時每 3 張看一次
        if batting_cam and self._pending is None and self._frame_idx % 3 == 0:
            en = read_energy(client_bgr)
            if en is not None:
                self._energy = (en, ts_sec)
                self.env["energy"] = en
        done, ws = self._take("wind")
        if done and ws is not None:
            self.env["wind_speed"] = ws

        if sb.valid and batting_cam and self._frame_idx % 8 == 1 and ts_sec >= self._info_next_ts:
            cb = client_bgr.copy()
            self._submit("info", self.ocr_engine.read_info_panel, cb, self.config, force=self._force_names)
            self._submit("inning", self.ocr_engine.read_inning_number, cb, self.config)
            self._submit("tab", self.ocr_engine.read_tab_panel_if_visible, cb, self.config, store=self.store)

        # 5. Handle post-finalization marker latch (wait until marker actually disappears)
        if self._latched_until_gone:
            if det is not None:
                self._latched_missing_frames = 0
                self._last_marker_gone_ts = ts_sec
                return None
            self._latched_missing_frames += 1
            if self._latched_missing_frames >= 3:
                self._latched_until_gone = False
                self._last_marker_gone_ts = ts_sec
            else:
                return None

        # 6. Stage 5: Post-pitch 5-consecutive-frame S/B light stabilization
        if (
            self._awaiting_post_count
            and sb.valid
            and batting_cam
            and self._pending is None
            and not self._latched_until_gone
        ):
            cur_sb_pair = (sb.balls, sb.strikes)
            if cur_sb_pair == self._stable_sb_target:
                self._stable_sb_count += 1
            else:
                self._stable_sb_target = cur_sb_pair
                self._stable_sb_count = 1

            hc = self._hit_check
            if self._stable_sb_count >= 5 and hc is not None and hc["id"] == self._last_recorded_pitch_id:
                # 沒標籤的球：看記分板決定是界外還是擊出
                foul_exp = hc["foul_exp"]
                if foul_exp is not None and cur_sb_pair == foul_exp and foul_exp != (0, 0):
                    self._resolve_hit("foul", f"球數 {cur_sb_pair[0]}-{cur_sb_pair[1]}")
                    self.tracked_balls, self.tracked_strikes = cur_sb_pair
                elif cur_sb_pair == (0, 0) and (hc["off"] >= 2.5 or hc["result"] == "in_play"):
                    self._resolve_hit("in_play", "球數歸零")
                    self.tracked_balls, self.tracked_strikes = 0, 0
                else:
                    self._hit_check = None
                    if cur_sb_pair != (0, 0):
                        self.tracked_balls, self.tracked_strikes = cur_sb_pair
                self._awaiting_post_count = False
                self._stable_sb_target = None
                self._stable_sb_count = 0
            elif self._stable_sb_count >= 5:
                balls_after, strikes_after = cur_sb_pair
                exp = self._last_recorded_expected_count
                self.audit["checked"] += 1
                if exp is not None and (balls_after, strikes_after) != exp and self._audit_fix(exp, (balls_after, strikes_after)):
                    pass  # 記分板審核已自動改正（好壞球判反或漏記一球）
                elif exp is not None and (balls_after, strikes_after) != exp:
                    # BBO2 0-0 pause bug or discrepancy detected
                    if balls_after == 0 and strikes_after == 0 and exp != (0, 0):
                        # Scoreboard was reset by pause/TAB bug -> keep expected count
                        self.tracked_balls, self.tracked_strikes = exp
                    else:
                        self.tracked_balls, self.tracked_strikes = balls_after, strikes_after
                    if self._last_recorded_pitch_id is not None:
                        note_add = f"球數燈號({balls_after}-{strikes_after})與推算({exp[0]}-{exp[1]})不一致"
                        self._emit("球數不一致", pitch_id=self._last_recorded_pitch_id, note=note_add)
                        combined = (
                            f"{self._last_recorded_note}; {note_add}"
                            if self._last_recorded_note
                            else note_add
                        )
                        self.store.update_pitch(
                            self._last_recorded_pitch_id, note=combined
                        )
                        if (
                            self.last_recorded_pitch
                            and self.last_recorded_pitch.get("id") == self._last_recorded_pitch_id
                        ):
                            self.last_recorded_pitch["note"] = combined
                else:
                    self.tracked_balls, self.tracked_strikes = balls_after, strikes_after

                self._awaiting_post_count = False
                self._stable_sb_target = None
                self._stable_sb_count = 0

        # 7. Update or finalize PendingPitch
        if self._pending is not None:
            p = self._pending
            if det is not None and not p.marker_gone:
                dist = float(np.hypot(det.cx - p.best_detection.cx, det.cy - p.best_detection.cy))
                step = float(np.hypot(det.cx - p.last_cx, det.cy - p.last_cy))
                # 標記還在移動（飛行中的藍圈被球蓋住時可能被誤認成 ◎）→ 以新位置重新起算
                move_tol = max(4.0, 6.0 * client_bgr.shape[0] / 1080.0)
                if det.kind == p.best_detection.kind and step > move_tol and dist <= 120.0:
                    p.first_seen_ts = ts_sec
                    p.confirm_frames = 1
                    p.missing_frames = 0
                    p.best_detection = det
                    p.best_frame_bgr = client_bgr.copy()
                    p.last_cx, p.last_cy = det.cx, det.cy
                    p.label_checked = False
                    p.label_ok = False
                    p.ocr_res = None
                elif dist <= 32.0 and det.kind == p.best_detection.kind:
                    p.last_cx, p.last_cy = det.cx, det.cy
                    p.last_seen_ts = ts_sec
                    p.confirm_frames += 1
                    p.missing_frames = 0
                    # Prefer detection with a visible label, or higher marker score
                    if (det.has_label and not p.best_detection.has_label) or (
                        det.has_label == p.best_detection.has_label
                        and det.score > p.best_detection.score
                    ):
                        p.best_detection = det
                        p.best_frame_bgr = client_bgr.copy()
                else:
                    p.missing_frames += 1
                    if p.missing_frames >= 3:
                        p.marker_gone = True
            else:
                if not p.marker_gone:
                    p.missing_frames += 1
                    if p.missing_frames >= 3:
                        p.marker_gone = True

            if not batting_cam:
                p.fielding_cut_frames += 1
            p.post_marker_frames += 1

            if (
                p.confirm_frames < self._min_confirm_frames
                and p.marker_gone
                and p.best_detection.kind == "bullseye"
                and p.fielding_cut_frames >= 2
            ):
                # 擊球後鏡頭馬上切到守備，◎ 只來得及看到一幀 → 仍記成擊出
                recorded = self._finalize_pending_pitch(p, ts_sec)
                self._pending = None
                self._last_marker_gone_ts = ts_sec
                return recorded
            if p.confirm_frames < self._min_confirm_frames and p.marker_gone:
                # Single-frame noise blip -> discard without triggering cooldown
                self._reject("只出現一幀")
                self._pending = None
                return None

            elapsed = ts_sec - p.first_seen_ts

            # 標籤 OCR（背景執行）：⊗ 一定有 BALL 標籤；◎ 旁邊看起來有標籤時，
            # 要確認真的讀到 STRIKE / BALL（暗色球場會把「擊出」的 ◎ 背景誤認成標籤）。
            bd = p.best_detection
            needs_label = bd.label_crop is not None and (bd.kind == "red_x" or bd.has_label)
            if needs_label and not p.label_checked:
                if p.label_job is None and (p.confirm_frames >= 2 or p.marker_gone):
                    self._job_seq += 1
                    p.label_job = f"label{self._job_seq}"
                    p.label_submit_ts = ts_sec
                    self._submit(
                        p.label_job, self.ocr_engine.recognize_pitch_label,
                        bd.label_crop, bd.pitch_text_crop,
                        store=self.store, pitcher=p.pitcher or self.pitcher, label=True,
                    )
                if p.label_job is not None:
                    done, res = self._take(p.label_job)
                    if done:
                        p.label_checked = True
                        p.ocr_res = res
                        p.label_ok = bool(res is not None and res.label_word in ("STRIKE", "BALL"))
                    elif ts_sec - p.label_submit_ts > 6.0:
                        p.label_checked = True
                        p.label_ok = False

            # 先記錄、晚點補球種球速（2026-11-10i）：
            #   ⊗ 一定是壞球 → 不等 OCR；◎ 旁邊的標籤用圖形比對認出「STRIKE」字樣就算好球 → 也不等 OCR。
            #   OCR 讀完後由 _poll_label_fill 補進同一筆。圖形比對認不出來時才照舊等 OCR 確認。
            if bd.kind == "bullseye" and bd.has_label and not p.label_checked and p.tpl_word is None:
                tw = find_label_text(client_bgr)
                if tw is not None:
                    p.tpl_word = tw[0]
            if bd.kind == "red_x":
                has_lbl_or_red = True
            elif bd.has_label and not p.label_checked:
                if p.tpl_word is None:
                    return None  # 等背景 OCR 確認是不是 STRIKE 標籤
                has_lbl_or_red = True
            else:
                has_lbl_or_red = p.label_ok

            should_finalize = False
            if has_lbl_or_red and (p.confirm_frames >= 3 or p.marker_gone):
                should_finalize = True
            elif not has_lbl_or_red:
                if p.fielding_cut_frames >= 3 and p.confirm_frames >= self._min_confirm_frames:
                    should_finalize = True
                elif (elapsed >= 0.75 or p.marker_gone) and p.confirm_frames >= self._min_confirm_frames:
                    should_finalize = True

            if should_finalize:
                recorded = self._finalize_pending_pitch(p, ts_sec)
                self._pending = None
                if not p.marker_gone:
                    self._latched_until_gone = True
                    self._latched_missing_frames = 0
                self._last_marker_gone_ts = ts_sec
                return recorded

            return None

        # 8. If no pending pitch, check if a new marker just appeared
        prev_det_present = self._prev_det_present
        self._prev_det_present = det is not None
        if det is None:
            self._replay_block_until_gone = False
        if det is not None:
            # Must be on the active batting/pitching camera with a visible scoreboard
            if not sb.valid or not cam_ok:
                self._reject("非打擊畫面")
                return None

            # 重播防護 1：鏡頭剛切回打擊畫面（不到 0.6 秒）時標記就已經在畫面上，
            # 代表是遊戲重新顯示上一顆球，不是新的投球 → 等這個標記消失前都不記
            if self._replay_block_until_gone:
                self._reject("重播")
                return None
            if (ts_sec - self._cam_on_since) < 0.6 or (prev_det_present and self._pending is None and (ts_sec - self._cam_on_since) < 1.5):
                self._replay_block_until_gone = True
                self._reject("重播")
                return None

            # 重播防護 2：和上一顆記錄的球位置、種類完全相同（30 秒內）→ 視為重播
            if self._last_rec_pos is not None:
                lx, ly, lk, lts = self._last_rec_pos
                # 只在鏡頭剛切回來的 3 秒內才算重播（電腦常常連續投同一個角落，位置會一模一樣）
                if (
                    det.kind == lk
                    and abs(det.cx - lx) <= 3
                    and abs(det.cy - ly) <= 3
                    and (ts_sec - lts) < 30.0
                    and (ts_sec - self._cam_on_since) < 3.0
                ):
                    self._replay_block_until_gone = True
                    return None

            # Check cooldown since last confirmed marker disappeared
            if (ts_sec - self._last_marker_gone_ts) < self._cooldown_sec:
                self._reject("冷卻")
                return None

            # Step 1 of Decision Tree: Is our team batting?
            if self._row_for_logic() not in ("top", "bottom"):
                return None
            if not self.record_both:
                # 自動模式還沒確定先攻／後攻時先不記錄，避免把我方投球誤記進來
                if self.row_mode == "auto" and self.row_status != "found":
                    self._reject("先後攻未定")
                    return None
                is_my_row = self._row_for_logic() == self.my_batting_row
                cursor_recent = (ts_sec - self._last_cursor_ts) <= self._cursor_memory_sec
                if not is_my_row:
                    self._reject("非我方打擊")
                    return None
                if self.require_cursor and not cursor_recent:
                    return None

            # （以前這裡會在球剛出現時讀名字，正好跟標籤 OCR 搶 CPU；名字改由一般排程在投球後讀）

            self._pending = PendingPitch(
                pitching_side=self._other(self._row_for_logic()),
                first_seen_ts=ts_sec,
                last_seen_ts=ts_sec,
                confirm_frames=1,
                best_detection=det,
                best_frame_bgr=client_bgr.copy(),
                inning=self.inning,
                half=self.half,
                outs_before=self.outs,
                batter=self.batter,
                pitcher=self.pitcher,
                balls_before=self.tracked_balls,
                strikes_before=self.tracked_strikes,
                last_cx=det.cx,
                last_cy=det.cy,
            )

        return None

    def flush_pending(self, ts_sec: float = 0.0) -> Optional[Dict[str, Any]]:
        """Finalize any pending pitch at the end of a video/session if confirmed."""
        if self._pending is not None and self._pending.confirm_frames >= self._min_confirm_frames:
            p = self._pending
            self._pending = None
            return self._finalize_pending_pitch(p, ts_sec or p.last_seen_ts)
        self._pending = None
        return None

    def _finalize_pending_pitch(
        self, p: PendingPitch, ts_sec: float
    ) -> Dict[str, Any]:
        det = p.best_detection
        notes: List[str] = []
        pitch_raw: Optional[str] = None
        pitch_group: Optional[str] = None
        speed_kmh: Optional[float] = None

        # Decision Tree Steps 2, 3, 4
        if det.kind == "red_x":
            result = "ball"
            ocr_res = p.ocr_res
            if ocr_res is not None:
                pitch_raw = ocr_res.pitch_raw
                pitch_group = ocr_res.pitch_group
                speed_kmh = ocr_res.speed_kmh
                if ocr_res.label_word == "STRIKE":
                    result = "strike"
                    notes.append("顏色與標籤不一致(紅⊗但STRIKE)")
        else:
            # det.kind == "bullseye" (◎)
            if p.tpl_word and not p.label_checked:
                # 圖形比對已經認出標籤字樣，球種球速等 OCR 讀完再補
                result = "ball" if p.tpl_word == "BALL" else "strike"
                if p.tpl_word == "BALL":
                    notes.append("顏色與標籤不一致(藍◎但BALL)")
            elif p.label_checked and p.label_ok and p.ocr_res is not None:
                result = "strike"
                ocr_res = p.ocr_res
                pitch_raw = ocr_res.pitch_raw
                pitch_group = ocr_res.pitch_group
                speed_kmh = ocr_res.speed_kmh
                if ocr_res.label_word == "BALL":
                    result = "ball"
                    notes.append("顏色與標籤不一致(藍◎但BALL)")
            else:
                # No label -> distinguish in_play (擊出) vs foul (界外)
                if (
                    p.fielding_cut_frames >= 2
                    or self.outs != p.outs_before
                    or self.half != p.half
                    or (self.batter and p.batter and self.batter != p.batter)
                ):
                    result = "in_play"
                else:
                    result = "foul"

        # Save snapshot image around strike zone + label for review/re-training
        h, w = p.best_frame_bgr.shape[:2]
        sx1 = max(0, int(w * 0.36))
        sy1 = max(0, int(h * 0.56))
        sx2 = min(w, int(w * 0.68))
        sy2 = min(h, int(h * 0.94))
        snap_crop = p.best_frame_bgr[sy1:sy2, sx1:sx2]
        snap_name = f"{self.game_id}_{int(ts_sec * 1000):08d}_{det.zone}_{result}.jpg"
        snap_path = self.snapshots_dir / snap_name
        imwrite_unicode(snap_path, snap_crop)

        note_str = "; ".join(notes)
        record = {
            "game_id": self.game_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "inning": p.inning,
            "half": p.half,
            "outs": p.outs_before,
            "pitcher": p.pitcher or self.pitcher or "未知投手",
            # 投球方玩家 ID（單人：對方 ID；2 對 2：對方兩位玩家）；讀不到時用手動標籤
            "opponent_tag": self._players_for(p.pitching_side, p.inning) or self.opponent_tag,
            "batter": p.batter or self.batter or "未知打者",
            "balls_before": p.balls_before,
            "strikes_before": p.strikes_before,
            "zone": det.zone,
            "x_norm": det.x_norm,
            "y_norm": det.y_norm,
            "result": result,
            "pitch_raw": pitch_raw,
            "pitch_group": pitch_group or classify_pitch_group(pitch_raw),
            "speed_kmh": speed_kmh,
            "note": note_str,
            "snapshot": f"/snapshots/{snap_name}",
            **{k: self.env.get(k) for k in ("weather", "wind_dir", "wind_angle", "wind_speed", "pitcher_attr", "batter_attr")},
            # 能量：這球投出前 10 秒內讀到的才算（每一球前能量條都會重新出現）
            "energy": self._energy[0] if self._energy and (ts_sec - self._energy[1]) < 10.0 else None,
        }
        pitch_id = self.store.insert_pitch(record)
        self._pitches_this_game += 1
        try:
            self.store.score_prediction(pitch_id)  # 自動對答案：用這球之前的資料預測，再跟實際落點比
        except Exception as e:
            print(f"  [預測計分失敗] {e}", flush=True)
        if not (p.pitcher or self.pitcher):
            self._emit("投手未知", pitch_id=pitch_id)
        if not record.get("opponent_tag"):
            self._emit("玩家未知", pitch_id=pitch_id)
        record["id"] = pitch_id
        self.last_recorded_pitch = record
        if p.label_job is not None and not p.label_checked:
            # 標籤 OCR 還在跑：讀完再補上球種、球速
            self._label_fill[p.label_job] = (pitch_id, ts_sec)
        self._last_rec_pos = (int(det.cx), int(det.cy), det.kind, ts_sec)

        # Stage 5: Update count tracking state
        exp = self._expected_count_after(p.balls_before, p.strikes_before, result)
        if result in ("foul", "in_play"):
            # 界外／擊出單看畫面分不準 → 先記下，等記分板球數確認（見步驟 6）
            foul_exp = self._expected_count_after(p.balls_before, p.strikes_before, "foul")
            self._hit_check = {
                "id": pitch_id, "result": result, "foul_exp": foul_exp,
                "off": float(p.fielding_cut_frames) * 0.1, "note": note_str,
            }
            if result == "in_play":
                self.tracked_balls, self.tracked_strikes = 0, 0
            elif foul_exp is not None:
                self.tracked_balls, self.tracked_strikes = foul_exp
            self._awaiting_post_count = True
            self._last_recorded_pitch_id = pitch_id
            self._last_recorded_expected_count = foul_exp
            self._last_recorded_note = note_str
            self._stable_sb_target = None
            self._stable_sb_count = 0
        elif exp == (0, 0):
            self._hit_check = None
            self.reset_count(known=True)
        else:
            self._hit_check = None
            if exp is not None:
                self.tracked_balls, self.tracked_strikes = exp
            self._awaiting_post_count = True
            self._last_recorded_pitch_id = pitch_id
            self._last_recorded_expected_count = exp
            self._last_recorded_note = note_str
            self._stable_sb_target = None
            self._stable_sb_count = 0

        if self.on_pitch_recorded is not None:
            self.on_pitch_recorded(record)

        return record
