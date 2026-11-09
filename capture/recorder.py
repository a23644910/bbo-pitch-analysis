"""
飛行記錄器：隨時保留最近 30 秒「程式實際看到的畫面」與每一幀的判斷過程。

- 發生可疑狀況（球數對不上、看到標記卻沒記錄、投手或玩家未知、攻守交換）時，
  自動把事件前 8 秒、後 3 秒存成一個資料夾。
- 使用者在網頁按「回報問題」時，存下前 30 秒＋後 3 秒，並附上選的原因與備註。
- 另外每場比賽把每一幀的判斷過程（純文字，沒有圖片）寫進 sessions\\<場次>.jsonl。

存檔位置：data\\feedback\\<時間>_<類型>\\
  frames\\000123.jpg   畫面（1280 寬，JPEG）
  timeline.jsonl      每幀一行：時間、畫面檔名、判斷過程
  info.json           類型、原因、備註、相關的那一球、程式版本
"""

from __future__ import annotations
import json
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

FEEDBACK_DIR = Path(__file__).resolve().parent.parent / "data" / "feedback"


class FlightRecorder:
    def __init__(
        self,
        keep_sec: float = 30.0,
        fps: float = 10.0,
        width: int = 1280,
        max_total_mb: int = 3000,
        enabled: bool = True,
    ) -> None:
        self.keep_sec = keep_sec
        self.interval = 1.0 / fps
        self.width = width
        self.max_total_mb = max_total_mb
        self.enabled = enabled
        self._buf: Deque[Tuple[float, bytes, Dict[str, Any]]] = deque()
        self._last_ts = -999.0
        self._lock = threading.Lock()
        self._pending_dumps: List[Dict[str, Any]] = []
        self._session_file = None
        self._session_id = ""
        self._seen_unrecorded = 0
        self._last_auto: Dict[str, float] = {}
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        (FEEDBACK_DIR / "sessions").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def add_frame(self, client_bgr: np.ndarray, ts: float, state: Dict[str, Any], game_id: str) -> None:
        if not self.enabled:
            return
        self._write_session(ts, state, game_id)
        if ts - self._last_ts < self.interval:
            self._flush_due(ts)
            return
        self._last_ts = ts
        h, w = client_bgr.shape[:2]
        small = cv2.resize(client_bgr, (self.width, int(round(h * self.width / w))), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._lock:
            self._buf.append((ts, jpg.tobytes(), state))
            while self._buf and ts - self._buf[0][0] > self.keep_sec + 5:
                self._buf.popleft()
        self._auto_detect(ts, state)
        self._flush_due(ts)

    def _write_session(self, ts: float, state: Dict[str, Any], game_id: str) -> None:
        try:
            if game_id != self._session_id:
                if self._session_file:
                    self._session_file.close()
                self._session_id = game_id
                self._session_file = open(FEEDBACK_DIR / "sessions" / f"{game_id}.jsonl", "a", encoding="utf-8")
            self._session_file.write(json.dumps({"t": round(ts, 2), **state}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _auto_detect(self, ts: float, state: Dict[str, Any]) -> None:
        """看到標記卻一直沒記錄（不是因為「非我方打擊」或「冷卻」）→ 存下來。"""
        rej = state.get("rej") or ""
        if state.get("det") and state.get("pend") is None and rej not in ("", "非我方打擊", "冷卻"):
            self._seen_unrecorded += 1
            if self._seen_unrecorded == 8:
                self.request("看到標記卻沒記錄", ts, note=f"原因：{rej}", auto=True)
        elif not state.get("det"):
            self._seen_unrecorded = 0

    # ------------------------------------------------------------------
    def request(self, kind: str, ts: float, note: str = "", pitch: Optional[Dict[str, Any]] = None,
                before: float = 8.0, after: float = 3.0, auto: bool = False) -> None:
        """排定存檔：等事件後 `after` 秒的畫面也錄到再寫出。自動事件同類型 20 秒內只存一次。"""
        if not self.enabled:
            return
        if auto:
            last = self._last_auto.get(kind, -999.0)
            if ts - last < 20.0:
                return
            self._last_auto[kind] = ts
        self._pending_dumps.append({
            "kind": kind, "ts": ts, "note": note, "pitch": pitch,
            "before": before, "after": after, "auto": auto,
            "created": datetime.now().strftime("%Y%m%d_%H%M%S"),
        })

    def _flush_due(self, now: float) -> None:
        due = [d for d in self._pending_dumps if now >= d["ts"] + d["after"]]
        if not due:
            return
        self._pending_dumps = [d for d in self._pending_dumps if d not in due]
        for d in due:
            threading.Thread(target=self._dump, args=(d,), daemon=True).start()

    def flush_all(self) -> None:
        for d in list(self._pending_dumps):
            self._dump(d)
        self._pending_dumps = []

    def _dump(self, d: Dict[str, Any]) -> None:
        with self._lock:
            items = [x for x in self._buf if d["ts"] - d["before"] <= x[0] <= d["ts"] + d["after"]]
        if not items:
            return
        safe_kind = "".join(ch for ch in d["kind"] if ch not in '\\/:*?"<>|')
        folder = FEEDBACK_DIR / f"{d['created']}_{safe_kind}"
        n = 2
        while folder.exists():
            folder = FEEDBACK_DIR / f"{d['created']}_{safe_kind}_{n}"
            n += 1
        try:
            (folder / "frames").mkdir(parents=True)
            with open(folder / "timeline.jsonl", "w", encoding="utf-8") as f:
                for i, (ts, jpg, state) in enumerate(items):
                    name = f"{i:05d}.jpg"
                    (folder / "frames" / name).write_bytes(jpg)
                    f.write(json.dumps({"t": round(ts, 2), "img": name, **state}, ensure_ascii=False) + "\n")
            info = {k: d[k] for k in ("kind", "ts", "note", "pitch", "auto", "created")}
            from state.tracker import APP_VERSION
            info["app_version"] = APP_VERSION
            info["frames"] = len(items)
            (folder / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(f"  [回報存檔] {folder.name}（{len(items)} 張畫面）", flush=True)
        except Exception as e:
            print(f"  [回報存檔失敗] {e}", flush=True)
        self._trim()

    def _trim(self) -> None:
        """總容量超過上限時，先刪最舊的自動存檔（手動回報保留）。"""
        try:
            dirs = [p for p in FEEDBACK_DIR.iterdir() if p.is_dir() and p.name != "sessions"]
            total = sum(f.stat().st_size for p in dirs for f in p.rglob("*") if f.is_file())
            limit = self.max_total_mb * 1024 * 1024
            if total <= limit:
                return
            autos = []
            for p in dirs:
                try:
                    if json.loads((p / "info.json").read_text(encoding="utf-8")).get("auto"):
                        autos.append(p)
                except Exception:
                    pass
            for p in sorted(autos, key=lambda x: x.name):
                if total <= limit:
                    break
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p, ignore_errors=True)
                total -= size
        except Exception:
            pass


RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "recordings"


class SessionVideo:
    """
    整場錄影：把程式實際截到的畫面存成影片（1280 寬、每秒 10 張），
    同時把每一張畫面的判斷過程寫進同名的 .jsonl，兩者以 frame 編號逐格對應。
    換場次（新的一場）會自動換一個新檔案。
    """

    def __init__(self, fps: float = 10.0, width: int = 1280) -> None:
        self.fps = fps
        self.width = width
        self.enabled = False
        self._writer = None
        self._log = None
        self._size = None
        self._game_id = ""
        self._frame_no = 0
        self._last_ts = -999.0
        self._start_wall = 0.0
        self.current_file = ""
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)
        if not self.enabled:
            self.close()

    def status(self) -> Dict[str, Any]:
        if not self._writer:
            return {"on": self.enabled, "file": "", "sec": 0}
        return {"on": self.enabled, "file": self.current_file, "sec": int(time.time() - self._start_wall)}

    def _open(self, game_id: str, size: Tuple[int, int]) -> None:
        self.close()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = RECORDINGS_DIR / f"{stamp}_{game_id}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(base) + ".mp4", fourcc, self.fps, size)
        if not self._writer.isOpened():
            print("  [錄影] 無法建立影片檔", flush=True)
            self._writer = None
            return
        self._log = open(str(base) + ".jsonl", "w", encoding="utf-8")
        self._size = size
        self._game_id = game_id
        self._frame_no = 0
        self._start_wall = time.time()
        self.current_file = base.name + ".mp4"
        print(f"  [錄影] 開始：{self.current_file}", flush=True)

    def add(self, client_bgr: np.ndarray, ts: float, state: Dict[str, Any], game_id: str) -> None:
        if not self.enabled:
            return
        if ts - self._last_ts < 1.0 / self.fps:
            return
        self._last_ts = ts
        h, w = client_bgr.shape[:2]
        size = (self.width, int(round(h * self.width / w)) // 2 * 2)
        if self._writer is None or game_id != self._game_id or size != self._size:
            self._open(game_id, size)
            if self._writer is None:
                return
        try:
            frame = cv2.resize(client_bgr, size, interpolation=cv2.INTER_AREA)
            self._writer.write(frame)
            self._log.write(json.dumps({"frame": self._frame_no, "t": round(ts, 2), **state}, ensure_ascii=False) + "\n")
            self._frame_no += 1
        except Exception as e:
            print(f"  [錄影錯誤] {e}", flush=True)

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            print(f"  [錄影] 結束：{self.current_file}（{self._frame_no} 張）", flush=True)
        if self._log is not None:
            try:
                self._log.close()
            except Exception:
                pass
        self._writer = None
        self._log = None
        self._size = None
        self._game_id = ""
