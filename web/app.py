"""
Stage 6 FastAPI + WebSocket server (`web/app.py`):
- Serves the mobile-friendly dark HUD web UI (`web/static/index.html`)
- Pushes real-time pitch detections and live game status over WebSocket (`/ws`)
- Provides REST APIs for:
  - Filtering analytics by scope (本場 / 歷史全部), pitcher, opponent tag, count, pitch group, result
  - Switching `my_batting_row` ("top" 先攻/上行 vs "bottom" 後攻/下行) and `opponent_tag`
  - Manual click-to-record on the 9-grid + 4 out-of-zone buttons + undo last pitch + clear opponent
  - Inline editing / deleting pitch records (and updating `pitch_dict` pHash cache)
  - Exporting CSV (`GET /api/export.csv`)
  - Viewing and adjusting normalized calibration coordinates (`GET/POST /api/calibration`)
"""

from __future__ import annotations
from contextlib import asynccontextmanager
import asyncio
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from capture.calibration import CalibrationConfig
from ocr.phash import classify_pitch_group
from state.tracker import GameSessionTracker
from store.db import PitchStore


STATIC_DIR = Path(__file__).resolve().parent / "static"


class SettingsUpdateRequest(BaseModel):
    my_batting_row: Optional[str] = None
    row_auto: Optional[bool] = None
    my_id: Optional[str] = None
    record_both: Optional[bool] = None
    record_video: Optional[bool] = None
    opponent_tag: Optional[str] = None
    require_cursor: Optional[bool] = None
    new_game: bool = False
    reset_count: bool = False


class ManualPitchRequest(BaseModel):
    zone: str
    opponent_tag: Optional[str] = None
    pitcher: Optional[str] = None
    result: str = "strike"
    pitch_raw: Optional[str] = None
    speed_kmh: Optional[float] = None


class PitchEditRequest(BaseModel):
    pitch_raw: Optional[str] = None
    pitch_group: Optional[str] = None
    zone: Optional[str] = None
    result: Optional[str] = None
    opponent_tag: Optional[str] = None
    note: Optional[str] = None
    phash: Optional[str] = None


class FeedbackRequest(BaseModel):
    reason: str = "其他"
    note: str = ""
    pitch_id: Optional[int] = None


class ClearRequest(BaseModel):
    pitcher: Optional[str] = None
    opponent_tag: Optional[str] = None
    scope: str = "current"


class WebUIBridge:
    """Thread-safe / async bridge between the capture loop and connected WebSocket clients."""

    def __init__(self, store: PitchStore, tracker: GameSessionTracker):
        self.store = store
        self.tracker = tracker
        self.clients: Set[WebSocket] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.connected_window: bool = False
        self.last_frame_bgr = None
        self.recorder = None
        self.video = None
        self.last_ts: float = 0.0

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        dead: List[WebSocket] = []
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def notify_from_thread(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        msg = {
            "type": event_type,
            "status": {**asdict(self.tracker.get_status(connected=self.connected_window)), "video": self.video.status() if self.video else None},
            "payload": payload or {},
        }
        if self.loop is not None and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(msg), self.loop)


def create_app(
    store: Optional[PitchStore] = None,
    tracker: Optional[GameSessionTracker] = None,
    bridge: Optional[WebUIBridge] = None,
) -> FastAPI:
    if store is None:
        store = PitchStore()
    if tracker is None:
        from ocr.engine import BBOOCREngine

        tracker = GameSessionTracker(store=store, ocr_engine=BBOOCREngine())
    if bridge is None:
        bridge = WebUIBridge(store=store, tracker=tracker)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        bridge.loop = asyncio.get_running_loop()
        yield

    app = FastAPI(title="全打投球落點分析 AI", version="1.0.0", lifespan=lifespan)

    app.state.store = store
    app.state.tracker = tracker
    app.state.bridge = bridge

    def _status() -> Dict[str, Any]:
        d = asdict(tracker.get_status(connected=bridge.connected_window))
        d["video"] = bridge.video.status() if getattr(bridge, "video", None) else None
        return d

    snapshots_dir = tracker.snapshots_dir
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/snapshots", StaticFiles(directory=str(snapshots_dir)), name="snapshots")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/api/state")
    async def get_state(
        scope: str = Query("all", description="'current' (本場) or 'all' (歷史全部)"),
        pitcher: str = Query("all"),
        opponent_tag: str = Query("all"),
        count_filter: str = Query("all"),
        pitch_group_filter: str = Query("all"),
        result_filter: str = Query("all"),
    ) -> Dict[str, Any]:
        game_id_filter = tracker.game_id if scope == "current" else "all"
        status_obj = tracker.get_status(connected=bridge.connected_window)
        status_full = _status()
        analytics = store.get_analytics(
            pitcher=pitcher,
            opponent_tag=opponent_tag,
            game_id=game_id_filter,
            count_filter=count_filter,
            pitch_group_filter=pitch_group_filter,
            result_filter=result_filter,
            current_balls=status_obj.balls,
            current_strikes=status_obj.strikes,
        )
        pitchers_list = store.list_pitchers()
        return {
            "status": status_full,
            "analytics": analytics,
            "pitchers": pitchers_list,
        }

    @app.get("/api/update/check")
    async def update_check() -> Dict[str, Any]:
        """檢查 GitHub 上有沒有新版本。"""
        from tools import updater
        return await asyncio.get_running_loop().run_in_executor(None, updater.check)

    @app.post("/api/update/apply")
    async def update_apply() -> Dict[str, Any]:
        """下載新版、換掉程式檔，然後自動重新啟動。紀錄與設定不會動。"""
        import os
        import subprocess
        from tools import updater
        msgs: List[str] = []
        ok = await asyncio.get_running_loop().run_in_executor(None, lambda: updater.apply_update(msgs.append))
        if ok:
            def _restart() -> None:
                import time as _t
                _t.sleep(1.0)
                try:
                    bridge.video.close()
                except Exception:
                    pass
                bat = updater.ROOT / "2_啟動.bat"
                subprocess.Popen(["cmd", "/c", "start", "", str(bat), "--after-update"], cwd=str(updater.ROOT))
                os._exit(0)
            threading.Thread(target=_restart, daemon=True).start()
        return {"ok": ok, "log": msgs}

    @app.get("/api/dashboard")
    async def dashboard(
        player: str = Query("", description="右下玩家數據：玩家 ID"),
        pitcher: str = Query("all"),
        scope: str = Query("all", description="'current' 本場 / 'all' 全部場次"),
    ) -> Dict[str, Any]:
        """正式版網頁一次拿齊：狀態、左邊即時數據、下一球參考、右下玩家數據。"""
        st = tracker.get_status(connected=bridge.connected_window)
        status_full = _status()
        cur_tag = st.pitching_players if st.pitching_players and " / " not in st.pitching_players else ""
        live = None
        if cur_tag:
            live = store.player_view(cur_tag, st.pitcher or "all", tracker.game_id)
            live["tag"] = cur_tag
            live["pitcher"] = st.pitcher
        pred = None
        if cur_tag:
            pred = store.predict(cur_tag, st.pitcher, st.balls, st.strikes)
            pred["accuracy_game"] = store.prediction_accuracy(game_id=tracker.game_id)
            pred["accuracy_player"] = store.prediction_accuracy(opponent_tag=cur_tag)
        known = store.known_players()
        sel = player or cur_tag or (known[0]["id"] if known else "")
        view = None
        if sel:
            view = store.player_view(sel, pitcher, tracker.game_id if scope == "current" else "all")
            view["tag"] = sel
        return {"status": status_full, "live": live, "predict": pred, "player": view, "known_players": known}

    @app.post("/api/settings")
    async def update_settings(req: SettingsUpdateRequest) -> Dict[str, Any]:
        if req.record_video is not None and getattr(bridge, "video", None) is not None:
            bridge.video.set_enabled(req.record_video)
            tracker.config.record_video = bool(req.record_video)
            try:
                tracker.config.save()
            except Exception:
                pass
        if req.record_both is not None:
            tracker.set_record_both(req.record_both)
        if req.my_id is not None:
            tracker.set_my_id(req.my_id)
        if req.row_auto:
            tracker.set_row_auto()
        elif req.my_batting_row in ("top", "bottom"):
            tracker.set_my_batting_row(req.my_batting_row)
        if req.opponent_tag is not None:
            tracker.set_opponent_tag(req.opponent_tag)
        if req.require_cursor is not None:
            tracker.require_cursor = req.require_cursor
        if req.new_game:
            tracker.start_new_game()
        if req.reset_count:
            tracker.reset_count(known=True)

        status_dict = _status()
        await bridge.broadcast({"type": "settings", "status": status_dict})
        return {"ok": True, "status": status_dict}

    @app.post("/api/manual_pitch")
    async def add_manual_pitch(req: ManualPitchRequest) -> Dict[str, Any]:
        zone = req.zone.strip()
        # Map zone to representative (x_norm, y_norm)
        zone_coords = {
            "1": (0.16, 0.16),
            "2": (0.50, 0.16),
            "3": (0.84, 0.16),
            "4": (0.16, 0.50),
            "5": (0.50, 0.50),
            "6": (0.84, 0.50),
            "7": (0.16, 0.84),
            "8": (0.50, 0.84),
            "9": (0.84, 0.84),
            "H": (0.50, -0.18),
            "L": (0.50, 1.18),
            "IN": (-0.18, 0.50),
            "OUT": (1.18, 0.50),
        }
        x_norm, y_norm = zone_coords.get(zone, (0.50, 0.50))
        opp = (
            req.opponent_tag
            if req.opponent_tag is not None
            else (tracker.opponent_tag or "對手1")
        )
        pitcher_name = (
            req.pitcher
            if req.pitcher is not None
            else (tracker.pitcher or opp or "手動記錄投手")
        )
        res_type = req.result if req.result else ("ball" if zone in ("H", "L", "IN", "OUT") else "strike")

        record = {
            "game_id": tracker.game_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "inning": tracker.inning,
            "half": tracker.half,
            "outs": tracker.outs,
            "pitcher": pitcher_name,
            "opponent_tag": opp,
            "batter": tracker.batter or "我方打者",
            "balls_before": tracker.tracked_balls,
            "strikes_before": tracker.tracked_strikes,
            "zone": zone,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "result": res_type,
            "pitch_raw": req.pitch_raw,
            "pitch_group": classify_pitch_group(req.pitch_raw),
            "speed_kmh": req.speed_kmh,
            "note": "手動點選",
            "snapshot": "",
        }
        pid = store.insert_pitch(record)
        record["id"] = pid
        tracker.last_recorded_pitch = record
        await bridge.broadcast(
            {
                "type": "pitch",
                "status": _status(),
                "payload": record,
            }
        )
        return {"ok": True, "pitch": record}

    @app.post("/api/undo")
    async def undo_last_pitch(
        pitcher: Optional[str] = Query(None),
        opponent_tag: Optional[str] = Query(None),
        scope: str = Query("all"),
    ) -> Dict[str, Any]:
        game_id_filter = tracker.game_id if scope == "current" else "all"
        rows = store.query_pitches(
            pitcher=pitcher, opponent_tag=opponent_tag, game_id=game_id_filter
        )
        if not rows:
            return {"ok": False, "message": "目前沒有可撤銷的投球紀錄"}
        last_pitch = rows[0]
        store.delete_pitch(int(last_pitch["id"]))
        await bridge.broadcast(
            {
                "type": "undo",
                "status": _status(),
                "payload": last_pitch,
            }
        )
        return {"ok": True, "deleted": last_pitch}

    @app.post("/api/clear")
    async def clear_pitches(req: ClearRequest) -> Dict[str, Any]:
        game_id_filter = tracker.game_id if req.scope == "current" else None
        pitcher = req.pitcher if (req.pitcher and req.pitcher != "all") else None
        opp = req.opponent_tag if (req.opponent_tag and req.opponent_tag != "all") else None
        deleted_cnt = store.delete_by_filter(
            pitcher=pitcher, opponent_tag=opp, game_id=game_id_filter
        )
        await bridge.broadcast(
            {
                "type": "clear",
                "status": _status(),
                "payload": {"deleted_count": deleted_cnt},
            }
        )
        return {"ok": True, "deleted_count": deleted_cnt}

    def _get_pitch(pid: int) -> Optional[Dict[str, Any]]:
        try:
            with store._connect() as conn:
                row = conn.execute("SELECT * FROM pitches WHERE id = ?", (pid,)).fetchone()
                return dict(row) if row else None
        except Exception:
            return None

    @app.post("/api/feedback")
    async def send_feedback(req: FeedbackRequest) -> Dict[str, Any]:
        """使用者回報問題：存下前 30 秒＋後 3 秒的畫面與判斷過程。"""
        if bridge.recorder is None:
            return {"ok": False, "message": "記錄器沒有啟動（離線重播模式不支援）"}
        pitch = _get_pitch(req.pitch_id) if req.pitch_id else None
        bridge.recorder.request(
            f"回報-{req.reason}", bridge.last_ts, note=req.note, pitch=pitch, before=30.0, after=3.0
        )
        return {"ok": True}

    @app.put("/api/pitches/{pitch_id}")
    async def edit_pitch(pitch_id: int, req: PitchEditRequest) -> Dict[str, Any]:
        # 使用者的修正就是正確答案：記下修正前後，之後用來檢驗辨識
        before = _get_pitch(pitch_id)
        try:
            import json as _json
            from capture.recorder import FEEDBACK_DIR
            with open(FEEDBACK_DIR / "corrections.jsonl", "a", encoding="utf-8") as f:
                f.write(_json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "id": pitch_id,
                                     "before": before, "after": req.dict(exclude_none=True)},
                                    ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass
        ok = store.update_pitch(
            pitch_id=pitch_id,
            pitch_raw=req.pitch_raw,
            pitch_group=req.pitch_group,
            zone=req.zone,
            result=req.result,
            opponent_tag=req.opponent_tag,
            note=req.note,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="找不到該筆投球紀錄")
        if req.phash and req.pitch_raw:
            store.save_phash(
                phash=req.phash,
                pitch_raw=req.pitch_raw,
                pitch_group=req.pitch_group or classify_pitch_group(req.pitch_raw),
            )
        await bridge.broadcast(
            {
                "type": "edit",
                "status": _status(),
                "payload": {"id": pitch_id},
            }
        )
        return {"ok": True}

    @app.delete("/api/pitches/{pitch_id}")
    async def remove_pitch(pitch_id: int) -> Dict[str, Any]:
        before = _get_pitch(pitch_id)
        try:
            import json as _json
            from capture.recorder import FEEDBACK_DIR
            with open(FEEDBACK_DIR / "corrections.jsonl", "a", encoding="utf-8") as f:
                f.write(_json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "id": pitch_id,
                                     "before": before, "after": "deleted"}, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass
        ok = store.delete_pitch(pitch_id)
        if not ok:
            raise HTTPException(status_code=404, detail="找不到該筆投球紀錄")
        await bridge.broadcast(
            {
                "type": "delete",
                "status": _status(),
                "payload": {"id": pitch_id},
            }
        )
        return {"ok": True}

    @app.get("/api/export.csv")
    async def export_csv(
        scope: str = Query("all"),
        pitcher: str = Query("all"),
        opponent_tag: str = Query("all"),
    ) -> Response:
        game_id_filter = tracker.game_id if scope == "current" else "all"
        csv_text = "\ufeff" + store.export_csv(
            pitcher=pitcher, opponent_tag=opponent_tag, game_id=game_id_filter
        )
        return Response(
            content=csv_text.encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="bbo_pitches.csv"'},
        )

    @app.get("/api/calibration")
    async def get_calibration() -> Dict[str, Any]:
        return asdict(tracker.config)

    @app.get("/api/preview.jpg")
    async def preview_frame() -> Response:
        import cv2
        import numpy as np

        frame = bridge.last_frame_bgr
        if frame is None or frame.size == 0:
            canvas = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(
                canvas,
                "No Frame Captured Yet",
                (140, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (180, 180, 200),
                2,
            )
            _, buf = cv2.imencode(".jpg", canvas)
            return Response(content=buf.tobytes(), media_type="image/jpeg")

        vis = frame.copy()
        h, w = vis.shape[:2]
        cfg = tracker.config
        active_sz, hand = cfg.detect_active_strike_zone(vis)
        for box, col in (
            (active_sz, (120, 255, 0)),
            (cfg.scoreboard, (255, 140, 60)),
            (cfg.info_panel, (60, 190, 255)),
            (cfg.tab_panel, (90, 90, 255)),
        ):
            x1, y1, x2, y2 = box.to_px(w, h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        # Draw 3x3 grid inside active strike zone
        sx1, sy1, sx2, sy2 = active_sz.to_px(w, h)
        for i in (1, 2):
            gx = sx1 + int((sx2 - sx1) * i / 3.0)
            gy = sy1 + int((sy2 - sy1) * i / 3.0)
            cv2.line(vis, (gx, sy1), (gx, sy2), (120, 255, 0), 1)
            cv2.line(vis, (sx1, gy), (sx2, gy), (120, 255, 0), 1)
        cv2.putText(
            vis,
            f"SZ ({hand})",
            (sx1 + 4, max(20, sy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (120, 255, 0),
            2,
        )
        _, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    @app.post("/api/calibration")
    async def save_calibration(data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = tracker.config
        from capture.calibration import BoxRatio

        for key in ("strike_zone", "scoreboard", "info_panel", "tab_panel"):
            if key in data and isinstance(data[key], dict):
                setattr(cfg, key, BoxRatio(**data[key]))
        if "lh_x_offset" in data:
            cfg.lh_x_offset = float(data["lh_x_offset"])
        cfg.save()
        return {"ok": True, "calibration": asdict(cfg)}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await bridge.connect(ws)
        try:
            await ws.send_json(
                {
                    "type": "hello",
                    "status": _status(),
                }
            )
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            bridge.disconnect(ws)
        except Exception:
            bridge.disconnect(ws)

    return app
