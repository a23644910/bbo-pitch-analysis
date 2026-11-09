"""
SQLite database store for 《全民打棒球2》 pitch records, pHash dictionary,
and pitcher repertoire from the TAB panel.
"""

from __future__ import annotations
import csv
import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ocr.phash import classify_pitch_group, hamming_distance


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bbo_pitches.db"

ZONE_LABELS_ZH = {
    "1": "左上(1)",
    "2": "中上(2)",
    "3": "右上(3)",
    "4": "左中(4)",
    "5": "正中(5)",
    "6": "右中(6)",
    "7": "左下(7)",
    "8": "中下(8)",
    "9": "右下(9)",
    "H": "偏高(H)",
    "L": "偏低(L)",
    "IN": "內角(IN)",
    "OUT": "外角(OUT)",
    "H-IN": "內角高(H-IN)",
    "H-OUT": "外角高(H-OUT)",
    "L-IN": "內角低(L-IN)",
    "L-OUT": "外角低(L-OUT)",
}


def matches_count_filter(balls: Optional[int], strikes: Optional[int], count_filter: str) -> bool:
    """
    Evaluate whether (balls, strikes) matches a count filter:
    - 'all' / '': all pitches
    - '0-0': balls == 0 and strikes == 0
    - 'ahead' (投手領先 / 打者落後): strikes > balls
    - 'behind' (打者領先 / 投手落後): balls > strikes
    - 'two_strikes' (兩好球): strikes == 2
    - 'full' (滿球數): balls == 3 and strikes == 2
    """
    if not count_filter or count_filter == "all":
        return True
    if balls is None or strikes is None:
        return False
    if count_filter == "0-0":
        return balls == 0 and strikes == 0
    if count_filter == "ahead":
        return strikes > balls
    if count_filter == "behind":
        return balls > strikes
    if count_filter == "two_strikes":
        return strikes == 2
    if count_filter == "full":
        return balls == 3 and strikes == 2
    if "-" in count_filter:
        parts = count_filter.split("-", 1)
        try:
            return balls == int(parts[0]) and strikes == int(parts[1])
        except ValueError:
            return True
    return True


class PitchStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._migrate_out_of_zone()
        self._add_env_columns()

    @staticmethod
    def _grid_from_xy(x: float, y: float) -> str:
        x = min(max(float(x), 0.0), 0.9999)
        y = min(max(float(y), 0.0), 0.9999)
        return str(int(y * 3) * 3 + int(x * 3) + 1)

    ENV_COLUMNS = ("weather", "wind_dir", "wind_angle", "wind_speed", "pitcher_attr", "batter_attr")

    def _add_env_columns(self) -> None:
        """比賽環境欄位（天氣、風、卡片天氣屬性），舊資料庫自動補上。"""
        with self._connect() as conn:
            have = {r[1] for r in conn.execute("PRAGMA table_info(pitches)").fetchall()}
            for col in self.ENV_COLUMNS:
                if col not in have:
                    typ = "INTEGER" if col in ("wind_angle", "wind_speed") else "TEXT"
                    conn.execute(f"ALTER TABLE pitches ADD COLUMN {col} {typ}")

    def _migrate_out_of_zone(self) -> None:
        """舊紀錄的框外代號（H / L / IN / OUT…）改成沿格線延伸後的九宮格號碼。"""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, x_norm, y_norm FROM pitches WHERE zone NOT IN ('1','2','3','4','5','6','7','8','9')"
                ).fetchall()
                for r in rows:
                    conn.execute(
                        "UPDATE pitches SET zone = ? WHERE id = ?",
                        (self._grid_from_xy(r["x_norm"], r["y_norm"]), r["id"]),
                    )
        except Exception:
            pass

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pitches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    inning INTEGER NOT NULL DEFAULT 1,
                    half TEXT NOT NULL DEFAULT '上',
                    outs INTEGER NOT NULL DEFAULT 0,
                    pitcher TEXT NOT NULL DEFAULT '',
                    opponent_tag TEXT NOT NULL DEFAULT '',
                    batter TEXT NOT NULL DEFAULT '',
                    balls_before INTEGER NULL,
                    strikes_before INTEGER NULL,
                    zone TEXT NOT NULL,
                    x_norm REAL NOT NULL,
                    y_norm REAL NOT NULL,
                    result TEXT NOT NULL,
                    pitch_raw TEXT NULL,
                    pitch_group TEXT NULL,
                    speed_kmh REAL NULL,
                    note TEXT NOT NULL DEFAULT '',
                    snapshot TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_pitches_game ON pitches(game_id);
                CREATE INDEX IF NOT EXISTS idx_pitches_pitcher ON pitches(pitcher);
                CREATE INDEX IF NOT EXISTS idx_pitches_opponent ON pitches(opponent_tag);

                CREATE TABLE IF NOT EXISTS pitch_dict (
                    phash TEXT PRIMARY KEY,
                    pitch_raw TEXT NOT NULL,
                    pitch_group TEXT NOT NULL,
                    sample_path TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    pitch_id INTEGER PRIMARY KEY,
                    game_id TEXT NOT NULL,
                    opponent_tag TEXT NOT NULL DEFAULT '',
                    pitcher TEXT NOT NULL DEFAULT '',
                    balls INTEGER NULL,
                    strikes INTEGER NULL,
                    level TEXT NOT NULL DEFAULT '',
                    sample INTEGER NOT NULL DEFAULT 0,
                    top_zones TEXT NOT NULL DEFAULT '',
                    actual_zone TEXT NOT NULL DEFAULT '',
                    hit1 INTEGER NOT NULL DEFAULT 0,
                    hit3 INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS pitcher_repertoire (
                    pitcher TEXT PRIMARY KEY,
                    pitches_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    # ------------------------------------------------------------------
    # pHash dictionary & Pitcher repertoire
    # ------------------------------------------------------------------
    def lookup_phash(self, phash: str, max_hamming: int = 4) -> Optional[Tuple[str, str]]:
        """
        Look up a pitch label pHash in `pitch_dict`.
        Returns (pitch_raw, pitch_group) on exact or near-exact hit (Hamming <= max_hamming).
        """
        if not phash or phash == "0000000000000000":
            return None
        with self._connect() as conn:
            exact = conn.execute(
                "SELECT pitch_raw, pitch_group FROM pitch_dict WHERE phash = ?",
                (phash,),
            ).fetchone()
            if exact:
                return (exact["pitch_raw"], exact["pitch_group"])

            rows = conn.execute(
                "SELECT phash, pitch_raw, pitch_group FROM pitch_dict"
            ).fetchall()
            best_row = None
            best_dist = max_hamming + 1
            for r in rows:
                d = hamming_distance(phash, r["phash"])
                if d < best_dist:
                    best_dist = d
                    best_row = r
            if best_row is not None and best_dist <= max_hamming:
                return (best_row["pitch_raw"], best_row["pitch_group"])
        return None

    def save_phash(
        self,
        phash: str,
        pitch_raw: str,
        pitch_group: Optional[str] = None,
        sample_path: str = "",
    ) -> None:
        if not phash or not pitch_raw or pitch_raw == "未知":
            return
        grp = pitch_group or classify_pitch_group(pitch_raw) or "其他"
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pitch_dict (phash, pitch_raw, pitch_group, sample_path, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(phash) DO UPDATE SET
                    pitch_raw = excluded.pitch_raw,
                    pitch_group = excluded.pitch_group,
                    sample_path = CASE WHEN excluded.sample_path != '' THEN excluded.sample_path ELSE pitch_dict.sample_path END,
                    updated_at = excluded.updated_at
                """,
                (phash, pitch_raw, grp, sample_path, now),
            )

    def get_pitcher_repertoire(self, pitcher: str) -> List[str]:
        if not pitcher:
            return []
        with self._connect() as conn:
            row = conn.execute(
                "SELECT pitches_json FROM pitcher_repertoire WHERE pitcher = ?",
                (pitcher,),
            ).fetchone()
            if not row:
                return []
            try:
                return list(json.loads(row["pitches_json"]))
            except Exception:
                return []

    def save_pitcher_repertoire(self, pitcher: str, pitches: List[str]) -> None:
        if not pitcher or not pitches:
            return
        clean = []
        for p in pitches:
            if p and p.strip() and p.strip() not in clean:
                clean.append(p.strip())
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pitcher_repertoire (pitcher, pitches_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(pitcher) DO UPDATE SET
                    pitches_json = excluded.pitches_json,
                    updated_at = excluded.updated_at
                """,
                (pitcher, json.dumps(clean, ensure_ascii=False), now),
            )

    # ------------------------------------------------------------------
    # Pitch CRUD
    # ------------------------------------------------------------------
    def insert_pitch(self, record: Dict[str, Any]) -> int:
        now = record.get("ts") or datetime.now().isoformat(timespec="seconds")
        pitch_raw = record.get("pitch_raw")
        pitch_group = record.get("pitch_group") or classify_pitch_group(pitch_raw)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO pitches (
                    game_id, ts, inning, half, outs, pitcher, opponent_tag, batter,
                    balls_before, strikes_before, zone, x_norm, y_norm, result,
                    pitch_raw, pitch_group, speed_kmh, note, snapshot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("game_id", "default"),
                    now,
                    int(record.get("inning", 1)),
                    str(record.get("half", "上")),
                    int(record.get("outs", 0)),
                    str(record.get("pitcher", "")),
                    str(record.get("opponent_tag", "")),
                    str(record.get("batter", "")),
                    record.get("balls_before"),
                    record.get("strikes_before"),
                    str(record.get("zone", "5")),
                    float(record.get("x_norm", 0.5)),
                    float(record.get("y_norm", 0.5)),
                    str(record.get("result", "strike")),
                    pitch_raw,
                    pitch_group,
                    record.get("speed_kmh"),
                    str(record.get("note", "")),
                    str(record.get("snapshot", "")),
                ),
            )
            pid = int(cur.lastrowid)
            env = {k: record.get(k) for k in self.ENV_COLUMNS if record.get(k) not in (None, "")}
            if env:
                conn.execute(
                    f"UPDATE pitches SET {', '.join(k + ' = ?' for k in env)} WHERE id = ?",
                    list(env.values()) + [pid],
                )
            return pid

    def update_pitch(
        self,
        pitch_id: int,
        pitch_raw: Optional[str] = None,
        pitch_group: Optional[str] = None,
        zone: Optional[str] = None,
        result: Optional[str] = None,
        opponent_tag: Optional[str] = None,
        note: Optional[str] = None,
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM pitches WHERE id = ?", (pitch_id,)).fetchone()
            if not row:
                return False
            new_raw = pitch_raw if pitch_raw is not None else row["pitch_raw"]
            new_grp = (
                pitch_group
                if pitch_group is not None
                else (classify_pitch_group(new_raw) if pitch_raw is not None else row["pitch_group"])
            )
            new_zone = zone if zone is not None else row["zone"]
            new_res = result if result is not None else row["result"]
            new_opp = opponent_tag if opponent_tag is not None else row["opponent_tag"]
            new_note = note if note is not None else row["note"]

            conn.execute(
                """
                UPDATE pitches
                SET pitch_raw = ?, pitch_group = ?, zone = ?, result = ?, opponent_tag = ?, note = ?
                WHERE id = ?
                """,
                (new_raw, new_grp, new_zone, new_res, new_opp, new_note, pitch_id),
            )
            return True

    def delete_pitch(self, pitch_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM pitches WHERE id = ?", (pitch_id,))
            return cur.rowcount > 0

    def delete_by_filter(
        self, pitcher: Optional[str] = None, opponent_tag: Optional[str] = None, game_id: Optional[str] = None
    ) -> int:
        clauses = []
        params: List[Any] = []
        if pitcher:
            clauses.append("pitcher = ?")
            params.append(pitcher)
        if opponent_tag:
            clauses.append("opponent_tag = ?")
            params.append(opponent_tag)
        if game_id:
            clauses.append("game_id = ?")
            params.append(game_id)
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            cur = conn.execute(f"DELETE FROM pitches {where_sql}", params)
            return cur.rowcount

    def list_pitchers(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT pitcher, opponent_tag, COUNT(*) as cnt, MAX(ts) as last_ts
                FROM pitches
                GROUP BY pitcher, opponent_tag
                ORDER BY last_ts DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def query_pitches(
        self,
        pitcher: Optional[str] = None,
        opponent_tag: Optional[str] = None,
        game_id: Optional[str] = None,
        count_filter: str = "all",
        pitch_group_filter: str = "all",
        result_filter: str = "all",
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if pitcher and pitcher != "all":
            clauses.append("pitcher = ?")
            params.append(pitcher)
        if opponent_tag and opponent_tag != "all":
            clauses.append("opponent_tag = ?")
            params.append(opponent_tag)
        if game_id and game_id != "all":
            clauses.append("game_id = ?")
            params.append(game_id)
        if pitch_group_filter and pitch_group_filter != "all":
            clauses.append("pitch_group = ?")
            params.append(pitch_group_filter)
        if result_filter and result_filter != "all":
            clauses.append("result = ?")
            params.append(result_filter)

        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM pitches {where_sql} ORDER BY id DESC", params
            ).fetchall()

        out = []
        for r in rows:
            d = dict(r)
            if matches_count_filter(d["balls_before"], d["strikes_before"], count_filter):
                out.append(d)
        return out

    def get_analytics(
        self,
        pitcher: Optional[str] = None,
        opponent_tag: Optional[str] = None,
        game_id: Optional[str] = None,
        count_filter: str = "all",
        pitch_group_filter: str = "all",
        result_filter: str = "all",
        current_balls: Optional[int] = None,
        current_strikes: Optional[int] = None,
    ) -> Dict[str, Any]:
        pitches = self.query_pitches(
            pitcher=pitcher,
            opponent_tag=opponent_tag,
            game_id=game_id,
            count_filter=count_filter,
            pitch_group_filter=pitch_group_filter,
            result_filter=result_filter,
        )
        total = len(pitches)

        # 1. 9-grid heatmap + 4 out-of-zone directions
        grid_counts = {str(i): 0 for i in range(1, 10)}
        out_counts = {"H": 0, "L": 0, "IN": 0, "OUT": 0}
        for p in pitches:
            z = str(p["zone"])
            if z in grid_counts:
                grid_counts[z] += 1
            else:
                if "H" in z:
                    out_counts["H"] += 1
                if "L" in z:
                    out_counts["L"] += 1
                if "IN" in z:
                    out_counts["IN"] += 1
                if "OUT" in z:
                    out_counts["OUT"] += 1

        # 2. Pitch breakdown (raw and group) with average speed
        raw_stats: Dict[str, Dict[str, Any]] = {}
        grp_stats: Dict[str, Dict[str, Any]] = {}
        for p in pitches:
            raw_name = p["pitch_raw"] or "未知"
            grp_name = p["pitch_group"] or "未知"
            spd = p["speed_kmh"]

            for bucket, key in ((raw_stats, raw_name), (grp_stats, grp_name)):
                if key not in bucket:
                    bucket[key] = {"name": key, "count": 0, "speeds": []}
                bucket[key]["count"] += 1
                if spd is not None:
                    bucket[key]["speeds"].append(float(spd))

        def _finalize_breakdown(bucket: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
            items = []
            for k, v in bucket.items():
                cnt = v["count"]
                spds = v["speeds"]
                items.append(
                    {
                        "name": k,
                        "count": cnt,
                        "pct": round(cnt * 100.0 / total, 1) if total > 0 else 0.0,
                        "avg_speed": round(sum(spds) / len(spds), 1) if spds else None,
                    }
                )
            items.sort(key=lambda x: x["count"], reverse=True)
            return items

        # 3. Count-based pitch tendencies (依球數的配球表)
        all_for_entity = self.query_pitches(
            pitcher=pitcher, opponent_tag=opponent_tag, game_id=game_id
        )
        count_buckets = [
            ("0-0", "初球 (0-0)"),
            ("ahead", "投手領先 (S>B)"),
            ("behind", "球數落後 (B>S)"),
            ("two_strikes", "兩好球 (2S)"),
            ("full", "滿球數 (3-2)"),
        ]
        tendencies = []
        for b_code, b_label in count_buckets:
            sub = [
                p
                for p in all_for_entity
                if matches_count_filter(p["balls_before"], p["strikes_before"], b_code)
            ]
            if not sub:
                continue
            combos: Dict[Tuple[str, str], int] = {}
            for p in sub:
                p_name = p["pitch_group"] or p["pitch_raw"] or "未知"
                z_name = ZONE_LABELS_ZH.get(str(p["zone"]), str(p["zone"]))
                combos[(p_name, z_name)] = combos.get((p_name, z_name), 0) + 1
            top_combos = sorted(combos.items(), key=lambda x: x[1], reverse=True)[:3]
            tendencies.append(
                {
                    "code": b_code,
                    "label": b_label,
                    "total": len(sub),
                    "top": [
                        {
                            "pitch": k[0],
                            "zone": k[1],
                            "count": v,
                            "pct": round(v * 100.0 / len(sub), 1),
                        }
                        for k, v in top_combos
                    ],
                }
            )

        # 4. Next-pitch reference (下一球參考: top 3 combinations for current count)
        if current_balls is not None and current_strikes is not None:
            ref_pool = [
                p
                for p in all_for_entity
                if p["balls_before"] == current_balls and p["strikes_before"] == current_strikes
            ]
            if not ref_pool:
                ref_pool = all_for_entity
        else:
            ref_pool = all_for_entity

        ref_combos: Dict[Tuple[str, str], int] = {}
        for p in ref_pool:
            p_name = p["pitch_raw"] or p["pitch_group"] or "未知"
            z_name = ZONE_LABELS_ZH.get(str(p["zone"]), str(p["zone"]))
            ref_combos[(p_name, z_name)] = ref_combos.get((p_name, z_name), 0) + 1
        next_pitch_ref = [
            {
                "pitch": k[0],
                "zone": k[1],
                "count": v,
                "pct": round(v * 100.0 / len(ref_pool), 1) if ref_pool else 0.0,
            }
            for k, v in sorted(ref_combos.items(), key=lambda x: x[1], reverse=True)[:3]
        ]

        return {
            "total": total,
            "grid_counts": grid_counts,
            "out_counts": out_counts,
            "by_raw": _finalize_breakdown(raw_stats),
            "by_group": _finalize_breakdown(grp_stats),
            "tendencies": tendencies,
            "next_pitch_ref": next_pitch_ref,
            "recent_pitches": pitches[:100],
        }

    def export_csv(
        self,
        pitcher: Optional[str] = None,
        opponent_tag: Optional[str] = None,
        game_id: Optional[str] = None,
    ) -> str:
        rows = self.query_pitches(pitcher=pitcher, opponent_tag=opponent_tag, game_id=game_id)
        buf = io.StringIO()
        fields = [
            "id",
            "game_id",
            "ts",
            "inning",
            "half",
            "outs",
            "pitcher",
            "opponent_tag",
            "batter",
            "balls_before",
            "strikes_before",
            "zone",
            "x_norm",
            "y_norm",
            "result",
            "pitch_raw",
            "pitch_group",
            "speed_kmh",
            "note",
            "snapshot",
        ]
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in reversed(rows):
            writer.writerow(r)
        return buf.getvalue()


# ----------------------------------------------------------------------
# 正式版：即時／玩家數據與下一球預測
# ----------------------------------------------------------------------
def _grid_and_types(pitches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """九宮格（比例＋好壞球比例）與球種分布。壞球＝結果是 ball，其餘（好球、界外、擊出）算好球。"""
    # 漏記補上的球（zone "0"）落點不明，不算進九宮格與比例
    pitches = [p for p in pitches if str(p.get("zone")) in ("1", "2", "3", "4", "5", "6", "7", "8", "9")]
    total = len(pitches)

    def outside(p) -> Optional[bool]:
        """依標記實際位置判斷是否在好球帶外（x_norm / y_norm 以好球帶為 0~1）。沒有位置資料回傳 None。"""
        x, y = p.get("x_norm"), p.get("y_norm")
        if x is None or y is None:
            return None
        return not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0)

    SWING = ("strike", "foul", "in_play")
    cells = []
    for z in range(1, 10):
        sub = [p for p in pitches if str(p["zone"]) == str(z)]
        n = len(sub)
        balls = sum(1 for p in sub if p["result"] == "ball")
        located = [p for p in sub if outside(p) is not None]
        outs = [p for p in located if outside(p)]
        cells.append({
            "zone": str(z), "n": n,
            "pct": round(n * 100.0 / total) if total else 0,
            "ball_pct": round(balls * 100.0 / n) if n else 0,
            "out_pct": round(len(outs) * 100.0 / len(located)) if located else 0,
        })
    located = [p for p in pitches if outside(p) is not None]
    out_all = [p for p in located if outside(p)]
    chased = [p for p in out_all if p["result"] in SWING]
    known = [p for p in pitches if p.get("pitch_raw")]
    tcount: Dict[str, int] = {}
    for p in known:
        tcount[p["pitch_raw"]] = tcount.get(p["pitch_raw"], 0) + 1
    types = [
        {"name": k, "n": v, "pct": round(v * 100.0 / len(known))}
        for k, v in sorted(tcount.items(), key=lambda x: -x[1])
    ]
    return {
        "total": total, "cells": cells, "types": types, "typed": len(known),
        # 帶外比例：投到好球帶外的球占多少；追打率：帶外球裡被揮棒（好球／界外／擊出）的比例
        "out_pct": round(len(out_all) * 100.0 / len(located)) if located else None,
        "chase_pct": round(len(chased) * 100.0 / len(out_all)) if out_all else None,
        "out_n": len(out_all),
    }


def _store_methods():
    def _rows(self, where: str, params: List[Any]) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(f"SELECT * FROM pitches WHERE {where} ORDER BY id DESC", params).fetchall()]

    def player_view(self, opponent_tag: str, pitcher: str = "all", game_id: str = "all") -> Dict[str, Any]:
        """某位玩家（可再指定投手、本場）的九宮格與球種。"""
        where, params = ["opponent_tag = ?"], [opponent_tag]
        if pitcher and pitcher != "all":
            where.append("pitcher = ?"); params.append(pitcher)
        if game_id and game_id != "all":
            where.append("game_id = ?"); params.append(game_id)
        rows = _rows(self, " AND ".join(where), params)
        out = _grid_and_types(rows)
        out["games"] = len({r["game_id"] for r in rows})
        with self._connect() as conn:
            out["pitchers"] = [
                {"name": r["pitcher"], "n": r["n"]}
                for r in conn.execute(
                    "SELECT pitcher, COUNT(*) n FROM pitches WHERE opponent_tag = ? GROUP BY pitcher ORDER BY n DESC",
                    [opponent_tag],
                ).fetchall()
            ]
        out["recent"] = rows[:40]
        with self._connect() as conn:
            out["total_all"] = conn.execute("SELECT COUNT(*) FROM pitches WHERE opponent_tag = ?", [opponent_tag]).fetchone()[0]
        return out

    def known_players(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT opponent_tag AS id, COUNT(*) n, MAX(ts) last FROM pitches WHERE opponent_tag != '' GROUP BY opponent_tag ORDER BY last DESC"
            ).fetchall()]

    def predict(self, opponent_tag: str, pitcher: str, balls: Optional[int], strikes: Optional[int],
                before_id: Optional[int] = None, min_sample: int = 8) -> Dict[str, Any]:
        """
        下一球預測（只分玩家，不分投手）：
        把「這位玩家全部」「同球數情境（領先/平手/落後/兩好球）」「完全同球數」三種統計加權混合，
        樣本越多權重越高；再加上「上一球投哪格 → 下一球接哪格」的習慣。
        """
        where, params = ["opponent_tag = ?"], [opponent_tag]
        if before_id is not None:
            where.append("id < ?"); params.append(before_id)
        hist = [h for h in reversed(_rows(self, " AND ".join(where), params)) if str(h["zone"]) != "0"]  # 舊 → 新，略過落點不明的漏記
        Z = [str(i) for i in range(1, 10)]

        def bucket(b, s):
            if b is None or s is None:
                return "x"
            return "2s" if s == 2 else ("behind" if b > s else ("ahead" if s > b else "even"))

        def freq(pool):
            n = len(pool)
            c: Dict[str, int] = {}
            for h in pool:
                c[str(h["zone"])] = c.get(str(h["zone"]), 0) + 1
            return {z: (c.get(z, 0) / n if n else 0.0) for z in Z}, n

        k = 6.0
        score = {z: 1.0 / 18 for z in Z}
        bk = bucket(balls, strikes)
        comps = [hist, [h for h in hist if bucket(h["balls_before"], h["strikes_before"]) == bk]]
        if balls is not None and strikes is not None:
            comps.append([h for h in hist if h["balls_before"] == balls and h["strikes_before"] == strikes])
        for pool in comps:
            fr, n = freq(pool)
            a = n / (n + k)
            for z in Z:
                score[z] += a * fr[z]
        # 上一球 → 下一球
        cur_game = None
        if before_id is not None:
            with self._connect() as conn:
                r = conn.execute("SELECT game_id FROM pitches WHERE id = ?", [before_id]).fetchone()
                cur_game = r["game_id"] if r else None
        elif hist:
            cur_game = hist[-1]["game_id"]
        if hist and hist[-1]["game_id"] == cur_game:
            pz = str(hist[-1]["zone"])
            tr: Dict[str, int] = {}
            n = 0
            for a_, b_ in zip(hist, hist[1:]):
                if str(a_["zone"]) == pz and a_["game_id"] == b_["game_id"]:
                    tr[str(b_["zone"])] = tr.get(str(b_["zone"]), 0) + 1
                    n += 1
            if n:
                for z in Z:
                    score[z] += (n / (n + k)) * tr.get(z, 0) / n
        tot = sum(score.values())
        zones = []
        for z in sorted(Z, key=lambda z: -score[z])[:3]:
            inz = [h for h in hist if str(h["zone"]) == z]
            loc = [h for h in inz if h.get("x_norm") is not None and h.get("y_norm") is not None]
            outn = sum(1 for h in loc if not (0.0 <= float(h["x_norm"]) <= 1.0 and 0.0 <= float(h["y_norm"]) <= 1.0))
            zones.append({
                "zone": z, "n": len(inz),
                "pct": round(score[z] * 100.0 / tot),
                "ball_pct": round(sum(1 for h in inz if h["result"] == "ball") * 100.0 / len(inz)) if inz else 0,
                "out_pct": round(outn * 100.0 / len(loc)) if loc else 0,
            })
        tpool = comps[1] if len([h for h in comps[1] if h.get("pitch_raw")]) >= 5 else hist
        g = _grid_and_types(tpool)
        return {"level": "這位玩家", "n": len(hist), "zones": zones if hist else [], "types": g["types"][:3],
                "balls": balls, "strikes": strikes}

    def score_prediction(self, pitch_id: int) -> None:
        """一球記錄後：用「這球之前」的資料預測，再跟實際落點對答案。"""
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM pitches WHERE id = ?", [pitch_id]).fetchone()
        if not r or not r["opponent_tag"] or str(r["zone"]) == "0":
            return
        pr = self.predict(r["opponent_tag"], r["pitcher"], r["balls_before"], r["strikes_before"], before_id=pitch_id)
        if pr["n"] == 0:
            return
        tops = [z["zone"] for z in pr["zones"]]
        actual = str(r["zone"])
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO predictions (pitch_id, game_id, opponent_tag, pitcher, balls, strikes, level, sample, top_zones, actual_zone, hit1, hit3) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [pitch_id, r["game_id"], r["opponent_tag"], r["pitcher"], r["balls_before"], r["strikes_before"],
                 pr["level"], pr["n"], ",".join(tops), actual, int(bool(tops) and tops[0] == actual), int(actual in tops)],
            )

    def prediction_accuracy(self, game_id: Optional[str] = None, opponent_tag: Optional[str] = None) -> Dict[str, Any]:
        where, params = ["p.pitch_id = x.id"], []
        if game_id:
            where.append("p.game_id = ?"); params.append(game_id)
        if opponent_tag:
            where.append("p.opponent_tag = ?"); params.append(opponent_tag)
        with self._connect() as conn:
            r = conn.execute(
                f"SELECT COUNT(*) n, SUM(hit1) h1, SUM(hit3) h3 FROM predictions p, pitches x WHERE {' AND '.join(where)}",
                params,
            ).fetchone()
        n = r["n"] or 0
        return {"n": n, "top1": round((r["h1"] or 0) * 100.0 / n) if n else None,
                "top3": round((r["h3"] or 0) * 100.0 / n) if n else None}

    def pitcher_name_counts(self) -> Dict[str, int]:
        with self._connect() as conn:
            return {r[0]: r[1] for r in conn.execute(
                "SELECT pitcher, COUNT(*) FROM pitches WHERE pitcher != '' AND pitcher != '未知投手' GROUP BY pitcher"
            ).fetchall()}

    for f in (pitcher_name_counts, player_view, known_players, predict, score_prediction, prediction_accuracy):
        setattr(PitchStore, f.__name__, f)


_store_methods()
