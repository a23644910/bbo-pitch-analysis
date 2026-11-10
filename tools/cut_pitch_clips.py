"""
從軌跡錄影切出「每一球」的小片段，給軌跡辨識（球種、球速）的可行性測試用。

做法：
- 讀 data\\trajectory\\<時間>.mp4 + .csv（每一張畫面實際被截到的時間）
- 對照資料庫每一球的記錄時間，切出「記錄前 7 秒 ～ 記錄後 1 秒」（含投手出手、球飛行、落點出現）
- 只留畫面中間（投手到好球帶那一塊），保留原解析度，檔案才小
- 輸出到 data\\trajectory\\clips\\：每球一個 <球的編號>.mp4、每球的畫面時間 <編號>.csv、總表 index.csv

用法：雙擊「5_切投球片段.bat」（預設切 120 球，各球種平均挑）。
指令：python tools\\cut_pitch_clips.py [--max 120] [--all]
"""
from __future__ import annotations
import argparse
import bisect
import csv
import random
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "data" / "trajectory"
OUT = TRAJ / "clips"
DB = ROOT / "data" / "bbo_pitches.db"

BEFORE, AFTER = 7.0, 1.0           # 記錄時間前 7 秒、後 1 秒
CROP = (0.30, 0.25, 0.70, 0.95)    # 投手 → 好球帶（含球種球速標籤）
FPS = 60

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def load_recordings():
    """每段錄影：(開始的真實時間 epoch 秒, 長度秒, mp4, 每張畫面的毫秒 list)。"""
    recs = []
    for mp4 in sorted(TRAJ.glob("*.mp4")):
        csvp = mp4.with_suffix(".csv")
        if not csvp.exists():
            continue
        try:
            stamp = datetime.strptime(mp4.stem, "%Y%m%d_%H%M%S").timestamp()
        except ValueError:
            continue
        ms = []
        with open(csvp, encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        ms.append(float(parts[1]))
                    except ValueError:
                        pass
        if len(ms) < FPS * 10:
            continue
        dur = ms[-1] / 1000.0
        # 錄影的時間 0 ＝ 開始截圖那一刻（比檔名時間晚約 1 秒）；用 csv 寫完的時間往回推比較準
        t0 = csvp.stat().st_mtime - dur
        if abs(t0 - stamp) > 10:
            t0 = stamp
        recs.append((t0, dur, mp4, ms))
    return recs


def load_pitches():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM pitches WHERE zone IN ('1','2','3','4','5','6','7','8','9') "
        "AND result IN ('strike','ball') AND pitch_raw IS NOT NULL AND pitch_raw != '' "
        "AND speed_kmh IS NOT NULL ORDER BY id"
    )]
    con.close()
    return rows


def pick(cands, n: int):
    """各球種平均挑，避免全是直球。"""
    by = defaultdict(list)
    for c in cands:
        by[c["p"].get("pitch_group") or c["p"].get("pitch_raw") or "?"].append(c)
    rnd = random.Random(7)
    for v in by.values():
        rnd.shuffle(v)
    out = []
    while len(out) < n and any(by.values()):
        for k in sorted(by, key=lambda k: len(by[k])):
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return sorted(out, key=lambda c: c["p"]["id"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=120)
    ap.add_argument("--all", action="store_true", help="全部的球都切")
    args = ap.parse_args()

    recs = load_recordings()
    if not recs:
        print("data\\trajectory 裡沒有錄影（.mp4 + .csv）")
        return 1
    pitches = load_pitches()
    cands = []
    for p in pitches:
        try:
            t = datetime.fromisoformat(p["ts"]).timestamp()
        except Exception:
            continue
        for t0, dur, mp4, ms in recs:
            rel = t - t0
            if BEFORE <= rel <= dur - AFTER:
                cands.append({"p": p, "mp4": mp4, "rel": rel, "ms": ms})
                break
    print(f"錄影 {len(recs)} 段、資料庫可用的球 {len(pitches)} 顆、落在錄影時間內 {len(cands)} 顆")
    if not cands:
        print("沒有對得上的球（錄影時分析程式要同時開著）")
        return 1
    chosen = cands if args.all else pick(cands, args.max)
    OUT.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()
    x1, y1, x2, y2 = CROP

    index_rows = []
    for i, c in enumerate(chosen, 1):
        p, ms = c["p"], c["ms"]
        # 真實時間 → 影片裡第幾張（掉幀時兩者不同，要用 csv 對）
        a = bisect.bisect_left(ms, (c["rel"] - BEFORE) * 1000.0)
        b = bisect.bisect_right(ms, (c["rel"] + AFTER) * 1000.0)
        name = f"{p['id']:06d}"
        out_mp4 = OUT / f"{name}.mp4"
        if not out_mp4.exists():
            cmd = [ff, "-hide_banner", "-loglevel", "error", "-y",
                   "-ss", f"{a / FPS:.3f}", "-i", str(c["mp4"]), "-frames:v", str(b - a),
                   "-vf", f"crop=iw*{x2 - x1:.2f}:ih*{y2 - y1:.2f}:iw*{x1:.2f}:ih*{y1:.2f}",
                   "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                   str(out_mp4)]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0 or not out_mp4.exists():
                print(f"  #{p['id']} 切失敗：{r.stderr.decode(errors='ignore')[:200]}")
                continue
        with open(OUT / f"{name}.csv", "w", encoding="utf-8") as f:
            f.write("frame,ms\n")
            for k in range(a, b):
                f.write(f"{k - a},{ms[k] - ms[a]:.1f}\n")
        index_rows.append({
            "pitch_id": p["id"], "clip": f"{name}.mp4", "video": c["mp4"].name, "start_frame": a,
            "record_at_s": round(c["rel"] - ms[a] / 1000.0, 3),
            **{k: p.get(k) for k in ("game_id", "ts", "inning", "half", "pitcher", "opponent_tag",
                                     "balls_before", "strikes_before", "zone", "x_norm", "y_norm",
                                     "result", "pitch_raw", "pitch_group", "speed_kmh",
                                     "weather", "wind_dir", "wind_angle", "wind_speed",
                                     "pitcher_attr", "batter_attr", "energy")},
        })
        if i % 10 == 0 or i == len(chosen):
            print(f"  已切 {i}/{len(chosen)}", flush=True)

    with open(OUT / "index.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()) if index_rows else ["pitch_id"])
        w.writeheader()
        w.writerows(index_rows)
    size = sum(f.stat().st_size for f in OUT.glob("*.mp4")) / 1024 / 1024
    print(f"完成：{len(index_rows)} 球 → {OUT}（共 {size:.0f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
