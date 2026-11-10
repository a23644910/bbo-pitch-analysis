"""
下一球落點預測 AI（小型多類別邏輯回歸，只用 numpy，CPU 一瞬間就跑完）。

輸入特徵（每一球投出「之前」就知道的東西）：
- 這位玩家以前的落點分布（全部、同球數情境、完全同球數），新玩家就是平均分布
- 這位玩家的「上一球落點 → 下一球落點」習慣
- 球數、出局數、上一球的落點與結果、這個打席第幾球
- 天氣、風（順／逆／側、大小）、投手能量
輸出：9 格各自的機率。

訓練：依時間順序重算每一球當時的特徵（不偷看未來），用全部歷史資料訓練。
每場結束重訓一次；只有在「最近的資料」上回測贏過統計算法時才啟用（見 train_and_save）。
"""
from __future__ import annotations
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

ZONES = [str(i) for i in range(1, 10)]
RESULTS = ["strike", "ball", "foul", "in_play"]
WEATHERS = ["晴", "雨", "陰", "雪"]
MODEL_VERSION = 2  # 2026-11-10n：加入打席內順序習慣特徵


def _bucket(b, s) -> str:
    if b is None or s is None:
        return "x"
    return "2s" if s == 2 else ("behind" if b > s else ("ahead" if s > b else "even"))


def _wind_kind(wind_dir: Optional[str]) -> str:
    w = wind_dir or ""
    for k in ("順風", "逆風", "側風"):
        if k in w:
            return k
    return ""


class _PlayerState:
    """一位玩家到目前為止的統計（逐球更新）。"""

    def __init__(self) -> None:
        self.all = np.zeros(9)
        self.bucket: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(9))
        self.count: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(9))
        self.trans: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(9))
        self.prev: Optional[Dict[str, Any]] = None  # 上一球（同一場才算）
        self.pa_index = 0

    @staticmethod
    def _dist(v: np.ndarray, k: float = 3.0) -> np.ndarray:
        return (v + k / 9.0) / (v.sum() + k)


class _States(defaultdict):
    """各玩家統計＋配球習慣（predict.habit.Habit）。"""

    def __init__(self) -> None:
        super().__init__(_PlayerState)
        from predict.habit import Habit
        self.habit = Habit()


def _features(st: _PlayerState, p: Dict[str, Any], habit=None) -> np.ndarray:
    b, s = p.get("balls_before"), p.get("strikes_before")
    f: List[float] = []
    # 配球習慣（打席內順序）綜合機率＋這是打席第幾球
    if habit is not None:
        hp, _ = habit.predict(p.get("opponent_tag") or "", p.get("game_id"), b, s)
        f += [hp[z] for z in ZONES]
        idx = habit.state(p.get("opponent_tag") or "", p.get("game_id"), b, s)[0]
    else:
        f += [1 / 9] * 9
        idx = 1
    f += [1.0 if min(idx, 4) == i else 0.0 for i in (1, 2, 3, 4)]
    f += list(st._dist(st.all))
    f += list(st._dist(st.bucket[_bucket(b, s)]))
    f += list(st._dist(st.count[(b, s)]))
    prev = st.prev if (st.prev and st.prev.get("game_id") == p.get("game_id")) else None
    pz = str(prev["zone"]) if prev else None
    f += list(st._dist(st.trans[pz]) if pz else np.full(9, 1 / 9))
    f += [1.0 if pz == z else 0.0 for z in ZONES]
    f += [1.0 if prev and prev.get("result") == r else 0.0 for r in RESULTS]
    f += [1.0 if b == i else 0.0 for i in range(4)]
    f += [1.0 if s == i else 0.0 for i in range(3)]
    f += [1.0 if p.get("outs") == i else 0.0 for i in range(3)]
    new_pa = prev is None or (b == 0 and s == 0)
    f += [min(st.pa_index if not new_pa else 0, 8) / 8.0]
    f += [1.0 if p.get("weather") == w else 0.0 for w in WEATHERS]
    wk = _wind_kind(p.get("wind_dir"))
    ws = p.get("wind_speed")
    f += [1.0 if wk == k else 0.0 for k in ("順風", "逆風", "側風")]
    f += [min(float(ws), 10.0) / 10.0 if ws is not None else 0.0]
    en = p.get("energy")
    f += [float(en) if en is not None else 0.0, 1.0 if en is None else 0.0]
    f += [math.log1p(st.all.sum()) / 6.0]  # 這位玩家累積了多少資料
    f += [1.0]  # 常數項
    return np.asarray(f, dtype=np.float64)


def _update(st: _PlayerState, p: Dict[str, Any]) -> None:
    z = str(p["zone"])
    if z not in ZONES:
        return
    i = ZONES.index(z)
    b, s = p.get("balls_before"), p.get("strikes_before")
    prev = st.prev if (st.prev and st.prev.get("game_id") == p.get("game_id")) else None
    st.all[i] += 1
    st.bucket[_bucket(b, s)][i] += 1
    st.count[(b, s)][i] += 1
    if prev:
        st.trans[str(prev["zone"])][i] += 1
    st.pa_index = 0 if (b == 0 and s == 0) else st.pa_index + 1
    st.prev = p


def build_dataset(rows: Iterable[Dict[str, Any]]):
    """依時間順序（id 由小到大）重算每一球當時的特徵。回傳 X, y, 對應的列。"""
    states = _States()
    X, y, used = [], [], []
    for p in rows:
        tag = p.get("opponent_tag") or ""
        z = str(p.get("zone"))
        if not tag:
            continue
        if z not in ZONES:
            states.habit.update(tag, p.get("game_id"), p.get("balls_before"), p.get("strikes_before"), z, p.get("result"))
            continue
        st = states[tag]
        X.append(_features(st, p, states.habit))
        y.append(ZONES.index(z))
        used.append(p)
        _update(st, p)
        states.habit.update(tag, p.get("game_id"), p.get("balls_before"), p.get("strikes_before"), z, p.get("result"))
    if not X:
        return np.zeros((0, 1)), np.zeros(0, dtype=int), [], states
    return np.vstack(X), np.asarray(y), used, states


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 400, lr: float = 0.5) -> np.ndarray:
    n, d = X.shape
    W = np.zeros((d, 9))
    Y = np.eye(9)[y]
    for _ in range(iters):
        P = _softmax(X @ W)
        g = X.T @ (P - Y) / n + l2 * W
        W -= lr * g
    return W


def topk_hits(P: np.ndarray, y: np.ndarray, k: int) -> float:
    top = np.argsort(-P, axis=1)[:, :k]
    return float(np.mean([y[i] in top[i] for i in range(len(y))])) if len(y) else 0.0


def stats_baseline(used: List[Dict[str, Any]], test_from: int) -> np.ndarray:
    """統計算法（store.predict 的同一套：整體／球數＋打席內順序習慣）在測試區間的機率，用來比較。"""
    from predict.habit import Habit
    h = Habit()
    out = []
    for idx, p in enumerate(used):
        if idx >= test_from:
            pr, _ = h.predict(p["opponent_tag"], p.get("game_id"), p.get("balls_before"), p.get("strikes_before"))
            out.append(np.array([pr[z] for z in ZONES]))
        h.update(p["opponent_tag"], p.get("game_id"), p.get("balls_before"), p.get("strikes_before"), p.get("zone"), p.get("result"))
    return np.vstack(out) if out else np.zeros((0, 9))


def _old_stats_baseline(used: List[Dict[str, Any]], test_from: int) -> np.ndarray:
    states: Dict[str, _PlayerState] = defaultdict(_PlayerState)
    out = []
    k = 6.0
    for idx, p in enumerate(used):
        st = states[p["opponent_tag"]]
        if idx >= test_from:
            b, s = p.get("balls_before"), p.get("strikes_before")
            sc = np.full(9, 1 / 18)
            for v in (st.all, st.bucket[_bucket(b, s)], st.count[(b, s)]):
                n = v.sum()
                if n:
                    sc += n / (n + k) * v / n
            prev = st.prev if (st.prev and st.prev.get("game_id") == p.get("game_id")) else None
            if prev:
                v = st.trans[str(prev["zone"])]
                n = v.sum()
                if n:
                    sc += n / (n + k) * v / n
            out.append(sc / sc.sum())
        _update(st, p)
    return np.vstack(out) if out else np.zeros((0, 9))


def evaluate(rows: List[Dict[str, Any]], test_frac: float = 0.25) -> Dict[str, Any]:
    """時間序回測：前 75% 訓練、後 25% 測試，跟統計算法比。"""
    X, y, used, _ = build_dataset(rows)
    n = len(y)
    if n < 200:
        return {"n": n, "enough": False}
    cut = int(n * (1 - test_frac))
    W = fit(X[:cut], y[:cut])
    P = _softmax(X[cut:] @ W)
    B = stats_baseline(used, cut)
    yt = y[cut:]
    return {
        "n": n, "enough": True, "test_n": int(len(yt)),
        "ai_top1": round(topk_hits(P, yt, 1) * 100, 1), "ai_top3": round(topk_hits(P, yt, 3) * 100, 1),
        "stats_top1": round(topk_hits(B, yt, 1) * 100, 1), "stats_top3": round(topk_hits(B, yt, 3) * 100, 1),
    }


class ZoneModel:
    """已訓練的模型＋每位玩家的即時統計，提供 predict()。"""

    def __init__(self, W: np.ndarray, meta: Dict[str, Any]) -> None:
        self.W = W
        self.meta = meta
        self.states = _States()

    @property
    def active(self) -> bool:
        return bool(self.meta.get("active"))

    def warm(self, rows: Iterable[Dict[str, Any]]) -> None:
        self.states = player_states(rows)

    def predict(self, ctx: Dict[str, Any], states=None) -> np.ndarray:
        S = states if states is not None else self.states
        st = S[ctx.get("opponent_tag") or ""]
        x = _features(st, ctx, getattr(S, "habit", None))[None, :]
        return _softmax(x @ self.W)[0]


def player_states(rows: Iterable[Dict[str, Any]]) -> "_States":
    states = _States()
    for p in rows:
        tag = p.get("opponent_tag")
        if not tag:
            continue
        if str(p.get("zone")) in ZONES:
            _update(states[tag], p)
        states.habit.update(tag, p.get("game_id"), p.get("balls_before"), p.get("strikes_before"), p.get("zone"), p.get("result"))
    return states


def train_and_save(rows: List[Dict[str, Any]], path: Path) -> Dict[str, Any]:
    """回測＋用全部資料重訓並存檔。AI 在回測贏過統計算法（前三名命中率）才啟用。"""
    ev = evaluate(rows)
    meta: Dict[str, Any] = {"version": MODEL_VERSION, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), **ev}
    X, y, _, _ = build_dataset(rows)
    if len(y) < 200:
        meta["active"] = False
        meta["reason"] = f"資料還不夠（{len(y)} 球，至少要 200 球）"
    else:
        W = fit(X, y)
        meta["active"] = (ev.get("ai_top1", 0) + ev.get("ai_top3", 0)) > (ev.get("stats_top1", 0) + ev.get("stats_top3", 0))
        meta["reason"] = "AI 回測贏過統計算法" if meta["active"] else "AI 回測沒有贏過統計算法，先用統計"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path.with_suffix(".npy")), W)
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load(path: Path) -> Optional[ZoneModel]:
    try:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("version") != MODEL_VERSION:
            return None
        W = np.load(str(path.with_suffix(".npy")))
        return ZoneModel(W, meta)
    except Exception:
        return None
