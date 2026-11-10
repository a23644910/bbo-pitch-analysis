"""
配球習慣（2026-11-10n）：除了「整體」「球數情境」，也看打席內的順序——
第幾球、上一球投哪格、上兩球的順序、上一球的結果（好球／壞球…），逐層加權混合；
樣本越多、越具體的習慣權重越高。統計算法與 AI 都會用到。

例：「這位玩家打席第 1 球投 1 號後，第 2 球常投 3 號（5/7 次）」。
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Tuple

Z = [str(i) for i in range(1, 10)]
WEIGHTS = {"ip1": 2.0, "p2": 2.0, "pr": 2.0, "idx": 2.0}   # 11/10 資料回測選的（電腦 857 球：前三 63.0 → 65.0%）
K = 3.0
RES_NAME = {"strike": "好球", "ball": "壞球", "foul": "界外", "in_play": "擊出"}


def bucket(b, s) -> str:
    if b is None or s is None:
        return "x"
    return "2s" if s == 2 else ("behind" if b > s else ("ahead" if s > b else "even"))


def contexts(st: Tuple, b, s):
    idx, p1, p2, p1r, pg = st
    out = [("all", ()), ("bk", (bucket(b, s),))]
    if b is not None and s is not None:
        out.append(("cnt", (b, s)))
    out.append(("idx", (min(idx, 4),)))
    if pg:
        out.append(("p1g", (pg,)))
    if p1:
        out.append(("ip1", (min(idx, 4), p1)))
        out.append(("pr", (p1, p1r)))
        if p2:
            out.append(("p2", (p2, p1)))
    return out


class Habit:
    """逐球更新的玩家配球統計（依時間順序 update，不偷看未來）。"""

    def __init__(self) -> None:
        self.c: Dict[Tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.st: Dict[str, Tuple] = {}

    def state(self, tag: str, game: Any, b, s) -> Tuple:
        g, pa, lastz = self.st.get(tag, (None, [], None))
        if g != game:
            pa, lastz = [], None
        if b == 0 and s == 0:
            pa = []
        idx = len(pa) + 1
        p1 = pa[-1][0] if pa else None
        p1r = pa[-1][1] if pa else None
        p2 = pa[-2][0] if len(pa) >= 2 else None
        return (idx, p1, p2, p1r, lastz)

    def update(self, tag: str, game: Any, b, s, zone: Any, result: Any) -> None:
        zone = str(zone)
        stt = self.state(tag, game, b, s)
        if zone in Z:
            for name, key in contexts(stt, b, s):
                self.c[(tag, name, key)][zone] += 1
        g, pa, lastz = self.st.get(tag, (None, [], None))
        if g != game:
            pa, lastz = [], None
        if b == 0 and s == 0:
            pa = []
        self.st[tag] = (game, pa + [(zone if zone in Z else None, result)], zone if zone in Z else None)

    def update_rows(self, rows: Iterable[Dict[str, Any]]) -> "Habit":
        for r in rows:
            if r.get("opponent_tag"):
                self.update(r["opponent_tag"], r.get("game_id"), r.get("balls_before"), r.get("strikes_before"),
                            r.get("zone"), r.get("result"))
        return self

    def n(self, tag: str) -> int:
        return int(sum(self.c.get((tag, "all", ()), {}).values()))

    def predict(self, tag: str, game: Any, b, s) -> Tuple[Dict[str, float], Optional[str]]:
        """回傳（9 格機率, 最明顯的一個順序習慣說明或 None）。"""
        stt = self.state(tag, game, b, s)
        sc = {z: 1.0 / 18 for z in Z}
        best = None
        for name, key in contexts(stt, b, s):
            d = self.c.get((tag, name, key))
            if not d:
                continue
            n = sum(d.values())
            w = WEIGHTS.get(name, 1.0) * n / (n + K)
            for z in Z:
                sc[z] += w * d.get(z, 0) / n
            if name in ("ip1", "p2", "pr", "idx", "p1g") and n >= 3:
                z0 = max(d, key=d.get)
                p = d[z0] / n
                if p >= 0.5 and (best is None or p > best[3]):
                    best = (name, key, z0, p, int(d[z0]), int(n))
        tot = sum(sc.values())
        return {z: v / tot for z, v in sc.items()}, _explain(best)


def _explain(best) -> Optional[str]:
    if not best:
        return None
    name, key, z, p, c, n = best
    tail = f"→ 常投 {z} 號（{c}/{n} 次）"
    if name == "ip1":
        return f"打席第 {key[0]}{'+' if key[0] >= 4 else ''} 球、上一球投 {key[1]} 號 {tail}"
    if name == "p2":
        return f"連續投 {key[0]}→{key[1]} 號之後 {tail}"
    if name == "pr":
        return f"上一球 {key[0]} 號（{RES_NAME.get(key[1], key[1] or '')}）之後 {tail}"
    if name == "idx":
        return f"打席第 {key[0]}{'+' if key[0] >= 4 else ''} 球 {tail}"
    if name == "p1g":
        return f"上一球投 {key[0]} 號之後 {tail}"
    return None
