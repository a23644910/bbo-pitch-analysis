"""
文字辨識（PaddleOCR）改在「獨立的程序」裡跑（2026-11-10k）。

原因：PaddleOCR 推論時會鎖住 Python（GIL），以前放在背景執行緒，主程式（看畫面、判斷落點）
每次讀球種球速都整個卡 2～3 秒，一場卡掉近 3 成時間 → 判定變慢、漏記、擊出被當界外。
獨立程序有自己的 Python，主程式完全不受影響。

用法：RemoteOCREngine() 的介面跟 BBOOCREngine 一樣（read_info_panel、recognize_pitch_label…），
呼叫時會等結果回來（呼叫的是 tracker 的背景執行緒，等待不會卡住主程式）。
- 讀球種球速（recognize_pitch_label）優先處理，其他（名字、ID、局數）排在後面。
- 參數裡的 PitchStore 不能跨程序傳，送過去時換成記號，由子程序自己開一個連到同一個資料庫。
- 子程序掛掉會自動重開。
"""
from __future__ import annotations

import heapq
import itertools
import multiprocessing as mp
import threading
import time
from concurrent.futures import Future
from typing import Any, Dict, Optional

_STORE = "__store__"
_PRIORITY = {"recognize_pitch_label": 0}  # 數字越小越先處理


def _worker_main(req_q, res_q, db_path: Optional[str], engine_spec: str) -> None:
    import os
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "3")
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:
        pass
    import importlib
    mod, cls = engine_spec.split(":")
    engine = getattr(importlib.import_module(mod), cls)()
    store = None

    heap: list = []
    cond = threading.Condition()
    seq = itertools.count()

    def reader() -> None:
        while True:
            item = req_q.get()
            with cond:
                if item is None:
                    heapq.heappush(heap, (-1, next(seq), None))
                else:
                    heapq.heappush(heap, (item[4], next(seq), item))
                cond.notify()

    threading.Thread(target=reader, daemon=True).start()
    while True:
        with cond:
            while not heap:
                cond.wait()
            _, _, item = heapq.heappop(heap)
        if item is None:
            return
        rid, name, args, kwargs, _prio = item
        try:
            if any(v == _STORE for v in kwargs.values()):
                if store is None and db_path:
                    from store.db import PitchStore
                    store = PitchStore(db_path)
                kwargs = {k: (store if v == _STORE else v) for k, v in kwargs.items()}
            out = getattr(engine, name)(*args, **kwargs)
            res_q.put((rid, True, out))
        except Exception as e:  # 讀不到就回傳錯誤，tracker 那邊當作沒讀到
            res_q.put((rid, False, repr(e)))


class RemoteOCREngine:
    """跟 BBOOCREngine 一樣的介面，實際在子程序執行。"""

    def __init__(self, db_path: Optional[str] = None, engine_spec: str = "ocr.engine:BBOOCREngine",
                 timeout: float = 60.0) -> None:
        self._db_path = str(db_path) if db_path else None
        self._spec = engine_spec
        self._timeout = timeout
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._pending: Dict[int, Future] = {}
        self._ids = itertools.count(1)
        self._proc = None
        self._start()

    def _start(self) -> None:
        self._req = self._ctx.Queue()
        self._res = self._ctx.Queue()
        self._proc = self._ctx.Process(target=_worker_main, args=(self._req, self._res, self._db_path, self._spec),
                                       daemon=True, name="bbo-ocr")
        self._proc.start()
        threading.Thread(target=self._collect, args=(self._res,), daemon=True).start()

    def _collect(self, res_q) -> None:
        while True:
            try:
                rid, ok, out = res_q.get()
            except (EOFError, OSError):
                return
            with self._lock:
                f = self._pending.pop(rid, None)
            if f is not None and not f.done():
                if ok:
                    f.set_result(out)
                else:
                    f.set_exception(RuntimeError(out))

    def _ensure_alive(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                return
            print("  [文字辨識] 子程序停止了，重新啟動", flush=True)
            for f in self._pending.values():
                if not f.done():
                    f.set_exception(RuntimeError("OCR 子程序重啟"))
            self._pending.clear()
        self._start()

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self._ensure_alive()
        kwargs = {k: (_STORE if (k == "store" and v is not None) else v) for k, v in kwargs.items()}
        f: Future = Future()
        rid = next(self._ids)
        with self._lock:
            self._pending[rid] = f
        self._req.put((rid, name, args, kwargs, _PRIORITY.get(name, 1)))
        try:
            return f.result(timeout=self._timeout)
        finally:
            with self._lock:
                self._pending.pop(rid, None)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self.call(name, *a, **k)

    def close(self) -> None:
        try:
            self._req.put(None)
        except Exception:
            pass
