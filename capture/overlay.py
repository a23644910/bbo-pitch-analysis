"""
Stage 1 semi-transparent calibration overlay (`capture/overlay.py`):
Draws normalized ROI boxes (strike zone, scoreboard, bottom-right info panel,
right-side TAB panel) over the 《全民打棒球2》 client area so the user can
visually verify and fine-tune calibration ratios.
"""

from __future__ import annotations
import tkinter as tk
from typing import Callable, Optional

from capture.calibration import CalibrationConfig
from capture.window import ClientRect, find_game_window


class CalibrationOverlay:
    """
    A click-through / semi-transparent Tkinter top-level overlay aligned to the BBO2 client area.
    """

    def __init__(
        self,
        config: CalibrationConfig,
        on_save: Optional[Callable[[CalibrationConfig], None]] = None,
    ):
        self.config = config
        self.on_save = on_save
        self.root: Optional[tk.Tk] = None

    def show(self, duration_ms: Optional[int] = None) -> None:
        win = find_game_window()
        if win is not None:
            x, y, w, h = win.left, win.top, win.width, win.height
        else:
            x, y, w, h = 120, 68, 1680, 945

        root = tk.Tk()
        self.root = root
        root.title("全打投球辨識 — 校正框預覽 (按 Esc 關閉 / 按 S 儲存)")
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.78)
        root.configure(bg="black")
        try:
            root.wm_attributes("-transparentcolor", "black")
        except Exception:
            pass

        canvas = tk.Canvas(root, width=w, height=h, bg="black", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        self._draw_boxes(canvas, w, h)

        def _close(_event=None):
            root.destroy()

        def _save(_event=None):
            self.config.save()
            if self.on_save:
                self.on_save(self.config)
            root.destroy()

        root.bind("<Escape>", _close)
        root.bind("<s>", _save)
        root.bind("<S>", _save)

        if duration_ms is not None:
            root.after(duration_ms, _close)

        root.mainloop()

    def _draw_boxes(self, canvas: tk.Canvas, w: int, h: int) -> None:
        canvas.delete("all")
        boxes = [
            ("好球帶外框 (Strike Zone)", self.config.strike_zone, "#00ff88"),
            ("計分板 (Scoreboard)", self.config.scoreboard, "#3d8bff"),
            ("右下資訊欄 (Info Panel)", self.config.info_panel, "#ffb84d"),
            ("TAB 投手球種欄 (TAB Panel)", self.config.tab_panel, "#ff5a5a"),
        ]
        for label, box, color in boxes:
            x1, y1, x2, y2 = box.to_px(w, h)
            canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            canvas.create_text(
                x1 + 6,
                max(14, y1 - 10),
                text=label,
                fill=color,
                anchor="w",
                font=("Microsoft JhengHei", 11, "bold"),
            )

        # Draw 3x3 grid lines inside strike zone
        sx1, sy1, sx2, sy2 = self.config.strike_zone.to_px(w, h)
        for i in (1, 2):
            gx = sx1 + int((sx2 - sx1) * i / 3.0)
            gy = sy1 + int((sy2 - sy1) * i / 3.0)
            canvas.create_line(gx, sy1, gx, sy2, fill="#00ff88", dash=(4, 4))
            canvas.create_line(sx1, gy, sx2, gy, fill="#00ff88", dash=(4, 4))
