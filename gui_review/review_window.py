"""Single-cell review GUI window.

Left: a list of every recording (flagged ones marked ⚑). Right: a zoomable
DIC view of the cell + its outline. Arrow keys step frames; PageUp/Down (or
n/p) change recording; F (or space) flags the current cell as problematic.
The view zooms to the cell (+padding) by default; scroll to zoom further.
Flags auto-save to a CSV.
"""
from __future__ import annotations

import os
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QPushButton, QLabel, QShortcut)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402

from gui_review.review_data import (  # noqa: E402
    discover_recordings, trajectory_bbox, ReviewRenderer, FlagStore)

_FLAG_COLOR = QColor(255, 214, 214)
_OK_COLOR = QColor(255, 255, 255)


class ReviewWindow(QMainWindow):
    def __init__(self, recordings_root, flags_path, exclude=frozenset(),
                 margin=30):
        super().__init__()
        self.setWindowTitle("CellScope — Single-cell review")
        self.resize(1100, 760)
        self.recs = discover_recordings(recordings_root, exclude)
        self.renderer = ReviewRenderer()
        self.flags = FlagStore(flags_path)
        self.margin = margin
        self.ri = -1                  # recording index
        self.fi = 0                   # frame index
        self.frames = self.labels = None
        self.xlim = self.ylim = None
        self._build_ui()
        self._build_shortcuts()
        if self.recs:
            self._load_recording(0)
        else:
            self.info.setText(f"No recordings found under {recordings_root}")

    # ---------- UI ----------
    def _build_ui(self):
        split = QSplitter(Qt.Horizontal)
        left = QWidget(); lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Recordings  (⚑ = flagged)"))
        self.list = QListWidget()
        for r in self.recs:
            it = QListWidgetItem(self._list_text(r))
            it.setBackground(_FLAG_COLOR if self.flags.is_flagged(r["label"])
                             else _OK_COLOR)
            self.list.addItem(it)
        self.list.currentRowChanged.connect(self._on_list_select)
        lv.addWidget(self.list, 1)
        self.counter = QLabel("")
        lv.addWidget(self.counter)
        split.addWidget(left)

        right = QWidget(); rv = QVBoxLayout(right)
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111); self.ax.axis("off")
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        rv.addWidget(self.canvas, 1)
        self.info = QLabel("—"); self.info.setWordWrap(True)
        rv.addWidget(self.info)
        brow = QHBoxLayout()
        self.flag_btn = QPushButton("⚑ Flag problem  (F)")
        self.flag_btn.clicked.connect(self._flag)
        for txt, fn in (("◀ Frame", lambda: self._step_frame(-1)),
                        ("Frame ▶", lambda: self._step_frame(1)),
                        ("▲ Prev rec", lambda: self._step_rec(-1)),
                        ("Next rec ▼", lambda: self._step_rec(1))):
            b = QPushButton(txt); b.clicked.connect(fn); brow.addWidget(b)
        brow.addStretch(); brow.addWidget(self.flag_btn)
        rv.addLayout(brow)
        split.addWidget(right)
        split.setSizes([320, 780])
        self.setCentralWidget(split)
        from gui.help_menu import install_help_menu
        install_help_menu(self, "CellScope — Single-cell review", [
            ("Navigation", [
                ("→  /  ←", "Next / previous frame"),
                ("PageDn / PageUp  (n / p)", "Next / previous recording"),
                ("Scroll", "Zoom in / out at the cursor"),
                ("Home", "Reset zoom to fit the cell")]),
            ("Review", [("F  /  Space", "Flag / unflag this cell as problematic")]),
            ("Help", [("F1  /  ?", "Show this shortcuts list")])])

    def _build_shortcuts(self):
        sc = [(Qt.Key_Right, lambda: self._step_frame(1)),
              (Qt.Key_Left, lambda: self._step_frame(-1)),
              (Qt.Key_PageDown, lambda: self._step_rec(1)),
              (Qt.Key_PageUp, lambda: self._step_rec(-1)),
              (Qt.Key_N, lambda: self._step_rec(1)),
              (Qt.Key_P, lambda: self._step_rec(-1)),
              (Qt.Key_F, self._flag),
              (Qt.Key_Space, self._flag),
              (Qt.Key_Home, self._fit_zoom)]
        for key, fn in sc:
            QShortcut(QKeySequence(key), self, activated=fn)

    def _list_text(self, r):
        mark = "⚑ " if self.flags.is_flagged(r["label"]) else "   "
        exc = "  [excluded]" if r.get("excluded") else ""
        return f"{mark}{r['label']}  ({r['condition']}){exc}"

    # ---------- data / display ----------
    def _load_recording(self, i):
        if not (0 <= i < len(self.recs)):
            return
        self.ri = i
        self.frames, self.labels = self.renderer.get(self.recs[i])
        self.fi = 0
        self._fit_zoom(redraw=False)
        self._show_frame()
        self.list.blockSignals(True); self.list.setCurrentRow(i)
        self.list.blockSignals(False)
        for k in (i + 1, i + 2):                     # prefetch ahead
            if k < len(self.recs):
                self.renderer.prefetch(self.recs[k])

    def _fit_zoom(self, redraw=True):
        if self.labels is None:
            return
        r0, r1, c0, c1 = trajectory_bbox(self.labels, self.margin)
        self.xlim, self.ylim = (c0, c1), (r1, r0)    # y inverted for images
        if redraw:
            self._show_frame()

    def _show_frame(self):
        self.ax.clear(); self.ax.axis("off")
        if self.labels is None:
            self.canvas.draw_idle(); return
        T = self.labels.shape[0]
        self.fi = max(0, min(T - 1, self.fi))
        mask = self.labels[self.fi] > 0
        base = (self.frames[self.fi] if self.frames is not None
                else mask.astype(np.uint16))
        # contrast from the zoomed region so the cell is visible
        r0, r1 = int(self.ylim[1]), int(self.ylim[0])
        c0, c1 = int(self.xlim[0]), int(self.xlim[1])
        win = base[max(0, r0):r1, max(0, c0):c1]
        lo, hi = (np.percentile(win, [2, 98]) if win.size else (0, 1))
        self.ax.imshow(base, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
        edge = self._touches_edge(mask)
        from skimage import measure
        col = "#f0a01e" if edge else "#39ff14"
        for ct in measure.find_contours(mask.astype(float), 0.5):
            self.ax.plot(ct[:, 1], ct[:, 0], "-", color=col, lw=1.5)
        self.ax.set_xlim(*self.xlim); self.ax.set_ylim(*self.ylim)
        self.canvas.draw_idle()
        self._update_info(edge)

    def _touches_edge(self, mask):
        try:
            from core.edge_filter import mask_touches_edge
            from core.cell_state import DEFAULT_THRESHOLDS
            return bool(mask_touches_edge(
                mask, DEFAULT_THRESHOLDS.get("edge_margin_px", 0)))
        except Exception:
            return False

    def _update_info(self, edge):
        r = self.recs[self.ri]
        T = self.labels.shape[0]
        nid = int(len({int(v) for v in np.unique(self.labels) if v > 0}))
        present = bool((self.labels[self.fi] > 0).any())
        flagged = self.flags.is_flagged(r["label"])
        bits = [f"{self.ri + 1}/{len(self.recs)}  {r['label']} ({r['condition']})",
                f"frame {self.fi + 1}/{T}",
                f"{nid} cell ID(s)",
                "● in view" if present else "○ ABSENT this frame"]
        if edge:
            bits.append("⚠ edge-touch")
        if flagged:
            bits.append("⚑ FLAGGED")
        if r.get("excluded"):
            bits.append("[excluded from analysis]")
        self.info.setText("    ".join(bits) + "    (←/→ frame · PgUp/Dn rec · "
                          "scroll zoom · F flag)")
        self.flag_btn.setText("⚑ Unflag  (F)" if flagged else "⚑ Flag problem  (F)")
        self.counter.setText(f"flagged: {self.flags.count()} / {len(self.recs)}")

    # ---------- navigation / actions ----------
    def _step_frame(self, d):
        if self.labels is None:
            return
        self.fi += d
        self._show_frame()

    def _step_rec(self, d):
        self._load_recording(max(0, min(len(self.recs) - 1, self.ri + d)))

    def _on_list_select(self, row):
        if row != self.ri:
            self._load_recording(row)

    def _flag(self):
        if not (0 <= self.ri < len(self.recs)):
            return
        r = self.recs[self.ri]
        self.flags.toggle(r["label"], r["condition"], self.fi,
                          datetime.now().isoformat(timespec="seconds"))
        it = self.list.item(self.ri)
        if it:
            it.setText(self._list_text(r))
            it.setBackground(_FLAG_COLOR if self.flags.is_flagged(r["label"])
                             else _OK_COLOR)
        self._update_info(self._touches_edge(self.labels[self.fi] > 0))

    def _on_scroll(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        s = (1 / 1.25) if ev.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim(); y0, y1 = self.ax.get_ylim()
        self.xlim = (ev.xdata - (ev.xdata - x0) * s,
                     ev.xdata + (x1 - ev.xdata) * s)
        self.ylim = (ev.ydata - (ev.ydata - y0) * s,
                     ev.ydata + (y1 - ev.ydata) * s)
        self.ax.set_xlim(*self.xlim); self.ax.set_ylim(*self.ylim)
        self.canvas.draw_idle()
