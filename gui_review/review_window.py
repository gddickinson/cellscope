"""Single-cell candidate-review GUI — per-frame, 4-up comparison.

For each difficult cell we generated outline candidates (Original / Moderate /
Conservative / SAM2 — see scripts/ic293_gen_candidates.py). The reviewer steps
through ONLY the frames where a candidate meaningfully differs from Original,
sees all four candidates side-by-side on each frame, and picks the best one
PER FRAME (the chosen panel is outlined). The final mask is assembled
frame-by-frame and is Original everywhere the reviewer didn't deliberately
improve it. Picks auto-save to a choices CSV that ic293_apply_choices.py then
assembles into masks.npz.

Keys:  ←/→ next/prev CHANGE frame · PgUp/Dn (n/p) recording ·
       1/2/3/4 pick Original/Moderate/Conservative/SAM2 for THIS frame ·
       click a panel to pick it · A apply current pick to all this cell's
       change frames · F flag for manual brushing · scroll zoom · Home fit.
"""
from __future__ import annotations

import os
from collections import Counter
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
    discover_recordings, discover_candidates, load_manifest, trajectory_bbox,
    ReviewRenderer, ChoiceStore)

_MANUAL_COLOR = QColor(255, 224, 178)        # amber list row = flagged manual
_CHOSEN_COLOR = QColor(214, 245, 214)        # green list row = has a pick
_OK_COLOR = QColor(255, 255, 255)
_SLOT_ORDER = ["original", "moderate", "conservative", "sam2"]
_CAND_COL = {"original": "#39ff14", "moderate": "#19c3ff",
             "conservative": "#ffd11e", "sam2": "#ff4fd8"}
_SEL_BORDER = "#ff2d2d"                       # outline around the chosen panel


class ReviewWindow(QMainWindow):
    def __init__(self, recordings_root, choices_path, manifest_path,
                 exclude=frozenset(), margin=30, min_change=2.0):
        super().__init__()
        self.setWindowTitle("CellScope — Candidate review (per-frame, 4-up)")
        self.resize(1240, 860)
        all_recs = discover_recordings(recordings_root, exclude)
        self.manifest = load_manifest(manifest_path)
        self.choices = ChoiceStore(choices_path)
        self.renderer = ReviewRenderer()
        self.margin = margin
        self.recs = self._review_set(all_recs, min_change)
        self.ri = -1
        self.fi = 0
        self.frames = self.orig_labels = None
        self.cands = {}
        self.cand_order = []
        self.xlim = self.ylim = None
        # only show frames where a candidate differs from Original by
        # >= MIN_CHANGE of the cell area (SAM2 jitters ~5% on every frame, so
        # the floor must exclude that); ALL such frames are shown.
        self.MIN_CHANGE = 0.10
        self.view_frames = [0]
        self.vidx = 0
        self._build_ui()
        self._build_shortcuts()
        if self.recs:
            self._load_recording(0)
        else:
            self.info.setText(f"No candidate recordings under {recordings_root} "
                              "— run ic293_gen_candidates.py first.")

    # ---------- review-set + frame selection ----------
    def _review_set(self, all_recs, min_change):
        out = []
        for r in all_recs:
            if r.get("excluded"):
                continue
            ent = self.manifest.get(r["label"]) or {}
            r["_change"] = float(ent.get("max_abs_change", 0.0))
            r["_px"] = int(ent.get("max_abs_px", 0))
            if ent.get("review"):
                out.append(r)
        out.sort(key=lambda r: -r["_px"])
        return out or [r for r in all_recs if not r.get("excluded")]

    def _cand_entry(self, label):
        return (self.manifest.get(label) or {}).get("candidates", {})

    def _compute_view_frames(self):
        """All frames where a candidate differs from Original by >= MIN_CHANGE
        of the cell area, chronological. Falls back to the single biggest
        change, else frame 0."""
        o = self.orig_labels
        if o is None or not self.cand_order:
            return [0]
        ob = o > 0
        cands = [self.cands[n] > 0 for n in self.cand_order if n != "original"]
        if not cands:
            return [0]
        view = []
        for t in range(ob.shape[0]):
            a = int(ob[t].sum())
            if a and max(int((c[t] ^ ob[t]).sum()) for c in cands) / a >= self.MIN_CHANGE:
                view.append(t)
        if view:
            return view
        allf = [(max(int((c[t] ^ ob[t]).sum()) / max(int(ob[t].sum()), 1)
                     for c in cands), t)
                for t in range(ob.shape[0]) if ob[t].any()]
        return [max(allf)[1]] if allf else [0]

    # ---------- UI ----------
    def _build_ui(self):
        split = QSplitter(Qt.Horizontal)
        left = QWidget(); lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Review set  (✎ manual · ● has a pick)"))
        self.list = QListWidget()
        for r in self.recs:
            self.list.addItem(QListWidgetItem(self._list_text(r)))
        self._recolor_all()
        self.list.currentRowChanged.connect(self._on_list_select)
        lv.addWidget(self.list, 1)
        self.counter = QLabel("")
        lv.addWidget(self.counter)
        split.addWidget(left)

        right = QWidget(); rv = QVBoxLayout(right)
        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.axes = list(self.fig.subplots(2, 2).ravel())   # 4 candidate panels
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        rv.addWidget(self.canvas, 1)
        self.info = QLabel("—"); self.info.setWordWrap(True)
        rv.addWidget(self.info)
        brow = QHBoxLayout()
        self.manual_btn = QPushButton("✎ Manual  (F)")
        self.manual_btn.clicked.connect(self._flag_manual)
        allb = QPushButton("Apply pick → all change-frames  (A)")
        allb.clicked.connect(self._apply_all)
        brow.addWidget(QLabel("click a panel or press 1-4 to pick"))
        brow.addStretch(); brow.addWidget(allb); brow.addWidget(self.manual_btn)
        rv.addLayout(brow)
        split.addWidget(right)
        split.setSizes([300, 940])
        self.setCentralWidget(split)
        from gui.help_menu import install_help_menu
        install_help_menu(self, "CellScope — Candidate review", [
            ("Navigation", [
                ("→ / ←", "Next / previous CHANGE frame (unchanged frames skipped)"),
                ("PageDn / PageUp  (n / p)", "Next / previous recording"),
                ("Scroll", "Zoom"), ("Home", "Fit the cell")]),
            ("Pick (per frame)", [
                ("1 / 2 / 3 / 4", "Original / Moderate / Conservative / SAM2"),
                ("Click a panel", "Pick that candidate for this frame"),
                ("A", "Apply this frame's pick to ALL of the cell's change frames"),
                ("F", "Flag the cell for manual brushing")]),
            ("Help", [("F1 / ?", "Shortcuts")])])

    def _build_shortcuts(self):
        sc = [(Qt.Key_Right, lambda: self._step_frame(1)),
              (Qt.Key_Left, lambda: self._step_frame(-1)),
              (Qt.Key_PageDown, lambda: self._step_rec(1)),
              (Qt.Key_PageUp, lambda: self._step_rec(-1)),
              (Qt.Key_N, lambda: self._step_rec(1)),
              (Qt.Key_P, lambda: self._step_rec(-1)),
              (Qt.Key_F, self._flag_manual),
              (Qt.Key_A, self._apply_all),
              (Qt.Key_Home, self._fit_zoom),
              (Qt.Key_1, lambda: self._pick_idx(0)),
              (Qt.Key_2, lambda: self._pick_idx(1)),
              (Qt.Key_3, lambda: self._pick_idx(2)),
              (Qt.Key_4, lambda: self._pick_idx(3))]
        for key, fn in sc:
            QShortcut(QKeySequence(key), self, activated=fn)

    # ---------- list ----------
    def _list_text(self, r):
        fc = self.choices.frame_choices(r["label"])
        man = "✎" if self.choices.is_manual(r["label"]) else " "
        pick = "●" if fc else " "
        if fc:
            c = Counter(fc.values())
            summ = " ".join(f"{n}×{k[:3]}" for k, n in c.items())
        else:
            summ = "—"
        return f"{man}{pick} {r['label']} ({r['condition']})  [{summ}]"

    def _refresh_item(self, i):
        r = self.recs[i]; it = self.list.item(i)
        if not it:
            return
        it.setText(self._list_text(r))
        if self.choices.is_manual(r["label"]):
            it.setBackground(_MANUAL_COLOR)
        elif self.choices.frame_choices(r["label"]):
            it.setBackground(_CHOSEN_COLOR)
        else:
            it.setBackground(_OK_COLOR)

    def _recolor_all(self):
        for i in range(len(self.recs)):
            self._refresh_item(i)

    # ---------- data / display ----------
    def _load_recording(self, i):
        if not (0 <= i < len(self.recs)):
            return
        self.ri = i
        r = self.recs[i]
        self.frames = self.renderer.get(r)[0]
        self.cands, self.cand_order = {}, []
        for name, path in discover_candidates(r["masks_path"]):
            try:
                self.cands[name] = np.load(path)["labels"]
                self.cand_order.append(name)
            except Exception:
                pass
        self.orig_labels = self.cands.get("original")
        self.view_frames = self._compute_view_frames()
        self.vidx = 0
        self.fi = self.view_frames[0]
        self._fit_zoom(redraw=False)
        self._show_grid()
        self.list.blockSignals(True); self.list.setCurrentRow(i)
        self.list.blockSignals(False)
        for k in (i + 1, i + 2):
            if k < len(self.recs):
                self.renderer.prefetch(self.recs[k])

    def _fit_zoom(self, redraw=True):
        base = self.orig_labels
        if base is None:
            return
        r0, r1, c0, c1 = trajectory_bbox(base, self.margin)
        self.xlim, self.ylim = (c0, c1), (r1, r0)
        if redraw:
            self._show_grid()

    def _show_grid(self):
        from skimage import measure
        if not self.cand_order or self.orig_labels is None:
            self.canvas.draw_idle(); return
        r = self.recs[self.ri]
        T = self.orig_labels.shape[0]
        self.fi = max(0, min(T - 1, self.fi))
        chosen = self.choices.choice(r["label"], self.fi, default="original")
        base = self.frames[self.fi] if self.frames is not None else None
        r0, r1 = int(self.ylim[1]), int(self.ylim[0])
        c0, c1 = int(self.xlim[0]), int(self.xlim[1])
        lo, hi = (0, 1)
        if base is not None:
            win = base[max(0, r0):r1, max(0, c0):c1]
            lo, hi = (np.percentile(win, [2, 98]) if win.size else (0, 1))
        ent = self._cand_entry(r["label"])
        for slot, ax in zip(_SLOT_ORDER, self.axes):
            ax.clear()
            avail = slot in self.cand_order
            if avail and base is not None:
                ax.imshow(base, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
                if slot != "original":
                    for ct in measure.find_contours(
                            (self.orig_labels[self.fi] > 0).astype(float), 0.5):
                        ax.plot(ct[:, 1], ct[:, 0], "-", color="#8a8a8a",
                                lw=0.8, alpha=0.7)
                for ct in measure.find_contours(
                        (self.cands[slot][self.fi] > 0).astype(float), 0.5):
                    ax.plot(ct[:, 1], ct[:, 0], "-", color=_CAND_COL[slot], lw=1.7)
                ax.set_xlim(*self.xlim); ax.set_ylim(*self.ylim)
            else:
                ax.text(0.5, 0.5, f"(no {slot})", ha="center", va="center",
                        transform=ax.transAxes, color="#999")
            ax.set_xticks([]); ax.set_yticks([])
            is_ch = (slot == chosen)
            d = ent.get(slot, {})
            dt = f"  Δ{d['agg_delta_pct']:+.0f}%" if d else ""
            n = _SLOT_ORDER.index(slot) + 1
            ax.set_title(("✓ " if is_ch else "") + f"{n}  {slot}{dt}",
                         fontsize=10, fontweight="bold" if is_ch else "normal",
                         color=(_CAND_COL[slot] if avail else "#999"))
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(_SEL_BORDER if is_ch else "#dddddd")
                sp.set_linewidth(4.0 if is_ch else 0.8)
        self.fig.suptitle(f"{r['label']} ({r['condition']})  —  "
                          f"change {self.vidx+1}/{len(self.view_frames)} "
                          f"(frame {self.fi+1}/{T})", fontsize=11)
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw_idle()
        self._update_info(chosen)

    def _update_info(self, chosen):
        r = self.recs[self.ri]
        present = bool((self.orig_labels[self.fi] > 0).any())
        man = self.choices.is_manual(r["label"])
        bits = [f"cell {self.ri+1}/{len(self.recs)}",
                f"THIS FRAME pick: {chosen.upper()}",
                "● in view" if present else "○ ABSENT"]
        if man:
            bits.append("✎ MANUAL")
        self.info.setText("    ".join(bits) + "    (1-4 / click pick · A=all · "
                          "←/→ change-frame · F manual)")
        self.manual_btn.setText("✎ Unflag manual (F)" if man else "✎ Manual  (F)")
        self.counter.setText(f"frames picked: {self.choices.count_frames()}  ·  "
                             f"cells touched: {self.choices.cells_touched()}"
                             f"/{len(self.recs)}")

    # ---------- navigation ----------
    def _step_frame(self, d):
        if not self.view_frames:
            return
        self.vidx = max(0, min(len(self.view_frames) - 1, self.vidx + d))
        self.fi = self.view_frames[self.vidx]
        self._show_grid()

    def _step_rec(self, d):
        self._load_recording(max(0, min(len(self.recs) - 1, self.ri + d)))

    def _on_list_select(self, row):
        if row != self.ri:
            self._load_recording(row)

    # ---------- picking (per frame) ----------
    def _pick_idx(self, idx):
        if 0 <= idx < len(_SLOT_ORDER):
            self._pick(_SLOT_ORDER[idx])

    def _pick(self, name):
        if name not in self.cand_order:
            return
        r = self.recs[self.ri]
        self.choices.set_choice(r["label"], r["condition"], self.fi, name,
                                datetime.now().isoformat(timespec="seconds"))
        self._refresh_item(self.ri)
        self._show_grid()

    def _apply_all(self):
        """Apply this frame's pick to every change-frame of the cell."""
        if not (0 <= self.ri < len(self.recs)):
            return
        r = self.recs[self.ri]
        name = self.choices.choice(r["label"], self.fi, default="original")
        ts = datetime.now().isoformat(timespec="seconds")
        for f in self.view_frames:
            self.choices.set_choice(r["label"], r["condition"], f, name, ts)
        self._refresh_item(self.ri)
        self._show_grid()

    def _flag_manual(self):
        if not (0 <= self.ri < len(self.recs)):
            return
        r = self.recs[self.ri]
        self.choices.toggle_manual(r["label"], r["condition"],
                                   datetime.now().isoformat(timespec="seconds"))
        self._refresh_item(self.ri)
        self._show_grid()

    def _on_click(self, ev):
        if ev.inaxes in self.axes:
            self._pick(_SLOT_ORDER[self.axes.index(ev.inaxes)])

    def _on_scroll(self, ev):
        if ev.inaxes not in self.axes or ev.xdata is None:
            return
        s = (1 / 1.25) if ev.button == "up" else 1.25
        x0, x1 = self.xlim; y0, y1 = self.ylim
        self.xlim = (ev.xdata - (ev.xdata - x0) * s, ev.xdata + (x1 - ev.xdata) * s)
        self.ylim = (ev.ydata - (ev.ydata - y0) * s, ev.ydata + (y1 - ev.ydata) * s)
        self._show_grid()
