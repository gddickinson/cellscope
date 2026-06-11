"""Annotation GUI window — label cell-frames into classes + review/edit.

Left: a filterable table of every cell-frame with its label (click to
jump, recolour by label). Right: a zoomable DIC crop + mask contour with
label buttons + r/s/u keys; labelling auto-advances and auto-saves.

Built on the IC295 state_labels CSV schema (see scripts/ic295_label_states
.py) so labels round-trip with the CLI `train` step.
"""
from __future__ import annotations

import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QComboBox, QPushButton, QLabel, QFileDialog,
    QShortcut, QAbstractItemView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402

from gui_annotate.annotate_data import (  # noqa: E402
    LabelStore, CropRenderer, CLASS_LABELS)

_LABEL_COLOR = {"r": QColor(255, 208, 208), "s": QColor(206, 236, 212),
                "u": QColor(232, 232, 232), "": QColor(255, 255, 255)}
_NAME_TO_CODE = {v: k for k, v in CLASS_LABELS.items()}


class AnnotateWindow(QMainWindow):
    def __init__(self, csv_path=None):
        super().__init__()
        self.setWindowTitle("CellScope — Annotation")
        self.resize(1180, 720)
        self.store = LabelStore()
        self.renderer = None
        self.size = 256
        self.view = []        # row indices, in display order
        self.cur = -1
        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        if csv_path and os.path.exists(csv_path):
            self._load_csv(csv_path)

    # ---------- UI ----------
    def _build_ui(self):
        split = QSplitter(Qt.Horizontal)
        left = QWidget(); lyt = QVBoxLayout(left)
        frow = QHBoxLayout()
        self.filter_label = QComboBox()
        self.filter_label.addItems(["All", "Unlabelled", "rounded",
                                    "spread", "unsure"])
        self.filter_label.currentTextChanged.connect(self._rebuild_view)
        self.filter_cond = QComboBox()
        self.filter_cond.addItem("All conditions")
        self.filter_cond.currentTextChanged.connect(self._rebuild_view)
        frow.addWidget(QLabel("Show:")); frow.addWidget(self.filter_label)
        frow.addWidget(self.filter_cond); frow.addStretch()
        lyt.addLayout(frow)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["idx", "cond", "cid", "frame", "rel_area", "label"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_table_select)
        lyt.addWidget(self.table, 1)
        self.progress = QLabel("Open a labels.csv (File ▸ Open).")
        lyt.addWidget(self.progress)
        split.addWidget(left)

        right = QWidget(); rl = QVBoxLayout(right)
        self.fig = Figure(figsize=(5, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111); self.ax.axis("off")
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        rl.addWidget(self.canvas, 1)
        self.info = QLabel("—"); self.info.setWordWrap(True)
        rl.addWidget(self.info)
        brow = QHBoxLayout()
        for code, name in CLASS_LABELS.items():
            b = QPushButton(f"{name}  ({code})")
            b.clicked.connect(lambda _, c=code: self.set_label(c))
            brow.addWidget(b)
        rl.addLayout(brow)
        nrow = QHBoxLayout()
        for txt, fn in (("◀ Prev", lambda: self._step(-1)),
                        ("Next ▶", lambda: self._step(1)),
                        ("Next unlabelled ⏭", self._next_unlabelled)):
            b = QPushButton(txt); b.clicked.connect(fn); nrow.addWidget(b)
        nrow.addStretch()
        rl.addLayout(nrow)
        split.addWidget(right)
        split.setSizes([440, 740])
        self.setCentralWidget(split)

    def _build_menu(self):
        m = self.menuBar().addMenu("File")
        m.addAction("Open labels.csv…").triggered.connect(self._on_open)
        a_save = m.addAction("Save"); a_save.setShortcut("Ctrl+S")
        a_save.triggered.connect(self._on_save)
        m.addAction("Save As…").triggered.connect(self._on_save_as)
        # Help menu: keyboard-shortcuts popup + docs link.
        from gui.help_menu import install_help_menu
        install_help_menu(self, "CellScope — Annotation", [
            ("Labelling", [
                ("R", "Label as rounded"),
                ("S", "Label as spread"),
                ("U", "Label as unsure"),
            ]),
            ("Navigation", [
                ("→  /  ←", "Next / previous cell-frame"),
            ]),
            ("File", [("Ctrl+S", "Save labels")]),
            ("Help", [("F1  /  ?", "Show this shortcuts list")]),
        ])

    def _build_shortcuts(self):
        for code in CLASS_LABELS:
            QShortcut(QKeySequence(code), self,
                      activated=lambda c=code: self.set_label(c))
        QShortcut(QKeySequence(Qt.Key_Right), self,
                  activated=lambda: self._step(1))
        QShortcut(QKeySequence(Qt.Key_Left), self,
                  activated=lambda: self._step(-1))

    # ---------- data ----------
    def _load_csv(self, path):
        self.store.load(path)
        root = self.store.recordings_root()
        self.renderer = CropRenderer(root) if root else None
        if self.renderer:
            self.size = self.renderer.fixed_size(self.store.rows)
        conds = sorted({r.get("condition", "") for r in self.store.rows} - {""})
        self.filter_cond.blockSignals(True)
        self.filter_cond.clear(); self.filter_cond.addItem("All conditions")
        self.filter_cond.addItems(conds)
        self.filter_cond.blockSignals(False)
        self.setWindowTitle(f"CellScope — Annotation — {os.path.basename(path)}")
        self._rebuild_view()

    def _rebuild_view(self):
        flt = self.filter_label.currentText()
        cond = self.filter_cond.currentText()
        self.view = []
        for i, r in enumerate(self.store.rows):
            if cond != "All conditions" and r.get("condition") != cond:
                continue
            lab = r.get("label", "")
            if flt == "Unlabelled" and lab:
                continue
            if flt in _NAME_TO_CODE and lab != _NAME_TO_CODE[flt]:
                continue
            self.view.append(i)
        # Order by recording so consecutive crops reuse the loaded TIFF
        # (avoids a multi-second reload on every keypress).
        self.view.sort(key=lambda i: (
            self.store.rows[i].get("condition", ""),
            self.store.rows[i].get("label_dir", ""),
            int(self.store.rows[i].get("frame", 0) or 0)))
        self._fill_table()
        self.cur = 0 if self.view else -1
        self._show()
        self._update_progress()

    def _fill_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.view))
        for row, i in enumerate(self.view):
            r = self.store.rows[i]
            ra = float(r.get("rel_area", 0) or 0)
            vals = [r.get("idx"), r.get("condition"), r.get("cid"),
                    r.get("frame"), f"{ra:.2f}", r.get("label", "")]
            for col, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setBackground(_LABEL_COLOR.get(r.get("label", ""),
                                                  _LABEL_COLOR[""]))
                self.table.setItem(row, col, it)
        self.table.blockSignals(False)

    # ---------- display ----------
    def _show(self):
        self.ax.clear(); self.ax.axis("off")
        if not (0 <= self.cur < len(self.view)):
            self.info.setText("—"); self.canvas.draw_idle(); return
        r = self.store.rows[self.view[self.cur]]
        imgw = valid = mw = None
        edge = False
        if self.renderer is not None:
            imgw, valid, mw, edge = self.renderer.render(r, self.size)
        if imgw is not None:
            vals = imgw[valid]
            lo, hi = (np.percentile(vals, [2, 98]) if vals.size else (0, 1))
            self.ax.imshow(imgw, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
            from skimage import measure
            # amber outline + banner when the cell is cut by the image edge:
            # its shape/state is excluded from analysis (your label is still
            # kept — see CLAUDE.md edge policy).
            ccol = "#f0a01e" if edge else "#ff3030"
            for ct in measure.find_contours(mw.astype(float), 0.5):
                self.ax.plot(ct[:, 1], ct[:, 0], "-", color=ccol, lw=1.4)
            if edge:
                self.ax.set_title("⚠ edge-truncated — shape excluded "
                                  "from analysis", color="#c8801a",
                                  fontsize=9)
            self.ax.set_xlim(0, self.size); self.ax.set_ylim(self.size, 0)
        else:
            self.ax.text(0.5, 0.5, "(recordings not found — can't render)",
                         ha="center", va="center", transform=self.ax.transAxes)
        lab = CLASS_LABELS.get(r.get("label", ""), "unlabelled")
        self.info.setText(
            f"{self.cur + 1}/{len(self.view)}    "
            f"{r.get('condition')}/{r.get('label_dir')}  cell {r.get('cid')} "
            f"frame {r.get('frame')}    rel_area="
            f"{float(r.get('rel_area', 0) or 0):.2f}    "
            f"label: {lab}    (scroll = zoom)")
        self.table.blockSignals(True)
        self.table.selectRow(self.cur)
        self.table.blockSignals(False)
        self.canvas.draw_idle()
        self._prefetch_ahead()

    def _prefetch_ahead(self):
        """Warm the next crop's recording + the next 1-2 DISTINCT upcoming
        recordings on background threads, so advancing is instant."""
        if self.renderer is None:
            return
        seen = set()
        for k in range(self.cur + 1, min(self.cur + 18, len(self.view))):
            r = self.store.rows[self.view[k]]
            key = (r.get("condition"), r.get("label_dir"))
            if key in seen:
                continue
            seen.add(key)
            self.renderer.prefetch(r)
            if len(seen) >= 3:
                break

    def _on_scroll(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        s = (1 / 1.25) if ev.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim(); y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(ev.xdata - (ev.xdata - x0) * s,
                         ev.xdata + (x1 - ev.xdata) * s)
        self.ax.set_ylim(ev.ydata - (ev.ydata - y0) * s,
                         ev.ydata + (y1 - ev.ydata) * s)
        self.canvas.draw_idle()

    # ---------- actions ----------
    def set_label(self, code):
        if not (0 <= self.cur < len(self.view)):
            return
        self.store.set_label(self.view[self.cur], code)
        if self.store.path:
            self.store.save()                       # auto-save (cheap)
        self._update_progress()
        if self.filter_label.currentText() == "Unlabelled":
            # labelled row leaves this view — drop just that row (no full
            # rebuild), keep position so we land on the next crop.
            self.view.pop(self.cur)
            self.table.removeRow(self.cur)
            if self.cur >= len(self.view):
                self.cur = len(self.view) - 1
            self._show()
        else:
            self._recolour_row(self.cur)            # update one row, not all
            self._step(1)

    def _recolour_row(self, table_row):
        if not (0 <= table_row < len(self.view)):
            return
        r = self.store.rows[self.view[table_row]]
        col = _LABEL_COLOR.get(r.get("label", ""), _LABEL_COLOR[""])
        self.table.blockSignals(True)
        if self.table.item(table_row, 5):
            self.table.item(table_row, 5).setText(r.get("label", ""))
        for c in range(self.table.columnCount()):
            it = self.table.item(table_row, c)
            if it:
                it.setBackground(col)
        self.table.blockSignals(False)

    def _step(self, d):
        if not self.view:
            return
        self.cur = max(0, min(len(self.view) - 1, self.cur + d))
        self._show()

    def _next_unlabelled(self):
        if not self.view:
            return
        for j in range(1, len(self.view) + 1):
            k = (self.cur + j) % len(self.view)
            if not self.store.rows[self.view[k]].get("label"):
                self.cur = k; self._show(); return
        self.info.setText("no unlabelled crops in this view")

    def _on_table_select(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.cur = rows[0].row(); self._show()

    def _update_progress(self):
        c = self.store.counts(); n = len(self.store.rows)
        done = c.get("r", 0) + c.get("s", 0) + c.get("u", 0)
        self.progress.setText(
            f"labelled {done}/{n}    rounded:{c.get('r', 0)}  "
            f"spread:{c.get('s', 0)}  unsure:{c.get('u', 0)}")

    # ---------- file ----------
    def _on_open(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open labels.csv", "",
                                           "CSV (*.csv)")
        if p:
            self._load_csv(p)

    def _on_save(self):
        if self.store.path:
            self.store.save()
            self.statusBar().showMessage("saved", 3000)

    def _on_save_as(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save labels.csv", "",
                                           "CSV (*.csv)")
        if p:
            self.store.save(p)
            self.statusBar().showMessage(f"saved {p}", 4000)
