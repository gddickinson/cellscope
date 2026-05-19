"""Single recording tracking and analysis view."""
import os
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton,
    QLabel, QComboBox, QTableWidget, QTableWidgetItem, QFileDialog,
    QHeaderView, QStatusBar, QMessageBox,
)
from gui_focused.image_viewer import ImageViewer
from gui_focused.analysis_plots import GRAPH_REGISTRY
from core.track_quality import compute_track_quality, quality_color
from core.pipeline_defaults import DEFAULTS as _PD
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class SingleTrackingView(QWidget):
    """Load recording + masks, track cells, analyze per-track."""

    def __init__(self, logger=None):
        super().__init__()
        self.recording = None
        self.masks = None
        self.tracks = None
        self.per_cell_results = None
        self.logger = logger
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        ll = QVBoxLayout(left)
        self.viewer = ImageViewer()
        ll.addWidget(self.viewer, stretch=1)
        btn_row = QHBoxLayout()
        btn_load = QPushButton("Load Recording")
        btn_load.clicked.connect(self._on_load)
        btn_row.addWidget(btn_load)
        btn_masks = QPushButton("Load Masks (.npz)")
        btn_masks.clicked.connect(self._on_load_masks)
        btn_row.addWidget(btn_masks)
        self.btn_track = QPushButton("Track Cells")
        self.btn_track.clicked.connect(self._on_track)
        self.btn_track.setEnabled(False)
        btn_row.addWidget(self.btn_track)
        self.btn_analyze = QPushButton("Analyze")
        self.btn_analyze.clicked.connect(self._on_analyze)
        self.btn_analyze.setEnabled(False)
        btn_row.addWidget(self.btn_analyze)
        btn_row.addStretch()
        ll.addLayout(btn_row)
        self.status_label = QLabel("Load a recording to begin")
        ll.addWidget(self.status_label)
        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("<b>Tracks</b>"))
        self.track_table = QTableWidget(0, 6)
        self.track_table.setHorizontalHeaderLabels(
            ["Track", "Quality", "Frames", "Area", "Speed", "Parent"])
        self.track_table.horizontalHeader().setStretchLastSection(True)
        self.track_table.setMaximumHeight(200)
        self.track_table.setToolTip(
            "Quality combines frames-present, area stability, and "
            "total path. Green = good (≥0.7), amber = ok (≥0.4), "
            "red = poor (<0.4). Click a row to highlight that cell.")
        self.track_table.currentCellChanged.connect(
            lambda row, *_: self._on_track_selected(row))
        rl.addWidget(self.track_table)

        plot_row = QHBoxLayout()
        plot_row.addWidget(QLabel("Plot:"))
        self.graph_combo = QComboBox()
        self.graph_combo.currentTextChanged.connect(self._on_graph)
        plot_row.addWidget(self.graph_combo, stretch=1)
        plot_row.addWidget(QLabel("Cell:"))
        self.cell_combo = QComboBox()
        self.cell_combo.currentIndexChanged.connect(
            lambda _: self._on_graph(self.graph_combo.currentText()))
        plot_row.addWidget(self.cell_combo)
        plot_row.addWidget(QLabel("Gap fill:"))
        self.gap_combo = QComboBox()
        self.gap_combo.setToolTip(
            "Linearly interpolate short NaN gaps in timeseries plots.\n"
            "Filled samples are drawn dotted so they're never confused\n"
            "with measured points. Off by default.")
        for label, val in [("off", 0), ("≤1", 1), ("≤2", 2),
                           ("≤3", 3), ("≤5", 5)]:
            self.gap_combo.addItem(label, val)
        self.gap_combo.currentIndexChanged.connect(
            lambda _: self._on_graph(self.graph_combo.currentText()))
        plot_row.addWidget(self.gap_combo)
        rl.addLayout(plot_row)

        self.plot_fig = Figure(figsize=(5, 4), dpi=100)
        self.plot_canvas = FigureCanvasQTAgg(self.plot_fig)
        rl.addWidget(self.plot_canvas, stretch=1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    def _on_drop_load(self, path):
        from core.io import load_recording
        try:
            self.recording = load_recording(path)
        except Exception as e:
            return
        self.viewer.set_data(self.recording["frames"])
        self.masks = None
        self.tracks = None
        self.per_cell_results = None
        self.track_table.setRowCount(0)
        self.btn_track.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.status_label.setText(
            f"Loaded {len(self.recording['frames'])} frames")

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Recording", "",
            "Video/Image (*.mp4 *.avi *.mov *.tif *.tiff)")
        if not path:
            return
        from core.io import load_recording
        try:
            self.recording = load_recording(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        self.viewer.set_data(self.recording["frames"])
        self.masks = None
        self.tracks = None
        self.per_cell_results = None
        self.track_table.setRowCount(0)
        self.btn_track.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        n = len(self.recording["frames"])
        self.status_label.setText(f"Loaded {n} frames — load masks next")

    def _on_load_masks(self):
        if self.recording is None:
            QMessageBox.information(self, "Masks", "Load a recording first")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Masks", "", "NPZ (*.npz)")
        if not path:
            return
        data = np.load(path)
        key = "labels" if "labels" in data else \
              "masks" if "masks" in data else list(data.keys())[0]
        self.masks = data[key]
        if self.masks.dtype == bool:
            self.masks = self.masks.astype(np.int32)
        self.viewer.update_masks(self.masks)
        self.btn_track.setEnabled(True)
        self.status_label.setText(
            f"Masks loaded ({self.masks.max()} labels) — click Track")

    def _on_track(self):
        if self.masks is None:
            return
        from core.multi_cell import track_all_cells
        # Use per-recording physical-unit thresholds when the recording's
        # scale is known; otherwise fall back to the px-space DEFAULTS.
        um_per_px = (self.recording or {}).get("um_per_px")
        t_min = (self.recording or {}).get("time_interval_min")
        px = _PD.pixel_thresholds(um_per_px=um_per_px,
                                   time_interval_min=t_min)
        self.tracks = track_all_cells(
            self.masks,
            min_area_px=px["min_area_px"],
            max_hop_px=px["max_hop_px"],
            spawn_new_tracks=True,
            min_track_length=_PD.min_track_length)
        self._populate_track_table()
        self.btn_analyze.setEnabled(True)
        self.status_label.setText(f"Found {len(self.tracks)} tracks")

    def _on_analyze(self):
        if not self.tracks or self.recording is None:
            return
        from core.pipeline import analyze_recording
        self.per_cell_results = []
        for tid, t in enumerate(self.tracks):
            r = analyze_recording(self.recording, t["stack"])
            r["cell_id"] = tid + 1
            r["track_info"] = {
                "first_frame": t["first_frame"],
                "frames_tracked": int(
                    t["stack"].any(axis=(1, 2)).sum()),
                "parent_id": t.get("parent_id"),
            }
            self.per_cell_results.append(r)
        self._populate_track_table()
        self._populate_graph_combo()
        self.status_label.setText(
            f"Analyzed {len(self.per_cell_results)} tracks")

    def _populate_track_table(self):
        self.track_table.setRowCount(len(self.tracks))
        n_total = (self.masks.shape[0] if self.masks is not None
                   else max((int(t["stack"].shape[0])
                             for t in self.tracks), default=1))
        for i, t in enumerate(self.tracks):
            active = int(t["stack"].any(axis=(1, 2)).sum())
            ar = (self.per_cell_results[i]
                  if self.per_cell_results
                  and i < len(self.per_cell_results) else None)
            q = compute_track_quality(t, n_total, analysis_result=ar)
            self.track_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            qitem = QTableWidgetItem(
                f"{q['composite']:.2f}  ({q['label']})")
            tip = (f"Composite {q['composite']:.2f} ({q['label']})\n"
                   f"  frames present: {q['frames_active']}/"
                   f"{q['frames_total']} → {q['frames_score']:.2f}\n"
                   f"  area stability: "
                   + (f"{q['area_score']:.2f}" if q['area_score'] is not None
                      else "n/a") + "\n"
                   f"  path: "
                   + (f"{q['path_score']:.2f}" if q['path_score'] is not None
                      else "n/a (analyse to compute)"))
            qitem.setToolTip(tip)
            self.track_table.setItem(i, 1, qitem)
            self.track_table.setItem(i, 2, QTableWidgetItem(str(active)))
            if ar is not None:
                ss = ar.get("shape_summary", {}).get("area_um2", {})
                self.track_table.setItem(
                    i, 3, QTableWidgetItem(
                        f"{ss.get('mean', 0):.0f}"))
                self.track_table.setItem(
                    i, 4, QTableWidgetItem(
                        f"{ar.get('mean_speed', 0):.3f}"))
            # Parent column: display as 1-based track ID (parent_id
            # is stored as a 0-based list index). Show split frame +
            # score in the tooltip.
            parent = t.get("parent_id")
            if parent is None:
                pcell = QTableWidgetItem("-")
            else:
                pcell = QTableWidgetItem(str(parent + 1))
                d_frame = t.get("division_frame")
                d_score = t.get("division_score")
                if d_frame is not None:
                    tip = f"Daughter of Track {parent + 1}"
                    tip += f" — split at F{d_frame}"
                    if d_score is not None:
                        tip += f" (score {d_score:.2f})"
                    pcell.setToolTip(tip)
            self.track_table.setItem(i, 5, pcell)
            color = QColor(*quality_color(q["label"]))
            for c in range(self.track_table.columnCount()):
                cell = self.track_table.item(i, c)
                if cell is not None:
                    cell.setBackground(color)

    def _on_track_selected(self, row):
        if row < 0 or not self.tracks:
            return
        labels = np.zeros(self.masks.shape, dtype=np.int32)
        labels[self.tracks[row]["stack"]] = 1
        self.viewer.update_masks(labels)

    def _populate_graph_combo(self):
        self.graph_combo.clear()
        self.cell_combo.clear()
        if not self.per_cell_results:
            return
        for name, (fn, multi) in GRAPH_REGISTRY.items():
            self.graph_combo.addItem(name)
        self.cell_combo.addItem("All Cells", None)
        for r in self.per_cell_results:
            self.cell_combo.addItem(
                f"Cell {r['cell_id']}", r["cell_id"])

    def _on_graph(self, name):
        if not name or name not in GRAPH_REGISTRY or not self.per_cell_results:
            return
        fn, requires_multi = GRAPH_REGISTRY[name]
        cell_id = self.cell_combo.currentData()
        gap_max = self.gap_combo.currentData() or 0
        if requires_multi:
            fn(self.plot_fig, self.per_cell_results,
               gap_interp_max=gap_max)
        elif cell_id is None:
            fn(self.plot_fig, self.per_cell_results[0],
               gap_interp_max=gap_max)
        else:
            r = next((r for r in self.per_cell_results
                       if r["cell_id"] == cell_id),
                      self.per_cell_results[0])
            fn(self.plot_fig, r, gap_interp_max=gap_max)
        self.plot_canvas.draw_idle()
