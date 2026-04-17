"""Batch tracking view with group statistical comparisons."""
import os
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QTreeWidget, QTreeWidgetItem, QFileDialog, QProgressBar,
    QPlainTextEdit, QGroupBox, QFormLayout, QMessageBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gui_tracking.stats_plots import plot_group_boxplot, plot_group_violin


COMPARISON_METRICS = [
    ("mean_speed", "Mean Speed (um/min)"),
    ("persistence", "Persistence"),
    ("mean_area", "Mean Area (um^2)"),
    ("boundary_confidence", "Boundary Confidence"),
    ("n_cells", "Cells per Recording"),
]


class BatchTrackingView(QWidget):
    """Batch analysis with inter-group statistical comparison."""

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
        self._worker = None
        self._recordings = []
        self._group_results = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Directory pickers
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Input:"))
        self.input_edit = QLineEdit()
        dir_row.addWidget(self.input_edit, stretch=1)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(lambda: self._pick_dir(self.input_edit))
        dir_row.addWidget(btn_in)
        layout.addLayout(dir_row)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_scan = QPushButton("Scan")
        btn_scan.clicked.connect(self._on_scan)
        btn_row.addWidget(btn_scan)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Single Cell", "hybrid_cpsam")
        self.mode_combo.addItem("Multi Cell", "hybrid_cpsam_multi")
        btn_row.addWidget(self.mode_combo)

        self.btn_run = QPushButton("Run All")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_run.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)
        self.count_label = QLabel("")
        btn_row.addWidget(self.count_label)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left: recording tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Group / Recording", "Status",
                                   "Speed", "Area"])
        self.tree.setColumnWidth(0, 250)
        splitter.addWidget(self.tree)

        # Right: comparison panel
        right = QWidget()
        rl = QVBoxLayout(right)
        comp_row = QHBoxLayout()
        comp_row.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        for key, label in COMPARISON_METRICS:
            self.metric_combo.addItem(label, key)
        self.metric_combo.currentIndexChanged.connect(self._on_metric)
        comp_row.addWidget(self.metric_combo, stretch=1)

        self.plot_type = QComboBox()
        self.plot_type.addItem("Box Plot", "box")
        self.plot_type.addItem("Violin Plot", "violin")
        self.plot_type.currentIndexChanged.connect(self._on_metric)
        comp_row.addWidget(self.plot_type)
        rl.addLayout(comp_row)

        self.stats_text = QPlainTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(120)
        self.stats_text.setPlaceholderText(
            "Run batch analysis to see group comparisons...")
        rl.addWidget(self.stats_text)

        self.comp_fig = Figure(figsize=(5, 4), dpi=100)
        self.comp_canvas = FigureCanvasQTAgg(self.comp_fig)
        rl.addWidget(self.comp_canvas, stretch=1)

        btn_save = QPushButton("Save Plot")
        btn_save.clicked.connect(self._on_save_plot)
        rl.addWidget(btn_save)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

    def _pick_dir(self, edit):
        path = QFileDialog.getExistingDirectory(self, "Select directory")
        if path:
            edit.setText(path)

    def _on_scan(self):
        input_dir = self.input_edit.text()
        if not input_dir or not os.path.isdir(input_dir):
            return
        from core.io import find_recordings
        groups = find_recordings(input_dir)
        self.tree.clear()
        self._recordings = []
        for group, paths in sorted(groups.items()):
            parent = QTreeWidgetItem(self.tree, [group])
            parent.setExpanded(True)
            for p in sorted(paths):
                name = os.path.splitext(os.path.basename(p))[0]
                child = QTreeWidgetItem(parent, [name, "pending"])
                self._recordings.append((group, p, child))
        self.count_label.setText(
            f"{len(self._recordings)} recordings, "
            f"{len(groups)} groups")
        self.btn_run.setEnabled(len(self._recordings) > 0)

    def _on_run(self):
        if not self._recordings:
            return
        from gui_tracking.batch_worker import TrackingBatchWorker
        recs = [(g, p) for g, p, _ in self._recordings]
        params = {
            "mode": self.mode_combo.currentData(),
            "min_area_px": 500,
        }
        self._worker = TrackingBatchWorker(
            recs, params, "results/tracking_batch")
        self._worker.progress.connect(
            lambda m, p: self.progress.setValue(p))
        self._worker.recording_done.connect(self._on_rec_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(
            lambda e: QMessageBox.critical(self, "Error", e))
        if self.logger:
            self._worker.log_event.connect(
                lambda k, m: self.logger.log(k, m))
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()

    def _on_rec_done(self, group, name, metrics):
        for g, p, item in self._recordings:
            if g == group and os.path.splitext(
                    os.path.basename(p))[0] == name:
                item.setText(1, "done")
                item.setText(2, f"{metrics.get('mean_speed', 0):.2f}")
                item.setText(3, f"{metrics.get('mean_area', 0):.0f}")
                break

    def _on_finished(self, group_results):
        self._group_results = group_results
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._worker = None
        self._on_metric()

    def _on_metric(self, _=None):
        if not self._group_results:
            return
        metric_key = self.metric_combo.currentData()
        metric_label = self.metric_combo.currentText()
        groups = {}
        for grp, recs in self._group_results.items():
            vals = [r.get(metric_key, 0) for r in recs
                    if r.get(metric_key) is not None]
            if vals:
                groups[grp] = vals

        if len(groups) < 2:
            self.stats_text.setPlainText("Need at least 2 groups")
            return

        from core.statistics import group_comparison, format_comparison_text
        result = group_comparison(groups, metric_label)
        self.stats_text.setPlainText(format_comparison_text(result))

        plot_type = self.plot_type.currentData()
        if plot_type == "violin":
            plot_group_violin(self.comp_fig, groups, metric_label, result)
        else:
            plot_group_boxplot(self.comp_fig, groups, metric_label, result)
        self.comp_canvas.draw_idle()

    def _on_save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "comparison.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            self.comp_fig.savefig(path, dpi=150, bbox_inches="tight")
