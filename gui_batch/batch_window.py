"""Batch analysis window — process multiple recordings with hybrid_cpsam."""
import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QDoubleSpinBox, QTreeWidget, QTreeWidgetItem, QFileDialog,
    QProgressBar, QStatusBar, QGroupBox, QFormLayout, QMessageBox,
)
from gui.run_log import RunLogger, RunLogWidget
from gui_batch.batch_worker import BatchAnalysisWorker


class BatchWindow(QMainWindow):
    """Batch analysis of multiple recordings grouped by folder."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CellScope — Batch Processing")
        self.resize(1200, 800)
        self.logger = RunLogger()
        self._worker = None
        self._recordings = []
        self._build_ui()
        from gui.drag_drop import setup_drag_drop
        setup_drag_drop(self, self._on_drop_folder,
                        (".tif", ".tiff", ".mp4", ".avi", ".mov"))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Directory pickers
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Input:"))
        self.input_edit = QLineEdit()
        dir_row.addWidget(self.input_edit, stretch=1)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self._pick_input)
        dir_row.addWidget(btn_in)
        dir_row.addWidget(QLabel("Output:"))
        self.output_edit = QLineEdit("results/batch")
        dir_row.addWidget(self.output_edit, stretch=1)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._pick_output)
        dir_row.addWidget(btn_out)
        layout.addLayout(dir_row)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_scan = QPushButton("Scan")
        btn_scan.clicked.connect(self._on_scan)
        btn_row.addWidget(btn_scan)
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
        self.tree.setHeaderLabels(
            ["Group / Recording", "Status", "Speed", "Area"])
        self.tree.setColumnWidth(0, 300)
        splitter.addWidget(self.tree)

        # Right: settings
        settings = QWidget()
        sf = QVBoxLayout(settings)
        sg = QGroupBox("Pipeline Settings")
        form = QFormLayout(sg)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Single Cell (hybrid_cpsam)", "hybrid_cpsam")
        self.mode_combo.addItem("Multi Cell (hybrid_cpsam_multi)",
                                "hybrid_cpsam_multi")
        form.addRow("Mode:", self.mode_combo)
        self.min_area = QSpinBox()
        self.min_area.setRange(50, 10000)
        self.min_area.setValue(500)
        form.addRow("Min area (px):", self.min_area)
        self.use_deepsea = QCheckBox()
        self.use_deepsea.setChecked(True)
        form.addRow("DeepSea refinement:", self.use_deepsea)
        self.use_fallback = QCheckBox()
        self.use_fallback.setChecked(True)
        form.addRow("Fallback detection:", self.use_fallback)
        self.use_gap_fill = QCheckBox()
        self.use_gap_fill.setChecked(True)
        form.addRow("Gap fill:", self.use_gap_fill)
        self.use_vampire = QCheckBox()
        self.use_vampire.setChecked(False)
        self.use_vampire.setToolTip("VAMPIRE shape mode analysis")
        form.addRow("VAMPIRE analysis:", self.use_vampire)
        self.vampire_clusters = QSpinBox()
        self.vampire_clusters.setRange(2, 15)
        self.vampire_clusters.setValue(5)
        form.addRow("Shape clusters:", self.vampire_clusters)
        sf.addWidget(sg)

        # Scale overrides
        scale_grp = QGroupBox("Scale")
        sf2 = QFormLayout(scale_grp)
        self.um_per_px = QDoubleSpinBox()
        self.um_per_px.setRange(0, 100)
        self.um_per_px.setDecimals(3)
        self.um_per_px.setSpecialValueText("Auto")
        sf2.addRow("um/px:", self.um_per_px)
        self.time_interval = QDoubleSpinBox()
        self.time_interval.setRange(0, 1000)
        self.time_interval.setDecimals(2)
        self.time_interval.setSpecialValueText("Auto")
        sf2.addRow("Time (min):", self.time_interval)
        sf.addWidget(scale_grp)
        sf.addStretch()

        splitter.addWidget(settings)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        # Progress + log
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        log_widget = RunLogWidget(self.logger)
        log_widget.setMaximumHeight(150)
        layout.addWidget(log_widget)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Select input directory and click Scan")

    def _on_drop_folder(self, path):
        import os
        d = os.path.dirname(path) if os.path.isfile(path) else path
        self.input_edit.setText(d)
        self._on_scan()

    def _pick_input(self):
        path = QFileDialog.getExistingDirectory(self, "Input directory")
        if path:
            self.input_edit.setText(path)

    def _pick_output(self):
        path = QFileDialog.getExistingDirectory(self, "Output directory")
        if path:
            self.output_edit.setText(path)

    def _on_scan(self):
        input_dir = self.input_edit.text()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Scan", "Select a valid input directory")
            return
        from core.io import find_recordings
        groups = find_recordings(input_dir)
        self.tree.clear()
        self._recordings = []
        for group, paths in sorted(groups.items()):
            parent = QTreeWidgetItem(self.tree, [group, "", "", ""])
            parent.setExpanded(True)
            for p in sorted(paths):
                name = os.path.splitext(os.path.basename(p))[0]
                child = QTreeWidgetItem(parent,
                                        [name, "pending", "", ""])
                self._recordings.append((group, p, child))
        total = len(self._recordings)
        self.count_label.setText(f"{total} recordings in "
                                 f"{len(groups)} groups")
        self.btn_run.setEnabled(total > 0)
        self.status.showMessage(f"Found {total} recordings")

    def _on_run(self):
        if not self._recordings:
            return
        out = self.output_edit.text() or "results/batch"
        params = {
            "mode": self.mode_combo.currentData(),
            "min_area_px": self.min_area.value(),
            "use_deepsea": self.use_deepsea.isChecked(),
            "use_fallback": self.use_fallback.isChecked(),
            "use_gap_fill": self.use_gap_fill.isChecked(),
            "vampire": {
                "enabled": self.use_vampire.isChecked(),
                "n_clusters": self.vampire_clusters.value(),
            },
        }
        recs = [(g, p) for g, p, _ in self._recordings]
        self._worker = BatchAnalysisWorker(recs, params, out)
        self._worker.progress.connect(
            lambda m, p: (self.progress.setValue(p),
                          self.status.showMessage(m)))
        self._worker.recording_done.connect(self._on_rec_done)
        self._worker.log_event.connect(
            lambda k, m: self.logger.log(k, m))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(
            lambda e: QMessageBox.critical(self, "Error", e))
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

    def _on_finished(self, out_dir):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.showMessage(f"Batch complete → {out_dir}")
        self._worker = None
