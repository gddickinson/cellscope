"""Batch analysis window — process multiple recordings with hybrid_cpsam.

All initial widget values come from core.pipeline_defaults.DEFAULTS so the
batch GUI stays in lockstep with the focused GUI and the pipeline
functions. Edit DEFAULTS, not the .setValue / .setChecked calls here.
"""
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
from core.pipeline_defaults import DEFAULTS as _PD
from core.cell_state import DEFAULT_THRESHOLDS as _STATE_TH


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
        self.min_area.setValue(_PD.min_area_px)
        form.addRow("Min area (px):", self.min_area)
        self.use_deepsea = QCheckBox()
        self.use_deepsea.setChecked(_PD.use_deepsea)
        form.addRow("DeepSea refinement:", self.use_deepsea)
        self.use_fallback = QCheckBox()
        self.use_fallback.setChecked(_PD.use_fallback)
        form.addRow("Fallback detection:", self.use_fallback)
        self.use_gap_fill = QCheckBox()
        self.use_gap_fill.setChecked(_PD.use_gap_fill)
        form.addRow("Gap fill:", self.use_gap_fill)
        self.use_vampire = QCheckBox()
        self.use_vampire.setChecked(_PD.compute_vampire)
        self.use_vampire.setToolTip("VAMPIRE shape mode analysis")
        form.addRow("VAMPIRE analysis:", self.use_vampire)
        # --- Multichannel group ---
        self.multichannel = QCheckBox()
        self.multichannel.setChecked(False)
        self.multichannel.setToolTip(
            "For multichannel TIFFs (e.g. DIC + SiR-actin Cy5):\n"
            "load DIC for detection and fluorescence for Cy5 recovery\n"
            "and per-cell scoring. Multi-cell mode only.\n"
            "Channel indices are 0-based.")
        form.addRow("Multichannel:", self.multichannel)
        self.dic_channel = QSpinBox()
        self.dic_channel.setRange(0, 7); self.dic_channel.setValue(1)
        self.dic_channel.setToolTip("DIC channel index (default 1 for IC295)")
        form.addRow("  DIC channel:", self.dic_channel)
        self.fluo_channel = QSpinBox()
        self.fluo_channel.setRange(0, 7); self.fluo_channel.setValue(0)
        self.fluo_channel.setToolTip("Fluorescence channel index (default 0 for IC295)")
        form.addRow("  Fluo channel:", self.fluo_channel)
        self.use_cy5_recovery = QCheckBox()
        # Default to ON in the batch GUI even though DEFAULTS.use_cy5_recovery
        # is False — batch users typically have multichannel data and this
        # is harmless without it. DEFAULTS.with_cy5_available(True) is the
        # canonical "Cy5-on" state.
        self.use_cy5_recovery.setChecked(
            _PD.with_cy5_available(True).use_cy5_recovery)
        self.use_cy5_recovery.setToolTip(
            "Use Cy5 to recover cells cpsam(DIC) missed.\n"
            "Free if no missed cells exist.")
        form.addRow("  Cy5 recovery:", self.use_cy5_recovery)
        self.cy5_filter_mode = QComboBox()
        self.cy5_filter_mode.addItems([
            "Off", "Conservative", "Conservative_strict",
            "Adaptive", "Adaptive_loose",
            "Multi_metric", "Composite_score", "Consensus",
            "Temporal_stability", "Threshold"])
        # Map DEFAULTS.cy5_filter_mode ("multi_metric") to its first-
        # cap variant in this dropdown ("Multi_metric").
        self.cy5_filter_mode.setCurrentText(
            _PD.cy5_filter_mode[:1].upper() + _PD.cy5_filter_mode[1:])
        self.cy5_filter_mode.setToolTip(
            "Tier-4 false-positive filter (post-detection):\n"
            "  Off: keep all detected tracks\n"
            "  Conservative: drop tracks with NO Cy5 (mean<0.05 AND p95<0.10)\n"
            "  Adaptive: per-recording bimodal cut\n"
            "  Threshold: cut at the value below")
        form.addRow("  Cy5 filter:", self.cy5_filter_mode)
        self.cy5_filter_threshold = QDoubleSpinBox()
        self.cy5_filter_threshold.setRange(0.0, 1.0)
        self.cy5_filter_threshold.setDecimals(3)
        self.cy5_filter_threshold.setSingleStep(0.05)
        self.cy5_filter_threshold.setValue(_PD.cy5_filter_threshold)
        self.cy5_filter_threshold.setToolTip(
            "Manual threshold (only used when filter mode is Threshold).\n"
            "Pilot data: real cells score 0.25-0.59, debris 0.05-0.15.")
        # --- Cell state classification ---
        self.compute_states = QCheckBox()
        self.compute_states.setChecked(_PD.compute_state_classification)
        self.compute_states.setToolTip(
            "Per cell-frame: classify balled (mitotic, rounded) vs\n"
            "attached (spread). Stratify motility (speed, MSD,\n"
            "persistence) by state to remove the dividing-cell\n"
            "composition confound.")
        form.addRow("State classification:", self.compute_states)
        self.state_balled_circ = QDoubleSpinBox()
        self.state_balled_circ.setRange(0.5, 1.0)
        self.state_balled_circ.setSingleStep(0.05)
        self.state_balled_circ.setDecimals(2)
        self.state_balled_circ.setValue(_STATE_TH["balled_circ"])
        form.addRow("  Balled circ ≥:", self.state_balled_circ)
        self.state_balled_solid = QDoubleSpinBox()
        self.state_balled_solid.setRange(0.5, 1.0)
        self.state_balled_solid.setSingleStep(0.02)
        self.state_balled_solid.setDecimals(2)
        self.state_balled_solid.setValue(_STATE_TH["balled_solid"])
        form.addRow("  Balled solid ≥:", self.state_balled_solid)
        self.state_attached_circ = QDoubleSpinBox()
        self.state_attached_circ.setRange(0.0, 1.0)
        self.state_attached_circ.setSingleStep(0.05)
        self.state_attached_circ.setDecimals(2)
        self.state_attached_circ.setValue(_STATE_TH["attached_circ"])
        form.addRow("  Attached circ ≤:", self.state_attached_circ)
        self.state_attached_solid = QDoubleSpinBox()
        self.state_attached_solid.setRange(0.0, 1.0)
        self.state_attached_solid.setSingleStep(0.02)
        self.state_attached_solid.setDecimals(2)
        self.state_attached_solid.setValue(_STATE_TH["attached_solid"])
        form.addRow("  Attached solid ≤:", self.state_attached_solid)
        form.addRow("  Filter threshold:", self.cy5_filter_threshold)
        self.vampire_clusters = QSpinBox()
        self.vampire_clusters.setRange(2, 15)
        self.vampire_clusters.setValue(_PD.vampire_n_clusters)
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
            "multichannel": self.multichannel.isChecked(),
            "dic_channel": self.dic_channel.value(),
            "fluo_channel": self.fluo_channel.value(),
            "use_cy5_recovery": self.use_cy5_recovery.isChecked(),
            "cy5_filter_mode": self.cy5_filter_mode.currentText().lower(),
            "cy5_filter_threshold": self.cy5_filter_threshold.value(),
            "compute_states": self.compute_states.isChecked(),
            "state_thresholds": {
                "balled_circ": self.state_balled_circ.value(),
                "balled_solid": self.state_balled_solid.value(),
                "attached_circ": self.state_attached_circ.value(),
                "attached_solid": self.state_attached_solid.value(),
            },
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
