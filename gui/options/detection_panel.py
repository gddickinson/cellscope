"""Detection options panel: cellpose mode, thresholds, retry settings."""
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit, QLabel,
    QPushButton, QFileDialog,
)

from gui.options.params import DetectionParams


DETECTION_MODES = [
    ("default", "Default (cellpose + optical-flow fusion)"),
    ("threshold_retry",
     "Threshold-retry (primary → retry lower ct → interpolate)"),
    ("cascade",
     "Cascade (GT model → original → temporal fill)"),
    ("cpsam",
     "Cellpose-SAM (cpsam, ViT) — needs cellpose4 env"),
    ("hybrid_cpsam",
     "Hybrid cpsam + cellpose fallback — needs cellpose4 env"),
    ("hybrid_cpsam_multi",
     "Hybrid cpsam multi-cell (2-3 cells) — needs cellpose4 env"),
]

FLOW_METHODS = [
    ("farneback", "Farneback (classic, default)"),
    ("farneback_clean", "Farneback on cleaned frames"),
    ("framediff", "Frame differencing (fast-moving cells)"),
]

PREPROCESS_TEMPORAL = [
    ("", "None (skip temporal subtraction)"),
    ("median", "Median background subtract (recommended)"),
    ("mean", "Mean background subtract"),
    ("min", "Min projection subtract"),
    ("max", "Max projection subtract"),
]


def _list_cellpose_models():
    """Scan data/models/ for cellpose model subdirectories."""
    from config import MODEL_DIR
    import os
    if not os.path.isdir(MODEL_DIR):
        return ["cellpose_dic"]
    out = []
    for name in sorted(os.listdir(MODEL_DIR)):
        path = os.path.join(MODEL_DIR, name)
        # Cellpose model is a plain file (or a directory for some variants)
        if name.startswith("cellpose_") and os.path.exists(path):
            out.append(name)
    return out or ["cellpose_dic"]


class DetectionPanel(QWidget):
    """Panel for configuring the detection step."""
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.set_params(DetectionParams())
        self._wire_signals()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # --- Mode + basic thresholds ---
        core = QGroupBox("Mode & thresholds")
        form = QFormLayout()
        self.mode = QComboBox()
        for key, label in DETECTION_MODES:
            self.mode.addItem(label, userData=key)
        self.mode.setToolTip(
            "default: cellpose + flow fusion (single pass)\n"
            "threshold_retry: retry missed frames at lower cellprob_threshold\n"
            "cascade: GT model → original → temporal interpolation"
        )
        form.addRow("Mode:", self.mode)

        self.flow_threshold = QDoubleSpinBox()
        self.flow_threshold.setRange(0.0, 5.0)
        self.flow_threshold.setSingleStep(0.1)
        self.flow_threshold.setDecimals(2)
        self.flow_threshold.setToolTip(
            "Cellpose flow error threshold. 0.0 disables the flow-consistency "
            "filter (best for pre-cropped single-cell frames like Jesse's). "
            "Default 0.0 (was 0.4 before Test 21)."
        )
        form.addRow("flow_threshold:", self.flow_threshold)

        self.cellprob_threshold = QDoubleSpinBox()
        self.cellprob_threshold.setRange(-6.0, 6.0)
        self.cellprob_threshold.setSingleStep(0.5)
        self.cellprob_threshold.setDecimals(2)
        self.cellprob_threshold.setToolTip(
            "Cellpose cellprob threshold. Lower = more permissive, catches "
            "more cells but degrades boundary quality on cKO. Default 0.0."
        )
        form.addRow("cellprob_threshold:", self.cellprob_threshold)

        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(0.0, 500.0)
        self.diameter.setSingleStep(5.0)
        self.diameter.setDecimals(1)
        self.diameter.setToolTip(
            "Cellpose diameter hint in pixels. 0 = auto. Crucial for cells "
            "outside the model's training scale (e.g. Ignasi IC293 needs "
            "diameter=50)."
        )
        form.addRow("diameter (0=auto):", self.diameter)

        self.min_area_px = QSpinBox()
        self.min_area_px.setRange(0, 100000)
        self.min_area_px.setToolTip(
            "Minimum mask area (pixels) to accept as a detection. Smaller "
            "masks are treated as failures and trigger fallback/retry."
        )
        form.addRow("min_area_px:", self.min_area_px)

        cell_row = QHBoxLayout()
        self.expected_cells = QSpinBox()
        self.expected_cells.setRange(0, 20)
        self.expected_cells.setValue(0)
        self.expected_cells.setSpecialValueText("Auto")
        self.expected_cells.setToolTip(
            "Expected number of cells per frame.\n"
            "0 (Auto) = no filtering, keep all detected cells.\n"
            "1 = single-cell pipeline.\n"
            "2+ = keep only the N largest cells, treat rest as debris."
        )
        cell_row.addWidget(self.expected_cells)
        self.btn_scan_cells = QPushButton("Scan")
        self.btn_scan_cells.setToolTip(
            "Auto-detect cell count by sampling ~5 frames with cpsam. "
            "Requires cellpose4 env."
        )
        self.btn_scan_cells.setFixedWidth(50)
        self.btn_scan_cells.clicked.connect(self._on_scan_cells)
        cell_row.addWidget(self.btn_scan_cells)
        form.addRow("expected_cells:", cell_row)

        core.setLayout(form)
        root.addWidget(core)

        # --- Model ---
        model_box = QGroupBox("Model")
        mform = QFormLayout()
        self.model_name = QComboBox()
        self.model_name.setEditable(True)  # user can still type custom
        for m in _list_cellpose_models():
            self.model_name.addItem(m, userData=m)
        self.model_name.setToolTip(
            "Cellpose model under data/models/. Discovered models:\n"
            "  cellpose_dic — default fine-tuned DIC model (best)\n"
            "  cellpose_manual_gt_all120 — GT-retrained (best ctrl boundaries)\n"
            "  cellpose_jesse_2021 — Jesse's model (reference)\n"
            "You can also type a custom name."
        )
        mform.addRow("model_name:", self.model_name)

        row = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("(use model_name)")
        self.model_path.setToolTip(
            "Full path override. Leave blank to use data/models/{model_name}."
        )
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._pick_model_path)
        row.addWidget(self.model_path, 1)
        row.addWidget(btn)
        mform.addRow("model_path:", row)

        model_box.setLayout(mform)
        root.addWidget(model_box)

        # --- Preprocessing (applied BEFORE cellpose) ---
        prep_box = QGroupBox("Preprocessing (applied before cellpose)")
        pform = QFormLayout()

        self.preprocess_temporal = QComboBox()
        for key, label in PREPROCESS_TEMPORAL:
            self.preprocess_temporal.addItem(label, userData=key)
        self.preprocess_temporal.setToolTip(
            "Subtract a per-pixel temporal statistic across the whole "
            "stack. Median is the classic choice — removes stationary "
            "background + illumination drift while preserving moving cells."
        )
        pform.addRow("Temporal bg subtract:", self.preprocess_temporal)

        self.preprocess_spatial_hp = QDoubleSpinBox()
        self.preprocess_spatial_hp.setRange(0.0, 500.0)
        self.preprocess_spatial_hp.setSingleStep(5.0)
        self.preprocess_spatial_hp.setDecimals(1)
        self.preprocess_spatial_hp.setToolTip(
            "Per-frame Gaussian high-pass σ in pixels. Sigma MUST be "
            "larger than a cell radius so the blurred image approximates "
            "background (not the cell). 0 disables. Typical: 40-80 px."
        )
        pform.addRow("Spatial high-pass σ (px):",
                     self.preprocess_spatial_hp)

        self.preprocess_debris_um = QDoubleSpinBox()
        self.preprocess_debris_um.setRange(0.0, 50.0)
        self.preprocess_debris_um.setSingleStep(0.5)
        self.preprocess_debris_um.setDecimals(2)
        self.preprocess_debris_um.setToolTip(
            "Morphological ASF removes bright/dark features smaller "
            "than this size in µm. 0 disables. Requires valid um_per_px "
            "on the recording (or the override below)."
        )
        pform.addRow("Debris filter size (µm):",
                     self.preprocess_debris_um)

        self.preprocess_pixel_size = QDoubleSpinBox()
        self.preprocess_pixel_size.setRange(0.0, 10.0)
        self.preprocess_pixel_size.setSingleStep(0.05)
        self.preprocess_pixel_size.setDecimals(4)
        self.preprocess_pixel_size.setToolTip(
            "Override pixel size (µm/px) used by the debris filter. "
            "0 = use the recording's JSON sidecar value."
        )
        pform.addRow("Pixel size (µm/px, override):",
                     self.preprocess_pixel_size)

        prep_box.setLayout(pform)
        root.addWidget(prep_box)

        # --- Optical flow (default mode) ---
        self.flow_box = QGroupBox("Optical flow (default mode)")
        fform = QFormLayout()
        self.use_smart_fusion = QCheckBox(
            "Use joint quality + per-pixel trust fusion"
        )
        self.use_smart_fusion.setToolTip(
            "Robitaille 2022-style fusion: image-gradient-weighted flow + "
            "per-pixel trust map. Off uses the simpler inside/outside ratio."
        )
        fform.addRow(self.use_smart_fusion)

        self.max_flow_weight = QDoubleSpinBox()
        self.max_flow_weight.setRange(0.0, 2.0)
        self.max_flow_weight.setSingleStep(0.1)
        self.max_flow_weight.setDecimals(2)
        fform.addRow("max_flow_weight:", self.max_flow_weight)

        self.flow_method = QComboBox()
        for key, label in FLOW_METHODS:
            self.flow_method.addItem(label, userData=key)
        self.flow_method.setToolTip(
            "farneback: classic 2-frame flow\n"
            "farneback_clean: on background-subtracted frames\n"
            "framediff: |I(t) - I(t-1)|, better for fast-moving cKO cells"
        )
        fform.addRow("flow_method:", self.flow_method)
        self.flow_box.setLayout(fform)
        root.addWidget(self.flow_box)

        # --- threshold_retry ---
        self.retry_box = QGroupBox("Threshold-retry fallback")
        rform = QFormLayout()
        self.retry_ct = QLineEdit()
        self.retry_ct.setPlaceholderText("-2.0, -4.0")
        self.retry_ct.setToolTip(
            "Comma-separated cellprob_threshold values to retry in order "
            "on frames where the primary pass returned an empty mask."
        )
        rform.addRow("retry ct values:", self.retry_ct)

        self.interpolate_gaps = QCheckBox(
            "Fill remaining gaps with temporal interpolation"
        )
        self.interpolate_gaps.setToolTip(
            "After all retries, any still-empty frames are filled by "
            "flow-warping or morphological interpolation from neighbors."
        )
        rform.addRow(self.interpolate_gaps)
        self.retry_box.setLayout(rform)
        root.addWidget(self.retry_box)

        # --- cascade ---
        self.cascade_box = QGroupBox("Cascade fallback")
        cform = QFormLayout()
        row = QHBoxLayout()
        self.cascade_gt_path = QLineEdit()
        self.cascade_gt_path.setPlaceholderText("(auto-detect)")
        self.cascade_gt_path.setToolTip(
            "GT-retrained model path. Auto-detects "
            "data/models/cellpose_manual_gt_all120 if blank."
        )
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._pick_gt_path)
        row.addWidget(self.cascade_gt_path, 1)
        row.addWidget(btn)
        cform.addRow("GT model:", row)
        self.cascade_box.setLayout(cform)
        root.addWidget(self.cascade_box)

        root.addStretch()

        # Mode drives which sub-boxes are visible
        self.mode.currentIndexChanged.connect(self._update_visibility)

    def _wire_signals(self):
        """Emit `changed` when any widget changes, for log capture."""
        widgets = [
            self.mode, self.flow_threshold, self.cellprob_threshold,
            self.diameter,
            self.min_area_px, self.expected_cells,
            self.model_name, self.model_path,
            self.use_smart_fusion, self.max_flow_weight, self.flow_method,
            self.retry_ct, self.interpolate_gaps, self.cascade_gt_path,
            self.preprocess_temporal, self.preprocess_spatial_hp,
            self.preprocess_debris_um, self.preprocess_pixel_size,
        ]
        for w in widgets:
            sig = (getattr(w, "valueChanged", None)
                   or getattr(w, "currentIndexChanged", None)
                   or getattr(w, "textChanged", None)
                   or getattr(w, "toggled", None))
            if sig is not None:
                sig.connect(self.changed)

    def _update_visibility(self):
        mode = self.mode.currentData()
        self.flow_box.setVisible(mode == "default")
        self.retry_box.setVisible(mode == "threshold_retry")
        self.cascade_box.setVisible(mode == "cascade")

    def _pick_model_path(self):
        path, _ = QFileDialog.getExistingDirectory(
            self, "Pick cellpose model directory"
        ), None
        if path:
            self.model_path.setText(path)

    def _pick_gt_path(self):
        path = QFileDialog.getExistingDirectory(
            self, "Pick GT-retrained model directory"
        )
        if path:
            self.cascade_gt_path.setText(path)

    def _on_scan_cells(self):
        """Auto-detect cell count by scanning sample frames with cpsam."""
        try:
            from core.hybrid_cpsam_multi import scan_cell_count
        except ImportError:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Scan",
                                "scan_cell_count not available. "
                                "Requires cellpose4 env.")
            return
        parent = self.window()
        rec = getattr(parent, "recording", None)
        if rec is None:
            for w in self._find_recording_holders():
                if w is not None:
                    rec = w
                    break
        if rec is None or "frames" not in (rec if isinstance(rec, dict) else {}):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "Scan",
                                    "Load a recording first.")
            return
        from PyQt5.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            count = scan_cell_count(rec["frames"], n_sample=5,
                                    min_area_px=self.min_area_px.value())
            self.expected_cells.setValue(max(1, count))
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Scan Error", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _find_recording_holders(self):
        """Walk up the widget tree looking for a recording dict."""
        w = self.parent()
        while w is not None:
            if hasattr(w, "recording"):
                yield w.recording
            w = w.parent() if hasattr(w, "parent") else None

    # --- Params I/O ---
    def get_params(self) -> DetectionParams:
        # Parse retry list
        try:
            retry = [
                float(x.strip())
                for x in self.retry_ct.text().split(",") if x.strip()
            ]
        except ValueError:
            retry = [-2.0, -4.0]

        temporal = self.preprocess_temporal.currentData() or None
        return DetectionParams(
            mode=self.mode.currentData(),
            flow_threshold=float(self.flow_threshold.value()),
            cellprob_threshold=float(self.cellprob_threshold.value()),
            model_name=(
                self.model_name.currentText().strip() or "cellpose_dic"
            ),
            model_path=(self.model_path.text().strip() or None),
            preprocess_temporal_method=(temporal if temporal else None),
            preprocess_spatial_highpass_sigma=float(
                self.preprocess_spatial_hp.value()
            ),
            preprocess_debris_diameter_um=float(
                self.preprocess_debris_um.value()
            ),
            preprocess_pixel_size_um=float(
                self.preprocess_pixel_size.value()
            ),
            min_area_px=int(self.min_area_px.value()),
            use_smart_fusion=self.use_smart_fusion.isChecked(),
            max_flow_weight=float(self.max_flow_weight.value()),
            flow_method=self.flow_method.currentData(),
            retry_cellprob_thresholds=retry or [-2.0, -4.0],
            interpolate_gaps=self.interpolate_gaps.isChecked(),
            cascade_gt_model_path=(
                self.cascade_gt_path.text().strip() or None
            ),
            diameter=float(self.diameter.value()),
            expected_cells=int(self.expected_cells.value()),
        )

    def set_params(self, p: DetectionParams):
        # Block signals while we update to avoid noisy `changed` bursts
        self.blockSignals(True)
        try:
            idx = max(0, self.mode.findData(p.mode))
            self.mode.setCurrentIndex(idx)
            self.flow_threshold.setValue(p.flow_threshold)
            self.cellprob_threshold.setValue(p.cellprob_threshold)
            self.diameter.setValue(getattr(p, "diameter", 0.0))
            self.min_area_px.setValue(p.min_area_px)
            # model_name is now an editable combo
            idx_m = self.model_name.findData(p.model_name)
            if idx_m >= 0:
                self.model_name.setCurrentIndex(idx_m)
            else:
                self.model_name.setEditText(p.model_name)
            self.model_path.setText(p.model_path or "")
            # Preprocessing
            temporal = getattr(p, "preprocess_temporal_method", None) or ""
            self.preprocess_temporal.setCurrentIndex(
                max(0, self.preprocess_temporal.findData(temporal))
            )
            self.preprocess_spatial_hp.setValue(
                getattr(p, "preprocess_spatial_highpass_sigma", 0.0)
            )
            self.preprocess_debris_um.setValue(
                getattr(p, "preprocess_debris_diameter_um", 0.0)
            )
            self.preprocess_pixel_size.setValue(
                getattr(p, "preprocess_pixel_size_um", 0.0)
            )
            self.use_smart_fusion.setChecked(p.use_smart_fusion)
            self.max_flow_weight.setValue(p.max_flow_weight)
            fidx = max(0, self.flow_method.findData(p.flow_method))
            self.flow_method.setCurrentIndex(fidx)
            self.retry_ct.setText(
                ", ".join(f"{v:g}" for v in p.retry_cellprob_thresholds)
            )
            self.interpolate_gaps.setChecked(p.interpolate_gaps)
            self.cascade_gt_path.setText(p.cascade_gt_model_path or "")
            self.expected_cells.setValue(getattr(p, "expected_cells", 1))
        finally:
            self.blockSignals(False)
        self._update_visibility()
        self.changed.emit()
