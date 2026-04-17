"""Refinement options panel: mode, crop, per-step toggles, step params."""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox,
)

from PyQt5.QtWidgets import QPushButton

from gui.options.params import RefinementParams


REFINEMENT_MODES = [
    ("none", "None (use detection masks as-is)"),
    ("snap_only", "Snap only (legacy, fastest)"),
    ("fixed", "Full stack (RF → snap → Fourier → CRF → temporal)"),
    ("iterative", "Iterative (try 3 presets, pick best)"),
    ("alt_method",
     "Alternative method (Otsu / GMM / Watershed / Morph-grad / …)"),
    ("hybrid",
     "Hybrid RF (classical seed → RF refines boundary; or ensemble)"),
    ("sam",
     "SAM (Segment Anything Model — foundation-model refinement)"),
    ("medsam",
     "MedSAM (biomedical SAM fine-tune — best foundation refiner)"),
]

CROP_MODES = [
    ("none", "None (refine on full frame)"),
    ("global", "Global bbox (union of all masks + padding)"),
    ("per_cell", "Per-cell bbox (each frame cropped independently)"),
]


class RefinementPanel(QWidget):
    """Panel for configuring the refinement step."""
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.set_params(RefinementParams())
        self._wire_signals()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # --- Mode + cropping ---
        top = QGroupBox("Mode & region")
        form = QFormLayout()
        self.mode = QComboBox()
        for key, label in REFINEMENT_MODES:
            self.mode.addItem(label, userData=key)
        self.mode.setToolTip(
            "none / snap_only / fixed / iterative / alt_method / "
            "hybrid / sam — see sub-group details below."
        )
        form.addRow("Mode:", self.mode)

        self.crop_mode = QComboBox()
        for key, label in CROP_MODES:
            self.crop_mode.addItem(label, userData=key)
        self.crop_mode.setToolTip(
            "none:     refine on the full frame (slow).\n"
            "global:   one union bbox across all frames (temporal-safe, "
            "~2-4x faster).\n"
            "per_cell: each frame gets its own tight bbox (best for cells "
            "that move a lot; only works with frame-local methods like "
            "alt_method)."
        )
        form.addRow("Crop mode:", self.crop_mode)

        self.crop_padding_px = QSpinBox()
        self.crop_padding_px.setRange(0, 500)
        self.crop_padding_px.setToolTip(
            "Pixels of padding around the GLOBAL union bbox. Must be "
            "large enough for RF filter banks and CRF context (default 50)."
        )
        form.addRow("Global padding px:", self.crop_padding_px)

        self.per_cell_padding_px = QSpinBox()
        self.per_cell_padding_px.setRange(0, 500)
        self.per_cell_padding_px.setToolTip(
            "Pixels of padding around each PER-CELL bbox (default 30)."
        )
        form.addRow("Per-cell padding px:", self.per_cell_padding_px)

        self.skip_cascade_gt = QCheckBox(
            "Skip refinement on cascade GT-model frames"
        )
        self.skip_cascade_gt.setToolTip(
            "When detection=cascade, frames produced by the GT model "
            "already have superior boundaries. Preserve them during "
            "refinement (temporal smoothing still applies)."
        )
        form.addRow(self.skip_cascade_gt)

        self.pre_refine_medsam = QCheckBox(
            "Pre-refine with MedSAM before main refinement"
        )
        self.pre_refine_medsam.setToolTip(
            "Run MedSAM on the detection masks first to tighten the "
            "boundaries, then pass that output as the seed to the "
            "configured refinement mode below. Useful when detection "
            "gives a rough/oversized seed and you want a foundation-"
            "model boundary plus a finishing refiner (e.g. snap or "
            "Fourier smoothing)."
        )
        form.addRow(self.pre_refine_medsam)

        self.union_with_deepsea = QCheckBox(
            "After MedSAM, union mask with DeepSea segmentation"
        )
        self.union_with_deepsea.setToolTip(
            "Run DeepSea (phase-contrast pretrained) on the same image "
            "and pixel-OR its mask with the MedSAM output, then fill "
            "holes + keep largest component. Recovers boundary pixels "
            "MedSAM under-segments. Verified +0.025 mean IoU on Ignasi "
            "(0.860 → 0.888). Requires DeepSea checkpoint at "
            "data/models/deepsea/segmentation.pth."
        )
        form.addRow(self.union_with_deepsea)

        top.setLayout(form)
        root.addWidget(top)

        # --- Per-step toggles ---
        steps = QGroupBox("Pipeline steps (fixed / iterative)")
        sform = QFormLayout()
        self.use_rf = QCheckBox(
            "RF isoline (texture-aware boundary shift)"
        )
        self.use_snap = QCheckBox(
            "Edge snap (pull contour to gradient maxima)"
        )
        self.use_fourier = QCheckBox(
            "Fourier contour smoothing"
        )
        self.use_crf = QCheckBox(
            "DenseCRF post-processing"
        )
        self.use_temporal = QCheckBox(
            "Temporal boundary smoothing"
        )
        for cb in (self.use_rf, self.use_snap, self.use_fourier,
                   self.use_crf, self.use_temporal):
            sform.addRow(cb)
        steps.setLayout(sform)
        root.addWidget(steps)

        # --- Alt-method (mode=alt_method) ---
        self.alt_box = QGroupBox(
            "Alternative segmentation method (mode=alt_method)"
        )
        alt_form = QFormLayout()
        self.alt_method = QComboBox()
        # Import registry lazily to avoid load-order issues
        from core.alt_segmentation import METHODS
        for key, (_, label) in METHODS.items():
            self.alt_method.addItem(label, userData=key)
        self.alt_method.setToolTip(
            "Classical (non-RF) method applied per-frame on the "
            "cellpose mask. GMM (3) was the strongest non-RF method "
            "(IoU 0.639). Works with per_cell crop."
        )
        alt_form.addRow("Method:", self.alt_method)
        self.alt_box.setLayout(alt_form)
        root.addWidget(self.alt_box)

        # --- Hybrid RF (mode=hybrid) ---
        self.hybrid_box = QGroupBox("Hybrid RF (mode=hybrid)")
        hy_form = QFormLayout()
        self.hybrid_method = QComboBox()
        from core.hybrid_rf import METHODS as HYBRID_METHODS
        for key, (_, label) in HYBRID_METHODS.items():
            self.hybrid_method.addItem(label, userData=key)
        self.hybrid_method.setToolTip(
            "localstd_rf / gmm_rf: classical seed + RF refines the "
            "boundary band. ensemble: averages RF + local_std + GMM "
            "probabilities. Uses the RF model selected in 'Filter bank'."
        )
        hy_form.addRow("Method:", self.hybrid_method)

        self.hybrid_rf_threshold = QDoubleSpinBox()
        self.hybrid_rf_threshold.setRange(0.0, 1.0)
        self.hybrid_rf_threshold.setSingleStep(0.05)
        self.hybrid_rf_threshold.setDecimals(3)
        self.hybrid_rf_threshold.setToolTip(
            "Threshold applied to the RF probability inside the "
            "boundary band (localstd_rf / gmm_rf). 0.5 = midline."
        )
        hy_form.addRow("RF threshold (band):", self.hybrid_rf_threshold)

        self.hybrid_ensemble_threshold = QDoubleSpinBox()
        self.hybrid_ensemble_threshold.setRange(0.0, 1.0)
        self.hybrid_ensemble_threshold.setSingleStep(0.05)
        self.hybrid_ensemble_threshold.setDecimals(3)
        self.hybrid_ensemble_threshold.setToolTip(
            "Threshold for the averaged probability map in 'ensemble' mode."
        )
        hy_form.addRow("Ensemble threshold:", self.hybrid_ensemble_threshold)

        self.hybrid_boundary_dilate = QSpinBox()
        self.hybrid_boundary_dilate.setRange(0, 50)
        self.hybrid_boundary_dilate.setToolTip(
            "Iterations to dilate the seed mask when defining the "
            "boundary band (where RF is trusted)."
        )
        hy_form.addRow("Boundary dilate (px):", self.hybrid_boundary_dilate)

        self.hybrid_boundary_erode = QSpinBox()
        self.hybrid_boundary_erode.setRange(0, 50)
        self.hybrid_boundary_erode.setToolTip(
            "Iterations to erode the seed mask when defining the "
            "boundary band."
        )
        hy_form.addRow("Boundary erode (px):", self.hybrid_boundary_erode)

        self.hybrid_box.setLayout(hy_form)
        root.addWidget(self.hybrid_box)

        # --- SAM (mode=sam) — widgets built in sam_panel.py ---
        from gui.options.sam_panel import build_sam_group
        self.sam_box, sam_widgets = build_sam_group()
        self.sam_model_type = sam_widgets["sam_model_type"]
        self.sam_use_mask_prompt = sam_widgets["sam_use_mask_prompt"]
        root.addWidget(self.sam_box)

        # --- RF filter bank (mode=fixed/iterative with use_rf) ---
        self.bank_box = QGroupBox("RF filter bank (choose trained model)")
        bank_form = QFormLayout()
        self.rf_filter_bank = QComboBox()
        self._populate_filter_banks()
        self.rf_filter_bank.setToolTip(
            "Which RF model to use. 'default' = winning model (IoU "
            "0.826). Others trained by scripts/train_rf_filter_banks.py."
        )
        bank_form.addRow("Filter bank:", self.rf_filter_bank)

        self.btn_refresh_banks = QPushButton("Refresh list")
        self.btn_refresh_banks.setToolTip(
            "Re-scan data/models/ for rf_bank_*.pkl files (useful after "
            "running scripts/train_rf_filter_banks.py)."
        )
        self.btn_refresh_banks.clicked.connect(self._populate_filter_banks)
        bank_form.addRow(self.btn_refresh_banks)
        self.bank_box.setLayout(bank_form)
        root.addWidget(self.bank_box)

        # --- RF params ---
        self.rf_box = QGroupBox("Random Forest isoline")
        rform = QFormLayout()
        self.rf_threshold = QDoubleSpinBox()
        self.rf_threshold.setRange(0.0, 1.0)
        self.rf_threshold.setSingleStep(0.05)
        self.rf_threshold.setDecimals(3)
        self.rf_threshold.setToolTip(
            "Cell-probability level at which the boundary is extracted. "
            "0.5=midline, higher=more conservative (closer to centre)."
        )
        rform.addRow("rf_threshold:", self.rf_threshold)

        self.rf_use_isoline = QCheckBox(
            "Isoline extraction (else use argmax > threshold)"
        )
        rform.addRow(self.rf_use_isoline)
        self.rf_box.setLayout(rform)
        root.addWidget(self.rf_box)

        # --- Snap params ---
        self.snap_box = QGroupBox("Edge snap")
        snform = QFormLayout()
        self.snap_search = QSpinBox()
        self.snap_search.setRange(1, 50)
        self.snap_search.setToolTip(
            "Pixels along the outward normal to search for a gradient peak."
        )
        snform.addRow("search_radius:", self.snap_search)

        self.snap_max_step = QSpinBox()
        self.snap_max_step.setRange(1, 50)
        self.snap_max_step.setToolTip(
            "Maximum allowed displacement per contour point per snap step."
        )
        snform.addRow("max_step:", self.snap_max_step)

        self.snap_smooth = QDoubleSpinBox()
        self.snap_smooth.setRange(0.0, 20.0)
        self.snap_smooth.setSingleStep(0.5)
        self.snap_smooth.setDecimals(2)
        self.snap_smooth.setToolTip(
            "Gaussian smoothing σ applied along the contour to the snap "
            "displacements. Larger = smoother contour."
        )
        snform.addRow("smooth_sigma:", self.snap_smooth)

        self.snap_use_membrane = QCheckBox(
            "Use membrane edge map (gradient × texture)"
        )
        self.snap_use_membrane.setToolTip(
            "Combine image gradient with a smoothed local-std map. "
            "Suppresses internal features (organelles) that would "
            "otherwise attract the snap."
        )
        snform.addRow(self.snap_use_membrane)

        self.snap_alpha = QDoubleSpinBox()
        self.snap_alpha.setRange(0.0, 1.0)
        self.snap_alpha.setSingleStep(0.1)
        self.snap_alpha.setDecimals(2)
        self.snap_alpha.setToolTip(
            "Weight of the texture-boundary signal. 0 = image gradient "
            "only, 1 = texture only."
        )
        snform.addRow("membrane_alpha:", self.snap_alpha)
        self.snap_box.setLayout(snform)
        root.addWidget(self.snap_box)

        # --- Fourier + temporal ---
        self.misc_box = QGroupBox("Fourier & temporal")
        mform = QFormLayout()
        self.fourier_n = QSpinBox()
        self.fourier_n.setRange(3, 200)
        self.fourier_n.setToolTip(
            "Number of Fourier descriptors retained. Lower = smoother "
            "(rounder) contour; higher = preserves fine features."
        )
        mform.addRow("fourier_n_descriptors:", self.fourier_n)

        self.temporal_sigma = QDoubleSpinBox()
        self.temporal_sigma.setRange(0.0, 20.0)
        self.temporal_sigma.setSingleStep(0.5)
        self.temporal_sigma.setDecimals(2)
        self.temporal_sigma.setToolTip(
            "Gaussian σ for temporal smoothing of polar boundary radii. "
            "0 disables it. Default 1.5."
        )
        mform.addRow("temporal_sigma:", self.temporal_sigma)
        self.misc_box.setLayout(mform)
        root.addWidget(self.misc_box)

        root.addStretch()

        # Visibility wiring — sub-boxes grey out when step disabled or
        # mode doesn't use them
        self.mode.currentIndexChanged.connect(self._update_visibility)
        self.crop_mode.currentIndexChanged.connect(self._update_visibility)
        self.use_rf.toggled.connect(self._update_visibility)
        self.use_snap.toggled.connect(self._update_visibility)
        self.use_fourier.toggled.connect(self._update_visibility)
        self.use_temporal.toggled.connect(self._update_visibility)

    def _populate_filter_banks(self):
        """Scan data/models/ for trained RF banks and populate dropdown."""
        from core.boundary_rf import list_available_rf_banks
        current = self.rf_filter_bank.currentData() \
            if self.rf_filter_bank.count() > 0 else "default"
        self.rf_filter_bank.blockSignals(True)
        self.rf_filter_bank.clear()
        for name in list_available_rf_banks():
            label = ("default (winning model)" if name == "default"
                     else f"{name}")
            self.rf_filter_bank.addItem(label, userData=name)
        idx = max(0, self.rf_filter_bank.findData(current))
        self.rf_filter_bank.setCurrentIndex(idx)
        self.rf_filter_bank.blockSignals(False)

    def _wire_signals(self):
        widgets = [
            self.mode, self.crop_mode, self.crop_padding_px,
            self.per_cell_padding_px, self.skip_cascade_gt,
            self.pre_refine_medsam, self.union_with_deepsea,
            self.use_rf, self.use_snap, self.use_fourier,
            self.use_crf, self.use_temporal,
            self.alt_method, self.rf_filter_bank,
            self.hybrid_method, self.hybrid_rf_threshold,
            self.hybrid_ensemble_threshold,
            self.hybrid_boundary_dilate, self.hybrid_boundary_erode,
            self.sam_model_type, self.sam_use_mask_prompt,
            self.rf_threshold, self.rf_use_isoline,
            self.snap_search, self.snap_max_step, self.snap_smooth,
            self.snap_use_membrane, self.snap_alpha,
            self.fourier_n, self.temporal_sigma,
        ]
        for w in widgets:
            sig = (getattr(w, "valueChanged", None)
                   or getattr(w, "currentIndexChanged", None)
                   or getattr(w, "toggled", None)
                   or getattr(w, "textChanged", None))
            if sig is not None:
                sig.connect(self.changed)

    def _update_visibility(self):
        mode = self.mode.currentData()
        stack_modes = mode in ("fixed", "iterative")
        self.rf_box.setEnabled(stack_modes and self.use_rf.isChecked())
        self.bank_box.setEnabled(stack_modes and self.use_rf.isChecked())
        self.snap_box.setEnabled(
            (stack_modes or mode == "snap_only") and self.use_snap.isChecked()
        )
        fourier_relevant = stack_modes or mode == "snap_only"
        self.misc_box.setEnabled(
            fourier_relevant and
            (self.use_fourier.isChecked() or self.use_temporal.isChecked())
        )
        # alt_method / hybrid / sam boxes only when matching mode is chosen
        self.alt_box.setVisible(mode == "alt_method")
        self.hybrid_box.setVisible(mode == "hybrid")
        self.sam_box.setVisible(mode == "sam")
        # Hybrid uses the RF model → enable the filter-bank group too
        if mode == "hybrid":
            self.bank_box.setEnabled(True)
        # per_cell valid for alt_method, hybrid, and sam; warn by disabling elsewhere
        crop = self.crop_mode.currentData()
        if crop == "per_cell" and mode not in ("alt_method", "hybrid", "sam"):
            self.crop_mode.blockSignals(True)
            self.crop_mode.setCurrentIndex(
                max(0, self.crop_mode.findData("global"))
            )
            self.crop_mode.blockSignals(False)
        self.crop_padding_px.setEnabled(
            self.crop_mode.currentData() == "global"
        )
        self.per_cell_padding_px.setEnabled(
            self.crop_mode.currentData() == "per_cell"
        )

    # --- Params I/O ---
    def get_params(self) -> RefinementParams:
        crop_mode = self.crop_mode.currentData()
        return RefinementParams(
            mode=self.mode.currentData(),
            crop_mode=crop_mode,
            use_crop=(crop_mode != "none"),  # kept for back-compat
            crop_padding_px=int(self.crop_padding_px.value()),
            per_cell_padding_px=int(self.per_cell_padding_px.value()),
            alt_method=self.alt_method.currentData(),
            rf_filter_bank=self.rf_filter_bank.currentData(),
            hybrid_method=self.hybrid_method.currentData(),
            hybrid_rf_threshold=float(self.hybrid_rf_threshold.value()),
            hybrid_ensemble_threshold=float(
                self.hybrid_ensemble_threshold.value()
            ),
            hybrid_boundary_dilate=int(self.hybrid_boundary_dilate.value()),
            hybrid_boundary_erode=int(self.hybrid_boundary_erode.value()),
            sam_model_type=self.sam_model_type.currentData() or "vit_b",
            sam_use_mask_prompt=self.sam_use_mask_prompt.isChecked(),
            use_rf=self.use_rf.isChecked(),
            use_snap=self.use_snap.isChecked(),
            use_fourier=self.use_fourier.isChecked(),
            use_crf=self.use_crf.isChecked(),
            use_temporal=self.use_temporal.isChecked(),
            rf_threshold=float(self.rf_threshold.value()),
            rf_use_isoline=self.rf_use_isoline.isChecked(),
            snap_search_radius=int(self.snap_search.value()),
            snap_max_step=int(self.snap_max_step.value()),
            snap_smooth_sigma=float(self.snap_smooth.value()),
            snap_use_membrane_edge=self.snap_use_membrane.isChecked(),
            snap_membrane_alpha=float(self.snap_alpha.value()),
            fourier_n_descriptors=int(self.fourier_n.value()),
            temporal_sigma=float(self.temporal_sigma.value()),
            skip_cascade_gt_frames=self.skip_cascade_gt.isChecked(),
            pre_refine_medsam=self.pre_refine_medsam.isChecked(),
            union_with_deepsea=self.union_with_deepsea.isChecked(),
        )

    def set_params(self, p: RefinementParams):
        self.blockSignals(True)
        try:
            idx = max(0, self.mode.findData(p.mode))
            self.mode.setCurrentIndex(idx)
            # Resolve crop_mode (fall back to use_crop mapping if missing)
            cm = getattr(p, "crop_mode", None) \
                or ("global" if p.use_crop else "none")
            self.crop_mode.setCurrentIndex(
                max(0, self.crop_mode.findData(cm))
            )
            self.crop_padding_px.setValue(p.crop_padding_px)
            self.per_cell_padding_px.setValue(
                getattr(p, "per_cell_padding_px", 30)
            )
            # Alt method + RF filter bank
            am = getattr(p, "alt_method", "gmm_3")
            self.alt_method.setCurrentIndex(
                max(0, self.alt_method.findData(am))
            )
            # Refresh bank list in case new ones were trained
            self._populate_filter_banks()
            bank = getattr(p, "rf_filter_bank", "default") or "default"
            self.rf_filter_bank.setCurrentIndex(
                max(0, self.rf_filter_bank.findData(bank))
            )
            # Hybrid params
            hm = getattr(p, "hybrid_method", "gmm_rf")
            self.hybrid_method.setCurrentIndex(
                max(0, self.hybrid_method.findData(hm))
            )
            self.hybrid_rf_threshold.setValue(
                getattr(p, "hybrid_rf_threshold", 0.6)
            )
            self.hybrid_ensemble_threshold.setValue(
                getattr(p, "hybrid_ensemble_threshold", 0.5)
            )
            self.hybrid_boundary_dilate.setValue(
                getattr(p, "hybrid_boundary_dilate", 10)
            )
            self.hybrid_boundary_erode.setValue(
                getattr(p, "hybrid_boundary_erode", 5)
            )
            sam_mt = getattr(p, "sam_model_type", "vit_b")
            self.sam_model_type.setCurrentIndex(
                max(0, self.sam_model_type.findData(sam_mt))
            )
            self.sam_use_mask_prompt.setChecked(
                getattr(p, "sam_use_mask_prompt", True)
            )
            self.skip_cascade_gt.setChecked(p.skip_cascade_gt_frames)
            self.pre_refine_medsam.setChecked(
                getattr(p, "pre_refine_medsam", False))
            self.union_with_deepsea.setChecked(
                getattr(p, "union_with_deepsea", False))
            self.use_rf.setChecked(p.use_rf)
            self.use_snap.setChecked(p.use_snap)
            self.use_fourier.setChecked(p.use_fourier)
            self.use_crf.setChecked(p.use_crf)
            self.use_temporal.setChecked(p.use_temporal)
            self.rf_threshold.setValue(p.rf_threshold)
            self.rf_use_isoline.setChecked(p.rf_use_isoline)
            self.snap_search.setValue(p.snap_search_radius)
            self.snap_max_step.setValue(p.snap_max_step)
            self.snap_smooth.setValue(p.snap_smooth_sigma)
            self.snap_use_membrane.setChecked(p.snap_use_membrane_edge)
            self.snap_alpha.setValue(p.snap_membrane_alpha)
            self.fourier_n.setValue(p.fourier_n_descriptors)
            self.temporal_sigma.setValue(p.temporal_sigma)
        finally:
            self.blockSignals(False)
        self._update_visibility()
        self.changed.emit()
