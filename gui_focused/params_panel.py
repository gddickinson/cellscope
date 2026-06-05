"""Parameter panel for the focused GUI.

All parameter groups are exposed as visible tabs so users can configure
any stage at any time. set_context() still auto-switches the active tab
as a workflow hint, but the user can also click any tab manually — this
is essential for options like VAMPIRE shape modes and state classification
which live on the Analysis tab.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QDoubleSpinBox, QCheckBox, QGroupBox, QTabWidget,
    QFormLayout, QPushButton, QComboBox, QScrollArea,
)


class ParamsPanel(QWidget):
    """All parameter pages exposed as visible tabs.

    Tabs (always visible, click to switch):
      0: Detection — modality, model, debris filter, Cy5 recovery + filter,
                     ROI, scale overrides
      1: Analysis  — per-metric toggles, VAMPIRE shape modes, cell state
                     classification thresholds
    """

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        # No internal header — the dock widget title bar already
        # labels the panel.

        self.stack = QTabWidget()
        self.stack.setDocumentMode(True)
        layout.addWidget(self.stack)

        # Wrap each page in a scroll area so long forms (analysis) don't
        # overflow on small screens.
        def _scroll(widget):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            scroll.setFrameShape(QScrollArea.NoFrame)
            return scroll

        self.stack.addTab(_scroll(self._build_detection_page()),
                          "Detection")                            # 0
        self.stack.addTab(_scroll(self._build_analysis_page()),
                          "Analysis")                             # 1
        # Edit and Export info pages folded into a single help tab —
        # the actions themselves live in the pipeline buttons.
        self.stack.addTab(_scroll(self._build_help_page()),
                          "Tips")                                 # 2

    def _build_detection_page(self):
        page = QWidget()
        form = QFormLayout(page)

        # --- Modality selector ---
        self.modality = QComboBox()
        self.modality.addItems(["Auto", "Phase-contrast", "DIC"])
        self.modality.setToolTip(
            "Imaging modality determines the detection pipeline:\n"
            "  Auto: auto-detect from image statistics\n"
            "  Phase-contrast: cpsam (ViT) + DeepSea union\n"
            "  DIC: cellpose_dic + preprocessing + DeepSea\n\n"
            "DIC recordings have textured backgrounds that confuse\n"
            "cpsam. The DIC pipeline uses a model trained specifically\n"
            "for differential interference contrast microscopy.")
        form.addRow("Modality:", self.modality)

        # --- DIC model selector (only relevant when modality=DIC/Auto) ---
        self.dic_model = QComboBox()
        self.dic_model.setToolTip(
            "Which DIC model to use when modality is DIC:\n"
            "  Auto (recommended): pick the strongest available\n"
            "      (cpsam_dic > cellpose_dic_v3 > v2 > v1)\n"
            "  cpsam_dic: ViT fine-tune on DIC — best overall\n"
            "      (IoU 0.795 on our-GT, 0.697 on VAMPIRE OOD)\n"
            "      Loads via cellpose4 env (delegated automatically).\n"
            "  cellpose_dic_v3: 448x448 CP3 fine-tune (IoU 0.74)\n"
            "  cellpose_dic_v2: raw VAMPIRE crops\n"
            "  cellpose_dic: original CP3 fine-tune\n"
            "Populated from data/models/{cpsam_dic,cellpose_dic}* "
            "at startup.")
        self._populate_dic_models()
        form.addRow("DIC model:", self.dic_model)

        # All defaults below are pulled from core.pipeline_defaults so
        # the GUI and the pipeline functions stay in lockstep. Edit
        # values there, not here.
        from core.pipeline_defaults import DEFAULTS as _PD

        self.downsample = QComboBox()
        self.downsample.addItem("Auto", "auto")
        self.downsample.addItem("Off (full resolution)", "off")
        self.downsample.addItem("2× (faster, loses fine boundaries)",
                                 "2")
        self.downsample.addItem("3× (much faster, lossy)", "3")
        self.downsample.setCurrentIndex(0)
        self.downsample.setToolTip(
            "How much to downsample before cpsam detection.\n"
            "Auto picks 1× for ≤1024² recordings (default 1100 px\n"
            "threshold), 2× for larger. Set to Off if cells are\n"
            "tightly touching and cpsam is fusing them at the auto\n"
            "factor — at full resolution the boundary between\n"
            "neighbouring cells stays resolvable.")
        form.addRow("Downsample:", self.downsample)

        self.min_area = QSpinBox()
        self.min_area.setRange(50, 10000)
        self.min_area.setValue(_PD.min_area_px)
        self.min_area.setToolTip("Minimum cell area (pixels) for debris filter")
        form.addRow("Min area (px):", self.min_area)

        cell_row = QHBoxLayout()
        self.expected_cells = QSpinBox()
        self.expected_cells.setRange(0, 20)
        self.expected_cells.setValue(_PD.expected_cells)
        self.expected_cells.setSpecialValueText("Auto")
        self.expected_cells.setToolTip(
            "Expected cells per frame.\n"
            "Auto (0) = no filtering, keep all detected cells.\n"
            "1+ = keep only the N largest, rest = debris.")
        cell_row.addWidget(self.expected_cells)
        self.btn_scan = QPushButton("Scan")
        self.btn_scan.setToolTip("Auto-detect cell count (~30s)")
        self.btn_scan.setFixedWidth(50)
        cell_row.addWidget(self.btn_scan)
        form.addRow("Expected cells:", cell_row)

        self.search_radius = QSpinBox()
        self.search_radius.setRange(20, 500)
        self.search_radius.setValue(_PD.search_radius_px)
        self.search_radius.setToolTip(
            "Max centroid hop (pixels) the multi-cell tracker will "
            "allow between consecutive frames. Default 150 suits "
            "fast-moving cells; for slow keratinocytes at long frame "
            "intervals, tighten to 30-50 to prevent ID swaps "
            "between nearby cells.")
        form.addRow("Search radius (px):", self.search_radius)

        self.min_track_len = QSpinBox()
        self.min_track_len.setRange(1, 50)
        self.min_track_len.setValue(_PD.min_track_length)
        self.min_track_len.setToolTip("Drop tracks shorter than this (multi-cell)")
        form.addRow("Min track length:", self.min_track_len)

        # --- ROI ---
        self.use_roi = QCheckBox()
        self.use_roi.setChecked(_PD.use_roi)
        self.use_roi.setToolTip(
            "Apply the drawn ROI to restrict analysis.\n"
            "Pixels outside the ROI are zeroed before detection.\n"
            "Draw an ROI first via Edit > Select ROI.")
        form.addRow("Apply ROI:", self.use_roi)

        # --- Refinement steps ---
        steps_label = QLabel("<b>Refinement steps:</b>")
        form.addRow(steps_label)

        self.use_deepsea = QCheckBox()
        self.use_deepsea.setChecked(_PD.use_deepsea)
        self.use_deepsea.setToolTip(
            "Run DeepSea union to fill under-segmented regions\n"
            "and remove debris (via largest connected component).\n"
            "Adds ~2s/frame. Recommended for best results.")
        form.addRow("DeepSea refinement:", self.use_deepsea)

        self.use_tta = QCheckBox()
        self.use_tta.setChecked(_PD.use_tta)
        self.use_tta.setToolTip(
            "Test-time augmentation: run cpsam on 4 rotations\n"
            "and average results. Slower (~4x) but may recover\n"
            "cells missed at default orientation.\n"
            "Effects vary by recording (validated 2026-05-21):\n"
            "  - helps very dense fields (Pos68_DMSO +0.03 F1)\n"
            "  - hurts most other recordings (-0.02 F1 typical)\n"
            "Off by default; enable for dense recordings.")
        form.addRow("TTA (augment):", self.use_tta)

        self.use_cpsam_cy5_union = QCheckBox()
        self.use_cpsam_cy5_union.setChecked(_PD.use_cpsam_cy5_union)
        self.use_cpsam_cy5_union.setToolTip(
            "EXPERIMENTAL (default off). Run cpsam on the Cy5\n"
            "channel too and union-merge with DIC detections.\n"
            "Bbox validation showed +0.11-0.26 F1 on dense\n"
            "recordings but the multi-metric Cy5 filter rejects\n"
            "the cpsam(Cy5) cells in the full pipeline, so net\n"
            "F1 change is ~0 currently. Will be re-enabled once\n"
            "the filter is tuned. Cost when on: ~1.5× runtime.")
        form.addRow("cpsam-on-Cy5 union:", self.use_cpsam_cy5_union)

        self.use_mirror_pad = QComboBox()
        self.use_mirror_pad.addItems(["auto", "on", "off"])
        idx = self.use_mirror_pad.findText(
            str(_PD.use_mirror_pad).lower())
        if idx >= 0:
            self.use_mirror_pad.setCurrentIndex(idx)
        self.use_mirror_pad.setToolTip(
            "Mirror-pad the image by 50 px before each cpsam call so\n"
            "cells at the FoV edge sit in the interior of a 'complete'\n"
            "reflection — cpsam segments them as full cells instead of\n"
            "partial profiles. Adds ~40% runtime.\n"
            "auto: enabled when detection image min-dim ≥ 1000 px\n"
            "      (skipped on small cropped recordings where the\n"
            "      relative padding ratio would be too large).\n"
            "Validated 2026-05-21: +0.018 mean F1 on 6 IC295 recordings.")
        form.addRow("Mirror-pad:", self.use_mirror_pad)

        self.use_tiling = QCheckBox()
        self.use_tiling.setChecked(_PD.use_tiling)
        self.use_tiling.setToolTip(
            "Split image into tiles before detection.\n"
            "Best for large FOV with 5+ cells per frame.\n"
            "NOT recommended for sparse recordings.")
        form.addRow("Tiled detection:", self.use_tiling)

        self.tile_grid = QSpinBox()
        self.tile_grid.setRange(2, 5)
        self.tile_grid.setValue(_PD.tile_grid)
        self.tile_grid.setToolTip("Tile grid size (NxN)")
        form.addRow("Tile grid (NxN):", self.tile_grid)

        self.use_fallback = QCheckBox()
        self.use_fallback.setChecked(_PD.use_fallback)
        self.use_fallback.setToolTip(
            "For frames where cpsam fails to detect a cell,\n"
            "fall back to cellpose + MedSAM + DeepSea.\n"
            "Requires the cellpose (CP3) env.")
        form.addRow("Fallback detection:", self.use_fallback)

        self.use_gap_fill = QCheckBox()
        self.use_gap_fill.setChecked(_PD.use_gap_fill)
        self.use_gap_fill.setToolTip(
            "After tracking, fill internal gaps where a cell\n"
            "was lost for a few frames. Uses cpsam(augment=True)\n"
            "then cellpose fallback. Multi-cell only.")
        form.addRow("Gap fill:", self.use_gap_fill)

        self.use_sam2_video_gap_fill = QCheckBox()
        self.use_sam2_video_gap_fill.setChecked(
            _PD.use_sam2_video_gap_fill)
        self.use_sam2_video_gap_fill.setEnabled(_PD.use_gap_fill)
        self.use_sam2_video_gap_fill.setToolTip(
            "Extra gap-fill stage: SAM2 video propagation between\n"
            "Phase 2 (CP3 fallback) and Phase 4 (mask translation).\n"
            "Catches cells that biologically retract or dim too much\n"
            "for cpsam to detect at the gap frame, but where SAM2's\n"
            "memory attention can follow them through the gap.\n"
            "~1 sec per gap frame; turn off if cpsam(augment) is\n"
            "enough for your data. Only effective when Gap fill is on.")
        form.addRow("  SAM2 video gap fill:",
                     self.use_sam2_video_gap_fill)
        # Disable SAM2 sub-toggle when gap fill is off
        self.use_gap_fill.toggled.connect(
            self.use_sam2_video_gap_fill.setEnabled)

        self.max_gap_frames = QSpinBox()
        self.max_gap_frames.setRange(1, 100)
        self.max_gap_frames.setValue(_PD.max_gap_frames)
        self.max_gap_frames.setToolTip(
            "Max consecutive frames a track can be undetected before\n"
            "the Hungarian tracker declares it dead. Default 15 —\n"
            "covers ~10 frames of biological transitions (mitosis,\n"
            "retraction, de-attachment) at our 10-min/frame interval.")
        form.addRow("  Max gap frames:", self.max_gap_frames)

        self.gap_fill_crop = QCheckBox()
        self.gap_fill_crop.setChecked(_PD.gap_fill_crop)
        self.gap_fill_crop.setEnabled(_PD.use_gap_fill)
        self.gap_fill_crop.setToolTip(
            "Fast gap fill (default ON). Phase 1 segments an adaptive\n"
            "crop around the interpolated centroid instead of the whole\n"
            "frame — ~12× faster per gap with no quality loss (validated\n"
            "against reviewed-mask GT; full-frame fallback on a crop-miss\n"
            "so recall is never worse). Uncheck to revert to the older,\n"
            "slower full-frame Phase 1 behaviour.")
        form.addRow("  Gap-fill crop (fast):", self.gap_fill_crop)
        self.use_gap_fill.toggled.connect(self.gap_fill_crop.setEnabled)

        self.gap_fill_augment = QCheckBox()
        self.gap_fill_augment.setChecked(_PD.gap_fill_augment)
        self.gap_fill_augment.setEnabled(_PD.use_gap_fill)
        self.gap_fill_augment.setToolTip(
            "Always run gap-fill Phase 1 with cpsam augment (4-rotation\n"
            "TTA). Default OFF (faster): the fast no-augment pass runs\n"
            "first and only falls back to augment on a miss — ~2× faster\n"
            "Phase 1 with no quality loss (GT-validated, 0 good fills\n"
            "dropped, IoU Δ −0.011). Check to always augment (old, slower\n"
            "behaviour).")
        form.addRow("  Gap-fill always augment:", self.gap_fill_augment)
        self.use_gap_fill.toggled.connect(self.gap_fill_augment.setEnabled)

        # --- DIC-pipeline-only params (cpsam_dic / hybrid_dic) ---
        dic_label = QLabel("<b>DIC pipeline (single-cell):</b>")
        form.addRow(dic_label)

        self.use_preprocess = QCheckBox()
        self.use_preprocess.setChecked(_PD.use_preprocess)
        self.use_preprocess.setToolTip(
            "DIC pipeline only: apply temporal background subtraction\n"
            "+ spatial high-pass before detection. Skipped for cpsam\n"
            "(it hurts ViT predictions). Default ON for cellpose_dic.")
        form.addRow("DIC preprocess:", self.use_preprocess)

        self.use_retry = QCheckBox()
        self.use_retry.setChecked(_PD.use_retry)
        self.use_retry.setToolTip(
            "DIC pipeline only: retry missed frames with progressively\n"
            "lower cellprob_threshold. Default ON.")
        form.addRow("DIC retry low-cp:", self.use_retry)

        self.cy5_fusion_jaccard_thresh = QDoubleSpinBox()
        self.cy5_fusion_jaccard_thresh.setRange(0.0, 1.0)
        self.cy5_fusion_jaccard_thresh.setSingleStep(0.05)
        self.cy5_fusion_jaccard_thresh.setDecimals(3)
        self.cy5_fusion_jaccard_thresh.setValue(
            _PD.cy5_fusion_jaccard_thresh)
        self.cy5_fusion_jaccard_thresh.setToolTip(
            "DIC pipeline only: Jaccard overlap threshold for\n"
            "merging a DIC mask with a Cy5-derived mask.\n"
            "Default 0.30 — pairs above this get unioned.")
        form.addRow("Cy5 fusion Jaccard:",
                     self.cy5_fusion_jaccard_thresh)

        self.cy5_fusion_max_overlap_frac = QDoubleSpinBox()
        self.cy5_fusion_max_overlap_frac.setRange(0.0, 1.0)
        self.cy5_fusion_max_overlap_frac.setSingleStep(0.05)
        self.cy5_fusion_max_overlap_frac.setDecimals(3)
        self.cy5_fusion_max_overlap_frac.setValue(
            _PD.cy5_fusion_max_overlap_frac)
        self.cy5_fusion_max_overlap_frac.setToolTip(
            "DIC pipeline only: max overlap fraction for fusion-\n"
            "candidate merging. Default 0.50.")
        form.addRow("Cy5 fusion max overlap:",
                     self.cy5_fusion_max_overlap_frac)

        self.cy5_fusion_augment_cpsam = QCheckBox()
        self.cy5_fusion_augment_cpsam.setChecked(
            _PD.cy5_fusion_augment_cpsam)
        self.cy5_fusion_augment_cpsam.setToolTip(
            "DIC pipeline only: enable cpsam augmentation pass\n"
            "during Cy5 fusion. Slower but may improve recall.")
        form.addRow("Cy5 fusion augment cpsam:",
                     self.cy5_fusion_augment_cpsam)

        # --- Cy5 fusion (Tier 1: detect on fluorescence) ---
        self.use_cy5_fusion = QCheckBox()
        self.use_cy5_fusion.setChecked(False)
        self.use_cy5_fusion.setEnabled(False)
        self.use_cy5_fusion.setToolTip(
            "Multichannel only — disabled until you load a recording\n"
            "with a fluorescence channel.\n\n"
            "Run cpsam DIRECTLY on the fluorescence channel as a\n"
            "parallel detection, then merge the result into the DIC\n"
            "labels at stage 1 (before DeepSea refinement and\n"
            "tracking). Cells visible in fluorescence but missed by\n"
            "the DIC detector are added with fresh label IDs.\n\n"
            "Adds ~30-60s on a 40-frame 1024² recording. Strongly\n"
            "recommended for DIC recordings where the F-actin /\n"
            "SiR-actin signal is informative — the conservative\n"
            "DIC-fine-tuned model alone can miss the majority of\n"
            "clearly-stained cells.")
        form.addRow("Cy5 fusion (Tier 1):", self.use_cy5_fusion)

        # --- Cy5 recovery (Tier 2 fail-safe, multichannel only) ---
        self.use_cy5_recovery = QCheckBox()
        self.use_cy5_recovery.setChecked(False)
        self.use_cy5_recovery.setEnabled(False)
        self.use_cy5_recovery.setToolTip(
            "Multichannel only — disabled until you load a recording\n"
            "with a fluorescence channel.\n\n"
            "After cpsam runs on the DIC channel, search the\n"
            "fluorescence channel for bright regions not covered by\n"
            "any cpsam mask, crop the DIC around them, and re-run\n"
            "cpsam(TTA) on the crop to recover cells the base model\n"
            "missed. Adds ~1s/frame when no recoveries fire,\n"
            "~10-20s per recovered cell.\n\n"
            "Cells added via this path go through the same DeepSea\n"
            "refinement + tracking pipeline as primary detections.")
        form.addRow("Cy5 recovery (Tier 2):", self.use_cy5_recovery)

        # --- Cy5 false-positive filter (Tier 4, multichannel only) ---
        self.cy5_filter_mode = QComboBox()
        self.cy5_filter_mode.addItems([
            "Off",
            "Persistence guard (mm-pass OR long+moving)",
            "Conservative (no signal)",
            "Conservative strict (tighter)",
            "Adaptive (bimodal-aware)",
            "Adaptive loose (less aggressive)",
            "Multi-metric (≥2/3 cellularity tests)",
            "Composite score (continuous)",
            "Consensus (multi_metric ∩ composite, safest)",
            "Temporal stability (drop noise)",
            "Threshold (manual)"])
        # Default to Persistence guard — recovers +70 GT cells (+7.1%)
        # across 13-recording corpus vs Multi-metric (which dropped
        # persistent cells in weak-Cy5 conditions: GOF/OT/DMSO).
        self.cy5_filter_mode.setCurrentText(
            "Persistence guard (mm-pass OR long+moving)")
        self.cy5_filter_mode.setEnabled(False)
        self.cy5_filter_mode.setToolTip(
            "After detection + tracking, drop tracks whose Cy5\n"
            "fluorescence signal indicates a DIC false positive\n"
            "(debris, dust, vignette artefact). Multichannel only.\n\n"
            "  Off: keep all detected tracks.\n"
            "  Conservative (default safe): drop only tracks with\n"
            "      NO Cy5 signal at all (mean<0.05 AND p95<0.10).\n"
            "      Recommended — keeps faint real cells.\n"
            "  Adaptive: per-recording bimodal cut. If track scores\n"
            "      cluster bimodally (debris vs cells), cut at the\n"
            "      gap. If unimodal, falls back to conservative.\n"
            "  Threshold: manual cut at the score below.")
        form.addRow("Cy5 filter (Tier 4):", self.cy5_filter_mode)
        self.cy5_filter_threshold = QDoubleSpinBox()
        self.cy5_filter_threshold.setRange(0.0, 1.0)
        self.cy5_filter_threshold.setDecimals(3)
        self.cy5_filter_threshold.setSingleStep(0.05)
        self.cy5_filter_threshold.setValue(_PD.cy5_filter_threshold)
        self.cy5_filter_threshold.setEnabled(False)
        self.cy5_filter_threshold.setToolTip(
            "Track-mean Cy5 score cut (only used when filter mode\n"
            "is 'Threshold'). Pilot data: real cells score 0.25-0.59,\n"
            "background detections 0.05-0.15.")
        form.addRow("  Threshold:", self.cy5_filter_threshold)
        # Enable threshold spinbox only when filter mode is "Threshold"
        self.cy5_filter_mode.currentTextChanged.connect(
            lambda txt: self.cy5_filter_threshold.setEnabled(
                self.cy5_filter_mode.isEnabled()
                and txt.startswith("Threshold")))

        # --- Persistence guard v2 sub-parameters ---
        # Enabled only when the Persistence guard mode is selected.
        # See core/cy5_filter.persistence_guard_filter docstring.
        self.cy5_pg_min_lifetime = QSpinBox()
        self.cy5_pg_min_lifetime.setRange(5, 100)
        self.cy5_pg_min_lifetime.setValue(_PD.cy5_pg_min_lifetime)
        self.cy5_pg_min_lifetime.setEnabled(False)
        self.cy5_pg_min_lifetime.setToolTip(
            "Persistence guard: minimum lifetime (frames) above which "
            "a track that fails the Cy5 multi-metric test still gets "
            "a chance to be kept via the motion/shape gate.\n"
            "Tracks shorter than this with mm<2 are dropped outright.\n"
            "Default 35; tuned on full 13-recording GT corpus.")
        form.addRow("  PG min lifetime:", self.cy5_pg_min_lifetime)

        self.cy5_pg_static_velocity_px = QDoubleSpinBox()
        self.cy5_pg_static_velocity_px.setRange(0.1, 20.0)
        self.cy5_pg_static_velocity_px.setSingleStep(0.5)
        self.cy5_pg_static_velocity_px.setDecimals(2)
        self.cy5_pg_static_velocity_px.setValue(
            _PD.cy5_pg_static_velocity_px)
        self.cy5_pg_static_velocity_px.setEnabled(False)
        self.cy5_pg_static_velocity_px.setToolTip(
            "Persistence guard: tracks with mean per-frame centroid "
            "displacement below this AND high shape stability are "
            "considered 'static' and dropped (phantom signature).\n"
            "Default 3.0 px/frame.")
        form.addRow("  PG static vel (px):",
                     self.cy5_pg_static_velocity_px)

        self.cy5_pg_static_shape_iou = QDoubleSpinBox()
        self.cy5_pg_static_shape_iou.setRange(0.5, 1.0)
        self.cy5_pg_static_shape_iou.setSingleStep(0.05)
        self.cy5_pg_static_shape_iou.setDecimals(3)
        self.cy5_pg_static_shape_iou.setValue(
            _PD.cy5_pg_static_shape_iou)
        self.cy5_pg_static_shape_iou.setEnabled(False)
        self.cy5_pg_static_shape_iou.setToolTip(
            "Persistence guard: tracks with median consecutive-frame "
            "mask IoU above this AND low velocity are 'static' "
            "(phantom signature).\n"
            "Default 0.85. Real cells deform (0.5-0.75) — phantoms "
            "are bit-stable (0.85-0.97).")
        form.addRow("  PG static shape IoU:",
                     self.cy5_pg_static_shape_iou)

        # Enable PG sub-spinboxes only when Persistence mode selected
        def _toggle_pg(_=None):
            on = (self.cy5_filter_mode.isEnabled()
                  and self.cy5_filter_mode.currentText().startswith(
                      "Persistence"))
            for w in (self.cy5_pg_min_lifetime,
                       self.cy5_pg_static_velocity_px,
                       self.cy5_pg_static_shape_iou):
                w.setEnabled(on)
        self.cy5_filter_mode.currentTextChanged.connect(_toggle_pg)
        # Also fire on enable-toggle (when Cy5 becomes available)
        self._toggle_pg_sub = _toggle_pg

        # --- Scale ---
        scale_label = QLabel("<b>Scale:</b>")
        form.addRow(scale_label)

        self.um_per_px = QDoubleSpinBox()
        self.um_per_px.setRange(0.0, 100.0)
        self.um_per_px.setDecimals(3)
        self.um_per_px.setValue(0.0)
        self.um_per_px.setToolTip("0 = use recording metadata")
        form.addRow("um/px:", self.um_per_px)

        self.time_interval = QDoubleSpinBox()
        self.time_interval.setRange(0.0, 1000.0)
        self.time_interval.setDecimals(2)
        self.time_interval.setValue(0.0)
        self.time_interval.setToolTip("0 = use recording metadata")
        form.addRow("Time interval (min):", self.time_interval)

        self._multi_widgets = [self.expected_cells, self.btn_scan,
                               self.search_radius, self.min_track_len,
                               self.use_gap_fill]
        return page

    def _build_analysis_page(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(6, 6, 6, 6)

        banner = QLabel(
            "These options take effect when you click "
            "<b>Analyze</b> in the pipeline panel. Re-run Analyze after "
            "changing them.")
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "QLabel { background: #fff3e0; border: 1px solid #e26d5c; "
            "padding: 6px; border-radius: 3px; color: #5a2a1e; }")
        outer.addWidget(banner)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        outer.addWidget(form_widget)
        outer.addStretch()

        form.addRow(QLabel("<b>Per-metric toggles:</b>"))
        self.compute_tracking = QCheckBox()
        self.compute_tracking.setChecked(True)
        self.compute_tracking.setToolTip(
            "Compute migration speed, MSD, persistence, "
            "direction autocorrelation.")
        form.addRow("Tracking:", self.compute_tracking)
        self.compute_morphology = QCheckBox()
        self.compute_morphology.setChecked(True)
        self.compute_morphology.setToolTip(
            "Compute area, perimeter, circularity, solidity, "
            "aspect ratio, eccentricity.")
        form.addRow("Morphology:", self.compute_morphology)
        self.compute_edge = QCheckBox()
        self.compute_edge.setChecked(True)
        self.compute_edge.setToolTip(
            "Compute polar boundary, radial edge velocity, "
            "protrusion/retraction kymograph.")
        form.addRow("Edge dynamics:", self.compute_edge)
        self.compute_boundary = QCheckBox()
        self.compute_boundary.setChecked(True)
        self.compute_boundary.setToolTip(
            "Compute boundary-confidence quality score per frame.")
        form.addRow("Boundary confidence:", self.compute_boundary)
        self.compute_area = QCheckBox()
        self.compute_area.setChecked(True)
        self.compute_area.setToolTip(
            "Compute area-stability score (coefficient of variation).")
        form.addRow("Area stability:", self.compute_area)
        self.compute_membrane = QCheckBox()
        self.compute_membrane.setChecked(False)
        self.compute_membrane.setToolTip(
            "Texture-aware membrane-quality metric. Off by default; "
            "adds ~1 s per frame for the texture-contrast computation.")
        form.addRow("Membrane quality:", self.compute_membrane)

        # Canonical analysis defaults live in core.pipeline_defaults —
        # read from there (rule 1) so the focused GUI, batch GUI, and
        # workers stay in lockstep instead of drifting on hardcoded
        # literals.
        from core.pipeline_defaults import DEFAULTS as _PD

        form.addRow(QLabel("<b>Shape analysis:</b>"))
        self.compute_vampire = QCheckBox()
        self.compute_vampire.setChecked(_PD.compute_vampire)
        self.compute_vampire.setToolTip(
            "VAMPIRE shape mode analysis: PCA + K-means clustering\n"
            "of cell contours. Identifies morphological phenotypes\n"
            "and computes heterogeneity (Shannon entropy).\n"
            "Adds ~1s to analysis. Requires vampire-analysis package.")
        form.addRow("VAMPIRE shape modes:", self.compute_vampire)

        self.vampire_clusters = QSpinBox()
        self.vampire_clusters.setRange(2, 15)
        self.vampire_clusters.setValue(_PD.vampire_n_clusters)
        self.vampire_clusters.setToolTip("Number of shape mode clusters")
        form.addRow("Shape mode clusters:", self.vampire_clusters)

        # --- Cell state classification (balled vs attached) ---
        form.addRow(QLabel("<b>Cell state classification:</b>"))
        self.compute_states = QCheckBox()
        self.compute_states.setChecked(_PD.compute_state_classification)
        self.compute_states.setToolTip(
            "Classify each cell-frame as balled (rounded, mitotic /\n"
            "pre-mitotic) or attached (spread, adherent), based on\n"
            "circularity + solidity. Required to stratify motility\n"
            "by state — without it, recordings with more dividing\n"
            "cells appear to migrate faster as a composition\n"
            "artefact.\n\n"
            "Outputs (CSV at export): per-cell state composition\n"
            "(% time balled / attached) plus per-state speed,\n"
            "directional persistence, MSD, total displacement.")
        form.addRow("Classify states:", self.compute_states)

        # State thresholds — canonical source is
        # core.cell_state.DEFAULT_THRESHOLDS. Pull from there so the
        # focused GUI, batch GUI, and core/cell_state.classify_*
        # functions stay in lockstep.
        from core.cell_state import DEFAULT_THRESHOLDS as _STH
        self.state_balled_circ = QDoubleSpinBox()
        self.state_balled_circ.setRange(0.5, 1.0)
        self.state_balled_circ.setSingleStep(0.05)
        self.state_balled_circ.setDecimals(2)
        self.state_balled_circ.setValue(_STH["balled_circ"])
        self.state_balled_circ.setToolTip(
            "Cells with circularity ≥ this AND solidity ≥ "
            "balled-solidity threshold are classified BALLED.\n"
            "Default validated on IC295 (cleanly separates "
            "rounded mitotic cells from spread cells).")
        form.addRow("  Balled circularity ≥:", self.state_balled_circ)
        self.state_balled_solid = QDoubleSpinBox()
        self.state_balled_solid.setRange(0.5, 1.0)
        self.state_balled_solid.setSingleStep(0.02)
        self.state_balled_solid.setDecimals(2)
        self.state_balled_solid.setValue(_STH["balled_solid"])
        self.state_balled_solid.setToolTip(
            "Cells with solidity ≥ this AND circularity ≥ "
            "balled-circ are classified BALLED.")
        form.addRow("  Balled solidity ≥:", self.state_balled_solid)
        self.state_attached_circ = QDoubleSpinBox()
        self.state_attached_circ.setRange(0.0, 1.0)
        self.state_attached_circ.setSingleStep(0.05)
        self.state_attached_circ.setDecimals(2)
        self.state_attached_circ.setValue(_STH["attached_circ"])
        self.state_attached_circ.setToolTip(
            "Cells with circularity ≤ this OR solidity ≤ "
            "attached-solidity are classified ATTACHED.")
        form.addRow("  Attached circularity ≤:", self.state_attached_circ)
        self.state_attached_solid = QDoubleSpinBox()
        self.state_attached_solid.setRange(0.0, 1.0)
        self.state_attached_solid.setSingleStep(0.02)
        self.state_attached_solid.setDecimals(2)
        self.state_attached_solid.setValue(_STH["attached_solid"])
        self.state_attached_solid.setToolTip(
            "Cells with solidity ≤ this OR circularity ≤ "
            "attached-circ are classified ATTACHED.")
        form.addRow("  Attached solidity ≤:", self.state_attached_solid)

        # --- Cell division ---
        form.addRow(QLabel("<b>Cell division:</b>"))
        self.compute_divisions = QCheckBox()
        self.compute_divisions.setChecked(True)
        self.compute_divisions.setToolTip(
            "Detect cell-division events (pre-mitotic swelling → balled →\n"
            "mask halves → two persisting daughters) and assign lineage\n"
            "(parent_id / division_frame / division_score) to daughter\n"
            "tracks. Powers the Cell Lineage Tree + Division Timeline\n"
            "graphs and the division-rate summary. Detection pipelines\n"
            "annotate divisions inline; this re-runs detection only when\n"
            "masks were loaded from a file with no divisions.json sidecar.")
        form.addRow("Detect divisions:", self.compute_divisions)
        return page

    def _build_help_page(self):
        """Quick reference for editing + export (those actions live in
        the pipeline-panel buttons, not as parameters)."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)

        tips = QLabel(
            "<b>Edit Masks</b><br/>"
            "Launch the mask editor from the pipeline panel to manually "
            "correct cell boundaries. Use number keys 1-9 to select "
            "different cells in multi-cell mode.<br/><br/>"
            "<b>Export</b><br/>"
            "Click Export in the pipeline panel to save masks, metrics, "
            "overlay TIFFs, plots, and per-cell CSV summaries.<br/><br/>"
            "<b>VAMPIRE plots</b><br/>"
            "To use the VAMPIRE Shape Modes / Distribution / Eigenshapes "
            "plots, switch to the <b>Analysis</b> tab and tick "
            "<i>VAMPIRE shape modes</i> before clicking Analyze."
        )
        tips.setWordWrap(True)
        tips.setTextFormat(Qt.RichText)
        layout.addWidget(tips)
        layout.addStretch()
        return page

    def set_context(self, stage, mode="single"):
        """Switch the active parameter tab as a workflow hint.

        Users can still click any tab manually — this just brings the
        most relevant one to the front for the current stage.
        """
        page_map = {
            "load": 0, "detect": 0,
            "edit": 2,
            "analyze": 1,
            "export": 2,
        }
        self.stack.setCurrentIndex(page_map.get(stage, 0))
        is_multi = mode == "multi"
        for w in self._multi_widgets:
            w.setEnabled(is_multi)
        if not is_multi:
            self.expected_cells.setValue(1)
        elif self.expected_cells.value() == 1:
            self.expected_cells.setValue(0)

    def set_from_recording(self, recording):
        """Populate scale fields from recording metadata."""
        um = recording.get("um_per_px", 0)
        dt = recording.get("time_interval_min", 0)
        if um and um > 0:
            self.um_per_px.setValue(um)
        if dt and dt > 0:
            self.time_interval.setValue(dt)

    def set_cy5_available(self, has_cy5):
        """Enable/disable the Cy5 fusion + recovery + filter widgets
        based on whether the loaded recording has a fluorescence
        channel."""
        self.use_cy5_fusion.setEnabled(has_cy5)
        self.use_cy5_recovery.setEnabled(has_cy5)
        self.cy5_filter_mode.setEnabled(has_cy5)
        if has_cy5:
            # Fusion ON by default — biggest single win for multichannel
            # DIC recordings where fluorescence is informative.
            self.use_cy5_fusion.setChecked(True)
            self.use_cy5_recovery.setChecked(True)
            self.cy5_filter_mode.setCurrentText(
                "Persistence guard (mm-pass OR long+moving)")
        else:
            self.use_cy5_fusion.setChecked(False)
            self.use_cy5_recovery.setChecked(False)
            self.cy5_filter_mode.setCurrentText("Off")
            self.cy5_filter_threshold.setEnabled(False)
        # Re-evaluate the v2 sub-control enable state for the
        # current mode (which may have just changed above).
        if hasattr(self, "_toggle_pg_sub"):
            self._toggle_pg_sub()

    def _populate_dic_models(self):
        """Scan data/models for cpsam_dic* and cellpose_dic* models.

        Order in dropdown: cpsam_dic* first (best, ViT fine-tune),
        then cellpose_dic_v3 / v2 / v1 (older CP3 fine-tunes).
        Default 'Auto' resolves to the first item (best available).
        """
        import os
        self.dic_model.clear()
        self.dic_model.addItem("Auto (best available)")
        base = "data/models"
        cpsam, cp3 = [], []
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                if name.startswith("cpsam_dic"):
                    cpsam.append(name)
                elif name.startswith("cellpose_dic"):
                    cp3.append(name)
        # cpsam_dic first (newest first), then cellpose_dic v3/v2/v1
        for name in sorted(cpsam, reverse=True):
            self.dic_model.addItem(name)
        for name in sorted(cp3, reverse=True):
            self.dic_model.addItem(name)

    def get_detect_params(self):
        modality_map = {"Auto": "auto", "Phase-contrast": "phase_contrast",
                        "DIC": "dic"}
        dic_choice = self.dic_model.currentText()
        if dic_choice.startswith("Auto"):
            dic_model_path = None  # let core pick newest
        else:
            dic_model_path = f"data/models/{dic_choice}"
        return {
            "modality": modality_map.get(self.modality.currentText(), "auto"),
            "dic_model_path": dic_model_path,
            "downsample": self.downsample.currentData(),
            "min_area_px": self.min_area.value(),
            "expected_cells": self.expected_cells.value(),
            "search_radius": self.search_radius.value(),
            "min_track_length": self.min_track_len.value(),
            "use_tta": self.use_tta.isChecked(),
            "use_mirror_pad": self.use_mirror_pad.currentText(),
            "use_cpsam_cy5_union":
                self.use_cpsam_cy5_union.isChecked(),
            "use_tiling": self.use_tiling.isChecked(),
            "tile_grid": self.tile_grid.value(),
            "use_deepsea": self.use_deepsea.isChecked(),
            "use_fallback": self.use_fallback.isChecked(),
            "use_gap_fill": self.use_gap_fill.isChecked(),
            "use_cy5_fusion": self.use_cy5_fusion.isChecked(),
            "use_cy5_recovery": self.use_cy5_recovery.isChecked(),
            "cy5_filter_mode": self._cy5_filter_mode_value(),
            "cy5_filter_threshold": self.cy5_filter_threshold.value(),
            "cy5_pg_min_lifetime": self.cy5_pg_min_lifetime.value(),
            "cy5_pg_static_velocity_px":
                self.cy5_pg_static_velocity_px.value(),
            "cy5_pg_static_shape_iou":
                self.cy5_pg_static_shape_iou.value(),
            "use_sam2_video_gap_fill":
                self.use_sam2_video_gap_fill.isChecked(),
            "gap_fill_crop": self.gap_fill_crop.isChecked(),
            "gap_fill_augment": self.gap_fill_augment.isChecked(),
            "max_gap_frames": self.max_gap_frames.value(),
            "use_preprocess": self.use_preprocess.isChecked(),
            "use_retry": self.use_retry.isChecked(),
            "cy5_fusion_jaccard_thresh":
                self.cy5_fusion_jaccard_thresh.value(),
            "cy5_fusion_max_overlap_frac":
                self.cy5_fusion_max_overlap_frac.value(),
            "cy5_fusion_augment_cpsam":
                self.cy5_fusion_augment_cpsam.isChecked(),
        }

    def _cy5_filter_mode_value(self):
        """Map dropdown text to the lowercase mode key expected by
        core.cy5_filter.apply_cy5_filter."""
        txt = self.cy5_filter_mode.currentText()
        if txt.startswith("Off"):
            return "off"
        if txt.startswith("Persistence"):
            return "persistence_guard"
        if txt.startswith("Conservative strict"):
            return "conservative_strict"
        if txt.startswith("Conservative"):
            return "conservative"
        if txt.startswith("Adaptive loose"):
            return "adaptive_loose"
        if txt.startswith("Adaptive"):
            return "adaptive"
        if txt.startswith("Multi-metric"):
            return "multi_metric"
        if txt.startswith("Composite"):
            return "composite_score"
        if txt.startswith("Consensus"):
            return "consensus"
        if txt.startswith("Temporal"):
            return "temporal_stability"
        if txt.startswith("Threshold"):
            return "threshold"
        return "off"

    def get_scale_overrides(self):
        return {
            "um_per_px": self.um_per_px.value(),
            "time_interval_min": self.time_interval.value(),
        }

    def get_state_params(self):
        """State-classification settings (returned only when enabled)."""
        if not self.compute_states.isChecked():
            return {"enabled": False}
        return {
            "enabled": True,
            "thresholds": {
                "balled_circ": self.state_balled_circ.value(),
                "balled_solid": self.state_balled_solid.value(),
                "attached_circ": self.state_attached_circ.value(),
                "attached_solid": self.state_attached_solid.value(),
            },
        }

    def get_division_params(self):
        """Cell-division detection settings."""
        return {"enabled": self.compute_divisions.isChecked()}

    def get_vampire_params(self):
        """VAMPIRE shape-mode-analysis settings.

        Returns a dict with `enabled` and `n_clusters`. Always includes
        both keys so callers can read them without checking enabled
        first.
        """
        return {
            "enabled": self.compute_vampire.isChecked(),
            "n_clusters": self.vampire_clusters.value(),
        }

    def get_analysis_compute_flags(self):
        """Which per-metric analyses the user has ticked on the
        Analysis tab. Stored on the result dict so downstream consumers
        (writers, plot panel) can suppress sections the user opted out
        of. Note: in v1, core.pipeline.analyze_recording always computes
        everything — these flags are advisory, not gating."""
        return {
            "tracking": self.compute_tracking.isChecked(),
            "morphology": self.compute_morphology.isChecked(),
            "edge_dynamics": self.compute_edge.isChecked(),
            "boundary_confidence": self.compute_boundary.isChecked(),
            "area_stability": self.compute_area.isChecked(),
            "membrane_quality": self.compute_membrane.isChecked(),
        }
