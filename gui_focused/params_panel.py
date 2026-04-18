"""Context-sensitive parameter panel for the focused GUI."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QDoubleSpinBox, QCheckBox, QGroupBox, QStackedWidget,
    QFormLayout, QPushButton,
)


class ParamsPanel(QWidget):
    """Shows relevant parameters based on current pipeline stage."""

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Parameters"))

        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.stack.addWidget(self._build_detection_page())  # 0
        self.stack.addWidget(self._build_analysis_page())    # 1
        self.stack.addWidget(self._build_info_page(
            "Edit Masks",
            "Launch the mask editor to manually correct\n"
            "cell boundaries. Use number keys 1-9 to\n"
            "select different cells in multi-cell mode."))  # 2
        self.stack.addWidget(self._build_info_page(
            "Export",
            "Choose which outputs to save:\n"
            "masks, metrics, plots, overlay TIFFs."))       # 3

    def _build_detection_page(self):
        page = QWidget()
        form = QFormLayout(page)
        self.min_area = QSpinBox()
        self.min_area.setRange(50, 10000)
        self.min_area.setValue(500)
        self.min_area.setToolTip("Minimum cell area (pixels) for debris filter")
        form.addRow("Min area (px):", self.min_area)

        cell_row = QHBoxLayout()
        self.expected_cells = QSpinBox()
        self.expected_cells.setRange(0, 20)
        self.expected_cells.setValue(0)
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
        self.search_radius.setRange(50, 500)
        self.search_radius.setValue(150)
        self.search_radius.setToolTip("Max centroid hop for tracking (multi-cell)")
        form.addRow("Search radius (px):", self.search_radius)

        self.min_track_len = QSpinBox()
        self.min_track_len.setRange(1, 50)
        self.min_track_len.setValue(3)
        self.min_track_len.setToolTip("Drop tracks shorter than this (multi-cell)")
        form.addRow("Min track length:", self.min_track_len)

        # --- ROI ---
        self.use_roi = QCheckBox()
        self.use_roi.setChecked(False)
        self.use_roi.setToolTip(
            "Apply the drawn ROI to restrict analysis.\n"
            "Pixels outside the ROI are zeroed before detection.\n"
            "Draw an ROI first via Edit > Select ROI.")
        form.addRow("Apply ROI:", self.use_roi)

        # --- Refinement steps ---
        steps_label = QLabel("<b>Refinement steps:</b>")
        form.addRow(steps_label)

        self.use_deepsea = QCheckBox()
        self.use_deepsea.setChecked(True)
        self.use_deepsea.setToolTip(
            "Run DeepSea union to fill under-segmented regions\n"
            "and remove debris (via largest connected component).\n"
            "Adds ~2s/frame. Recommended for best results.")
        form.addRow("DeepSea refinement:", self.use_deepsea)

        self.use_tta = QCheckBox()
        self.use_tta.setChecked(False)
        self.use_tta.setToolTip(
            "Test-time augmentation: run cpsam on 4 rotations\n"
            "and average results. Slower (~4x) but may recover\n"
            "cells missed at default orientation.")
        form.addRow("TTA (augment):", self.use_tta)

        self.use_fallback = QCheckBox()
        self.use_fallback.setChecked(True)
        self.use_fallback.setToolTip(
            "For frames where cpsam fails to detect a cell,\n"
            "fall back to cellpose + MedSAM + DeepSea.\n"
            "Requires the cellpose (CP3) env.")
        form.addRow("Fallback detection:", self.use_fallback)

        self.use_gap_fill = QCheckBox()
        self.use_gap_fill.setChecked(True)
        self.use_gap_fill.setToolTip(
            "After tracking, fill internal gaps where a cell\n"
            "was lost for a few frames. Uses cpsam(augment=True)\n"
            "then cellpose fallback. Multi-cell only.")
        form.addRow("Gap fill:", self.use_gap_fill)

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
        form = QFormLayout(page)
        self.compute_tracking = QCheckBox()
        self.compute_tracking.setChecked(True)
        form.addRow("Tracking:", self.compute_tracking)
        self.compute_morphology = QCheckBox()
        self.compute_morphology.setChecked(True)
        form.addRow("Morphology:", self.compute_morphology)
        self.compute_edge = QCheckBox()
        self.compute_edge.setChecked(True)
        form.addRow("Edge dynamics:", self.compute_edge)
        self.compute_boundary = QCheckBox()
        self.compute_boundary.setChecked(True)
        form.addRow("Boundary confidence:", self.compute_boundary)
        self.compute_area = QCheckBox()
        self.compute_area.setChecked(True)
        form.addRow("Area stability:", self.compute_area)

        form.addRow(QLabel("<b>Shape analysis:</b>"))
        self.compute_vampire = QCheckBox()
        self.compute_vampire.setChecked(False)
        self.compute_vampire.setToolTip(
            "VAMPIRE shape mode analysis: PCA + K-means clustering\n"
            "of cell contours. Identifies morphological phenotypes\n"
            "and computes heterogeneity (Shannon entropy).\n"
            "Adds ~1s to analysis. Requires vampire-analysis package.")
        form.addRow("VAMPIRE shape modes:", self.compute_vampire)

        self.vampire_clusters = QSpinBox()
        self.vampire_clusters.setRange(2, 15)
        self.vampire_clusters.setValue(5)
        self.vampire_clusters.setToolTip("Number of shape mode clusters")
        form.addRow("Shape mode clusters:", self.vampire_clusters)
        return page

    def _build_info_page(self, title, text):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        layout.addWidget(QLabel(text))
        layout.addStretch()
        return page

    def set_context(self, stage, mode="single"):
        """Switch parameter page based on current stage."""
        page_map = {
            "load": 0, "detect": 0,
            "edit": 2,
            "analyze": 1,
            "export": 3,
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

    def get_detect_params(self):
        return {
            "min_area_px": self.min_area.value(),
            "expected_cells": self.expected_cells.value(),
            "search_radius": self.search_radius.value(),
            "min_track_length": self.min_track_len.value(),
            "use_tta": self.use_tta.isChecked(),
            "use_deepsea": self.use_deepsea.isChecked(),
            "use_fallback": self.use_fallback.isChecked(),
            "use_gap_fill": self.use_gap_fill.isChecked(),
        }

    def get_scale_overrides(self):
        return {
            "um_per_px": self.um_per_px.value(),
            "time_interval_min": self.time_interval.value(),
        }
