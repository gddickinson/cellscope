"""Analysis options panel: which analytics to compute + scale overrides."""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QCheckBox,
)

from gui.options.params import AnalysisParams


class AnalysisPanel(QWidget):
    """Panel for choosing which analytics to run + optional overrides."""
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.set_params(AnalysisParams())
        self._wire_signals()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # --- Analytics checkboxes ---
        a = QGroupBox("Analytics to compute")
        aform = QFormLayout()
        self.compute_tracking = QCheckBox(
            "Tracking (centroids, speed, MSD, persistence)"
        )
        self.compute_morphology = QCheckBox(
            "Morphology (area, perimeter, circularity, AR)"
        )
        self.compute_edge_dynamics = QCheckBox(
            "Edge dynamics (protrusion/retraction kymograph)"
        )
        self.compute_boundary_confidence = QCheckBox(
            "Boundary confidence (image gradient along contour)"
        )
        self.compute_area_stability = QCheckBox(
            "Area stability (mean, CV, drift)"
        )
        self.compute_membrane_quality = QCheckBox(
            "Membrane quality (texture-aware, slower)"
        )
        for cb in (self.compute_tracking, self.compute_morphology,
                   self.compute_edge_dynamics,
                   self.compute_boundary_confidence,
                   self.compute_area_stability,
                   self.compute_membrane_quality):
            aform.addRow(cb)
        a.setLayout(aform)
        root.addWidget(a)

        # --- Scale overrides ---
        s = QGroupBox("Scale overrides (leave at 0 to use recording JSON)")
        sform = QFormLayout()
        self.um_per_px = QDoubleSpinBox()
        self.um_per_px.setRange(0.0, 10.0)
        self.um_per_px.setSingleStep(0.05)
        self.um_per_px.setDecimals(4)
        self.um_per_px.setToolTip(
            "Override pixel size in μm. 0 = use the recording's JSON "
            "sidecar value."
        )
        sform.addRow("um_per_px:", self.um_per_px)

        self.time_interval_min = QDoubleSpinBox()
        self.time_interval_min.setRange(0.0, 120.0)
        self.time_interval_min.setSingleStep(0.5)
        self.time_interval_min.setDecimals(3)
        self.time_interval_min.setToolTip(
            "Override time interval in minutes. 0 = use JSON sidecar."
        )
        sform.addRow("time_interval_min:", self.time_interval_min)
        s.setLayout(sform)
        root.addWidget(s)

        root.addStretch()

    def _wire_signals(self):
        widgets = [
            self.compute_tracking, self.compute_morphology,
            self.compute_edge_dynamics,
            self.compute_boundary_confidence,
            self.compute_area_stability,
            self.compute_membrane_quality,
            self.um_per_px, self.time_interval_min,
        ]
        for w in widgets:
            sig = (getattr(w, "valueChanged", None)
                   or getattr(w, "toggled", None))
            if sig is not None:
                sig.connect(self.changed)

    def get_params(self) -> AnalysisParams:
        um = float(self.um_per_px.value())
        dt = float(self.time_interval_min.value())
        return AnalysisParams(
            compute_tracking=self.compute_tracking.isChecked(),
            compute_morphology=self.compute_morphology.isChecked(),
            compute_edge_dynamics=self.compute_edge_dynamics.isChecked(),
            compute_boundary_confidence=(
                self.compute_boundary_confidence.isChecked()
            ),
            compute_area_stability=self.compute_area_stability.isChecked(),
            compute_membrane_quality=(
                self.compute_membrane_quality.isChecked()
            ),
            override_um_per_px=(um if um > 0 else None),
            override_time_interval_min=(dt if dt > 0 else None),
        )

    def set_params(self, p: AnalysisParams):
        self.blockSignals(True)
        try:
            self.compute_tracking.setChecked(p.compute_tracking)
            self.compute_morphology.setChecked(p.compute_morphology)
            self.compute_edge_dynamics.setChecked(p.compute_edge_dynamics)
            self.compute_boundary_confidence.setChecked(
                p.compute_boundary_confidence
            )
            self.compute_area_stability.setChecked(p.compute_area_stability)
            self.compute_membrane_quality.setChecked(
                p.compute_membrane_quality
            )
            self.um_per_px.setValue(p.override_um_per_px or 0.0)
            self.time_interval_min.setValue(
                p.override_time_interval_min or 0.0
            )
        finally:
            self.blockSignals(False)
        self.changed.emit()
