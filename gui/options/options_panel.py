"""Combined options panel: detection + refinement + analysis + presets.

A single tab widget that both GUIs can embed. Exposes:
    get_params() -> RunParams
    set_params(RunParams)
    changed signal (any sub-widget edited)
"""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QScrollArea, QHBoxLayout, QLabel,
)

from gui.options.params import RunParams
from gui.options.detection_panel import DetectionPanel
from gui.options.refinement_panel import RefinementPanel
from gui.options.analysis_panel import AnalysisPanel
from gui.options.presets_widget import PresetSelector


class OptionsPanel(QWidget):
    """Top-level options widget with internal tabs for each panel."""
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._wire_signals()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # --- Preset bar ---
        top = QHBoxLayout()
        top.addWidget(QLabel("Preset:"))
        self.presets = PresetSelector()
        self.presets.preset_applied.connect(self.set_params)
        self.presets.save_requested.connect(self._on_save_requested)
        top.addWidget(self.presets, 1)
        root.addLayout(top)

        # --- Tabs for detection / refinement / analysis ---
        self.tabs = QTabWidget()
        self.detection = DetectionPanel()
        self.refinement = RefinementPanel()
        self.analysis = AnalysisPanel()

        for panel, title in [
            (self.detection, "Detection"),
            (self.refinement, "Refinement"),
            (self.analysis, "Analysis"),
        ]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(panel)
            self.tabs.addTab(scroll, title)

        root.addWidget(self.tabs, 1)

    def _wire_signals(self):
        self.detection.changed.connect(self.changed)
        self.refinement.changed.connect(self.changed)
        self.analysis.changed.connect(self.changed)

    # --- Params I/O ---
    def get_params(self) -> RunParams:
        return RunParams(
            detection=self.detection.get_params(),
            refinement=self.refinement.get_params(),
            analysis=self.analysis.get_params(),
            preset_name=self.presets.current_name() or "custom",
        )

    def set_params(self, p: RunParams):
        self.detection.set_params(p.detection)
        self.refinement.set_params(p.refinement)
        self.analysis.set_params(p.analysis)

    # --- Preset save handler ---
    def _on_save_requested(self, name: str):
        self.presets.save_current(name, self.get_params())
