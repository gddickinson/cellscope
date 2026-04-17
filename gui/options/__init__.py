"""Shared option panels for both GUIs.

Contents:
- params.py              — RunParams / DetectionParams / RefinementParams /
                           AnalysisParams dataclasses
- detection_panel.py     — DetectionPanel widget
- refinement_panel.py    — RefinementPanel widget
- analysis_panel.py      — AnalysisPanel widget
- presets.py             — built-in + user preset I/O
- presets_widget.py      — PresetSelector widget
- options_panel.py       — OptionsPanel combining all of the above
"""
from gui.options.params import (
    RunParams, DetectionParams, RefinementParams, AnalysisParams,
)
from gui.options.detection_panel import DetectionPanel
from gui.options.refinement_panel import RefinementPanel
from gui.options.analysis_panel import AnalysisPanel
from gui.options.presets_widget import PresetSelector
from gui.options.options_panel import OptionsPanel

__all__ = [
    "RunParams", "DetectionParams", "RefinementParams", "AnalysisParams",
    "DetectionPanel", "RefinementPanel", "AnalysisPanel",
    "PresetSelector", "OptionsPanel",
]
