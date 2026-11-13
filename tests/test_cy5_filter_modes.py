"""CY5_FILTER_MODES is the one list of Tier-4 Cy5 filter strategies.

It said "keep in lockstep" with core.cy5_filter.apply_cy5_filter and the GUI dropdowns but lacked
persistence_guard (the default mode) and nothing used it. Now apply_cy5_filter validates against it and
both GUIs' dropdowns map onto exactly this set, with the initial choice taken from DEFAULTS.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cy5_filter import apply_cy5_filter  # noqa: E402
from core.pipeline_defaults import CY5_FILTER_MODES, DEFAULTS  # noqa: E402


def test_modes_cover_the_default_and_dispatcher():
    assert "persistence_guard" in CY5_FILTER_MODES and DEFAULTS.cy5_filter_mode in CY5_FILTER_MODES
    assert len(set(CY5_FILTER_MODES)) == len(CY5_FILTER_MODES) == 11


def test_apply_cy5_filter_rejects_unknown_mode_even_without_tracks():
    with pytest.raises(ValueError):
        apply_cy5_filter([], mode="persistance_guard")       # typo used to pass silently on empty input
    for mode in CY5_FILTER_MODES:
        kept, dropped, info = apply_cy5_filter([], mode=mode)
        assert kept == [] and dropped == []


_APP = []


def _app():
    """The QApplication, kept referenced for the session (an unreferenced one is collected at once)."""
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def test_focused_dropdown_maps_onto_the_constant():
    _app()
    from gui_focused.params_panel import ParamsPanel
    p = ParamsPanel()
    ids = []
    for i in range(p.cy5_filter_mode.count()):
        p.cy5_filter_mode.setCurrentIndex(i)
        ids.append(p._cy5_filter_mode_value())
    assert sorted(ids) == sorted(CY5_FILTER_MODES)
    p.set_cy5_available(True)
    assert p._cy5_filter_mode_value() == DEFAULTS.cy5_filter_mode
    p.close()


def test_batch_dropdown_maps_onto_the_constant():
    _app()
    from gui_batch.batch_window import BatchWindow
    w = BatchWindow()
    items = [w.cy5_filter_mode.itemText(i).lower() for i in range(w.cy5_filter_mode.count())]
    assert items == list(CY5_FILTER_MODES)
    assert w.cy5_filter_mode.currentText().lower() == DEFAULTS.cy5_filter_mode
    w.close()
