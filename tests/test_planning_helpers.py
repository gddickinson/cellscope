"""Pure helpers factored out for the web planner: the auto-select backbone decision
(core.pipeline_defaults.decide_backbone) and the Test-on-frame runtime estimate
(core.runtime_estimate), plus the focused GUI still reporting the same estimate."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline_defaults import MULTICELL_THRESHOLD, decide_backbone  # noqa: E402
from core.runtime_estimate import estimate_full_runtime, format_duration  # noqa: E402


def test_decide_backbone_sparse_and_dense():
    sparse = decide_backbone([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])   # docstring example
    assert sparse["choice"] == "cpsam_dic" and sparse["p75_count"] == 1.0
    dense = decide_backbone([9, 10, 12, 14, 11, 9, 13, 12, 10, 11, 14])
    assert dense["choice"] == "cpsam" and dense["threshold"] == MULTICELL_THRESHOLD
    edge = decide_backbone([1, 1, 2, 2])                          # p75 = 2.0 >= 1.5
    assert edge["choice"] == "cpsam" and edge["p75_count"] == pytest.approx(2.0)
    assert edge["median_count"] == 1.5


@pytest.mark.parametrize("n, mult, dens", [(0, 1.5, "sparse"), (4, 1.5, "sparse"), (5, 2.0, "medium"),
                                            (9, 2.0, "medium"), (10, 2.5, "dense"), (17, 2.5, "dense")])
def test_runtime_multiplier(n, mult, dens):
    est = estimate_full_runtime(40.0, n, 97)
    assert est["multiplier"] == mult and est["density"] == dens
    assert est["detect_only_s"] == pytest.approx(40.0 * 97)
    assert est["full_s"] == pytest.approx(40.0 * 97 * mult)


def test_format_duration():
    assert format_duration(42) == "42s"
    assert format_duration(210) == "3.5 min"
    assert format_duration(9720) == "2.7 h"


def test_focused_gui_test_frame_reports_estimate(monkeypatch):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import core.modality
    import core.unified_detection
    from gui_focused.main_window import FocusedMainWindow

    lab = np.zeros((1, 64, 64), np.int32)
    for k in range(6):                                   # 6 cells -> "medium", x2.0
        lab[0, 4 + 10 * k:10 + 10 * k, 5:15] = k + 1
    monkeypatch.setattr(core.modality, "detect_modality", lambda frames: "dic")
    monkeypatch.setattr(core.unified_detection, "detect_recording", lambda *a, **k: {"labels": lab})
    w = FocusedMainWindow()
    w.recording = {"frames": np.zeros((20, 64, 64), np.uint8), "um_per_px": 1.0,
                   "time_interval_min": 10.0}
    w._on_test_frame()
    msg = w.status.currentMessage()
    assert "6 cell(s)" in msg and "(20 frames)" in msg and "× 2.0 medium post-proc" in msg, msg
    app.processEvents()
