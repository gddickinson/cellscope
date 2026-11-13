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


_APP = []


def _gui_window(monkeypatch, detect):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    import core.modality
    import core.unified_detection
    from gui_focused.main_window import FocusedMainWindow
    monkeypatch.setattr(core.modality, "detect_modality", lambda frames: "dic")
    monkeypatch.setattr(core.unified_detection, "detect_recording", detect)
    w = FocusedMainWindow()
    w.recording = {"frames": np.zeros((20, 64, 64), np.uint8), "um_per_px": 1.0,
                   "time_interval_min": 10.0}
    return _APP[0], w


def _six_cells(*a, **k):
    lab = np.zeros((1, 64, 64), np.int32)
    for c in range(6):                                   # 6 cells -> "medium", x2.0
        lab[0, 4 + 10 * c:10 + 10 * c, 5:15] = c + 1
    return {"labels": lab}


def _finish(app, w):
    assert w._test_worker.wait(10000)
    app.processEvents()                                  # deliver the queued finished/error signal


def test_focused_gui_test_frame_reports_estimate(monkeypatch):
    app, w = _gui_window(monkeypatch, _six_cells)
    w._on_test_frame()
    _finish(app, w)
    msg = w.status.currentMessage()
    assert "6 cell(s)" in msg and "(20 frames)" in msg and "× 2.0 medium post-proc" in msg, msg


def test_focused_gui_test_frame_runs_off_the_gui_thread(monkeypatch):
    """_on_test_frame returns while detection is still running (worker thread), and the button is
    disabled until it finishes."""
    import threading
    gate, ran_on = threading.Event(), []

    def slow(*a, **k):
        ran_on.append(threading.current_thread() is threading.main_thread())
        gate.wait(10)
        return _six_cells()
    app, w = _gui_window(monkeypatch, slow)
    w.pipeline.enable_stage("detect", True)
    w._on_test_frame()
    assert w._test_worker.isRunning() and not w.pipeline.btn_test_frame.isEnabled()
    assert w.status.currentMessage().startswith("Testing frame")
    gate.set()
    _finish(app, w)
    assert ran_on == [False]
    assert "6 cell(s)" in w.status.currentMessage() and w.pipeline.btn_test_frame.isEnabled()


def test_focused_gui_test_frame_error_is_reported(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no model")
    app, w = _gui_window(monkeypatch, boom)
    w._on_test_frame()
    _finish(app, w)
    assert "Test failed on frame 0: no model" in w.status.currentMessage()
