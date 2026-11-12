"""The Hungarian tracker's cost weights come from DEFAULTS (the single source of truth).

README / INTERFACE / ROADMAP said the weights live in ``DEFAULTS.track_w_dist / track_w_iou /
track_w_area``; they were bare literals in the ``track_all_cells`` signature instead. The
values are kept as they were (1.0 / 1.0 / 0.5, the weights the Pos7-WT validation ran with,
commit 60ea19c), so tracking output does not change.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.multi_cell import track_all_cells  # noqa: E402
from core.pipeline_defaults import DEFAULTS  # noqa: E402


def _disc(shape, cy, cx, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _two_cells_crossing():
    T, H, W = 8, 80, 80
    lab = np.zeros((T, H, W), np.int32)
    for t in range(T):
        a = _disc((H, W), 30, 15 + 6 * t, 9)          # big cell moving right
        b = _disc((H, W), 44, 60 - 5 * t, 6)          # small cell moving left
        lab[t][b] = 1 + (t % 2)                       # per-frame ids shuffle
        lab[t][a & ~b] = 2 - (t % 2)
    return lab


def test_defaults_hold_tracker_weights():
    assert DEFAULTS.track_w_dist == 1.0
    assert DEFAULTS.track_w_iou == 1.0
    assert DEFAULTS.track_w_area == 0.5


def test_track_all_cells_uses_defaults_weights():
    lab = _two_cells_crossing()
    implicit = track_all_cells(lab, min_area_px=20, spawn_new_tracks=True)
    explicit = track_all_cells(lab, min_area_px=20, spawn_new_tracks=True,
                               w_dist=DEFAULTS.track_w_dist, w_iou=DEFAULTS.track_w_iou,
                               w_area=DEFAULTS.track_w_area,
                               max_gap_frames=DEFAULTS.max_gap_frames)
    assert len(implicit) == len(explicit) == 2
    for a, b in zip(implicit, explicit):
        assert np.array_equal(a["stack"], b["stack"])


def test_track_all_cells_signature_defaults_are_none():
    import inspect
    sig = inspect.signature(track_all_cells)
    for k in ("w_dist", "w_iou", "w_area", "max_gap_frames"):
        assert sig.parameters[k].default is None, k
