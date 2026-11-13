"""The tracker's max hop in detect_recording / hybrid_multichannel comes from DEFAULTS and the pixel size.

detect_recording used a literal 150 px when no max_hop_px was passed (and the scale-aware
pixel_thresholds()["max_hop_px"] it computed went unused); hybrid_multichannel re-tracked with literals
200 / 150 / 3. Now: DEFAULTS.tracker_hop_mode = "scale_aware" -> max_hop_um_per_min x frame interval /
µm-per-px at the detection resolution; "legacy_fixed_px" -> DEFAULTS.search_radius_px (the old 150 px).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline_defaults import DEFAULTS, TRACKER_HOP_MODES  # noqa: E402


def test_modes_and_default():
    assert TRACKER_HOP_MODES == ("scale_aware", "legacy_fixed_px")
    assert DEFAULTS.tracker_hop_mode == "scale_aware"


@pytest.mark.parametrize("um, dt, mode, px, source", [
    (1.3046, 10.0, None, 115, "scale_aware"),                 # IC295 at ds=2: 15 µm/min x 10 min / 1.3046
    (0.6523, 10.0, None, 230, "scale_aware"),
    (1.3046, 10.0, "legacy_fixed_px", 150, "legacy_fixed_px"),
    (None, 10.0, None, 150, "fixed (no pixel size or frame interval)"),
    (1.3, None, None, 150, "fixed (no pixel size or frame interval)"),
])
def test_tracker_max_hop_px(um, dt, mode, px, source):
    assert DEFAULTS.tracker_max_hop_px(um, dt, mode) == (px, source)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        DEFAULTS.tracker_max_hop_px(1.0, 10.0, "fixed")


def _fake_cpsam_multi(record):
    def fake(frames, **kw):
        record.update(kw)
        lab = np.zeros(frames.shape, np.int32)
        return {"labels": lab, "masks": lab > 0, "tracks": []}
    return fake


@pytest.mark.parametrize("mode, expect", [(None, 115), ("legacy_fixed_px", 150)])
def test_detect_recording_uses_scale_aware_hop(monkeypatch, mode, expect):
    import core.hybrid_cpsam_multi
    import core.unified_detection as ud
    rec = {}
    monkeypatch.setattr(core.hybrid_cpsam_multi, "detect_hybrid_cpsam_multi", _fake_cpsam_multi(rec))
    frames = np.zeros((3, 1200, 1200), np.uint8)                 # auto-downsample -> 2x
    out = ud.detect_recording(frames, um_per_px=0.6523, time_interval_min=10.0, pipeline_kind="cpsam",
                              tracker_hop_mode=mode)
    assert rec["max_hop_px"] == expect
    assert out["auto"]["tracker_max_hop_px"] == expect
    rec.clear()
    ud.detect_recording(frames, um_per_px=0.6523, time_interval_min=10.0, pipeline_kind="cpsam", max_hop_px=42)
    assert rec["max_hop_px"] == 42                                # an explicit value still wins


@pytest.mark.parametrize("kw, expect", [
    ({"um_per_px": 1.3, "time_interval_min": 10.0}, {"min_area_px": 50, "max_hop_px": 115, "min_track_length": 3}),
    ({"tracker_hop_mode": "legacy_fixed_px"}, {"min_area_px": 200, "max_hop_px": 150, "min_track_length": 3}),
    ({"um_per_px": 1.3, "time_interval_min": 10.0, "min_area_px": 77, "max_hop_px": 33},
     {"min_area_px": 77, "max_hop_px": 33, "min_track_length": 3}),
])
def test_multichannel_retrack_params(monkeypatch, kw, expect):
    import types
    import core.cy5_fallbacks
    import core.hybrid_cpsam_multi
    import core.multi_cell
    from core.hybrid_multichannel import detect_hybrid_multichannel
    monkeypatch.setattr(core.hybrid_cpsam_multi, "detect_hybrid_cpsam_multi", _fake_cpsam_multi({}))
    monkeypatch.setattr(core.cy5_fallbacks, "recover_missed_cells_via_dic_crop", lambda d, lab, f, m, use_tta: (lab, 1, None))
    fake_cp = types.ModuleType("cellpose")
    fake_cp.models = types.SimpleNamespace(CellposeModel=lambda gpu=True: None)
    monkeypatch.setitem(sys.modules, "cellpose", fake_cp)
    seen = {}
    monkeypatch.setattr(core.multi_cell, "track_all_cells", lambda labels, **k: seen.update(k) or [])
    frames = np.zeros((2, 32, 32), np.uint8)
    detect_hybrid_multichannel(frames, frames, use_cy5_recovery=True, **kw)
    for k, v in expect.items():
        assert seen[k] == v, k
