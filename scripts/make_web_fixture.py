"""Write the fixtures the web engines' node tests check against (tests/fixtures/web_engine/).

Every expected value is computed by CellScope's own Python code on SYNTHETIC masks:
  masks.npz      ``labels`` = the web demo's per-frame detections (web/demo_data.py),
                 ``shapes`` = a few hand-built cells (a hole, a detached speck, a border cell, a tie)
  expected.json  core.multi_cell.track_all_cells, core.cell_state.shape_metrics_for_mask /
                 classify_state, core.mask_metrics.compute_label_metrics, core.pipeline_defaults
                 (resolve_downsample, should_use_mirror_pad, decide_backbone, pixel_thresholds),
                 core.runtime_estimate and scipy.optimize.linear_sum_assignment.

Run: python scripts/make_web_fixture.py   (tests/test_web.py checks the fixture is current)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.cell_state import DEFAULT_THRESHOLDS, classify_state, shape_metrics_for_mask  # noqa: E402
from core.mask_metrics import compute_label_metrics  # noqa: E402
from core.multi_cell import track_all_cells  # noqa: E402
from core.pipeline_defaults import (DEFAULTS, decide_backbone, resolve_downsample,  # noqa: E402
                                    should_use_mirror_pad)
from core.runtime_estimate import estimate_full_runtime  # noqa: E402
from web import demo_data  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "web_engine"


def clean(v):
    """JSON-safe: NaN -> None, numpy -> python."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    return v


def shapes_stack():
    """(4, 60, 60) hand-built label frames that exercise mask cleanup, edges and ties."""
    H = W = 60
    yy, xx = np.mgrid[:H, :W]
    s = np.zeros((4, H, W), np.int32)
    disc = lambda cy, cx, r: (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r   # noqa: E731
    s[0][disc(30, 30, 14) & ~disc(28, 32, 4)] = 1                      # cell with a hole
    s[0][disc(8, 8, 3)] = 1                                            # detached speck, same label
    s[1][(np.abs(yy - 30) <= 6) & (np.abs(xx - 28) <= 20)] = 2         # elongated rectangle
    s[1][disc(52, 3, 9)] = 3                                           # touches the border
    s[2][disc(20, 20, 9)] = 4                                          # two equal components:
    s[2][disc(42, 42, 9)] = 4                                          # the first one is kept
    s[3][(yy - 30) ** 2 / 400 + (xx - 30) ** 2 / 100 <= 1] = 5         # ellipse, rounded by size
    return s


def track_case(labels, **kw):
    tracks = track_all_cells(labels, **kw)
    out = []
    for t in tracks:
        per = [int(np.bincount(labels[f][t["stack"][f]]).argmax()) if t["stack"][f].any() else 0
               for f in range(len(labels))]
        out.append({"labels": per, "first": t["first_frame"], "parent": t.get("parent_id"),
                    "centroids": [list(c) for c in t["centroid_history"]]})
    return {"params": kw, "tracks": out}


def build():
    """(arrays for masks.npz, expected dict) — everything computed by the repo's Python."""
    _, labels, _ = demo_data.simulate()
    shapes = shapes_stack()
    um, dt = demo_data.UM_PER_PX, demo_data.DT_MIN
    px = DEFAULTS.pixel_thresholds(um, dt)
    gui = dict(min_area_px=px["min_area_px"], max_hop_px=px["max_hop_px"], spawn_new_tracks=True,
               min_track_length=DEFAULTS.min_track_length)
    tracking = [track_case(labels, **gui),
                track_case(labels, min_area_px=200, max_hop_px=40, spawn_new_tracks=True, min_track_length=5,
                           w_dist=1.0, w_iou=0.0, w_area=0.0, max_gap_frames=1),
                track_case(labels, min_area_px=200, max_hop_px=150, spawn_new_tracks=False)]
    tracked = np.zeros_like(labels)
    for k, t in enumerate(track_all_cells(labels, **gui)):
        tracked[t["stack"]] = k + 1

    shape_rows = []
    for name, stack in (("labels", labels[::7]), ("shapes", shapes)):
        for f in range(len(stack)):
            for lab in np.unique(stack[f]):
                if lab == 0:
                    continue
                d = shape_metrics_for_mask(stack[f] == lab)
                shape_rows.append({"stack": name, "frame": f * (7 if name == "labels" else 1), "label": int(lab),
                                   "metrics": d, "state_um": classify_state(d, um_per_px=um),
                                   "state_noscale": classify_state(d)})

    rng = np.random.default_rng(7)
    from scipy.optimize import linear_sum_assignment
    lsa = []
    for n in (1, 3, 6, 12):
        c = rng.uniform(0, 100, (n, n))
        c[rng.random((n, n)) < 0.3] = 1e9
        r, k = linear_sum_assignment(c)
        lsa.append({"cost": c, "total": float(c[r, k].sum())})

    planner = {
        "downsample": [{"spec": s, "shape": sh, "out": list(resolve_downsample(s, sh))}
                       for s in ("auto", "off", 3, "2") for sh in ((438, 759), (1028, 1028), (1100, 900),
                                                                    (1499, 1499), (1500, 10), (2048, 2048))],
        "mirror": [{"spec": s, "shape": sh, "out": should_use_mirror_pad(s, sh)}
                   for s in ("auto", "on", "off") for sh in ((999, 2000), (1000, 1000), (512, 512))],
        "backbone": [{"counts": c, "out": decide_backbone(c)}
                     for c in ([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], [1, 1, 2, 2], [0, 0, 0], [3, 9, 14, 2, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2])],
        "pixel": [{"um": u, "dt": d, "out": DEFAULTS.pixel_thresholds(u, d)}
                  for u in (None, 0.4, 0.6523, 1.3, 2.5) for d in (None, 10.0, 2.5)],
        "probe": [{"n": n, "k": 11, "out": [int(i) for i in np.linspace(0, n - 1, min(11, n), dtype=int)]}
                  for n in (1, 5, 11, 12, 40, 97, 300, 1001)],
        "runtime": [{"args": a, "out": estimate_full_runtime(*a)}
                    for a in ((40.0, 4, 97), (39.74, 14, 97), (2.0, 5, 10), (2.0, 9, 10), (1.0, 10, 1))],
    }
    expected = {
        "source": "scripts/make_web_fixture.py (CellScope Python on synthetic masks)",
        "um_per_px": um, "dt_min": dt, "defaults": DEFAULTS.as_dict(), "thresholds": DEFAULT_THRESHOLDS,
        "tracking": tracking, "shapes": shape_rows,
        "metrics_scaled": compute_label_metrics(tracked, um, dt),
        "metrics_noscale": compute_label_metrics(tracked),
        "lsa": lsa, "planner": planner,
    }
    return {"labels": labels, "shapes": shapes}, clean(expected)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, expected = build()
    np.savez_compressed(OUT / "masks.npz", **arrays)
    (OUT / "expected.json").write_text(json.dumps(expected, separators=(",", ":")))
    print(f"wrote {OUT}/masks.npz and expected.json ({len(expected['tracking'])} tracking cases, "
          f"{len(expected['shapes'])} shapes)")


if __name__ == "__main__":
    main()
