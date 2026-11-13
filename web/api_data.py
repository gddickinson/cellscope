"""Data behind the web API: project counts, the defaults/presets/thresholds (via defaults_doc),
the synthetic demo files and the Python reference results on the demo.

The reference results are what the desktop Tracking GUI does with a masks file
(gui_tracking/single_view.py::_on_track: ``DEFAULTS.pixel_thresholds`` then
``core.multi_cell.track_all_cells``) followed by ``core.mask_metrics.compute_label_metrics``
(the colour-by-result metrics). The JS check page compares its own results with these.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np

from . import defaults_doc
from .demo_data import DT_MIN, NAME, NOTE, SOURCE, UM_PER_PX, ensure_demo

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = Path(os.environ.get("CELLSCOPE_WEB_DEMO", ROOT / "web_demo"))


@functools.lru_cache(maxsize=1)
def demo_paths() -> dict:
    return ensure_demo(str(DEMO_DIR))


def demo_info() -> dict:
    """Demo file URLs (relative to the site root) + the simulation's ground truth."""
    p = demo_paths()
    with open(p["truth"]) as f:
        truth = json.load(f)
    rel = {k: "demo/" + os.path.basename(v) for k, v in p.items()}
    return {"name": NAME, "synthetic": True, "note": NOTE, "source": SOURCE, "um_per_px": UM_PER_PX,
            "dt_min": DT_MIN, "files": rel, "truth": truth}


def gui_track_params(um_per_px, dt_min) -> dict:
    """The tracker arguments the Tracking GUI uses for a recording with this scale."""
    from core.pipeline_defaults import DEFAULTS
    px = DEFAULTS.pixel_thresholds(um_per_px=um_per_px, time_interval_min=dt_min)
    return {"min_area_px": px["min_area_px"], "max_hop_px": px["max_hop_px"], "spawn_new_tracks": True,
            "min_track_length": DEFAULTS.min_track_length}


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


@functools.lru_cache(maxsize=1)
def demo_reference() -> dict:
    """Python tracking + colour-by metrics on the demo detections (as the Tracking GUI does)."""
    from core.mask_metrics import compute_label_metrics
    from core.multi_cell import track_all_cells
    labels = np.load(demo_paths()["detections"])["labels"]
    params = gui_track_params(UM_PER_PX, DT_MIN)
    tracks = track_all_cells(labels, **params)
    tracked = np.zeros_like(labels)
    rows = []
    for k, t in enumerate(tracks):
        tracked[t["stack"]] = k + 1
        per = [int(np.bincount(labels[f][t["stack"][f]]).argmax()) if t["stack"][f].any() else 0
               for f in range(len(labels))]
        rows.append({"labels": per, "first": t["first_frame"], "parent": t.get("parent_id")})
    m = compute_label_metrics(tracked, UM_PER_PX, DT_MIN)
    return _clean({"source": "core.multi_cell.track_all_cells + core.mask_metrics.compute_label_metrics",
                   "params": params, "tracks": rows, "per_track": m["per_track"], "states": m["states"],
                   "cell_ids": m["cell_ids"]})


def defaults() -> dict:
    from core.pipeline_defaults import DEFAULTS
    return {"defaults": DEFAULTS.as_dict(), "fields": defaults_doc.pipeline_defaults_doc(),
            "thresholds": defaults_doc.thresholds_doc(), "constants": defaults_doc.module_constants(),
            "presets": defaults_doc.presets_doc()}


def info() -> dict:
    d = defaults()
    truth = demo_info()["truth"]
    gui_mains = sorted(p.name for p in ROOT.glob("main_*.py"))
    return {"title": "CellScope", "repo": "cellscope",
            "n_defaults": len(d["fields"]), "n_presets": len(d["presets"]["order"]),
            "n_core_modules": len([p for p in (ROOT / "core").glob("*.py") if p.name != "__init__.py"]),
            "n_apps": len(gui_mains), "apps": gui_mains,
            "demo": {"n_cells": len(truth["cells"]), "n_frames": len(truth["frames"]), "synthetic": True,
                     "note": NOTE}}
