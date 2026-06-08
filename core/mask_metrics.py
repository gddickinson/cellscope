"""Lightweight per-cell metrics computed directly from a label stack.

Powers the "Colour masks by result" overlay in BOTH the focused GUI
viewer (`gui_focused/image_viewer.py`) and the mask editor
(`gui/mask_editor.py`). Computed from masks alone — so recolouring works
immediately after detection or after loading a `masks.npz`, with NO full
`Analyze` run required.

Reuses `core.cell_state.classify_track_states` (per-frame shape metrics
+ rounded/spread state in a single regionprops pass) and
`core.tracking.extract_centroids` (centroid-based motion). The heavier,
authoritative analysis lives in `core.pipeline.analyze_recording`; this
is the fast subset needed for interactive recolouring.

NB: the per-track scalars here (mean_speed, mean_area, …) are
WHOLE-TRACK — they exist purely to colour cells for visual inspection,
not for statistical comparison. The state-stratified, comparison-grade
metrics live in `core.state_analysis` / the per_cell.csv schema.

`compute_label_metrics(labels, um_per_px, dt_min)` returns:
    per_track:  {cell_id: {metric_key: float}}      # one value per cell
    per_frame:  {cell_id: {metric_key: (N,) array}} # one value per frame
    states:     {cell_id: (N,) int8 array}           # STATE_CODE values
    n_frames:   int
    cell_ids:   sorted list of present cell IDs
"""
from __future__ import annotations

import numpy as np

# Per-frame state → compact int code (kept tiny for per-frame storage).
# Binary model, ordered by roundedness: spread → rounded.
STATE_CODE = {"unknown": 0, "spread": 1, "rounded": 2}
# Edge-truncated frames carry the `unknown` state string but get their own
# overlay code so the GUI can paint them distinctly (excluded from shape).
STATE_CODE_EDGE = 3

# The per-track scalars surfaced for colouring. Order is the order the
# colour-by dropdown presents them (see gui/metric_coloring.py).
PER_TRACK_KEYS = (
    "mean_speed", "total_distance", "net_displacement", "persistence",
    "mean_area", "mean_circularity", "mean_solidity",
    "mean_aspect_ratio", "mean_eccentricity", "frames_tracked",
    "frac_rounded",
)
PER_FRAME_KEYS = ("area", "circularity", "solidity", "speed")


def _empty():
    return {"per_track": {}, "per_frame": {}, "states": {},
            "n_frames": 0, "cell_ids": []}


def compute_label_metrics(labels, um_per_px=None, dt_min=None,
                          thresholds=None, max_cells=200):
    """Compute per-cell colour-by metrics from a labelled mask stack.

    Args:
        labels: (N, H, W) int label stack (0 = background) — or a bool
            mask stack (treated as a single cell, ID 1).
        um_per_px: physical pixel size (µm); None → values stay in px.
        dt_min: frame interval (min); None → speed stays in px/frame.
        thresholds: optional core.cell_state threshold override.
        max_cells: safety cap on the number of distinct IDs processed.
    """
    out = _empty()
    if labels is None or getattr(labels, "ndim", 0) != 3:
        return out
    if labels.dtype == bool:
        labels = labels.astype(np.int32)

    n = len(labels)
    out["n_frames"] = n
    has_scale = bool(um_per_px and dt_min)
    um = float(um_per_px) if um_per_px else 1.0
    dt = float(dt_min) if dt_min else 1.0

    ids = np.unique(labels)
    ids = [int(i) for i in ids.tolist() if i > 0][:max_cells]
    out["cell_ids"] = ids
    if not ids:
        return out

    from core.cell_state import (
        classify_track_states, state_fraction, STATE_ROUNDED)
    from core.tracking import extract_centroids

    for cid in ids:
        stack = labels == cid
        present = stack.any(axis=(1, 2))
        if not present.any():
            continue
        sd = classify_track_states(stack, thresholds)
        states = sd["states"]
        m = sd["metrics"]  # (N,) arrays: area(px²), circularity, solidity…
        cents = extract_centroids(stack)  # (N, 2) px, NaN where absent

        area = m["area"].astype(np.float32)
        area_um2 = area * (um * um) if has_scale else area

        # Per-frame instantaneous speed, aligned so speed[i] is the
        # step from frame i → i+1 (last frame left NaN).
        step = np.linalg.norm(np.diff(cents, axis=0), axis=1)
        speed = np.full(n, np.nan, dtype=np.float32)
        with np.errstate(invalid="ignore"):
            speed[:-1] = step * (um / dt)

        codes = np.array(
            [STATE_CODE.get(s, 0) for s in states], dtype=np.int8)
        edge = sd.get("edge")
        if edge is not None:
            codes[np.asarray(edge, dtype=bool)] = STATE_CODE_EDGE
        out["states"][cid] = codes
        out["per_frame"][cid] = {
            "area": area_um2,
            "circularity": m["circularity"].astype(np.float32),
            "solidity": m["solidity"].astype(np.float32),
            "speed": speed,
        }
        out["per_track"][cid] = _track_scalars(
            cents, step, speed, m, area_um2, present, um, has_scale,
            states, state_fraction, STATE_ROUNDED)
    return out


def _track_scalars(cents, step, speed, m, area_um2, present, um, has_scale,
                   states, state_fraction, rounded):
    """Reduce one cell's per-frame data to the per-track scalar dict."""
    valid = ~np.isnan(cents[:, 0])
    cv = cents[valid]
    if len(cv) >= 2:
        seg = np.linalg.norm(np.diff(cv, axis=0), axis=1) * (
            um if has_scale else 1.0)
        total_dist = float(seg.sum())
        net = float(np.linalg.norm(cv[-1] - cv[0]) * (um if has_scale else 1.0))
        persistence = (net / total_dist) if total_dist > 0 else 0.0
    else:
        total_dist = net = persistence = 0.0
    fin = speed[np.isfinite(speed)]
    mean_speed = float(np.mean(fin)) if fin.size else float("nan")

    def _nanmean(a):
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if a.size else float("nan")

    return {
        "mean_speed":         mean_speed,
        "total_distance":     total_dist,
        "net_displacement":   net,
        "persistence":        float(persistence),
        "mean_area":          _nanmean(area_um2),
        "mean_circularity":   _nanmean(m["circularity"]),
        "mean_solidity":      _nanmean(m["solidity"]),
        "mean_aspect_ratio":  _nanmean(m["aspect_ratio"]),
        "mean_eccentricity":  _nanmean(m["eccentricity"]),
        "frames_tracked":     float(int(present.sum())),
        "frac_rounded":       float(state_fraction(states, rounded)),
    }
