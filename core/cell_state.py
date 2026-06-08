"""Classify each cell-frame as ROUNDED (balled-up) vs SPREAD (adherent).

Mitotic / pre-mitotic / dying cells detach from the substrate and round
up: high circularity, high solidity, near-1 aspect ratio. Adherent
spread cells are irregular: protrusions, lower circularity/solidity.

Why a binary split + per-state metrics: a whole-track average (e.g.
"mean speed" over all frames) is a TIME-WEIGHTED blend of the two states,
so a condition that simply spends more time rounded looks slower/rounder
even if its cells behave identically within each state. To measure the
state itself, every per-frame metric is computed SEPARATELY over rounded
frames and over spread frames. Only lifetime quantities — cell count,
divisions, and % time rounded — are taken over the whole track.

Binary classification rule (per cell-frame), learned from hand labels
(0.90 CV; size / "footprint collapse" tracks the labeller far better than
circularity — see scripts/ic295_eval_state_rule.py):
    rounded  if area_um2 ≤ 960 AND eccentricity ≤ 0.85   (needs um_per_px)
    spread   if the shape is measurable but not rounded
    unknown  if shape metrics are undefined (empty / sub-min-area mask)
             or the mask is truncated by the image border (edge_touch)
    fallback when no scale is known: circularity ≥ 0.80 AND solidity ≥ 0.92

Tunable via classify_state(thresholds=..., um_per_px=...).
"""
from __future__ import annotations

import numpy as np
from skimage import measure


STATE_ROUNDED = "rounded"
STATE_SPREAD = "spread"
STATE_UNKNOWN = "unknown"

# --- Deprecated 3-state aliases ---------------------------------------
# Kept so older standalone state scripts (analyze_state_motility.py,
# compare_state_datasets.py, …) still import without error. They now
# resolve to the binary model: balled→rounded, attached/transitional→
# spread. DO NOT use in new code — use STATE_ROUNDED / STATE_SPREAD.
STATE_BALLED = STATE_ROUNDED
STATE_ATTACHED = STATE_SPREAD
STATE_TRANSITIONAL = STATE_SPREAD

DEFAULT_THRESHOLDS = {
    # --- PRIMARY rule (learned from hand labels, 0.90 CV vs the old 0.60) ---
    # A cell-frame is `rounded` iff its physical footprint is small AND it is
    # not elongated: area_um2 <= rounded_area_um2 AND eccentricity <=
    # rounded_eccentricity. Equivalent to the depth-2 decision tree fit on
    # ic295 state_labels (see scripts/ic295_eval_state_rule.py). Uses µm² so
    # it is robust across magnifications; needs the recording's um_per_px.
    "rounded_area_um2": 960.0,
    "rounded_eccentricity": 0.85,
    # --- FALLBACK rule (no scale available) — legacy circularity/solidity ---
    "rounded_circ": 0.80,
    "rounded_solid": 0.92,
    "min_area_px": 200,
    # Cells whose mask reaches within this many px of the image border are
    # truncated by the field of view; their shape metrics are unreliable so
    # they are classified `unknown` and excluded from SHAPE/STATE analysis
    # (but still counted + tracked). 0 = literal border contact.
    "edge_margin_px": 0,
}


def shape_metrics_for_mask(mask_bool):
    """Compute the shape metrics needed for state classification.

    Returns dict with: area, perimeter, circularity, solidity,
    eccentricity, aspect_ratio. Returns NaN dict if mask is empty
    or below min_area_px.
    """
    if not mask_bool.any():
        return _nan_metrics()
    # Fill small holes + drop rogue specks before measuring shape — both
    # otherwise deflate circularity / solidity (see core.mask_cleanup).
    from core.mask_cleanup import clean_cell_mask
    mask_bool = clean_cell_mask(mask_bool)
    # Edge-truncation flag — measured on the cleaned FULL-FRAME mask, so a
    # speck that touched the border but got removed doesn't falsely flag an
    # interior cell. Carried in the result so classify_state can void it.
    from core.edge_filter import mask_touches_edge
    edge = mask_touches_edge(mask_bool, DEFAULT_THRESHOLDS["edge_margin_px"])
    props = measure.regionprops(mask_bool.astype(np.uint8))
    if not props:
        return _nan_metrics(edge)
    p = props[0]
    if p.area < DEFAULT_THRESHOLDS["min_area_px"]:
        return _nan_metrics(edge)
    area = float(p.area)
    perimeter = float(p.perimeter)
    circ = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else np.nan
    minr, minc, maxr, maxc = p.bbox
    aspect = (maxc - minc) / max(maxr - minr, 1e-6)
    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": float(circ),
        "solidity": float(p.solidity),
        "eccentricity": float(p.eccentricity),
        "aspect_ratio": float(aspect),
        "edge_touch": edge,
    }


def _nan_metrics(edge=False):
    d = {k: float("nan") for k in
         ("area", "perimeter", "circularity", "solidity",
          "eccentricity", "aspect_ratio")}
    d["edge_touch"] = bool(edge)
    return d


def classify_state(metrics, thresholds=None, um_per_px=None):
    """Classify a single cell-frame as rounded / spread / unknown.

    `metrics` is the dict returned by `shape_metrics_for_mask`. Returns one
    of STATE_ROUNDED / STATE_SPREAD / STATE_UNKNOWN.

    PRIMARY rule (when `um_per_px` is known): `rounded` iff the physical
    footprint is small AND the cell is not elongated —
    `area_um2 <= rounded_area_um2 AND eccentricity <= rounded_eccentricity`.
    This is the decision tree learned from the hand labels (0.90 CV; size /
    "footprint collapse" tracks the labeller far better than circularity).
    FALLBACK rule (no scale): the legacy `circ >= rounded_circ AND
    solid >= rounded_solid` shape gate.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    # Edge-truncated cell-frames have an unreliable outline — void the shape
    # state so they drop out of every per-state aggregation (they are still
    # counted + tracked elsewhere).
    if metrics.get("edge_touch"):
        return STATE_UNKNOWN
    area = metrics.get("area", np.nan)
    ecc = metrics.get("eccentricity", np.nan)
    # Primary: physical size + elongation. `area` is NaN for empty / sub-
    # min-area masks, so those fall through to `unknown` below.
    if um_per_px and np.isfinite(area) and np.isfinite(ecc):
        area_um2 = float(area) * float(um_per_px) ** 2
        if (area_um2 <= th["rounded_area_um2"]
                and ecc <= th["rounded_eccentricity"]):
            return STATE_ROUNDED
        return STATE_SPREAD
    # Fallback: legacy circularity / solidity gate (no scale available).
    circ = metrics.get("circularity", np.nan)
    solid = metrics.get("solidity", np.nan)
    if np.isnan(circ) or np.isnan(solid):
        return STATE_UNKNOWN
    if circ >= th["rounded_circ"] and solid >= th["rounded_solid"]:
        return STATE_ROUNDED
    return STATE_SPREAD


def classify_track_states(track_stack, thresholds=None, um_per_px=None):
    """Classify each frame of a track stack.

    `um_per_px` enables the primary physical-size state rule (see
    `classify_state`); without it the legacy circularity/solidity gate is
    used. Pass the recording's pixel size whenever it is known.

    Returns dict with:
        states:   length-T list of state labels
        metrics:  dict of length-T arrays (area, circularity, ...)
        edge:     length-T bool array — True where the mask is truncated by
                  the image border (those frames are state `unknown`).
    """
    n = track_stack.shape[0]
    keys = ("area", "perimeter", "circularity", "solidity",
            "eccentricity", "aspect_ratio")
    metrics = {k: np.full(n, np.nan, dtype=np.float32) for k in keys}
    states = [STATE_UNKNOWN] * n
    edge = np.zeros(n, dtype=bool)
    for i in range(n):
        m = track_stack[i].astype(bool)
        d = shape_metrics_for_mask(m)
        for k in keys:
            metrics[k][i] = d[k]
        edge[i] = bool(d.get("edge_touch", False))
        states[i] = classify_state(d, thresholds, um_per_px=um_per_px)
    return {"states": states, "metrics": metrics, "edge": edge}


def state_segments(states):
    """Find contiguous runs of the same state.

    Returns list of (state, start_frame, end_frame_inclusive).
    Useful for picking pure-state segments for motility analysis.

    Accepts either a Python list or a numpy array — the previous
    `if not states:` truthiness check raised ValueError on numpy
    arrays, silently breaking per-state speed analysis (the
    ic295_analyze_one caller swallowed the exception and wrote
    mean_speed_attached/balled = None for every cell in the corpus).
    """
    if len(states) == 0:
        return []
    out = []
    cur = states[0]
    cur_start = 0
    for i in range(1, len(states)):
        if states[i] != cur:
            out.append((cur, cur_start, i - 1))
            cur = states[i]
            cur_start = i
    out.append((cur, cur_start, len(states) - 1))
    return out


def state_fraction(states, target_state):
    """Fraction of valid (non-unknown) frames in target_state."""
    valid = [s for s in states if s != STATE_UNKNOWN]
    if not valid:
        return 0.0
    return sum(1 for s in valid if s == target_state) / len(valid)


def per_state_means(metrics, states, target_state):
    """For one state, return per-shape-metric mean across frames in
    that state. Returns NaN dict if no frames in target_state."""
    out = {}
    in_state = np.array([s == target_state for s in states])
    for k, v in metrics.items():
        valid = in_state & ~np.isnan(v)
        out[k] = float(np.mean(v[valid])) if valid.any() else float("nan")
    return out
