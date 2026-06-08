"""Shared cell-state classification + per-state metric annotation.

Single source of truth used by BOTH analysis entry points:
  - scripts/ic295_analyze_one.py  (IC295 batch analysis)
  - gui_focused/workers.py        (focused GUI "Analyze")

so the GUI and the batch script compute IDENTICAL per-state metrics.

State model (see core.cell_state): a binary split per cell-frame —
`rounded` (balled-up / mitotic) vs `spread` (adherent), plus `unknown`
for frames with no measurable shape. EVERY per-frame metric is computed
separately over the cell's rounded frames and its spread frames, so a
comparison is never confounded by how much time a cell spends in each
state. Only the lifetime quantities (% time rounded, division counts,
track length, cell count) are taken over the whole track.

`annotate_state(result, cell_stack, um, dt)` mutates `result` in place,
adding the keys below.

Lifetime (state-independent):
  state_per_frame
  frac_rounded, frac_spread

Per-state (one value per state, computed only over that state's frames):
  mean_speed_{rounded,spread}        µm/min, per-frame step speeds
  persistence_{rounded,spread}       lag-1 directional autocorrelation
  straightness_{rounded,spread}      net / path displacement
  mean_area_um2_{rounded,spread}
  mean_circularity_{rounded,spread}
  mean_solidity_{rounded,spread}
  mean_aspect_ratio_{rounded,spread}
  mean_eccentricity_{rounded,spread}

Display aliases (focused-GUI metrics panel / metrics.json):
  {rounded,spread}_mean_speed_um_per_min
  {rounded,spread}_persistence_lag1
  {rounded,spread}_n_speed_samples
"""
from __future__ import annotations

import numpy as np

# Speed cap (µm/min) for the per-frame speed — discards tracking glitches
# (a keratinocyte cannot exceed ~15 µm/min).
_PF_SPEED_CAP = 15.0

_SHAPE_KEYS = ("circularity", "solidity", "aspect_ratio", "eccentricity")


def _safe(fn, default=None):
    """Call fn(); swallow any exception, return `default`. The per-state
    primitives legitimately fail (e.g. a state with no contiguous segment
    long enough to measure); we want None there, not an aborted run."""
    try:
        return fn()
    except Exception:
        return default


def _f(x):
    """float-or-None, dropping NaN/inf."""
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def annotate_state(result, cell_stack, um_per_px, dt_min, thresholds=None):
    """Classify each frame rounded/spread and attach per-state metrics.

    Args:
        result: dict mutated in place (a per-cell analysis result).
        cell_stack: (N, H, W) bool/int mask stack for ONE cell.
        um_per_px: physical pixel size (µm); px → µm conversion.
        dt_min: frame interval (min); displacement → speed.
        thresholds: optional override for core.cell_state thresholds.

    Returns the same `result` dict (for chaining).
    """
    from core.cell_state import (
        classify_track_states, state_fraction, per_state_means,
        STATE_ROUNDED, STATE_SPREAD)
    from core.motility_state import state_persistence, state_total_displacement
    from core.tracking import extract_centroids

    um = float(um_per_px) if um_per_px else 1.0
    dt = float(dt_min) if dt_min else 1.0

    stack = cell_stack.astype(bool)
    sd = classify_track_states(stack, thresholds)
    states = np.asarray(sd["states"])
    shape = sd["metrics"]   # dict of (N,) arrays incl. area (px), circ, …
    result["state_per_frame"] = states.tolist()
    result["frac_rounded"] = float(state_fraction(states, STATE_ROUNDED))
    result["frac_spread"] = float(state_fraction(states, STATE_SPREAD))

    cents = extract_centroids(stack)

    # Per-frame step speeds (µm/min) paired with the state at the step's
    # start frame — no contiguous-segment requirement, so brief rounded
    # events are not lost.
    if len(cents) > 1:
        step = np.linalg.norm(np.diff(cents, axis=0), axis=1) * um / dt
        step_state = states[:-1]
    else:
        step = np.array([])
        step_state = np.array([])

    for target, prefix in ((STATE_ROUNDED, "rounded"),
                           (STATE_SPREAD, "spread")):
        # --- motility ---
        sm = (step[step_state == target] if len(step) else np.array([]))
        sm = sm[np.isfinite(sm)]
        sm = sm[sm <= _PF_SPEED_CAP]
        result[f"mean_speed_{prefix}"] = (
            float(np.mean(sm)) if len(sm) else None)
        result[f"n_speed_samples_{prefix}"] = int(len(sm))

        per = _safe(lambda t=target: state_persistence(cents, states, t, um))
        result[f"persistence_{prefix}"] = (
            _f(per["lag1"]) if per else None)

        disp = _safe(lambda t=target:
                     state_total_displacement(cents, states, t, um))
        result[f"straightness_{prefix}"] = (
            _f(disp["straightness"])
            if disp and disp.get("total_path_um", 0) > 0 else None)

        # --- shape (per-frame means over this state's frames) ---
        means = per_state_means(shape, states, target)
        area_px = means.get("area")
        result[f"mean_area_um2_{prefix}"] = (
            _f(area_px * um * um) if _f(area_px) is not None else None)
        for k in _SHAPE_KEYS:
            result[f"mean_{k}_{prefix}"] = _f(means.get(k))

        # --- display aliases (focused GUI metrics panel) ---
        result[f"{prefix}_mean_speed_um_per_min"] = (
            result[f"mean_speed_{prefix}"]
            if result[f"mean_speed_{prefix}"] is not None else float("nan"))
        result[f"{prefix}_persistence_lag1"] = (
            result[f"persistence_{prefix}"]
            if result[f"persistence_{prefix}"] is not None else float("nan"))
        result[f"{prefix}_n_speed_samples"] = result[f"n_speed_samples_{prefix}"]

    return result
