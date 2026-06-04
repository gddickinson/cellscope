"""Shared cell-state classification + per-state motility annotation.

Single source of truth used by BOTH analysis entry points:
  - scripts/ic295_analyze_one.py  (IC295 batch analysis)
  - gui_focused/workers.py        (focused GUI "Analyze")

so the GUI and the batch script compute IDENTICAL per-state metrics.
Before this module the two had drifted: the script computed the
compound `unattached` / `non_balled` states + per-frame speed variants
+ per-state straightness, while the GUI only did `balled` / `attached`.

Classification rule lives in `core.cell_state`; the per-state speed /
persistence / displacement primitives live in `core.motility_state`.

`annotate_state(result, cell_stack, um, dt)` mutates `result` in place,
adding the keys below.

Canonical (IC295 schema — read by `core.cell_metrics_table.per_cell_row`):
  state_per_frame
  state_frac_{balled,attached,transitional}
  mean_speed_{attached,balled,unattached,non_balled}
  persistence_{attached,balled,unattached,non_balled}
  straightness_{attached,balled,unattached,non_balled}
  mean_speed_balled_pf, mean_speed_non_balled_pf
  n_frames_balled_pf,   n_frames_non_balled_pf

Extended (focused-GUI display + metrics.json — keys prefixed
`balled_` / `attached_`, picked up by export_dialog._build_metrics):
  {balled,attached}_n_speed_samples
  {balled,attached}_mean_speed_um_per_min
  {balled,attached}_median_speed_um_per_min
  {balled,attached}_persistence_lag1
  {balled,attached}_msd_lag5_um2
  {balled,attached}_total_displacement_um
  {balled,attached}_straightness
"""
from __future__ import annotations

import numpy as np

# Compound (synthetic) state labels for cleaner balled-vs-rest comparisons:
#   "unattached" = balled + transitional  (the "not currently spread +
#                  migrating attached" cohort; dominated by transitional)
#   "non_balled" = attached + transitional (everything except the binary
#                  "rounded mitotic / dying" balled state — the cleanest
#                  split for comparing balled cells against the rest)
STATE_UNATTACHED = "unattached"
STATE_NON_BALLED = "non_balled"

# Speed cap (µm/min) for the per-frame variant — mirrors the cap inside
# core.motility_state.state_speeds so the two speed families are
# directly comparable.
_PF_SPEED_CAP = 15.0


def _safe(fn, default=None):
    """Call fn(); swallow any exception and return `default`.

    The per-state primitives can legitimately fail (e.g. a track with
    no contiguous segment of the target state long enough to measure);
    we want a None / NaN in that slot rather than aborting the whole
    annotation.
    """
    try:
        return fn()
    except Exception:
        return default


def annotate_state(result, cell_stack, um_per_px, dt_min, thresholds=None):
    """Classify each frame's state and attach per-state motility metrics.

    Args:
        result: dict mutated in place (a per-cell analysis result).
        cell_stack: (N, H, W) bool/int mask stack for ONE cell.
        um_per_px: physical pixel size (µm). Used to convert px → µm.
        dt_min: frame interval (min). Used to convert displacement → speed.
        thresholds: optional override for core.cell_state thresholds.

    Returns the same `result` dict (for chaining).
    """
    from core.cell_state import (
        classify_track_states, state_fraction,
        STATE_BALLED, STATE_ATTACHED, STATE_TRANSITIONAL)
    from core.motility_state import (
        state_speeds, state_msd, state_persistence,
        state_total_displacement)
    from core.tracking import extract_centroids

    um = float(um_per_px) if um_per_px else 1.0
    dt = float(dt_min) if dt_min else 1.0

    stack = cell_stack.astype(bool)
    sd = classify_track_states(stack, thresholds)
    states = np.asarray(sd["states"])
    result["state_per_frame"] = states.tolist()
    result["state_frac_balled"] = float(state_fraction(states, STATE_BALLED))
    result["state_frac_attached"] = float(
        state_fraction(states, STATE_ATTACHED))
    result["state_frac_transitional"] = float(
        state_fraction(states, STATE_TRANSITIONAL))

    cents = extract_centroids(stack)

    # Compound-state label arrays (same length as `states`).
    states_unatt = np.where(
        (states == STATE_BALLED) | (states == STATE_TRANSITIONAL),
        STATE_UNATTACHED, states)
    states_nonbal = np.where(
        states == STATE_BALLED, STATE_BALLED, STATE_NON_BALLED)

    _annotate_per_frame_speed(result, cents, states, um, dt, STATE_BALLED)
    _annotate_segment_states(
        result, cents, states, states_unatt, states_nonbal, um, dt,
        STATE_ATTACHED, STATE_BALLED)
    _annotate_extended(
        result, cents, states, um, dt, STATE_BALLED, STATE_ATTACHED)
    return result


def _annotate_per_frame_speed(result, cents, states, um, dt, balled):
    """Per-frame speed split: pair each step velocity v_i (cent[i+1]-cent[i])
    with state[i]. No contiguous-segment requirement, so it captures the
    isolated balled frames that the segment-based metrics miss."""
    if len(cents) > 1:
        step = (np.linalg.norm(np.diff(cents, axis=0), axis=1) * um / dt)
        step_state = states[:-1]
        for is_balled, prefix in ((True, "balled"), (False, "non_balled")):
            mask = (step_state == balled) if is_balled else (
                step_state != balled)
            vals = step[mask]
            vals = vals[np.isfinite(vals)]
            vals = vals[vals <= _PF_SPEED_CAP]
            result[f"mean_speed_{prefix}_pf"] = (
                float(np.mean(vals)) if len(vals) else None)
            result[f"n_frames_{prefix}_pf"] = int(len(vals))
    else:
        for prefix in ("balled", "non_balled"):
            result[f"mean_speed_{prefix}_pf"] = None
            result[f"n_frames_{prefix}_pf"] = 0


def _annotate_segment_states(result, cents, states, states_unatt,
                             states_nonbal, um, dt, attached, balled):
    """Per-state segment metrics for the four states (IC295 canonical keys):
    mean_speed_<p>, persistence_<p>, straightness_<p>."""
    from core.motility_state import (
        state_speeds, state_persistence, state_total_displacement)
    for target, prefix, st in (
            (attached,          "attached",   states),
            (balled,            "balled",     states),
            (STATE_UNATTACHED,  "unattached", states_unatt),
            (STATE_NON_BALLED,  "non_balled", states_nonbal)):
        sp = _safe(lambda t=target, s=st:
                   state_speeds(cents, s, t, um, dt))
        vals = [v for v in (sp if sp is not None else [])
                if v is not None and np.isfinite(v)]
        result[f"mean_speed_{prefix}"] = (
            float(np.mean(vals)) if vals else None)

        per = _safe(lambda t=target, s=st:
                    state_persistence(cents, s, t, um))
        result[f"persistence_{prefix}"] = (
            float(per["lag1"]) if per and not np.isnan(per["lag1"]) else None)

        disp = _safe(lambda t=target, s=st:
                     state_total_displacement(cents, s, t, um))
        result[f"straightness_{prefix}"] = (
            float(disp["straightness"])
            if disp and disp["total_path_um"] > 0 else None)


def _annotate_extended(result, cents, states, um, dt, balled, attached):
    """Extended balled_/attached_ keys for the focused GUI display + the
    metrics.json export. Mirrors the previous FocusedAnalyzeWorker output
    so nothing downstream regresses."""
    from core.motility_state import (
        state_speeds, state_msd, state_persistence,
        state_total_displacement)
    for target, prefix in ((balled, "balled"), (attached, "attached")):
        sp = _safe(lambda t=target: state_speeds(cents, states, t, um, dt),
                   default=np.array([]))
        sp = sp if sp is not None else np.array([])
        msd = _safe(lambda t=target:
                    state_msd(cents, states, t, um, max_lag=20))
        per = _safe(lambda t=target: state_persistence(cents, states, t, um))
        td = _safe(lambda t=target:
                   state_total_displacement(cents, states, t, um))
        result[f"{prefix}_n_speed_samples"] = int(len(sp))
        result[f"{prefix}_mean_speed_um_per_min"] = (
            float(np.mean(sp)) if len(sp) else float("nan"))
        result[f"{prefix}_median_speed_um_per_min"] = (
            float(np.median(sp)) if len(sp) else float("nan"))
        result[f"{prefix}_persistence_lag1"] = (
            float(per["lag1"]) if per else float("nan"))
        result[f"{prefix}_msd_lag5_um2"] = (
            float(msd["msd"][4])
            if msd is not None and len(msd["msd"]) > 4
            and not np.isnan(msd["msd"][4]) else float("nan"))
        result[f"{prefix}_total_displacement_um"] = (
            float(td["total_displacement_um"]) if td else float("nan"))
        result[f"{prefix}_straightness"] = (
            float(td["straightness"]) if td else float("nan"))
