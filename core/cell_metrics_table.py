"""Per-cell metric rows + recording-level aggregation.

Single source of truth shared by:
  - scripts/ic295_analyze_one.py   (IC295 batch → per_cell.csv + summary)
  - gui_focused/export_dialog.py   (focused GUI export → the same files)

Keeping the column set here guarantees the focused GUI's exported
`per_cell.csv` / `recording_summary.json` carry EXACTLY the columns the
IC295 cross-treatment comparison (`scripts/ic295_compare.py`) expects.

`c` is a per-cell analysis result dict (from `core.pipeline.analyze_recording`
augmented by `core.state_analysis.annotate_state`); `ti` is its
`track_info` sub-dict.
"""
from __future__ import annotations

import os
import csv

import numpy as np


def per_cell_row(c, ti):
    """Flatten one cell's analysis result into a CSV-friendly row.

    The state-aware columns (state_frac_*, mean_speed_*, persistence_*,
    straightness_*, *_pf) are populated by core.state_analysis; they are
    None when state classification was skipped.
    """
    ss = c.get("shape_summary", {}) or {}
    es = c.get("edge_summary", {}) or {}
    area = ss.get("area_um2") or {}
    return {
        "cell_id":                  c.get("cell_id"),
        "first_frame":              ti.get("first_frame"),
        "frames_tracked":           ti.get("frames_tracked"),
        "parent_id":                ti.get("parent_id"),
        "division_frame":           ti.get("division_frame"),
        "division_score":           ti.get("division_score"),
        "mean_speed":               c.get("mean_speed"),
        "total_distance":           c.get("total_distance"),
        "net_displacement":         c.get("net_displacement"),
        "persistence":              c.get("persistence"),
        "mean_area_um2":            area.get("mean"),
        "median_area_um2":          area.get("median"),
        "mean_circularity":         (ss.get("circularity") or {}).get("mean"),
        "mean_solidity":            (ss.get("solidity") or {}).get("mean"),
        "mean_aspect_ratio":        (ss.get("aspect_ratio") or {}).get("mean"),
        "mean_eccentricity":        (ss.get("eccentricity") or {}).get("mean"),
        "mean_protrusion_velocity": es.get("mean_protrusion_velocity"),
        "mean_retraction_velocity": es.get("mean_retraction_velocity"),
        "mean_boundary_confidence": c.get("mean_boundary_confidence"),
        "state_frac_balled":        c.get("state_frac_balled"),
        "state_frac_attached":      c.get("state_frac_attached"),
        "mean_speed_balled":        c.get("mean_speed_balled"),
        "mean_speed_attached":      c.get("mean_speed_attached"),
        "mean_speed_unattached":    c.get("mean_speed_unattached"),
        "mean_speed_non_balled":    c.get("mean_speed_non_balled"),
        "persistence_attached":     c.get("persistence_attached"),
        "persistence_balled":       c.get("persistence_balled"),
        "persistence_unattached":   c.get("persistence_unattached"),
        "persistence_non_balled":   c.get("persistence_non_balled"),
        "straightness_attached":    c.get("straightness_attached"),
        "straightness_balled":      c.get("straightness_balled"),
        "straightness_unattached":  c.get("straightness_unattached"),
        "straightness_non_balled":  c.get("straightness_non_balled"),
        # Per-frame speeds (no contiguous-segment requirement —
        # catches the brief balled events that segment metrics miss)
        "mean_speed_balled_pf":     c.get("mean_speed_balled_pf"),
        "mean_speed_non_balled_pf": c.get("mean_speed_non_balled_pf"),
        "n_frames_balled_pf":       c.get("n_frames_balled_pf"),
        "n_frames_non_balled_pf":   c.get("n_frames_non_balled_pf"),
    }


# Per-cell identity columns that are NOT reduced across cells (they're
# bookkeeping, not measurements).
_ID_COLS = ("cell_id", "first_frame", "frames_tracked",
            "parent_id", "division_frame", "division_score")


def aggregate_recording(rows, n_divisions):
    """Reduce across cells → one-row recording summary.

    Each cell is treated as an independent observation within this
    recording; the cross-recording phase treats each recording as one
    experiment. For each measured column emits `{col}_mean`,
    `{col}_median`, `{col}_std`, `{col}_n` over the finite values.
    """
    n = len(rows)
    out = {"n_cells": n, "n_divisions": int(n_divisions),
           "division_rate": (float(n_divisions) / n) if n else 0.0}
    if not n:
        return out
    cols = [k for k in rows[0].keys() if k not in _ID_COLS]
    for k in cols:
        vals = [r[k] for r in rows
                if isinstance(r.get(k), (int, float))
                and r[k] is not None and np.isfinite(r[k])]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{k}_median"] = float(np.median(vals)) if vals else None
        out[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else None
        out[f"{k}_n"] = len(vals)
    return out


def write_per_cell_csv(rows, path):
    """Write the per-cell rows to a CSV at `path` (no-op if empty)."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
