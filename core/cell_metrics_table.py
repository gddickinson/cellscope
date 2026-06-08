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

    Binary-state schema (see core.state_analysis). Two kinds of column:
      * LIFETIME (state-independent) — frames_tracked, division*, and
        frac_rounded / frac_spread (% time in each state). Safe to
        compare directly.
      * PER-STATE — every motility / shape metric, computed SEPARATELY
        over the cell's rounded frames and its spread frames, so a
        comparison is not confounded by time-in-state. None when that
        state never occurred for the cell.

    Deliberately omits whole-track state-MIXED averages (overall mean
    speed / area / circularity …): they blend the two states and must
    not be compared across conditions.
    """
    row = {
        # --- identity / lifetime (state-independent) ---
        "cell_id":                  c.get("cell_id"),
        "first_frame":              ti.get("first_frame"),
        "frames_tracked":           ti.get("frames_tracked"),
        "parent_id":                ti.get("parent_id"),
        "division_frame":           ti.get("division_frame"),
        "division_score":           ti.get("division_score"),
        "mean_boundary_confidence": c.get("mean_boundary_confidence"),
        "frac_rounded":             c.get("frac_rounded"),
        "frac_spread":              c.get("frac_spread"),
    }
    # --- per-state metrics (rounded + spread, each over its own frames) ---
    for st in ("rounded", "spread"):
        row[f"mean_speed_{st}"]      = c.get(f"mean_speed_{st}")
        row[f"n_speed_samples_{st}"] = c.get(f"n_speed_samples_{st}")
        row[f"persistence_{st}"]     = c.get(f"persistence_{st}")
        row[f"straightness_{st}"]    = c.get(f"straightness_{st}")
        row[f"mean_area_um2_{st}"]   = c.get(f"mean_area_um2_{st}")
        row[f"mean_circularity_{st}"] = c.get(f"mean_circularity_{st}")
        row[f"mean_solidity_{st}"]   = c.get(f"mean_solidity_{st}")
        row[f"mean_aspect_ratio_{st}"] = c.get(f"mean_aspect_ratio_{st}")
        row[f"mean_eccentricity_{st}"] = c.get(f"mean_eccentricity_{st}")
    return row


# Per-cell identity / bookkeeping columns that are NOT reduced across
# cells (they're not measurements). frac_rounded / frac_spread and the
# per-state metrics ARE reduced (→ <col>_mean/_median/_std/_n).
_ID_COLS = ("cell_id", "first_frame", "frames_tracked",
            "parent_id", "division_frame", "division_score",
            "n_speed_samples_rounded", "n_speed_samples_spread")


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
