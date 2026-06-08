"""Cell-POOLED mirror of ic295_compare.py.

MIRROR of scripts/ic295_compare.py, but the experimental unit is the
**cell**, not the recording: every cell across every recording in a
condition is pooled, so n = total cells per condition. The statistics
machinery is REUSED verbatim from ic295_compare (`_summary_stats`,
`_kw_and_pairs`, `_plot_metric`) so the ONLY difference between the two
analyses is the unit of replication.

⚠️  CAVEAT — this is PSEUDOREPLICATION. Cells within a recording are not
independent, so pooling inflates n and makes p-values anti-conservative
(too significant). The recording-level `ic295_compare.py` is the
statistically correct PRIMARY analysis; this pooled view exists only for
side-by-side comparison (it typically shows much larger n and smaller
p-values for the same effect).

Outputs (under ic295_analysis/compare_pooled/):
  per_cell_pooled.csv   one row per cell, condition-tagged
  per_treatment.csv     mean / SEM / n(cells) per (condition, metric)
  stats.json            per-metric Kruskal-Wallis + pairwise MWU (n=cells)
  plots/<metric>.png    box + per-cell strip (one dot per cell)

Usage:
  python scripts/ic295_compare_pooled.py
  python scripts/ic295_compare_pooled.py --metrics mean_speed,persistence
"""
import os
import sys
import csv
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    RECORDINGS_ROOT, COMPARE_DIR, CONDITIONS, parse_condition,
    atomic_write_json,
)
# Reuse the EXACT stats + plotting from the recording-level analysis so
# the two only differ in the unit of replication.
from scripts.ic295_compare import (  # noqa: E402
    _summary_stats, _kw_and_pairs, _plot_metric,
)
import numpy as np  # noqa: E402

COMPARE_POOLED_DIR = os.path.join(os.path.dirname(COMPARE_DIR),
                                   "compare_pooled")

# Per-cell metrics mirroring the recording-level DEFAULT_METRICS (the
# recording version aggregates these to `<metric>_mean`; here we pool the
# raw per-cell values). Binary-state schema: lifetime % time rounded +
# the per-state (rounded / spread) metrics. Recording-only counts
# (n_cells, n_divisions) have no per-cell analogue and are omitted.
_PER_STATE = [
    "mean_speed", "persistence", "straightness", "mean_area_um2",
    "mean_circularity", "mean_solidity", "mean_aspect_ratio",
    "mean_eccentricity",
]
DEFAULT_CELL_METRICS = (
    ["frac_rounded"]                                  # lifetime per-cell
    + [f"{m}_rounded" for m in _PER_STATE]
    + [f"{m}_spread" for m in _PER_STATE]
)


def _collect_cells():
    """Read every per_cell.csv, tag each cell with its condition."""
    cells = []
    for path in sorted(glob.glob(
            os.path.join(RECORDINGS_ROOT, "*", "*", "per_cell.csv"))):
        label = os.path.basename(os.path.dirname(path))
        cond = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if cond not in CONDITIONS:
            cond = parse_condition(label)
        try:
            with open(path) as f:
                for row in csv.DictReader(f):
                    row["label"] = label
                    row["condition"] = cond
                    cells.append(row)
        except Exception as e:
            print(f"  WARN: {path}: {e}")
    return cells


def _to_float(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _per_treatment_cells(cells, metric):
    """Pool per-cell metric values by condition (n = cells)."""
    out = {c: [] for c in CONDITIONS}
    for r in cells:
        c = r.get("condition")
        if c not in out:
            continue
        v = _to_float(r.get(metric))
        if v is not None:
            out[c].append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None,
                    help="comma-separated per-cell metric names")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cells = _collect_cells()
    if not cells:
        print(f"No per_cell.csv under {RECORDINGS_ROOT}. Run analyze first.")
        return 1

    cnt = {}
    for r in cells:
        cnt[r["condition"]] = cnt.get(r["condition"], 0) + 1
    print(f"Pooled {len(cells)} cells (PSEUDOREPLICATION — see header):")
    for c in CONDITIONS:
        if cnt.get(c):
            print(f"  {c:>5}: n_cells={cnt[c]}")

    os.makedirs(COMPARE_POOLED_DIR, exist_ok=True)

    # per_cell_pooled.csv (every cell, condition-tagged)
    keys = ["label", "condition"] + [
        k for k in cells[0] if k not in ("label", "condition")]
    with open(os.path.join(COMPARE_POOLED_DIR, "per_cell_pooled.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in cells:
            w.writerow(r)

    metrics = (args.metrics.split(",") if args.metrics
               else DEFAULT_CELL_METRICS)
    metrics = [m.strip() for m in metrics if m.strip()]

    # per_treatment.csv (n = cells)
    with open(os.path.join(COMPARE_POOLED_DIR, "per_treatment.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "metric", "n_cells", "mean", "sem",
                    "std", "median"])
        for m in metrics:
            groups = _per_treatment_cells(cells, m)
            for c in CONDITIONS:
                st = _summary_stats(groups[c])
                w.writerow([c, m, st["n"], st["mean"], st["sem"],
                            st["std"], st["median"]])

    # stats: K-W + pairwise MWU per metric (n = cells)
    stats_out = {}
    for m in metrics:
        stats_out[m] = _kw_and_pairs(_per_treatment_cells(cells, m))
    atomic_write_json(os.path.join(COMPARE_POOLED_DIR, "stats.json"),
                      stats_out)

    if not args.no_plots:
        plots_dir = os.path.join(COMPARE_POOLED_DIR, "plots")
        for m in metrics:
            groups = _per_treatment_cells(cells, m)
            kw_p = (stats_out[m].get("kw") or {}).get("p_value")
            _plot_metric(m, groups,
                          os.path.join(plots_dir, f"{m}.png"), kw_p=kw_p)

    print(f"\nWrote: {COMPARE_POOLED_DIR}/per_cell_pooled.csv")
    print(f"Wrote: {COMPARE_POOLED_DIR}/per_treatment.csv")
    print(f"Wrote: {COMPARE_POOLED_DIR}/stats.json")
    if not args.no_plots:
        print(f"Wrote: {COMPARE_POOLED_DIR}/plots/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
