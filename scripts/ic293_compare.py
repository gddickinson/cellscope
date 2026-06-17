"""IC293 shape/state comparison — arm-structured, position as the unit.

Reads every recording_summary.json (one per crop) written by
ic293_analyze_one, reduces each metric to ONE value per POSITION (median
over the position's crops — the field of view is the cluster), then runs
the IC295 arm-structured tests (per-arm Kruskal-Wallis + within-arm
Bonferroni vs the arm control, plus the WT-vs-DMSO vehicle MWU).

EVERY metric here is state-dependent (frac_rounded + per-state shape /
motility means). The state rule is the IC295 keratinocyte fit, UNVALIDATED
on these endothelial cells — so this whole comparison is EXPLORATORY. The
state-agnostic primaries (speed, net displacement, area, MSD, α, D, P)
live in ic293_motility_stats.

Usage:
  conda run -n cellpose4 python scripts/ic293_compare.py
"""
import os
import sys
import csv
import glob
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic293_common import (  # noqa: E402
    RECORDINGS_ROOT, COMPARE_DIR, CONDITIONS, ANALYSIS_EXCLUDE,
    parse_condition, parse_position, atomic_write_json)
from scripts.ic295_compare import _summary_stats  # noqa: E402
from scripts.ic295_compare_arms import (  # noqa: E402
    ARMS, _arm_stats, _plot_arms, _print_summary)
import numpy as np  # noqa: E402

OUT_DIR = COMPARE_DIR
_PER_STATE = ["mean_speed", "persistence", "straightness", "mean_area_um2",
              "mean_circularity", "mean_solidity", "mean_aspect_ratio",
              "mean_eccentricity"]
# Count metrics (n_cells / n_divisions) are cropping artifacts here (one
# hand-picked cell per crop) — omitted. Everything kept is state-dependent.
METRICS = (["frac_rounded_mean"]
           + [f"{m}_rounded_mean" for m in _PER_STATE]
           + [f"{m}_spread_mean" for m in _PER_STATE])


def _collect_summaries():
    out = []
    for path in sorted(glob.glob(os.path.join(
            RECORDINGS_ROOT, "*", "*", "recording_summary.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
            d.setdefault("label", os.path.basename(os.path.dirname(path)))
            if d["label"] in ANALYSIS_EXCLUDE:
                continue
            d.setdefault("condition", parse_condition(d["label"]))
            d.setdefault("position", parse_position(d["label"]))
            out.append(d)
        except Exception as e:
            print(f"  WARN {path}: {e}")
    return out


def _per_position_values(rows, metric):
    """{condition: [one median-over-crops value per position]} for a metric."""
    by_pos = defaultdict(lambda: defaultdict(list))   # cond -> pos -> [vals]
    for r in rows:
        v = r.get(metric)
        c = r.get("condition")
        if c not in CONDITIONS or not isinstance(v, (int, float)):
            continue
        if v is None or not np.isfinite(v):
            continue
        by_pos[c][r.get("position")].append(float(v))
    out = {c: [] for c in CONDITIONS}
    for c in CONDITIONS:
        for _pos, vals in by_pos[c].items():
            out[c].append(float(np.median(vals)))
    return out


def _write_per_position_csv(rows, path):
    """One row per (position) with the per-position median of each metric."""
    by_pos = defaultdict(lambda: defaultdict(list))
    meta = {}
    for r in rows:
        key = (r.get("condition"), r.get("position"))
        meta[key] = key
        for m in METRICS:
            v = r.get(m)
            if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                by_pos[key][m].append(float(v))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "position", "n_crops"] + METRICS)
        for (cond, pos) in sorted(meta):
            n_crops = max((len(by_pos[(cond, pos)].get(m, [])) for m in METRICS),
                          default=0)
            vals = [(np.median(by_pos[(cond, pos)][m])
                     if by_pos[(cond, pos)].get(m) else "")
                    for m in METRICS]
            w.writerow([cond, pos, n_crops] + vals)


def main():
    argparse.ArgumentParser().parse_args()
    rows = _collect_summaries()
    if not rows:
        print(f"No recording_summary.json under {RECORDINGS_ROOT} — run "
              "analyze first.")
        return 1
    cnt = defaultdict(int)
    for r in rows:
        cnt[r.get("condition", "?")] += 1
    print(f"Collected {len(rows)} crops: "
          + ", ".join(f"{c}={cnt[c]}" for c in CONDITIONS if cnt[c]))

    os.makedirs(OUT_DIR, exist_ok=True)
    _write_per_position_csv(rows, os.path.join(OUT_DIR, "per_position.csv"))

    all_stats = {}
    for m in METRICS:
        groups = _per_position_values(rows, m)
        all_stats[m] = _arm_stats(groups)
        _plot_arms(m, groups, all_stats[m],
                   os.path.join(OUT_DIR, "plots_arms", f"{m}.png"))
    atomic_write_json(os.path.join(OUT_DIR, "stats_arms_shape.json"), all_stats)

    # Per-treatment summary CSV (position-level n)
    with open(os.path.join(OUT_DIR, "per_treatment.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "metric", "n_positions", "mean", "sem",
                    "std", "median"])
        for m in METRICS:
            g = _per_position_values(rows, m)
            for c in CONDITIONS:
                st = _summary_stats(g[c])
                w.writerow([c, m, st["n"], st["mean"], st["sem"],
                            st["std"], st["median"]])

    print(f"\nWrote shape/state comparison → {OUT_DIR}/ "
          "(per_position.csv, stats_arms_shape.json, plots_arms/*.png)")
    print("NOTE: all rows are state-rule-dependent → EXPLORATORY on EC.")
    _print_summary("position (shape/state) — EXPLORATORY", all_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
