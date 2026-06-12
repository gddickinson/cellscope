"""Mean ± SEM bar plots with individual-value points, per condition.

Reads the comparison outputs and, for every metric, draws a bar at the
condition mean with SEM error bars and the individual values overlaid as
jittered points. Works on either analysis:

  recording-level (ic295_analysis/compare/):     one point per RECORDING
  cell-pooled      (ic295_analysis/compare_pooled/): one point per CELL

For each analysis it reads:
  per_treatment.csv  -> mean + SEM per (condition, metric)
  per_recording.csv / per_cell_pooled.csv -> the individual values
  stats.json         -> Kruskal-Wallis p (annotated in the title)
and writes plots_mean_sem/<metric>.png alongside the existing plots/.

Usage:
  python scripts/ic295_plot_mean_sem.py                 # both analyses
  python scripts/ic295_plot_mean_sem.py --analysis recording
  python scripts/ic295_plot_mean_sem.py --analysis pooled
"""
import os
import sys
import csv
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import COMPARE_DIR, CONDITIONS  # noqa: E402
import numpy as np  # noqa: E402

COMPARE_POOLED_DIR = os.path.join(os.path.dirname(COMPARE_DIR),
                                   "compare_pooled")

# (key, compare_dir, individual_csv, point-label)
ANALYSES = {
    "recording": (COMPARE_DIR, "per_recording.csv", "recording"),
    "pooled":    (COMPARE_POOLED_DIR, "per_cell_pooled.csv", "cell"),
}


def _to_float(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _load_treatment(path):
    """per_treatment.csv -> {metric: {cond: {mean, sem, n}}} (ordered)."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m = r["metric"]
            n_raw = r.get("n") or r.get("n_cells") or 0  # recording vs pooled
            out.setdefault(m, {})[r["condition"]] = {
                "mean": _to_float(r["mean"]),
                "sem": _to_float(r["sem"]),
                "n": int(n_raw) if n_raw else 0,
            }
    return out


def _load_points(path, metrics):
    """individual CSV -> {metric: {cond: [values]}}."""
    pts = {m: {c: [] for c in CONDITIONS} for m in metrics}
    with open(path) as f:
        for r in csv.DictReader(f):
            c = r.get("condition")
            if c not in CONDITIONS:
                continue
            for m in metrics:
                v = _to_float(r.get(m))
                if v is not None:
                    pts[m][c].append(v)
    return pts


def _plot(metric, stats, points, kw_p, out_path, point_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ic295_plot_utils import apply_ybreak, axis_label
    conds = [c for c in CONDITIONS
             if points.get(c) or (stats.get(c) or {}).get("mean") is not None]
    if not conds:
        return False
    rng = np.random.default_rng(0)
    x = np.arange(1, len(conds) + 1)
    means = [(stats.get(c) or {}).get("mean") for c in conds]
    sems = [(stats.get(c) or {}).get("sem") or 0.0 for c in conds]
    bar_h = [m if m is not None else 0 for m in means]
    pts = [points.get(c) or [] for c in conds]
    # Jitter once so both broken panels place points identically.
    jit = [rng.normal(xi, 0.075, size=len(v)) for xi, v in zip(x, pts)]

    def draw(ax):
        ax.bar(x, bar_h, width=0.62, color="#cfe3f7", edgecolor="#3a5a78",
               linewidth=1.1, zorder=1)
        ax.errorbar(x, bar_h, yerr=sems, fmt="none", ecolor="#1b2b3a",
                    elinewidth=1.4, capsize=5, capthick=1.4, zorder=3)
        for jx, vals in zip(jit, pts):
            if len(vals):
                ax.scatter(jx, vals, s=26, color="#21405a", alpha=0.55,
                           edgecolor="white", linewidth=0.4, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n(n={len(v)})" for c, v in zip(conds, pts)])
        ax.margins(x=0.02)

    fig = plt.figure(figsize=(7, 4.8))
    title = f"{metric}  —  mean ± SEM, points = {point_label}s"
    if kw_p is not None:
        title += f"\nKruskal-Wallis p = {kw_p:.2g}"
    # break decision off the individual points (the values that squish)
    allv = [v for vals in pts for v in vals] or bar_h
    apply_ybreak(fig, draw, allv, ylabel=axis_label(metric),
                 xlabel="Condition", title=title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def run_analysis(key):
    cdir, indiv_csv, point_label = ANALYSES[key]
    tr = os.path.join(cdir, "per_treatment.csv")
    ind = os.path.join(cdir, indiv_csv)
    if not (os.path.exists(tr) and os.path.exists(ind)):
        print(f"  [{key}] missing {tr} or {ind} — run the comparison first.")
        return 0
    treat = _load_treatment(tr)
    metrics = list(treat.keys())
    points = _load_points(ind, metrics)
    stats_p = os.path.join(cdir, "stats.json")
    kw = {}
    if os.path.exists(stats_p):
        sj = json.load(open(stats_p))
        kw = {m: ((sj.get(m) or {}).get("kw") or {}).get("p_value")
              for m in metrics}
    out_dir = os.path.join(cdir, "plots_mean_sem")
    n = 0
    for m in metrics:
        if _plot(m, treat[m], points[m], kw.get(m),
                 os.path.join(out_dir, f"{m}.png"), point_label):
            n += 1
    print(f"  [{key}] wrote {n} plots → {out_dir}/  "
          f"(points = one per {point_label})")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", choices=["recording", "pooled", "both"],
                    default="both")
    args = ap.parse_args()
    keys = (["recording", "pooled"] if args.analysis == "both"
            else [args.analysis])
    total = 0
    for k in keys:
        total += run_analysis(k)
    print(f"\nDone — {total} mean±SEM plots.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
