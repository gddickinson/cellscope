"""Compile per-recording summaries → treatment comparison.

For each recording with a recording_summary.json, compile one row of
aggregated metrics. Group by condition; each recording is one
experimental replicate. Compute mean ± SEM per condition, run
Kruskal-Wallis across conditions per metric, then pairwise
Mann-Whitney with Bonferroni correction. Emit CSVs + box/strip plots.

Outputs (under ic295_analysis/compare/):
  per_recording.csv         one row per recording (n × m)
  per_treatment.csv         mean / SEM / n per (condition, metric)
  stats.json                per-metric K-W p + pairwise post-hoc
  plots/<metric>.png        box+strip plot, condition vs metric

Usage:
  python scripts/ic295_compare.py
  python scripts/ic295_compare.py --metrics mean_speed,persistence
"""
import os
import sys
import json
import glob
import argparse
import csv
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    inventory_drive, recording_dir, atomic_write_json,
    COMPARE_DIR, RECORDINGS_ROOT, CONDITIONS, parse_condition,
)

import numpy as np

# Metrics to ship in plots / pairwise post-hoc by default. The CSVs
# include EVERY field from recording_summary.json (so nothing is lost),
# but plots/stats focus on the biology-relevant scalars.
DEFAULT_METRICS = [
    "mean_speed_mean",            # avg cell speed in this recording
    "median_speed_mean",
    "persistence_mean",
    "total_distance_mean",
    "net_displacement_mean",
    "mean_area_um2_mean",
    "mean_circularity_mean",
    "mean_solidity_mean",
    "mean_aspect_ratio_mean",
    "mean_protrusion_velocity_mean",
    "mean_retraction_velocity_mean",
    "state_frac_balled_mean",
    "state_frac_attached_mean",
    "mean_speed_balled_mean",
    "mean_speed_attached_mean",
    "division_rate",
    "n_cells",
    "n_divisions",
]


def _collect_summaries():
    """Return list of recording_summary.json dicts present on disk."""
    out = []
    for path in sorted(glob.glob(
            os.path.join(RECORDINGS_ROOT, "*", "*",
                          "recording_summary.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
            d.setdefault("label",
                          os.path.basename(os.path.dirname(path)))
            d.setdefault("condition",
                          parse_condition(d["label"]))
            d["_summary_path"] = path
            out.append(d)
        except Exception as e:
            print(f"  WARN: {path}: {e}")
    return out


def _write_per_recording_csv(rows, path):
    if not rows:
        return
    # Union of keys across all rows (per_cell.csv style)
    all_keys = ["label", "condition"]
    for r in rows:
        for k in r:
            if k not in all_keys and not k.startswith("_"):
                all_keys.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _per_treatment(rows, metric):
    """Group recording-level metric values by condition. Returns
    dict condition -> list[float] (NaN/None dropped)."""
    out = {c: [] for c in CONDITIONS}
    for r in rows:
        c = r.get("condition")
        v = r.get(metric)
        if c not in out:
            continue
        if not isinstance(v, (int, float)) or v is None:
            continue
        if not np.isfinite(v):
            continue
        out[c].append(float(v))
    return out


def _summary_stats(values):
    if not values:
        return {"n": 0, "mean": None, "sem": None, "std": None,
                "median": None}
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    return {
        "n":      n,
        "mean":   float(np.mean(arr)),
        "sem":    sd / np.sqrt(n) if n else None,
        "std":    sd,
        "median": float(np.median(arr)),
    }


def _kw_and_pairs(groups, alpha=0.05):
    """Kruskal-Wallis across all conditions + pairwise Mann-Whitney
    with Bonferroni correction. Returns dict."""
    from scipy import stats as sps
    nonempty = [(c, v) for c, v in groups.items() if len(v) >= 2]
    out = {"kw": None, "pairs": {}}
    if len(nonempty) >= 2:
        try:
            stat, p = sps.kruskal(*(v for _, v in nonempty))
            out["kw"] = {"statistic": float(stat), "p_value": float(p)}
        except Exception as e:
            out["kw"] = {"error": str(e)}
    # Pairwise MWU with Bonferroni
    pairs = list(combinations([c for c, _ in nonempty], 2))
    n_pairs = max(len(pairs), 1)
    for a, b in pairs:
        try:
            stat, p = sps.mannwhitneyu(groups[a], groups[b],
                                        alternative="two-sided")
            out["pairs"][f"{a}_vs_{b}"] = {
                "n_a":       len(groups[a]),
                "n_b":       len(groups[b]),
                "u":         float(stat),
                "p_raw":     float(p),
                "p_bonf":    float(min(p * n_pairs, 1.0)),
                "sig_bonf":  bool(p * n_pairs < alpha),
            }
        except Exception as e:
            out["pairs"][f"{a}_vs_{b}"] = {"error": str(e)}
    return out


def _plot_metric(metric, groups, out_path, kw_p=None):
    """Box plot per condition + scatter overlay (one dot per recording)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    conds = [c for c in CONDITIONS if groups.get(c)]
    if not conds:
        return
    data = [groups[c] for c in conds]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, labels=conds, widths=0.6, showfliers=False,
                     patch_artist=True)
    for patch in bp["boxes"]:
        patch.set(facecolor="#cce4ff", edgecolor="#446")
    # Overlay individual recording points
    for i, vals in enumerate(data):
        x = np.random.default_rng(0).normal(i + 1, 0.06, size=len(vals))
        ax.scatter(x, vals, s=22, color="#234", alpha=0.7, zorder=3)
    ax.set_xlabel("Condition")
    ax.set_ylabel(metric)
    title = metric
    if kw_p is not None:
        title += f"   (Kruskal-Wallis p={kw_p:.2e})"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None,
                    help="comma-separated metric names to plot/test "
                    "(default = a curated set)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    rows = _collect_summaries()
    if not rows:
        print("No recording_summary.json files found under "
              f"{RECORDINGS_ROOT}.  Run --phase analyze first.")
        return 1

    print(f"Collected {len(rows)} recordings:")
    cnt = {c: 0 for c in CONDITIONS}
    for r in rows:
        cnt[r.get("condition", "?")] = cnt.get(r.get("condition", "?"), 0) + 1
    for c, n in cnt.items():
        if n:
            print(f"  {c:>5}: n={n}")

    os.makedirs(COMPARE_DIR, exist_ok=True)
    _write_per_recording_csv(rows,
                              os.path.join(COMPARE_DIR, "per_recording.csv"))

    metrics = (args.metrics.split(",") if args.metrics
               else DEFAULT_METRICS)
    metrics = [m.strip() for m in metrics if m.strip()]

    # Per-treatment summary CSV
    tr_path = os.path.join(COMPARE_DIR, "per_treatment.csv")
    with open(tr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "metric", "n", "mean", "sem",
                    "std", "median"])
        for m in metrics:
            groups = _per_treatment(rows, m)
            for c in CONDITIONS:
                st = _summary_stats(groups[c])
                w.writerow([c, m, st["n"], st["mean"], st["sem"],
                            st["std"], st["median"]])

    # Stats: K-W + pairwise per metric
    stats_out = {}
    for m in metrics:
        groups = _per_treatment(rows, m)
        stats_out[m] = _kw_and_pairs(groups)
    atomic_write_json(os.path.join(COMPARE_DIR, "stats.json"), stats_out)

    if not args.no_plots:
        plots_dir = os.path.join(COMPARE_DIR, "plots")
        for m in metrics:
            groups = _per_treatment(rows, m)
            kw_p = (stats_out[m].get("kw") or {}).get("p_value")
            _plot_metric(m, groups,
                          os.path.join(plots_dir, f"{m}.png"),
                          kw_p=kw_p)

    print(f"\nWrote: {COMPARE_DIR}/per_recording.csv")
    print(f"Wrote: {COMPARE_DIR}/per_treatment.csv")
    print(f"Wrote: {COMPARE_DIR}/stats.json")
    if not args.no_plots:
        print(f"Wrote: {COMPARE_DIR}/plots/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
