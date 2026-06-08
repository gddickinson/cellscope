"""Per-metric histograms split by condition, at both replication levels.

Compiles every experiment (all recordings) and draws, for each metric, a
condition-split histogram so the full distribution — not just the mean —
is visible:

  recording level (ic295_analysis/compare/per_recording.csv)
      → one value per recording   → compare/histograms/<metric>.png
  cell level      (ic295_analysis/compare_pooled/per_cell_pooled.csv)
      → one value per cell        → compare_pooled/histograms/<metric>.png

Conditions are overlaid as step histograms over shared bins. Shape
metrics get the rounded threshold drawn (a sanity check on the cut).
Run after the comparison scripts (which write the CSVs). For the raw
per-frame circ/solid threshold confirmation see ic295_state_diagnostic.py.

Usage:
  conda run -n cellpose4 python scripts/ic295_histograms.py
  conda run -n cellpose4 python scripts/ic295_histograms.py --level cell
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import COMPARE_DIR, CONDITIONS  # noqa: E402
from scripts.ic295_compare import DEFAULT_METRICS  # noqa: E402
from scripts.ic295_compare_pooled import (  # noqa: E402
    DEFAULT_CELL_METRICS, COMPARE_POOLED_DIR)
import numpy as np  # noqa: E402

_COND_COLORS = {"WT": "#1f77b4", "KO": "#d62728", "GOF": "#2ca02c",
                "Y1": "#9467bd", "OT": "#ff7f0e", "DMSO": "#7f7f7f"}

# Shape metrics whose rounded threshold is worth drawing on the hist.
try:
    from core.cell_state import DEFAULT_THRESHOLDS as _TH
    _THRESH = {"circularity": _TH["rounded_circ"],
               "solidity": _TH["rounded_solid"]}
except Exception:
    _THRESH = {}


def _to_float(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _load(path, metrics):
    """csv → {metric: {cond: [values]}} for the requested metrics."""
    out = {m: {c: [] for c in CONDITIONS} for m in metrics}
    with open(path) as f:
        for r in csv.DictReader(f):
            c = r.get("condition")
            if c not in CONDITIONS:
                continue
            for m in metrics:
                v = _to_float(r.get(m))
                if v is not None:
                    out[m][c].append(v)
    return out


def _threshold_for(metric):
    for key, thr in _THRESH.items():
        if key in metric:
            return thr
    return None


def _shared_bins(by_cond, density):
    pooled = np.concatenate(
        [np.asarray(by_cond[c]) for c in CONDITIONS if by_cond[c]]
        or [np.array([])])
    if pooled.size < 2:
        return None
    lo, hi = float(np.min(pooled)), float(np.max(pooled))
    if hi <= lo:
        hi = lo + 1e-6
    nb = 12 if not density else 30
    return np.linspace(lo, hi, nb + 1)


def _hist_facet(metric, by_cond, out_path, density, unit_label):
    """One panel per condition (shared x-bins + shared y) — easy to
    compare distributions side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = _shared_bins(by_cond, density)
    if bins is None:
        return False
    conds = [c for c in CONDITIONS if by_cond.get(c)]
    thr = _threshold_for(metric)
    ncols = min(3, len(conds))
    nrows = int(np.ceil(len(conds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for i, c in enumerate(conds):
        ax = axes[i // ncols][i % ncols]
        v = np.asarray(by_cond[c])
        ax.hist(v, bins=bins, density=density, color=_COND_COLORS.get(c),
                edgecolor="#333", linewidth=0.5, alpha=0.9)
        if thr is not None:
            ax.axvline(thr, color="#c00", lw=1.8, ls="--")
        ax.set_title(f"{c}  (n={v.size})", fontsize=10)
        ax.grid(alpha=0.3)
    for j in range(len(conds), nrows * ncols):   # hide unused panels
        axes[j // ncols][j % ncols].axis("off")
    fig.supxlabel(metric)
    fig.supylabel("density" if density else "count")
    ttl = f"{metric} — {unit_label}, by condition"
    if thr is not None:
        ttl += f"   (rounded thr = {thr})"
    fig.suptitle(ttl, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def _hist_overlay(metric, by_cond, out_path, density, unit_label):
    """All conditions overlaid as step histograms (compact)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = _shared_bins(by_cond, density)
    if bins is None:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for c in CONDITIONS:
        v = np.asarray(by_cond[c])
        if v.size < 1:
            continue
        ax.hist(v, bins=bins, histtype="step", density=density, lw=1.8,
                color=_COND_COLORS.get(c), label=f"{c} (n={v.size})")
    thr = _threshold_for(metric)
    if thr is not None:
        ax.axvline(thr, color="#c00", lw=2, ls="--",
                   label=f"rounded thr = {thr}")
    ax.set_xlabel(metric)
    ax.set_ylabel("density" if density else "count")
    ax.set_title(f"{metric} — {unit_label}, split by condition")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def run_level(level, overlay=False):
    plot_fn = _hist_overlay if overlay else _hist_facet
    if level == "recording":
        path = os.path.join(COMPARE_DIR, "per_recording.csv")
        metrics = DEFAULT_METRICS
        out_dir = os.path.join(COMPARE_DIR, "histograms")
        density, unit = False, "one value per recording"
    else:
        path = os.path.join(COMPARE_POOLED_DIR, "per_cell_pooled.csv")
        metrics = DEFAULT_CELL_METRICS
        out_dir = os.path.join(COMPARE_POOLED_DIR, "histograms")
        density, unit = True, "one value per cell"
    if not os.path.exists(path):
        print(f"  [{level}] missing {path} — run the comparison first.")
        return 0
    data = _load(path, metrics)
    n = 0
    for m in metrics:
        if plot_fn(m, data[m], os.path.join(out_dir, f"{m}.png"),
                   density, unit):
            n += 1
    style = "overlay" if overlay else "faceted (one panel/condition)"
    print(f"  [{level}] wrote {n} {style} histograms → {out_dir}/")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["recording", "cell", "both"],
                    default="both")
    ap.add_argument("--overlay", action="store_true",
                    help="overlay conditions on one axis instead of "
                    "one panel per condition (default = faceted)")
    args = ap.parse_args()
    levels = (["recording", "cell"] if args.level == "both"
              else [args.level])
    total = sum(run_level(l, overlay=args.overlay) for l in levels)
    print(f"\nDone — {total} histograms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
