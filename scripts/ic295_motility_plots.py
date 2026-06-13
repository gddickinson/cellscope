"""Presentation plots for the IC295 motility analysis — the three figures
that actually carry the conclusions (not one-per-metric box dumps).

  motility_forest.png       the NULL result: covariate-adjusted treatment
                            effects (recording OLS, 95% CI) on net
                            displacement + step speed — every interval
                            crosses zero.
  motility_covariates.png   what DOES structure dispersal: net displacement
                            vs time-spread and vs local crowding, per-cell,
                            coloured by treatment (which intermix). Both
                            negative — a confluence/jamming signature, NOT
                            "spread cells migrate more".
  msd_mean_vs_median.png    why the ensemble mean misleads: genetic arm,
                            KO is LOWEST by mean but HIGHEST by median.

Reads the shared enriched cache; reuses the stats/feature code so the
numbers match REPORT.md exactly.

Usage:
  conda run -n cellpose4 python scripts/ic295_motility_plots.py --from-cache
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import CONDITIONS  # noqa: E402
from scripts.ic295_track_data import load_or_build, DEFAULT_DT  # noqa: E402
from scripts.ic295_motility_stats import (  # noqa: E402
    OUT_DIR, all_cells, per_recording, _ols_adjusted, _COND_COLOR)
from scripts.ic295_compare_arms import ARMS  # noqa: E402
from scripts.ic295_flower_plots import (  # noqa: E402
    _ensemble_msd, _cell_msd, _full_track, _MSD_MAXLAG)
import numpy as np  # noqa: E402


def _forest(rec_rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import t as tdist
    outcomes = [("netdisp_med", "adjusted Δ net displacement  (µm)"),
                ("speed_med", "adjusted Δ mean step speed  (µm/min)")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, (oc, xlab) in zip(axes, outcomes):
        items = []
        for arm in ARMS:
            o = _ols_adjusted(rec_rows, arm, oc)
            if not o:
                continue
            ctrl = ARMS[arm]["control"]
            tc = tdist.ppf(0.975, o["dof"])
            for t in ARMS[arm]["conditions"]:
                if t == ctrl or t not in o:
                    continue
                items.append((f"{t} vs {ctrl}", o[t]["coef"], tc * o[t]["se"],
                              _COND_COLOR.get(t)))
        y = np.arange(len(items))
        for yi, (_lab, coef, ci, col) in zip(y, items):
            ax.errorbar(coef, yi, xerr=ci, fmt="o", color=col, capsize=4,
                        ms=8, elinewidth=2, zorder=3)
        ax.axvline(0, color="#444", ls="--", lw=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels([it[0] for it in items])
        ax.invert_yaxis()
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("No treatment effect on motility — covariate-adjusted "
                 "(recording-level OLS, 95% CI)\nadjusted for time-in-state + "
                 "local density; every interval crosses 0", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _covariate_scatter(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    panels = [("frac_spread", "fraction of time spread"),
              ("med_neighbors", "median neighbours within 100 µm (crowding)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (xk, xlab) in zip(axes, panels):
        allx, ally = [], []
        for c in CONDITIONS:
            xs = [r[xk] for r in rows if r["cond"] == c
                  and np.isfinite(r[xk]) and np.isfinite(r["netdisp"])]
            ys = [r["netdisp"] for r in rows if r["cond"] == c
                  and np.isfinite(r[xk]) and np.isfinite(r["netdisp"])]
            ax.scatter(xs, ys, s=16, alpha=0.5, color=_COND_COLOR.get(c),
                       label=f"{c} (n={len(xs)})")
            allx += xs
            ally += ys
        rho, p = spearmanr(allx, ally)
        b = np.polyfit(allx, ally, 1)
        xx = np.linspace(min(allx), max(allx), 50)
        ax.plot(xx, np.polyval(b, xx), color="#222", lw=2, ls="--")
        ax.set_xlabel(xlab)
        ax.set_ylabel("net displacement  (µm)")
        ax.set_title(f"Spearman ρ={rho:+.2f}, p={p:.1e}  (n={len(allx)} cells)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Dispersal is structured by STATE + CROWDING, not treatment "
                 "(colours intermix)\nmore time spread → LESS net displacement; "
                 "more crowded → less — a confluence/jamming signature",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _msd_panel(ax, data, conds, stat, dt, maxlag):
    lag = np.arange(maxlag + 1) * dt
    sl = slice(1, maxlag + 1)
    for c in conds:
        recs = [r for r in data[c]["all"] if _full_track(r["cents"])]
        if not recs:
            continue
        arr = np.vstack([_cell_msd(r["cents"], maxlag) for r in recs])
        center, lo, hi = _ensemble_msd(arr, stat)
        ax.plot(lag[sl], center[sl], color=_COND_COLOR.get(c), lw=1.9,
                label=f"{c} (n={arr.shape[0]})")
        ax.fill_between(lag[sl], lo[sl], hi[sl], color=_COND_COLOR.get(c),
                        alpha=0.15, lw=0)
    ax.set_xlabel("lag time τ  (min)")
    ax.grid(alpha=0.3)


def _mean_vs_median_msd(data, out_path, dt=DEFAULT_DT, maxlag=_MSD_MAXLAG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = ["WT", "GOF", "KO"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    _msd_panel(axes[0], data, conds, "mean", dt, maxlag)
    axes[0].set_title("MEAN ± SEM  (outlier-driven)")
    axes[0].set_ylabel("MSD  (µm²)")
    axes[0].legend(fontsize=9)
    _msd_panel(axes[1], data, conds, "median", dt, maxlag)
    axes[1].set_title("MEDIAN + 95% CI  (robust)")
    axes[1].legend(fontsize=9)
    fig.suptitle("Genetic arm — why the ensemble mean misleads: KO is LOWEST "
                 "by mean but HIGHEST by median\n(a few hyper-motile WT/GOF "
                 "cells inflate the mean; neither difference is significant)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_or_build(from_cache="--from-cache" in sys.argv)
    if "cells" not in next(iter(data.values())):
        print("ERROR: enriched cache required — rebuild ic295_track_data.")
        return 1
    rows = all_cells(data)
    _groups, rec_rows = per_recording(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    _forest(rec_rows, os.path.join(OUT_DIR, "motility_forest.png"))
    _covariate_scatter(rows, os.path.join(OUT_DIR, "motility_covariates.png"))
    _mean_vs_median_msd(data, os.path.join(OUT_DIR, "msd_mean_vs_median.png"))
    print(f"Wrote 3 summary plots → {OUT_DIR}/ "
          "(motility_forest, motility_covariates, msd_mean_vs_median)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
