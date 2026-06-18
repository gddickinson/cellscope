"""Slide-ready composite figure for the IC295 motility story (one panel).

Stitches the conclusion-carrying views into a single 16:9 figure (PNG +
vector PDF) with lettered panels, so it drops straight into a slide / the
writeup. Reuses the exact plotting + stats code so the numbers match
REPORT.md / MATCHED.md.

  A  ensemble MEAN MSD (genetic) — the naive view, outlier-driven
  B  ensemble MEDIAN MSD (genetic) — robust; the KO rank inverts
  C  covariate-adjusted treatment effect (recording OLS, 95% CI) — null
  D  net displacement vs time-spread — state structures dispersal
  E  net displacement vs crowding — density structures dispersal
  F  like-vs-like (matching ● + normalization □) — three methods, one null

Usage:
  conda run -n cellpose4 python scripts/ic295_motility_panel.py --from-cache
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
from scripts.ic295_motility_plots import _msd_panel  # noqa: E402
from scripts.ic295_motility_matched import analyse, _stars  # noqa: E402
from scripts.ic295_compare_arms import ARMS  # noqa: E402
from scripts.ic295_flower_plots import _MSD_MAXLAG  # noqa: E402
import numpy as np  # noqa: E402


def _letter(ax, s):
    ax.text(-0.13, 1.05, s, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="right")


def _forest_ols(ax, rec_rows, outcome="netdisp_med"):
    from scipy.stats import t as tdist
    items = []
    for arm in ARMS:
        o = _ols_adjusted(rec_rows, arm, outcome)
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
        ax.errorbar(coef, yi, xerr=ci, fmt="o", color=col, capsize=4, ms=7,
                    elinewidth=2, zorder=3)
    ax.axvline(0, color="#444", ls="--", lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([it[0] for it in items], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("adjusted Δ net displacement  (µm)", fontsize=9)
    ax.grid(alpha=0.3, axis="x")


def _forest_matched(ax, res, outcome="netdisp"):
    keys = list(res.keys())
    y = np.arange(len(keys))
    for yi, k in zip(y, keys):
        d = res[k][outcome]
        col = _COND_COLOR.get(res[k]["test"])
        eff, (lo, hi) = d["matched_effect"], d["ci"]
        if np.isfinite(eff):
            xerr = [[eff - lo], [hi - eff]] if np.isfinite(lo) else None
            ax.errorbar(eff, yi - 0.12, xerr=xerr, fmt="o", color=col,
                        capsize=3, ms=7, elinewidth=2, zorder=3)
            ax.text(eff, yi + 0.34, _stars(d["van_elteren_p"]), ha="center",
                    fontsize=8, color="#333")
        nz = d.get("normalized")
        if nz and np.isfinite(nz["effect"]):
            ax.scatter(nz["effect"], yi + 0.12, marker="s", s=45,
                       facecolor="none", edgecolor=col, linewidth=1.6, zorder=3)
    ax.axvline(0, color="#444", ls="--", lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([res[k]["test"] + " vs " + res[k]["control"]
                        for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Δ net displacement  (µm)", fontsize=9)
    ax.grid(alpha=0.3, axis="x")


def _scatter(ax, rows, xk, xlabel, legend=False):
    from scipy.stats import spearmanr
    allx, ally = [], []
    for c in CONDITIONS:
        xs = [r[xk] for r in rows if r["cond"] == c
              and np.isfinite(r[xk]) and np.isfinite(r["netdisp"])]
        ys = [r["netdisp"] for r in rows if r["cond"] == c
              and np.isfinite(r[xk]) and np.isfinite(r["netdisp"])]
        ax.scatter(xs, ys, s=12, alpha=0.5, color=_COND_COLOR.get(c), label=c)
        allx += xs
        ally += ys
    rho, p = spearmanr(allx, ally)
    b = np.polyfit(allx, ally, 1)
    xx = np.linspace(min(allx), max(allx), 50)
    ax.plot(xx, np.polyval(b, xx), "--", color="#222", lw=1.8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("net displacement  (µm)", fontsize=9)
    ax.set_title(f"Spearman ρ={rho:+.2f}, p={p:.1e}", fontsize=9)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(fontsize=6, ncol=3, loc="upper right", framealpha=0.9)


def main():
    data = load_or_build(from_cache="--from-cache" in sys.argv)
    if "cells" not in next(iter(data.values())):
        print("ERROR: enriched cache required — rebuild ic295_track_data.")
        return 1
    rows = all_cells(data)
    _g, rec_rows = per_recording(rows)
    res = analyse(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                          left=0.06, right=0.985, top=0.86, bottom=0.08)

    axA = fig.add_subplot(gs[0, 0])
    _msd_panel(axA, data, ["WT", "GOF", "KO"], "mean", DEFAULT_DT, _MSD_MAXLAG)
    axA.set_ylabel("MSD  (µm²)")
    axA.legend(fontsize=7)
    axA.set_title("Ensemble MEAN MSD (genetic) — outlier-driven", fontsize=10)
    _letter(axA, "A")

    axB = fig.add_subplot(gs[0, 1])
    _msd_panel(axB, data, ["WT", "GOF", "KO"], "median", DEFAULT_DT, _MSD_MAXLAG)
    axB.set_ylabel("MSD  (µm²)")
    axB.legend(fontsize=7)
    axB.set_title("Ensemble MEDIAN MSD — KO rank inverts (robust)", fontsize=10)
    _letter(axB, "B")

    axC = fig.add_subplot(gs[0, 2])
    _forest_ols(axC, rec_rows)
    axC.set_title("No treatment effect — adjusted (OLS, 95% CI)", fontsize=10)
    _letter(axC, "C")

    axD = fig.add_subplot(gs[1, 0])
    _scatter(axD, rows, "frac_spread", "fraction of time spread", legend=True)
    axD.set_title("State structures dispersal " + axD.get_title(), fontsize=9)
    _letter(axD, "D")

    axE = fig.add_subplot(gs[1, 1])
    _scatter(axE, rows, "med_neighbors", "median neighbours within 100 µm")
    axE.set_title("Crowding structures dispersal " + axE.get_title(), fontsize=9)
    _letter(axE, "E")

    axF = fig.add_subplot(gs[1, 2])
    _forest_matched(axF, res)
    axF.set_title("Like-vs-like: matching ● + normalization □ (null)",
                  fontsize=10)
    _letter(axF, "F")

    fig.suptitle("IC295 endothelial cell motility (n=8/condition): NO treatment "
                 "effect on migration — dispersal is structured by cell "
                 "STATE + CROWDING, not genotype/drug\n"
                 "A naive ensemble mean (A) suggests differences but inverts "
                 "under the robust median (B); the effect is null when tested "
                 "design-correctly (C) and survives matching + normalization "
                 "(F); what varies tracks state (D) and crowding (E).",
                 fontsize=12.5, y=0.985)

    png = os.path.join(OUT_DIR, "motility_panel.png")
    pdf = os.path.join(OUT_DIR, "motility_panel.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote slide panel → {png}\n                  → {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
