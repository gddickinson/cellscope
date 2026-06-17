"""Design-correct, confounder-aware motility statistics for IC293.

The IC293 recordings are single hand-cropped cells, DIC-only, from
endothelial-cell migration assays. This mirrors ic295_motility_stats but
makes the three single-cell-crop adaptations explicit:

  UNIT = POSITION. Each crop is one cell; several crops share a field of
    view (Pos{N}). The position is the cluster, so each metric is reduced
    to one value per position (median over its cells) before the arm test
    — the exact analog of IC295's "recording is the unit". A cell-level
    LMM with a per-position random intercept is the confounder-aware
    secondary.

  NO CROWDING CONFOUNDER. Every crop is one isolated cell, so neighbour
    count / nearest-neighbour distance have no between-unit variation. The
    density metrics and the density covariate are dropped (they would be
    singular); frac_spread remains as the state-time covariate.

  PER-CELL Fürth FIT. IC295 fits one Fürth PRW to each recording's
    *ensemble* MSD (needs ≥3 full cells/recording) — impossible when a
    recording is one cell. Here D and P are fit to each cell's OWN MSD
    curve, then reduced to the per-position median.

PRIMARY metrics are state-agnostic (speed, net displacement, area, MSD,
α, D, P). frac_spread / speed_spread use the IC295 keratinocyte-fit state
rule and are EXPLORATORY on these EC crops — reported in a separate,
flagged block.

Usage:
  conda run -n cellpose4 python scripts/ic293_motility_stats.py
  conda run -n cellpose4 python scripts/ic293_motility_stats.py --rebuild
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic293_common import (  # noqa: E402
    COMPARE_DIR, CONDITIONS, atomic_write_json, DEFAULT_UM_PER_PX,
    DEFAULT_DT_MIN)
from scripts.ic293_track_data import load_or_build  # noqa: E402
from scripts.ic295_motility_stats import (  # noqa: E402
    cell_features, _agg, _paired, _spearman, _fmt_p)
from scripts.ic295_compare_arms import (  # noqa: E402
    ARMS, _arm_stats, _plot_arms, _print_summary)
from scripts.ic295_motility_models import (  # noqa: E402
    furth_fit, fit_lmm, LMM_AVAILABLE)
import numpy as np  # noqa: E402

OUT_DIR = os.path.join(COMPARE_DIR, "motility_stats")
DEFAULT_UM = DEFAULT_UM_PER_PX
DEFAULT_DT = DEFAULT_DT_MIN
# Crops are 30–97 frames; cap the MSD lag so even a 30-frame full-duration
# cell reaches the endpoint (lag 24 × 10 min = 240 min, ≥6 displacement
# pairs at the shortest crop).
MSD_MAXLAG = 24

# Per-position metrics for the arm test. (key, label, how-to-reduce-cells)
# PRIMARY = state-agnostic. The two flagged rows depend on the (unvalidated
# on EC) state rule and are reported separately.
REC_METRICS = [
    ("netdisp_med",      "net displacement (µm)",                 "median"),
    ("speed_med",        "mean step speed (µm/min)",              "median"),
    ("area_med",         "mean cell area (µm²)",                  "median"),
    ("endpoint_msd_med", f"MSD @ {MSD_MAXLAG*int(DEFAULT_DT)} min (µm²)",
                                                                  "median_full"),
    ("msd_alpha_med",    "MSD exponent α (>1 super-diff)",        "median_full"),
    ("frac_spread_mean", "fraction of time spread [state-rule: exploratory]",
                                                                  "mean"),
    ("speed_spread_med", "speed when spread (µm/min) [exploratory]", "median"),
]
MODEL_METRICS = [("prw_D_med", "motility coefficient D (µm²/min)"),
                 ("prw_P_med", "persistence time P (min)")]
ALL_ARM_METRICS = [(k, lab) for k, lab, _ in REC_METRICS] + MODEL_METRICS
PRIMARY_KEYS = {"netdisp_med", "speed_med", "area_med", "endpoint_msd_med",
                "msd_alpha_med", "prw_D_med", "prw_P_med"}

_SRC = {"netdisp_med": "netdisp", "speed_med": "speed", "area_med": "area",
        "endpoint_msd_med": "endpoint_msd", "msd_alpha_med": "alpha",
        "frac_spread_mean": "frac_spread", "speed_spread_med": "speed_spread"}


def all_cells(data, dt, maxlag):
    """Per-cell feature dicts, augmented with pos + state-agnostic area."""
    rows = []
    for c in CONDITIONS:
        for rec in data[c].get("cells", data[c]["all"]):
            feat = cell_features(rec, dt, maxlag)
            feat["pos"] = rec["pos"]
            feat["area"] = rec["area"]
            rows.append(feat)
    return rows


def per_position(rows, dt, maxlag):
    """{metric: {cond: [one value per position]}} + per-position rows.

    Per-cell D,P come from a Fürth fit to each FULL cell's own MSD curve,
    reduced to the position median."""
    by_pos = {}
    for r in rows:
        by_pos.setdefault((r["cond"], r["pos"]), []).append(r)
    keys = [k for k, _ in ALL_ARM_METRICS]
    groups = {m: {c: [] for c in CONDITIONS} for m in keys}
    lag_min = np.arange(maxlag + 1) * dt
    pos_rows = []
    n_full = n_tot = 0
    for (cond, pos), cells in by_pos.items():
        n_tot += len(cells)
        n_full += sum(1 for c in cells if c["full"])
        rec = {"cond": cond, "pos": pos, "n_cells": len(cells)}
        for key, _lab, how in REC_METRICS:
            src = _SRC[key]
            vals = ([c[src] for c in cells if c["full"]] if how == "median_full"
                    else [c[src] for c in cells])
            rec[key] = _agg(vals, how)
            groups[key][cond].append(rec[key])
        DP = [furth_fit(lag_min, c["msd_curve"]) for c in cells
              if c["full"] and c["msd_curve"] is not None]
        Ds = [d for d, p in DP if np.isfinite(d)]
        Ps = [p for d, p in DP if np.isfinite(p)]
        rec["prw_D_med"] = float(np.median(Ds)) if Ds else np.nan
        rec["prw_P_med"] = float(np.median(Ps)) if Ps else np.nan
        groups["prw_D_med"][cond].append(rec["prw_D_med"])
        groups["prw_P_med"][cond].append(rec["prw_P_med"])
        pos_rows.append(rec)
    return groups, pos_rows, (n_full, n_tot)


def _ols_nd(pos_rows, arm, outcome):
    """Position-level OLS: outcome ~ treatment dummies + frac_spread (no
    density — single isolated cells). Returns per-test adjusted coef/p."""
    from scipy.stats import t as tdist
    spec = ARMS[arm]
    conds, ctrl = spec["conditions"], spec["control"]
    tests = [c for c in conds if c != ctrl]
    rows = [r for r in pos_rows if r["cond"] in conds]
    y = np.array([r[outcome] for r in rows])
    fs = np.array([r["frac_spread_mean"] for r in rows])
    cols = [np.ones(len(rows))]
    cols += [np.array([1.0 if r["cond"] == t else 0.0 for r in rows])
             for t in tests]
    cols.append(fs)
    X = np.column_stack(cols)
    good = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[good], y[good]
    dof = len(y) - X.shape[1]
    if dof <= 0:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float(resid @ resid) / dof
    try:
        cov = np.linalg.inv(X.T @ X) * sigma2
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.diag(cov))
    out = {"n_positions": int(len(y)), "dof": int(dof),
           "frac_spread_coef": float(beta[-1])}
    for i, t in enumerate(tests, start=1):
        tv = beta[i] / se[i] if se[i] > 0 else np.nan
        out[t] = {"coef": float(beta[i]), "se": float(se[i]),
                  "p": (float(2 * tdist.sf(abs(tv), dof))
                        if np.isfinite(tv) else None)}
    return out


def write_report(arm_stats, rows, pos_rows, ols, lmm, full_frac, path):
    nf, nt = full_frac
    L = ["# IC293 motility & dispersal — single-cell crops (EC migration)",
         "",
         "Per-position arm-structured tests (the field of view is the unit; "
         "several single-cell crops share a position). Within-arm Bonferroni "
         "vs the arm control; vehicle is a single WT-vs-DMSO MWU. "
         f"Full-duration cells: {nf}/{nt} ({100*nf/max(nt,1):.0f}%).",
         "",
         "> **State caveat.** `frac_spread` / `speed_spread` use the IC295 "
         "keratinocyte-fit rounded/spread rule (area ≤ 960 µm² AND ecc ≤ "
         "0.85). These are endothelial cells with no IC293 hand-labels, so "
         "those two rows are EXPLORATORY. Everything else is state-agnostic.",
         "",
         "## PRIMARY — state-agnostic motility/shape (test vs control)",
         "",
         "| metric | KO vs WT | GOF vs WT | Y1 vs DMSO | OT vs DMSO | WT vs DMSO (veh) |",
         "|---|---|---|---|---|---|"]

    def row_for(key, lab):
        st = arm_stats[key]
        def pr(arm, a, b):
            return st[arm]["pairs"].get(f"{a}_vs_{b}", {}).get("p_bonf")
        veh = st["vehicle"]["pairs"].get("WT_vs_DMSO", {}).get("p_raw")
        return (f"| {lab} | {_fmt_p(pr('genetic','WT','KO'))} | "
                f"{_fmt_p(pr('genetic','WT','GOF'))} | "
                f"{_fmt_p(pr('drug','DMSO','Y1'))} | "
                f"{_fmt_p(pr('drug','DMSO','OT'))} | {_fmt_p(veh)} |")

    for key, lab in ALL_ARM_METRICS:
        if key in PRIMARY_KEYS:
            L.append(row_for(key, lab))
    L += ["", "## EXPLORATORY — state-dependent (keratinocyte rule on EC)", "",
          "| metric | KO vs WT | GOF vs WT | Y1 vs DMSO | OT vs DMSO | WT vs DMSO (veh) |",
          "|---|---|---|---|---|---|"]
    for key, lab in ALL_ARM_METRICS:
        if key not in PRIMARY_KEYS:
            L.append(row_for(key, lab))

    L += ["", "## Confounder — state-time (frac_spread)", ""]
    nsp, ms, mr, p = _paired(rows, "speed_spread", "speed_rounded")
    L.append(f"- Within-cell paired step speed (n={nsp} cells with both "
             f"states): spread median **{ms:.2f}** vs rounded **{mr:.2f}** "
             f"µm/min, Wilcoxon p={_fmt_p(p)}. (State labels exploratory.)")
    nd, rho_nd, p_nd = _spearman(rows, "frac_spread", "netdisp")
    L.append(f"- Net displacement vs time-spread (Spearman, n={nd}): "
             f"ρ={rho_nd:+.2f}, p={_fmt_p(p_nd)}.")
    L += ["", "## Pseudoreplication — position-level OLS "
          "(outcome ~ treatment + frac_spread)", "",
          "Density is omitted: every crop is one isolated cell, so crowding "
          "has no between-position variation.", ""]
    for outcome in ("speed_med", "netdisp_med"):
        L.append(f"**{outcome}**")
        for arm in ARMS:
            o = ols[outcome][arm]
            if not o:
                L.append(f"- {arm}: insufficient dof"); continue
            ctrl = ARMS[arm]["control"]
            bits = [f"{t} vs {ctrl}: adj. coef={o[t]['coef']:.2f}, "
                    f"p={_fmt_p(o[t]['p'])}"
                    for t in ARMS[arm]["conditions"] if t != ctrl and t in o]
            L.append(f"- {arm} (n={o['n_positions']} pos, dof={o['dof']}): "
                     + "; ".join(bits))
        L.append("")
    L += ["## Pseudoreplication — cell-level LMM "
          "(speed ~ C(treatment) + frac_spread + (1 | position))", ""]
    if not LMM_AVAILABLE:
        L.append("- statsmodels unavailable — LMM skipped (OLS above stands in).")
    for arm in ARMS:
        m = (lmm or {}).get(arm)
        if not m:
            L.append(f"- {arm}: LMM unavailable / did not converge."); continue
        ctrl = ARMS[arm]["control"]
        bits = [f"{t} vs {ctrl}: β={m[t]['coef']:.2f} µm/min, p={_fmt_p(m[t]['p'])}"
                for t in ARMS[arm]["conditions"] if t != ctrl and t in m]
        L.append(f"- {arm} (n={m['n']} cells / {m['groups']} positions): "
                 + "; ".join(bits))
    L += ["",
          "_Caveats: (1) endothelial cells analysed with a keratinocyte-fit "
          "state rule — state rows exploratory only. (2) Full-duration cohort "
          "under-samples cells that leave the crop FOV, so absolute dispersal "
          "is conservative. (3) Cross-dataset comparison with IC295 absolute "
          "values is not meaningful; only within-IC293 arm contrasts are._"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--from-cache", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_or_build(DEFAULT_UM, DEFAULT_DT,
                         from_cache=args.from_cache, rebuild=args.rebuild)
    rows = all_cells(data, DEFAULT_DT, MSD_MAXLAG)
    if not rows:
        print("ERROR: no per-cell records — run detect + analyze first, then "
              "rebuild: python scripts/ic293_track_data.py --rebuild")
        return 1
    groups, pos_rows, full_frac = per_position(rows, DEFAULT_DT, MSD_MAXLAG)
    groups = {k: {c: [v for v in vals if v is not None and np.isfinite(v)]
                  for c, vals in g.items()}
              for k, g in groups.items()}

    arm_stats = {key: _arm_stats(groups[key]) for key, _ in ALL_ARM_METRICS}
    for key, _lab in ALL_ARM_METRICS:
        _plot_arms(key, groups[key], arm_stats[key],
                   os.path.join(OUT_DIR, "plots_arms", f"{key}.png"))
    atomic_write_json(os.path.join(OUT_DIR, "stats_arms_motility.json"),
                      arm_stats)

    ols = {o: {arm: _ols_nd(pos_rows, arm, o) for arm in ARMS}
           for o in ("speed_med", "netdisp_med")}
    lmm = {arm: fit_lmm(rows, ARMS[arm]["conditions"], ARMS[arm]["control"],
                        include_density=False, group_key="pos")
           for arm in ARMS}

    write_report(arm_stats, rows, pos_rows, ols, lmm, full_frac,
                 os.path.join(OUT_DIR, "REPORT.md"))
    print(f"\nWrote IC293 motility stats → {OUT_DIR}/")
    _print_summary("position (motility)", arm_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
