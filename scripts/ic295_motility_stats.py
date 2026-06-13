"""Design-correct, confounder-aware motility & dispersal statistics (IC295).

Reads the enriched per-cell cache (`ic295_track_data`) and answers two
things the ensemble-MSD curves cannot:

(b) PER-RECORDING, ARM-STRUCTURED dispersal test
    Each recording's full-duration cells are reduced to ONE value
    (median), so the experimental unit is the recording — no
    pseudoreplication. Those per-recording values then go through the
    same arm-structured stats as everything else (per-arm Kruskal-Wallis
    + within-arm Bonferroni pairwise vs the arm control, plus the WT-vs-
    DMSO vehicle MWU). Reuses `ic295_compare_arms`.

CONFOUNDERS — motility is not just "treatment". Three things that move
with treatment are accounted for:

  STATE (rounded vs spread). Spread cells migrate; rounded cells barely
    move. A treatment can change dispersal by (i) changing how fast cells
    move IN a state, or (ii) changing how much TIME cells spend spread.
    We separate these: paired within-cell speed (spread vs rounded) +
    per-recording frac_spread as its own arm metric.

  CONTACT / LOCAL DENSITY. Crowded cells undergo contact inhibition of
    locomotion. We summarise each cell's neighbourhood (median neighbour
    count within 100 µm, nearest-neighbour distance, fraction of frames
    isolated), test density across treatments (is a "dispersal" effect
    really a crowding effect?), correlate speed vs density, and contrast
    each cell's speed when isolated vs in contact (paired).

  PSEUDOREPLICATION. Handled two ways: the primary tests aggregate to the
    recording; and a recording-level OLS regresses speed / net-displacement
    on treatment WHILE adjusting for frac_spread + density, so a treatment
    effect must survive the confounders. (A cell-level linear mixed model
    with a recording random effect is the textbook tool but needs
    statsmodels, which isn't in this env — the recording-level OLS is the
    dependency-free equivalent given the recording is already the unit.)

Outputs (compare/motility_stats/):
  stats_arms_motility.json   per-recording arm stats, every metric
  plots_arms/<metric>.png    genetic | drug arm panels (reused machinery)
  speed_vs_density.png       contact-inhibition scatter, by condition
  REPORT.md                  human-readable synthesis of everything

Usage:
  conda run -n cellpose4 python scripts/ic295_motility_stats.py
  conda run -n cellpose4 python scripts/ic295_motility_stats.py --rebuild
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    COMPARE_DIR, CONDITIONS, atomic_write_json)
from scripts.ic295_track_data import (  # noqa: E402
    load_or_build, DEFAULT_UM, DEFAULT_DT, SPEED_CAP, NEIGHBOR_RADIUS_UM)
from scripts.ic295_flower_plots import _full_track, _cell_msd, _MSD_MAXLAG
from scripts.ic295_compare_arms import (  # noqa: E402
    ARMS, VEHICLE, _arm_stats, _plot_arms, _print_summary)
from scripts.ic295_motility_models import (  # noqa: E402
    msd_alpha, furth_fit, fit_lmm, LMM_AVAILABLE)
from core.cell_state import STATE_ROUNDED, STATE_SPREAD  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = os.path.join(COMPARE_DIR, "motility_stats")
_COND_COLOR = {"WT": "#1f77b4", "KO": "#d62728", "GOF": "#2ca02c",
               "Y1": "#9467bd", "OT": "#ff7f0e", "DMSO": "#7f7f7f"}

# Per-recording metrics fed to the arm-structured test. (key, label, agg)
# agg = how the recording's cells are reduced to one value.
REC_METRICS = [
    ("netdisp_med",        "net displacement (µm)",          "median"),
    ("endpoint_msd_med",   "MSD @ 600 min (µm²)",            "median_full"),
    ("speed_med",          "mean step speed (µm/min)",        "median"),
    ("speed_spread_med",   "speed when spread (µm/min)",      "median"),
    ("frac_spread_mean",   "fraction of time spread",         "mean"),
    ("msd_alpha_med",      "MSD exponent α (>1 super-diff)",  "median_full"),
    ("n_neighbors_med",    "neighbours within 100 µm",        "median"),
    ("nn_dist_med",        "nearest-neighbour dist (µm)",     "median"),
    ("frac_isolated_mean", "fraction of frames isolated",     "mean"),
]
# Per-recording ENSEMBLE-FIT metrics (one Fürth fit to the recording's mean
# MSD, not a per-cell aggregate) — filled directly in per_recording().
MODEL_METRICS = [
    ("prw_D", "motility coefficient D (µm²/min)"),
    ("prw_P", "persistence time P (min)"),
]
ALL_ARM_METRICS = ([(k, lab) for k, lab, _ in REC_METRICS] + MODEL_METRICS)


# ---------------------------------------------------------------- per cell
def _steps(cents, dt):
    """(origin-frame indices, step speed µm/min) over consecutive present frames."""
    spd = np.linalg.norm(np.diff(cents, axis=0), axis=1) / dt    # cents already µm
    ok = np.isfinite(spd) & (spd <= SPEED_CAP)
    return np.nonzero(ok)[0], spd[ok]


def _frac_spread(states):
    cls = states[(states == STATE_ROUNDED) | (states == STATE_SPREAD)]
    return float(np.mean(cls == STATE_SPREAD)) if cls.size else np.nan


def _mean_or_nan(a):
    a = np.asarray(a, float)
    return float(np.mean(a)) if a.size else np.nan


def cell_features(rec, dt, maxlag):
    """Reduce one per-cell record to the scalars the stats need."""
    cents, states = rec["cents"], rec["states"]
    f, sp = _steps(cents, dt)
    st_f = states[f]                                  # state at each step's origin
    nb = rec["n_neighbors"]
    nb_f = nb[f]
    pres = np.isfinite(cents[:, 0])
    full = _full_track(cents)
    nnd_pres = rec["nn_dist"][pres]
    curve = _cell_msd(cents, maxlag) if full else None    # per-cell MSD(τ)
    return {
        "label": rec["label"], "cond": rec["cond"],
        "full": full,
        "msd_curve": curve,                               # used for ensemble PRW fit
        "netdisp": rec["netdisp"],
        "speed": rec["speed"],
        "endpoint_msd": float(curve[maxlag]) if full else np.nan,
        "alpha": msd_alpha(curve, np.arange(maxlag + 1)) if full else np.nan,
        "frac_spread": _frac_spread(states),
        "speed_spread": _mean_or_nan(sp[st_f == STATE_SPREAD]),
        "speed_rounded": _mean_or_nan(sp[st_f == STATE_ROUNDED]),
        "speed_isolated": _mean_or_nan(sp[nb_f == 0]),
        "speed_crowded": _mean_or_nan(sp[nb_f >= 1]),
        "med_neighbors": float(np.nanmedian(nb[pres])) if pres.any() else np.nan,
        "med_nndist": (float(np.nanmedian(nnd_pres))
                       if np.isfinite(nnd_pres).any() else np.nan),
        "frac_isolated": (float(np.nanmean(nb[pres] == 0))
                          if pres.any() else np.nan),
    }


def all_cells(data, dt=DEFAULT_DT, maxlag=_MSD_MAXLAG):
    rows = []
    for c in CONDITIONS:
        for rec in data[c].get("cells", data[c]["all"]):
            rows.append(cell_features(rec, dt, maxlag))
    return rows


# ----------------------------------------------------- recording aggregation
def _agg(values, how):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not v.size:
        return np.nan
    return float(np.median(v)) if how.startswith("median") else float(np.mean(v))


_SRC = {"netdisp": "netdisp", "endpoint_msd": "endpoint_msd", "speed": "speed",
        "speed_spread": "speed_spread", "frac_spread": "frac_spread",
        "msd_alpha": "alpha", "n_neighbors": "med_neighbors",
        "nn_dist": "med_nndist", "frac_isolated": "frac_isolated"}


def per_recording(rows, dt=DEFAULT_DT, maxlag=_MSD_MAXLAG):
    """{metric: {cond: [one value per recording]}} for the arm tests.

    Per-cell metrics are aggregated (median/mean); the PRW D & P come from
    a single Fürth fit to each recording's mean MSD over its full cells."""
    by_rec = {}
    for r in rows:
        by_rec.setdefault((r["cond"], r["label"]), []).append(r)
    keys = [k for k, _ in ALL_ARM_METRICS]
    groups = {m: {c: [] for c in CONDITIONS} for m in keys}
    lag_min = np.arange(maxlag + 1) * dt
    rec_rows = []                                     # for the OLS adjustment
    for (cond, label), cells in by_rec.items():
        rec = {"cond": cond, "label": label}
        for key, _lab, how in REC_METRICS:
            src = _SRC[key.replace("_med", "").replace("_mean", "")]
            vals = ([c[src] for c in cells if c["full"]] if how == "median_full"
                    else [c[src] for c in cells])
            rec[key] = _agg(vals, how)
            groups[key][cond].append(rec[key])
        curves = [c["msd_curve"] for c in cells
                  if c["full"] and c["msd_curve"] is not None]
        if len(curves) >= 3:
            ens = np.nanmean(np.vstack(curves), axis=0)
            D, P = furth_fit(lag_min, ens)
        else:
            D, P = np.nan, np.nan
        rec["prw_D"], rec["prw_P"] = D, P
        groups["prw_D"][cond].append(D)
        groups["prw_P"][cond].append(P)
        rec_rows.append(rec)
    return groups, rec_rows


# ------------------------------------------------------ confounder analytics
def _paired(rows, a, b):
    """Wilcoxon signed-rank on per-cell (a, b) where both finite. Returns
    (n, median_a, median_b, p)."""
    from scipy.stats import wilcoxon
    pa, pb = [], []
    for r in rows:
        if np.isfinite(r[a]) and np.isfinite(r[b]):
            pa.append(r[a]); pb.append(r[b])
    if len(pa) < 8:
        return len(pa), np.nan, np.nan, None
    try:
        p = float(wilcoxon(pa, pb).pvalue)
    except ValueError:
        p = None
    return len(pa), float(np.median(pa)), float(np.median(pb)), p


def _spearman(rows, x, y):
    from scipy.stats import spearmanr
    xs = [r[x] for r in rows if np.isfinite(r[x]) and np.isfinite(r[y])]
    ys = [r[y] for r in rows if np.isfinite(r[x]) and np.isfinite(r[y])]
    if len(xs) < 8:
        return len(xs), np.nan, None
    rho, p = spearmanr(xs, ys)
    return len(xs), float(rho), float(p)


def _ols_adjusted(rec_rows, arm, outcome):
    """Recording-level OLS: outcome ~ treatment dummies + frac_spread +
    density. Returns per-test-condition adjusted coef/SE/t/p (treatment
    effect after removing the state-time + crowding confounds)."""
    from scipy.stats import t as tdist
    spec = ARMS[arm]
    conds, ctrl = spec["conditions"], spec["control"]
    tests = [c for c in conds if c != ctrl]
    rows = [r for r in rec_rows if r["cond"] in conds]
    y = np.array([r[outcome] for r in rows])
    fs = np.array([r["frac_spread_mean"] for r in rows])
    de = np.array([r["n_neighbors_med"] for r in rows])
    cols = [np.ones(len(rows))]
    cols += [np.array([1.0 if r["cond"] == t else 0.0 for r in rows])
             for t in tests]
    cols += [fs, de]
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
    out = {"n_recordings": int(len(y)), "dof": int(dof),
           "frac_spread_coef": float(beta[-2]), "density_coef": float(beta[-1])}
    for i, t in enumerate(tests, start=1):
        tv = beta[i] / se[i] if se[i] > 0 else np.nan
        out[t] = {"coef": float(beta[i]), "se": float(se[i]),
                  "t": float(tv) if np.isfinite(tv) else None,
                  "p": (float(2 * tdist.sf(abs(tv), dof))
                        if np.isfinite(tv) else None)}
    return out


# ------------------------------------------------------------------- plots
def _scatter_speed_density(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for c in CONDITIONS:
        xs = [r["med_neighbors"] for r in rows
              if r["cond"] == c and np.isfinite(r["med_neighbors"])
              and np.isfinite(r["speed"])]
        ys = [r["speed"] for r in rows
              if r["cond"] == c and np.isfinite(r["med_neighbors"])
              and np.isfinite(r["speed"])]
        ax.scatter(xs, ys, s=18, alpha=0.55, color=_COND_COLOR.get(c),
                   label=f"{c} (n={len(xs)})")
    n, rho, p = _spearman(rows, "med_neighbors", "speed")
    ax.set_xlabel(f"median neighbours within {NEIGHBOR_RADIUS_UM:.0f} µm "
                  "(local crowding)")
    ax.set_ylabel("mean step speed  (µm/min)")
    ax.set_title("Contact inhibition: cell speed vs local crowding\n"
                 f"Spearman ρ={rho:.2f}, p={p:.1e}  (n={n} cells, all treatments)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)


# ------------------------------------------------------------------ report
def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    star = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05
            else "ns")
    return f"{p:.3f} {star}"


def write_report(groups, arm_stats, rows, rec_rows, ols, lmm, path):
    L = ["# IC295 motility & dispersal — design-correct + confounder-aware",
         "",
         "Per-recording arm-structured tests (the recording is the unit, "
         "n≈8/condition). Within-arm Bonferroni vs the arm control; vehicle "
         "is a single WT-vs-DMSO MWU.", ""]
    L.append("## (b) Per-recording dispersal/motility — test vs control")
    L.append("")
    L.append("| metric | KO vs WT | GOF vs WT | Y1 vs DMSO | OT vs DMSO | "
             "WT vs DMSO (veh) |")
    L.append("|---|---|---|---|---|---|")
    for key, lab in ALL_ARM_METRICS:
        st = arm_stats[key]
        def pr(arm, a, b):
            return st[arm]["pairs"].get(f"{a}_vs_{b}", {}).get("p_bonf")
        veh = st["vehicle"]["pairs"].get("WT_vs_DMSO", {}).get("p_raw")
        L.append(f"| {lab} | {_fmt_p(pr('genetic','WT','KO'))} | "
                 f"{_fmt_p(pr('genetic','WT','GOF'))} | "
                 f"{_fmt_p(pr('drug','DMSO','Y1'))} | "
                 f"{_fmt_p(pr('drug','DMSO','OT'))} | {_fmt_p(veh)} |")
    L += ["", "## Confounder 1 — cell STATE (rounded vs spread)", ""]
    n, ms, mr, p = _paired(rows, "speed_spread", "speed_rounded")
    L.append(f"- Within-cell paired speed (n={n} cells with both states): "
             f"spread median **{ms:.2f}** vs rounded **{mr:.2f}** µm/min, "
             f"Wilcoxon p={_fmt_p(p)}. Spread cells migrate; rounded barely "
             "move — so any treatment that shifts the rounded:spread balance "
             "shifts dispersal even with identical per-state motility.")
    L.append("- `frac_spread` is therefore tested as its own per-recording "
             "metric in the table above (time-in-state channel).")
    L += ["", "## Confounder 2 — CONTACT / local density", ""]
    n, rho, p = _spearman(rows, "med_neighbors", "speed")
    L.append(f"- Speed vs crowding (Spearman, n={n} cells): ρ={rho:.2f}, "
             f"p={_fmt_p(p)} — negative ρ = contact inhibition (crowded cells "
             "slower). See `speed_vs_density.png`.")
    n, mi, mc, p = _paired(rows, "speed_isolated", "speed_crowded")
    L.append(f"- Within-cell paired speed (n={n}): isolated median "
             f"**{mi:.2f}** vs in-contact **{mc:.2f}** µm/min, Wilcoxon "
             f"p={_fmt_p(p)}.")
    L.append("- Crowding itself differs by treatment (`n_neighbors_med` / "
             "`nn_dist_med` rows above) — so a raw dispersal difference may be "
             "a density difference. The OLS below adjusts for it.")
    L += ["", "## Confounder 3 — pseudoreplication: covariate-adjusted OLS", "",
          "Recording-level OLS: outcome ~ treatment + frac_spread + density. "
          "A treatment effect is only believable if it survives here.", ""]
    for outcome in ("speed_med", "netdisp_med"):
        L.append(f"**{outcome}**")
        for arm in ARMS:
            o = ols[outcome][arm]
            if not o:
                L.append(f"- {arm}: insufficient dof"); continue
            ctrl = ARMS[arm]["control"]
            bits = []
            for t in ARMS[arm]["conditions"]:
                if t == ctrl or t not in o:
                    continue
                bits.append(f"{t} vs {ctrl}: adj. coef={o[t]['coef']:.2f}, "
                            f"p={_fmt_p(o[t]['p'])}")
            L.append(f"- {arm} (n={o['n_recordings']} rec, dof={o['dof']}): "
                     + "; ".join(bits))
        L.append("")
    L += ["## Confounder 3b — cell-level linear mixed model (gold standard)",
          "",
          "`speed ~ C(treatment) + frac_spread + density + (1 | recording)`. "
          "Per-cell rows; the recording random intercept handles "
          "pseudoreplication while frac_spread + density partial out the "
          "state-time and crowding confounds. Coefficients read µm/min vs the "
          "arm control.", ""]
    if not LMM_AVAILABLE:
        L.append("- statsmodels unavailable — LMM skipped (OLS above stands in).")
    for arm in ARMS:
        m = (lmm or {}).get(arm)
        if not m:
            L.append(f"- {arm}: LMM unavailable / did not converge."); continue
        ctrl = ARMS[arm]["control"]
        bits = [f"{t} vs {ctrl}: β={m[t]['coef']:.2f} µm/min, p={_fmt_p(m[t]['p'])}"
                for t in ARMS[arm]["conditions"] if t != ctrl and t in m]
        cov = m.get("_covariates", {})
        covbits = "; ".join(
            f"{k}: β={v['coef']:.3g}, p={_fmt_p(v['p'])}" for k, v in cov.items())
        L.append(f"- {arm} (n={m['n']} cells / {m['groups']} recordings): "
                 + "; ".join(bits))
        if covbits:
            L.append(f"    - covariates — {covbits}")
    L.append("")
    L.append("_Caveat: full-duration cohort under-samples the most motile "
             "cells (they leave the field of view), so absolute dispersal is "
             "conservative. Cell-level numbers are pseudoreplicated and shown "
             "for description only; inference is the per-recording tests + OLS._")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore the cache and recollect from masks")
    ap.add_argument("--from-cache", action="store_true",
                    help="require the cache (fail if absent)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_or_build(DEFAULT_UM, DEFAULT_DT,
                         from_cache=args.from_cache, rebuild=args.rebuild)
    if "cells" not in next(iter(data.values())):
        print("ERROR: cache lacks the enriched per-cell records. Rebuild "
              "with: python scripts/ic295_track_data.py --rebuild")
        return 1

    rows = all_cells(data)
    groups, rec_rows = per_recording(rows)

    arm_stats = {key: _arm_stats(groups[key]) for key, _ in ALL_ARM_METRICS}
    for key, _lab in ALL_ARM_METRICS:
        _plot_arms(key, groups[key], arm_stats[key],
                   os.path.join(OUT_DIR, "plots_arms", f"{key}.png"))
    atomic_write_json(os.path.join(OUT_DIR, "stats_arms_motility.json"),
                      arm_stats)

    ols = {o: {arm: _ols_adjusted(rec_rows, arm, o) for arm in ARMS}
           for o in ("speed_med", "netdisp_med")}
    lmm = {arm: fit_lmm(rows, ARMS[arm]["conditions"], ARMS[arm]["control"])
           for arm in ARMS}

    _scatter_speed_density(rows, os.path.join(OUT_DIR, "speed_vs_density.png"))
    write_report(groups, arm_stats, rows, rec_rows, ols, lmm,
                 os.path.join(OUT_DIR, "REPORT.md"))

    print(f"\nWrote motility stats → {OUT_DIR}/")
    _print_summary("recording (motility)", arm_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
