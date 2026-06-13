"""Migration-model fits for the IC295 motility analysis (pure functions).

Three model-based descriptors that separate *how directed* / *how
persistent* migration is from raw dispersal magnitude (which is outlier-
driven and confounds speed with directionality):

  msd_alpha(msd, lag)    log-log slope α of MSD(τ). α≈1 diffusive, α>1
                         super-diffusive / directionally persistent, α<1
                         confined. A shape parameter — robust to the few
                         hyper-motile cells that dominate MSD magnitude.

  furth_fit(lag, msd)    persistent-random-walk (Fürth) fit in 2D:
                         MSD(t) = 4 D [ t − P (1 − e^(−t/P)) ]
                         → motility coefficient D (µm²/min) and persistence
                         time P (min). Decomposes dispersal into a random-
                         motility part (D) and a directional-memory part (P).

  fit_lmm(table, arm)    cell-level linear mixed model with a recording
                         random intercept (statsmodels):
                         speed ~ C(treatment, ref=control) + frac_spread
                                 + density + (1 | recording)
                         the gold-standard pseudoreplication + confounder
                         control. Returns None if statsmodels is absent.

No I/O — `ic295_motility_stats` calls these and owns the cache/plots/report.
"""
import numpy as np

__all__ = ["msd_alpha", "furth_fit", "fit_lmm", "LMM_AVAILABLE"]

try:
    import statsmodels.formula.api as _smf       # noqa: F401
    LMM_AVAILABLE = True
except Exception:
    LMM_AVAILABLE = False


def msd_alpha(msd, lag, max_frac=0.5):
    """log-log slope of MSD(τ). Fit over the first `max_frac` of lags
    (the small-τ regime is where the power law is cleanest; the long-τ
    tail is noisy with few independent samples). Returns NaN if < 4 usable
    points."""
    msd = np.asarray(msd, float)
    lag = np.asarray(lag, float)
    hi = max(4, int(len(lag) * max_frac))
    sl = slice(1, hi + 1)
    x, y = lag[sl], msd[sl]
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 4:
        return float("nan")
    slope = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0]
    return float(slope)


def _furth_msd(t, D, P):
    # guard the exponential against P→0
    return 4.0 * D * (t - P * (1.0 - np.exp(-t / np.maximum(P, 1e-9))))


def furth_fit(lag, msd):
    """Fit the 2D persistent-random-walk MSD. Returns (D µm²/min, P min),
    (nan, nan) on failure. Initial guesses from the curve's tail/early
    slope; D,P bounded positive."""
    from scipy.optimize import curve_fit
    lag = np.asarray(lag, float)
    msd = np.asarray(msd, float)
    ok = np.isfinite(lag) & np.isfinite(msd) & (lag > 0) & (msd >= 0)
    if ok.sum() < 5:
        return float("nan"), float("nan")
    t, y = lag[ok], msd[ok]
    D0 = max(y[-1] / (4.0 * t[-1]), 1e-3)            # long-time slope ≈ 4D
    P0 = max(t[len(t) // 4], 1.0)
    try:
        popt, _ = curve_fit(_furth_msd, t, y, p0=[D0, P0],
                            bounds=([1e-6, 1e-3], [1e6, t[-1] * 5]),
                            maxfev=20000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return float("nan"), float("nan")


def fit_lmm(table, arm_conditions, control, outcome="speed"):
    """Cell-level linear mixed model, recording as random intercept.

    table: list of per-cell dicts with keys cond, label, <outcome>,
           frac_spread, med_neighbors. Rows outside `arm_conditions` or
           with non-finite fields are dropped. The control level is the
           reference so each coefficient reads "<test> vs control".

    Returns {test_cond: {coef, p, se}, '_covariates': {...}, 'n', 'groups'}
    or None if statsmodels is unavailable / the fit fails."""
    if not LMM_AVAILABLE:
        return None
    import statsmodels.formula.api as smf
    import pandas as pd
    rows = []
    for r in table:
        if r["cond"] not in arm_conditions:
            continue
        vals = (r.get(outcome), r.get("frac_spread"), r.get("med_neighbors"))
        if not all(v is not None and np.isfinite(v) for v in vals):
            continue
        rows.append({"y": float(r[outcome]), "frac_spread": float(r["frac_spread"]),
                     "density": float(r["med_neighbors"]),
                     "treatment": r["cond"], "recording": r["label"]})
    if len(rows) < len(arm_conditions) * 3:
        return None
    df = pd.DataFrame(rows)
    df["treatment"] = pd.Categorical(
        df["treatment"],
        categories=[control] + [c for c in arm_conditions if c != control])
    try:
        md = smf.mixedlm("y ~ C(treatment) + frac_spread + density", df,
                         groups=df["recording"])
        fit = md.fit(reml=True, method="lbfgs")
    except Exception:
        return None
    out = {"n": int(len(df)), "groups": int(df["recording"].nunique()),
           "_covariates": {}}
    for name in fit.params.index:
        coef = float(fit.params[name]); p = float(fit.pvalues[name])
        se = float(fit.bse[name])
        if name.startswith("C(treatment)[T."):
            cond = name.split("[T.")[1].rstrip("]")
            out[cond] = {"coef": coef, "p": p, "se": se}
        elif name in ("frac_spread", "density"):
            out["_covariates"][name] = {"coef": coef, "p": p}
    return out
