"""Advanced analysis functions: MSD fitting, bootstrap CIs,
trajectory smoothing, frame quality flagging.
"""
import numpy as np
from scipy import stats


def fit_msd_diffusion(lags, msd, dt=1.0):
    """Fit MSD curve to extract diffusion coefficient and exponent.

    Fits MSD = 4D * t^alpha in log-log space.

    Returns:
        dict with D (diffusion coefficient), alpha (anomalous exponent),
        r_squared, and fit line.
    """
    valid = (lags > 0) & (msd > 0) & ~np.isnan(msd)
    if valid.sum() < 3:
        return {"D": 0, "alpha": 1.0, "r_squared": 0,
                "fit_lags": [], "fit_msd": []}
    t = np.log10(lags[valid] * dt)
    y = np.log10(msd[valid])
    slope, intercept, r, p, se = stats.linregress(t, y)
    D = 10**intercept / 4.0
    alpha = slope
    fit_lags = lags[valid] * dt
    fit_msd = 4 * D * fit_lags**alpha
    return {
        "D": float(D),
        "alpha": float(alpha),
        "r_squared": float(r**2),
        "fit_lags": fit_lags.tolist(),
        "fit_msd": fit_msd.tolist(),
    }


def bootstrap_ci(values, n_boot=1000, ci=0.95, statistic=np.mean):
    """Compute bootstrap confidence interval for a statistic.

    Returns:
        dict with mean, ci_low, ci_high, se
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        v = float(statistic(arr)) if len(arr) else 0
        return {"mean": v, "ci_low": v, "ci_high": v, "se": 0}
    rng = np.random.default_rng(42)
    boot = np.array([statistic(rng.choice(arr, len(arr), replace=True))
                     for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(boot, 100 * alpha))
    ci_high = float(np.percentile(boot, 100 * (1 - alpha)))
    return {
        "mean": float(statistic(arr)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": float(np.std(boot)),
    }


def smooth_trajectory(centroids_px, sigma=1.5):
    """Gaussian-smooth a trajectory to reduce centroid jitter.

    Args:
        centroids_px: (N, 2) array of (row, col) centroids
        sigma: smoothing width in frames

    Returns:
        (N, 2) smoothed centroids (NaN frames preserved)
    """
    from scipy.ndimage import gaussian_filter1d
    out = centroids_px.copy()
    for dim in range(2):
        col = out[:, dim]
        valid = ~np.isnan(col)
        if valid.sum() < 3:
            continue
        smoothed = gaussian_filter1d(col[valid], sigma)
        out[valid, dim] = smoothed
    return out


def flag_quality_issues(masks, area_change_threshold=0.5):
    """Flag frames with potential detection quality issues.

    Returns:
        dict with:
            suspect_frames: list of frame indices
            reasons: list of reason strings
            area_changes: (N-1,) array of relative area changes
    """
    n = len(masks)
    areas = np.array([m.sum() for m in masks], dtype=float)
    suspect = []
    reasons = []

    area_changes = np.zeros(n - 1)
    for i in range(1, n):
        if areas[i-1] == 0 or areas[i] == 0:
            area_changes[i-1] = 1.0
            if areas[i] == 0 and areas[i-1] > 0:
                suspect.append(i)
                reasons.append(f"frame {i}: cell lost (area=0)")
            continue
        change = abs(areas[i] - areas[i-1]) / areas[i-1]
        area_changes[i-1] = change
        if change > area_change_threshold:
            suspect.append(i)
            reasons.append(
                f"frame {i}: area change {change:.0%} "
                f"({areas[i-1]:.0f} → {areas[i]:.0f})")

    return {
        "suspect_frames": suspect,
        "reasons": reasons,
        "area_changes": area_changes,
        "n_suspect": len(suspect),
    }


def check_normality(values, alpha=0.05):
    """Shapiro-Wilk normality test.

    Returns:
        dict with is_normal, statistic, p_value
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3 or len(arr) > 5000:
        return {"is_normal": len(arr) < 3, "statistic": 0,
                "p_value": 1.0}
    stat, p = stats.shapiro(arr)
    return {
        "is_normal": p > alpha,
        "statistic": float(stat),
        "p_value": float(p),
    }
