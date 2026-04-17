"""Centroid tracking and migration metrics."""
import numpy as np
from skimage import measure


def extract_centroids(masks):
    """Centroid (row, col) per frame; NaN where no cell."""
    n = len(masks)
    out = np.full((n, 2), np.nan)
    for i in range(n):
        if masks[i].any():
            props = measure.regionprops(masks[i].astype(np.uint8))
            if props:
                out[i] = props[0].centroid
    return out


def centroids_to_um(centroids_px, um_per_px):
    """Convert pixel centroids to micrometres."""
    return centroids_px * um_per_px


def trajectory_origin_normalized(centroids_um):
    """Translate trajectory so the first valid frame is at origin."""
    valid = ~np.isnan(centroids_um[:, 0])
    if not valid.any():
        return centroids_um.copy()
    origin = centroids_um[valid][0]
    out = centroids_um - origin
    out[~valid] = np.nan
    return out


def instantaneous_speed(centroids_um, dt):
    """Speed (μm/min) between consecutive valid frames."""
    n = len(centroids_um)
    speed = np.full(n - 1, np.nan)
    for i in range(n - 1):
        if not (np.isnan(centroids_um[i, 0]) or np.isnan(centroids_um[i + 1, 0])):
            d = np.linalg.norm(centroids_um[i + 1] - centroids_um[i])
            speed[i] = d / dt
    return speed


def total_distance(centroids_um):
    speed = instantaneous_speed(centroids_um, 1.0)  # use unit dt
    return float(np.nansum(speed))


def net_displacement(centroids_um):
    valid = ~np.isnan(centroids_um[:, 0])
    if valid.sum() < 2:
        return float("nan")
    pts = centroids_um[valid]
    return float(np.linalg.norm(pts[-1] - pts[0]))


def persistence_ratio(centroids_um):
    td = total_distance(centroids_um)
    nd = net_displacement(centroids_um)
    return float(nd / td) if td > 0 else float("nan")


def mean_squared_displacement(centroids_um, max_lag=None):
    """MSD curve via overlapping windows."""
    valid = ~np.isnan(centroids_um[:, 0])
    pts = centroids_um[valid]
    n = len(pts)
    if n < 2:
        return np.array([]), np.array([]), np.array([])
    if max_lag is None:
        max_lag = n // 2
    lags = np.arange(1, min(max_lag, n))
    msd = np.zeros(len(lags))
    sem = np.zeros(len(lags))
    for j, lag in enumerate(lags):
        d2 = np.sum((pts[lag:] - pts[:-lag]) ** 2, axis=1)
        msd[j] = np.mean(d2)
        sem[j] = np.std(d2) / np.sqrt(len(d2)) if len(d2) > 1 else 0
    return lags, msd, sem


def direction_autocorrelation(centroids_um, max_lag=None):
    """Direction autocorrelation (DiPer method)."""
    valid = ~np.isnan(centroids_um[:, 0])
    pts = centroids_um[valid]
    n = len(pts)
    if n < 3:
        return np.array([]), np.array([]), np.array([])
    vecs = pts[1:] - pts[:-1]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    units = vecs / np.maximum(norms, 1e-8)
    if max_lag is None:
        max_lag = (n - 1) // 2
    lags = np.arange(0, min(max_lag, n - 1))
    ac = np.zeros(len(lags))
    sem = np.zeros(len(lags))
    for j, lag in enumerate(lags):
        if lag == 0:
            cosines = np.ones(len(units))
        else:
            cosines = np.sum(units[:-lag] * units[lag:], axis=1)
        ac[j] = np.mean(cosines)
        sem[j] = np.std(cosines) / np.sqrt(len(cosines)) if len(cosines) > 1 else 0
    return lags, ac, sem
