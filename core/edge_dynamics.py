"""Polar boundary representation and edge velocity kymograph."""
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter

from core.contour import get_contour, resample_polar
from config import (
    N_ANGULAR_SECTORS, EDGE_AGG_METHOD,
    EDGE_ANGULAR_SMOOTH_WINDOW, EDGE_TEMPORAL_SIGMA,
)


def _smooth_polar_radii(radii, window=5, polyorder=2):
    n = len(radii)
    if window < 3 or window >= n:
        return radii
    if window % 2 == 0:
        window += 1
    padded = np.concatenate([radii[-window:], radii, radii[:window]])
    return savgol_filter(padded, window, polyorder)[window:window + n]


def _smooth_kymograph(vel, temporal_sigma=1.0):
    if vel.size == 0:
        return vel
    nan = np.isnan(vel)
    if nan.all():
        return vel
    filled = vel.copy()
    filled[nan] = 0.0
    if temporal_sigma > 0:
        filled = ndimage.gaussian_filter1d(filled, temporal_sigma, axis=0)
    filled[nan] = np.nan
    return filled


def edge_velocity_pair(mask_prev, mask_curr, centroid_prev, centroid_curr,
                       um_per_px, dt, n_sectors=N_ANGULAR_SECTORS,
                       agg=EDGE_AGG_METHOD, angular_smooth=EDGE_ANGULAR_SMOOTH_WINDOW):
    """Radial edge velocity between two consecutive frames."""
    cp = get_contour(mask_prev)
    cc = get_contour(mask_curr)
    if cp is None or cc is None:
        return None, None
    mid = (centroid_prev + centroid_curr) / 2
    _, rp = resample_polar(cp, mid, n_sectors, agg)
    angles, rc = resample_polar(cc, mid, n_sectors, agg)
    if angular_smooth > 0:
        rp = _smooth_polar_radii(rp, angular_smooth)
        rc = _smooth_polar_radii(rc, angular_smooth)
    return angles, (rc - rp) * um_per_px / dt


def edge_velocity_kymograph(masks, centroids_px, um_per_px, dt,
                            agg=EDGE_AGG_METHOD,
                            angular_smooth=EDGE_ANGULAR_SMOOTH_WINDOW,
                            temporal_sigma=EDGE_TEMPORAL_SIGMA):
    """Full edge velocity kymograph (n_frames-1, n_sectors)."""
    n = len(masks)
    vel = np.full((n - 1, N_ANGULAR_SECTORS), np.nan)
    sector_angles = np.linspace(-np.pi, np.pi, N_ANGULAR_SECTORS, endpoint=False)
    for i in range(n - 1):
        if np.isnan(centroids_px[i, 0]) or np.isnan(centroids_px[i + 1, 0]):
            continue
        angles, v = edge_velocity_pair(
            masks[i], masks[i + 1],
            centroids_px[i], centroids_px[i + 1],
            um_per_px, dt, agg=agg, angular_smooth=angular_smooth,
        )
        if v is not None:
            vel[i] = v
            sector_angles = angles
    if temporal_sigma > 0:
        vel = _smooth_kymograph(vel, temporal_sigma)
    return sector_angles, vel


def edge_summary(velocity_map):
    """Protrusion/retraction summary stats."""
    valid = velocity_map[~np.isnan(velocity_map)]
    if len(valid) == 0:
        return {k: float("nan") for k in [
            "mean_protrusion_velocity", "mean_retraction_velocity",
            "protrusion_fraction", "net_velocity",
            "max_protrusion", "max_retraction",
        ]}
    prot = valid[valid > 0]
    retr = valid[valid < 0]
    return {
        "mean_protrusion_velocity":
            float(np.mean(prot)) if len(prot) else 0.0,
        "mean_retraction_velocity":
            float(np.mean(retr)) if len(retr) else 0.0,
        "protrusion_fraction": float(len(prot) / len(valid)),
        "net_velocity": float(np.mean(valid)),
        "max_protrusion": float(np.max(prot)) if len(prot) else 0.0,
        "max_retraction": float(np.min(retr)) if len(retr) else 0.0,
    }
