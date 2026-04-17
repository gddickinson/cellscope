"""Proper membrane-quality metrics that distinguish true cell boundaries
from internal features.

The naive `boundary_confidence` metric rewards high image gradient along
the contour — but interior features (organelles, nucleus, vesicles) also
have high gradient. We need metrics that test whether each contour point
sits on a real outside-inside transition.

Key principle: a true membrane separates two distinguishable regions.
The cell interior has high texture/variance and a particular intensity
distribution; the background has low texture and a different distribution.
At an internal feature, both sides look like cell interior — same texture,
similar intensity.

Three metrics provided:
  - `intensity_contrast_score`: mean |inside - outside| intensity along
    a sampled normal direction at multiple scales
  - `texture_contrast_score`: mean (inside_variance - outside_variance);
    positive at real boundaries (cell texture > background)
  - `membrane_score` (composite): combines both
"""
import numpy as np
import cv2
from scipy import ndimage

from core.contour import get_contour


def _outward_normals(contour):
    """Unit outward normals for a closed contour."""
    fwd = np.roll(contour, -1, axis=0)
    bwd = np.roll(contour, 1, axis=0)
    tan = fwd - bwd
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    norms = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(norms, 1e-8)
    centroid = contour.mean(axis=0)
    radial = contour - centroid
    radial_unit = radial / np.maximum(
        np.linalg.norm(radial, axis=1, keepdims=True), 1e-8
    )
    flip = np.sum(nrm * radial_unit, axis=1) < 0
    nrm[flip] *= -1
    return nrm


def _sample_along_normals(image, contour, normals, distances):
    """Sample image at offsets along outward normals.

    Args:
        image: 2D array
        contour: (N, 2) (row, col)
        normals: (N, 2) outward unit normals
        distances: list of signed distances; negative = inward

    Returns:
        samples: (len(distances), N) sampled values
    """
    h, w = image.shape[:2]
    out = np.zeros((len(distances), len(contour)), dtype=np.float64)
    for i, d in enumerate(distances):
        rows = contour[:, 0] + d * normals[:, 0]
        cols = contour[:, 1] + d * normals[:, 1]
        rows = np.clip(rows, 0, h - 1)
        cols = np.clip(cols, 0, w - 1)
        out[i] = ndimage.map_coordinates(
            image.astype(np.float64), [rows, cols], order=1, mode="nearest"
        )
    return out


def intensity_contrast_score(image, mask, distances=(5, 10, 15)):
    """Mean inside-vs-outside intensity contrast along the contour.

    For each contour point, samples the image at +d and -d pixels along
    the outward normal at multiple scales. Computes |inside - outside|
    averaged over the contour and over scales.

    Returns:
        score: float — higher = stronger inside/outside separation
    """
    contour = get_contour(mask)
    if contour is None or len(contour) < 8:
        return float("nan")
    normals = _outward_normals(contour)

    diffs = []
    for d in distances:
        inside = _sample_along_normals(image, contour, normals, [-d])[0]
        outside = _sample_along_normals(image, contour, normals, [d])[0]
        diffs.append(np.abs(inside - outside))
    return float(np.mean(np.stack(diffs)))


def texture_contrast_score(image, mask, distances=(5, 10, 15),
                           local_window=7):
    """Mean texture (local std) difference between inside and outside.

    Cell interior has high local variance (organelles, DIC features);
    background has low variance. A true membrane separates these.

    Returns:
        score: float — positive when interior is more textured than outside
    """
    contour = get_contour(mask)
    if contour is None or len(contour) < 8:
        return float("nan")

    # Pre-compute local std map (faster than per-point computation)
    img = image.astype(np.float64)
    k = local_window
    mean = cv2.blur(img, (k, k))
    sq = cv2.blur(img ** 2, (k, k))
    local_std = np.sqrt(np.clip(sq - mean ** 2, 0, None))

    normals = _outward_normals(contour)
    inside_stds = []
    outside_stds = []
    for d in distances:
        inside = _sample_along_normals(local_std, contour, normals, [-d])[0]
        outside = _sample_along_normals(local_std, contour, normals, [d])[0]
        inside_stds.append(inside)
        outside_stds.append(outside)
    inside_mean = np.mean(np.stack(inside_stds))
    outside_mean = np.mean(np.stack(outside_stds))
    return float(inside_mean - outside_mean)


def membrane_score(image, mask, intensity_weight=0.15,
                    texture_weight=0.85):
    """Composite membrane quality: intensity contrast + texture contrast.

    Texture contrast is the dominant signal (it reliably distinguishes
    cell interior from background). Intensity contrast is a small
    secondary signal — it can be misled by DIC intensity transitions
    across organelles inside the cell, so it gets a low weight.

    Both are computed at multiple distances (5, 10, 15 px).

    Returns:
        score: float — higher = better membrane location
    """
    if not mask.any():
        return float("nan")
    ic = intensity_contrast_score(image, mask)
    tc = texture_contrast_score(image, mask)
    if np.isnan(ic) or np.isnan(tc):
        return float("nan")
    # Texture contrast can go negative (wrong direction); clip at 0
    return intensity_weight * ic + texture_weight * max(tc, 0.0)


def membrane_score_timeseries(frames, masks):
    """Per-frame membrane scores."""
    n = len(frames)
    out = np.full(n, np.nan)
    for i in range(n):
        if masks[i].any():
            out[i] = membrane_score(frames[i], masks[i])
    return out
