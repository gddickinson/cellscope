"""Contour utilities: extract, smooth, polar conversion."""
import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, measure


def get_contour(mask):
    """Extract the outer contour of a binary mask.

    Returns:
        (N, 2) array of (row, col) points, or None if no contour.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze()
    if pts.ndim == 1:
        return None
    return np.column_stack([pts[:, 1], pts[:, 0]])


def keep_largest_component(mask):
    """Keep only the largest connected component."""
    labeled = measure.label(mask.astype(np.uint8))
    if labeled.max() == 0:
        return mask.astype(bool)
    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    return (labeled == largest.label)


def fourier_smooth_contour(contour, n_descriptors=40, n_output=360):
    """Smooth a closed contour by Fourier descriptor filtering.

    Args:
        contour: (N, 2) (row, col) points
        n_descriptors: number of low-frequency coefficients to keep
        n_output: number of points in the smoothed contour

    Returns:
        (n_output, 2) smoothed contour
    """
    if contour is None or len(contour) < 4:
        return contour

    rows = contour[:, 0].astype(float)
    cols = contour[:, 1].astype(float)

    # Resample uniformly
    t = np.linspace(0, 1, len(rows), endpoint=False)
    t_new = np.linspace(0, 1, n_output, endpoint=False)
    rows_r = np.interp(t_new, t, rows, period=1)
    cols_r = np.interp(t_new, t, cols, period=1)

    # FFT, low-pass, IFFT
    z = cols_r + 1j * rows_r
    Z = np.fft.fft(z)
    n_keep = min(n_descriptors // 2, len(Z) // 2)
    mask = np.zeros(len(Z), dtype=bool)
    mask[:n_keep + 1] = True
    mask[-n_keep:] = True
    Z[~mask] = 0
    z_smooth = np.fft.ifft(Z)

    return np.column_stack([z_smooth.imag, z_smooth.real])


def smooth_mask_fourier(mask, n_descriptors=40):
    """Apply Fourier smoothing to a mask boundary."""
    if not mask.any():
        return mask
    contour = get_contour(mask)
    if contour is None:
        return mask
    smoothed = fourier_smooth_contour(contour, n_descriptors)
    if smoothed is None:
        return mask
    pts = smoothed[:, ::-1].astype(np.int32)
    new = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(new, [pts], 1)
    return new.astype(bool)


def smooth_all_masks_fourier(masks, n_descriptors=40, progress_fn=None):
    """Batch Fourier smoothing."""
    out = np.zeros_like(masks)
    for i in range(len(masks)):
        if progress_fn:
            progress_fn(i)
        out[i] = smooth_mask_fourier(masks[i], n_descriptors)
    return out


def temporal_smooth_polar_boundaries(masks, centroids_px,
                                     temporal_sigma=1.5,
                                     n_sectors=72, progress_fn=None):
    """Smooth boundaries temporally via polar (radius) representation.

    Converts each contour to polar radii (one value per angular sector),
    Gaussian-smooths each sector's radius across frames, and reconstructs
    the masks. This is independent of the per-frame metric — it just
    averages out frame-to-frame jitter and gives a substantial IoU
    improvement at virtually no quality cost.

    Args:
        masks: (N, H, W) bool array
        centroids_px: (N, 2) array of (row, col) centroids
        temporal_sigma: Gaussian sigma in frames (1-2 recommended)
        n_sectors: number of angular sectors for polar representation

    Returns:
        smoothed_masks: (N, H, W) bool array
    """
    n_frames = len(masks)
    h, w = masks.shape[1], masks.shape[2]
    sector_angles = np.linspace(-np.pi, np.pi, n_sectors, endpoint=False)
    radii_matrix = np.full((n_frames, n_sectors), np.nan)

    # Step 1: convert all contours to polar radii
    for i in range(n_frames):
        if progress_fn:
            progress_fn(i)
        contour = get_contour(masks[i])
        if contour is None or np.isnan(centroids_px[i, 0]):
            continue
        _, radii = resample_polar(
            contour, centroids_px[i], n_sectors, agg="median"
        )
        radii_matrix[i] = radii

    # Step 2: interpolate NaN frames per sector
    for s in range(n_sectors):
        col = radii_matrix[:, s]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        idx = np.arange(n_frames)
        col[~valid] = np.interp(idx[~valid], idx[valid], col[valid])
        radii_matrix[:, s] = col

    # Step 3: temporal Gaussian smoothing per sector
    if temporal_sigma > 0:
        for s in range(n_sectors):
            radii_matrix[:, s] = ndimage.gaussian_filter1d(
                radii_matrix[:, s], sigma=temporal_sigma
            )

    # Step 4: reconstruct masks from smoothed polar radii
    smoothed = np.zeros_like(masks)
    for i in range(n_frames):
        if np.isnan(centroids_px[i, 0]) or np.all(np.isnan(radii_matrix[i])):
            smoothed[i] = masks[i]
            continue
        cx, cy = centroids_px[i]
        radii = radii_matrix[i]
        rows = cx + radii * np.sin(sector_angles)
        cols = cy + radii * np.cos(sector_angles)
        rows = np.clip(rows, 0, h - 1)
        cols = np.clip(cols, 0, w - 1)
        pts = np.column_stack([cols, rows]).astype(np.int32)
        new_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(new_mask, [pts], 1)
        smoothed[i] = new_mask.astype(bool) if new_mask.any() else masks[i]
    return smoothed


def contour_to_polar(contour, centroid):
    """Polar coordinates of contour points relative to centroid."""
    dy = contour[:, 0] - centroid[0]
    dx = contour[:, 1] - centroid[1]
    return np.arctan2(dy, dx), np.sqrt(dy ** 2 + dx ** 2)


def resample_polar(contour, centroid, n_sectors=72, agg="median"):
    """Bin contour radii into fixed angular sectors."""
    angles, radii = contour_to_polar(contour, centroid)
    sector_angles = np.linspace(-np.pi, np.pi, n_sectors, endpoint=False)
    sector_w = 2 * np.pi / n_sectors
    sector_radii = np.zeros(n_sectors)
    fn = {"median": np.median, "mean": np.mean, "max": np.max}.get(agg, np.median)

    for i, sa in enumerate(sector_angles):
        lo = sa - sector_w / 2
        hi = sa + sector_w / 2
        if lo < -np.pi:
            in_bin = (angles >= lo + 2 * np.pi) | (angles < hi)
        elif hi > np.pi:
            in_bin = (angles >= lo) | (angles < hi - 2 * np.pi)
        else:
            in_bin = (angles >= lo) & (angles < hi)
        sector_radii[i] = fn(radii[in_bin]) if in_bin.any() else np.nan

    # Fill NaN gaps by circular interpolation
    if np.any(np.isnan(sector_radii)):
        valid = ~np.isnan(sector_radii)
        if valid.sum() > 2:
            idx = np.arange(n_sectors)
            sector_radii[~valid] = np.interp(
                idx[~valid], idx[valid], sector_radii[valid]
            )
    return sector_angles, sector_radii
