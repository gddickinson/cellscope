"""Sequence-level preprocessing: background subtraction, high-pass, debris.

Ported from `claude_test/optical_flow/src/{background,debris_removal}.py`
which validated these on the same DIC keratinocyte data we use here.

Filters that run ONCE before per-frame analysis:

1. **ROI masking** (`apply_roi_mask`) — black out pixels outside a
   rectangular region in every frame. Keeps the full (N, H, W) shape so
   downstream code doesn't need to know about the ROI. Applied FIRST by
   `pipeline.detect()` so the zeroed-out region doesn't skew the
   temporal-statistic steps below.

2. **Temporal background subtraction** — per-pixel statistic (median by
   default) across the whole stack is subtracted. Stationary content,
   slowly drifting illumination, and fixed imaging artifacts vanish.

3. **Spatial high-pass** — per-frame Gaussian high-pass (image minus a
   heavily-blurred copy of itself) flattens smooth illumination gradients
   and vignetting. Sigma must be much larger than a cell radius so the
   blurred image approximates the background, not the cell.

4. **Physical-unit debris removal** — morphological alternating-sequence
   filter with a disk sized in micrometers. Wipes bright AND dark features
   smaller than `min_cell_diameter_um` (default 5 μm; safely below
   keratinocyte size). Requires `pixel_size_um`.

These are SAFE additions: each step has a graceful disable (None or 0)
and the output is always uint8 the same shape as input.
"""
import math
import cv2
import numpy as np


_TEMPORAL_METHODS = {"median", "mean", "min", "max"}


# ---------------------------------------------------------------------------
# Temporal background subtraction
# ---------------------------------------------------------------------------

def temporal_background(frames, method="median"):
    """Per-pixel temporal statistic across an (N, H, W) stack."""
    if method not in _TEMPORAL_METHODS:
        raise ValueError(
            f"unknown method: {method!r}, expected one of {_TEMPORAL_METHODS}"
        )
    arr = np.asarray(frames)
    if method == "median":
        return np.median(arr, axis=0).astype(np.float32)
    if method == "mean":
        return arr.astype(np.float32).mean(axis=0)
    if method == "min":
        return arr.min(axis=0).astype(np.float32)
    return arr.max(axis=0).astype(np.float32)


def subtract_background(frames, background, offset=128.0):
    """Subtract `background` from each frame and recenter at `offset`."""
    out = frames.astype(np.float32) - background[None, ...] + float(offset)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Region-of-interest masking
# ---------------------------------------------------------------------------

def apply_roi_mask(frames, roi, fill_value=0):
    """Black out pixels outside a rectangular ROI across every frame.

    The returned stack keeps the ORIGINAL (N, H, W) shape — downstream
    analysis code doesn't need to know about the ROI. Pixels outside
    `roi` are set to `fill_value` (default 0 for DIC, which creates a
    clear "outside is black" visual).

    Args:
        frames: (N, H, W) array-like.
        roi: (r0, r1, c0, c1) half-open bbox. If None or any bound is
            out of range, the frames are returned unchanged.
        fill_value: value to write outside the ROI (default 0).

    Returns:
        (N, H, W) array of the same dtype as `frames`.
    """
    if roi is None:
        return frames
    arr = np.asarray(frames)
    if arr.ndim != 3:
        return arr
    n, h, w = arr.shape
    r0, r1, c0, c1 = roi
    # Clamp and sanity-check
    r0 = max(0, int(r0))
    r1 = min(h, int(r1))
    c0 = max(0, int(c0))
    c1 = min(w, int(c1))
    if r1 <= r0 or c1 <= c0:
        return arr
    # Build the output with fill_value outside, original values inside
    out = np.full_like(arr, fill_value)
    out[:, r0:r1, c0:c1] = arr[:, r0:r1, c0:c1]
    return out


# ---------------------------------------------------------------------------
# Spatial high-pass
# ---------------------------------------------------------------------------

def spatial_highpass(img, sigma, offset=128.0):
    """Per-frame Gaussian high-pass; sigma should be >> cell radius."""
    if sigma is None or sigma <= 0:
        return img
    src = img.astype(np.float32)
    blur = cv2.GaussianBlur(src, (0, 0), float(sigma))
    out = src - blur + float(offset)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def spatial_highpass_sequence(frames, sigma, offset=128.0):
    """Apply `spatial_highpass` to every frame."""
    if sigma is None or sigma <= 0:
        return frames
    return np.stack(
        [spatial_highpass(f, sigma, offset) for f in frames], axis=0
    )


# ---------------------------------------------------------------------------
# Physical-unit debris removal
# ---------------------------------------------------------------------------

def diameter_to_radius_px(diameter_um, pixel_size_um):
    """Convert micron diameter to integer pixel radius (>=1)."""
    if pixel_size_um is None or pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be > 0")
    r = 0.5 * float(diameter_um) / float(pixel_size_um)
    return max(1, int(round(r)))


def remove_small_blobs(img, diameter_um, pixel_size_um):
    """Morphological ASF removing bright + dark features < diameter_um.

    Returns the input unchanged if `pixel_size_um` is missing.
    """
    if (diameter_um is None or diameter_um <= 0
            or pixel_size_um is None or pixel_size_um <= 0):
        return img
    radius = diameter_to_radius_px(diameter_um, pixel_size_um)
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def remove_small_blobs_sequence(frames, diameter_um, pixel_size_um):
    if (diameter_um is None or diameter_um <= 0
            or pixel_size_um is None or pixel_size_um <= 0):
        return frames
    return np.stack(
        [remove_small_blobs(f, diameter_um, pixel_size_um) for f in frames],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Combined sequence preprocessor
# ---------------------------------------------------------------------------

def preprocess_sequence(
    frames,
    temporal_method="median",
    spatial_highpass_sigma=40.0,
    debris_diameter_um=5.0,
    pixel_size_um=None,
    offset=128.0,
):
    """Apply all three filters in order: debris → temporal bg → spatial HP.

    Each step is skipped when its control parameter is falsy.

    Args:
        frames: (N, H, W) uint8 stack
        temporal_method: 'median' | 'mean' | 'min' | 'max' | None
        spatial_highpass_sigma: Gaussian sigma in pixels (None to skip)
        debris_diameter_um: features smaller than this are wiped (μm)
        pixel_size_um: required for debris removal
        offset: recentering offset for the background subtraction (default 128)

    Returns:
        cleaned: (N, H, W) uint8 stack, same shape as input
    """
    processed = frames

    # Step 1: debris removal (per-frame)
    if (debris_diameter_um and debris_diameter_um > 0
            and pixel_size_um and pixel_size_um > 0):
        processed = remove_small_blobs_sequence(
            processed, debris_diameter_um, pixel_size_um,
        )

    # Step 2: temporal background subtraction
    if temporal_method:
        bg = temporal_background(processed, method=temporal_method)
        processed = subtract_background(processed, bg, offset=offset)

    # Step 3: spatial high-pass
    if spatial_highpass_sigma and spatial_highpass_sigma > 0:
        processed = spatial_highpass_sequence(
            processed, spatial_highpass_sigma, offset=offset,
        )

    return processed
