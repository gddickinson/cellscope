"""Multi-channel (DIC + SiR-actin/Cy5) detection utilities.

Designed for the IC295 dataset (and similar multi-channel time-lapse
recordings) where cells are labelled with **SiR-actin** in addition
to DIC. SiR-actin is a fluorogenic F-actin probe — it's only bright
when bound to polymerised F-actin, so intensity is a biologically
meaningful proxy for cytoskeletal organisation, not just a "cell is
present" indicator. The Cy5 channel acts as a "is this a viable
cell?" filter — DIC catches everything cell-shaped (real cells AND
debris/dust), and we use Cy5 presence to drop the debris.

Key design choices:

* DIC stays the canonical boundary source — the user (Ignasi)
  confirmed no fully un-stained cells, so DIC's broader detection
  doesn't lose cells.
* Cy5 score is **adaptive per mask**: z-score of inside-mask intensity
  vs a local 30-px annulus background, not an absolute threshold.
  This handles the "large intensity range, some cells faintly
  labelled" observation — faint cells still have inside > local
  background.
* Use the 75th percentile of inside-mask Cy5 (not mean) so that low-
  actin filopodia don't drag the score down for cells with large
  protrusions.

Generic over N channels via the loader: pass `dic_ch`/`fluo_ch`
indices. A future calcium channel slots in without touching this
module — a separate "calcium analysis pipeline" is planned and
will be its own module.
"""
from __future__ import annotations

import os
import re

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, binary_dilation


# ────────────────────────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────────────────────────

def parse_n_channels(tif_path):
    """Read N channels from the metadata sidecar, default to 1."""
    sidecar = tif_path.replace(".ome.tif", "_metadata.txt")
    if not os.path.exists(sidecar):
        return 1
    with open(sidecar) as f:
        m = re.search(r'"Channels"\s*:\s*(\d+)', f.read())
    return int(m.group(1)) if m else 1


def parse_n_frames(tif_path, n_channels):
    sidecar = tif_path.replace(".ome.tif", "_metadata.txt")
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            m = re.search(r'"Frames"\s*:\s*(\d+)', f.read())
        if m:
            return int(m.group(1))
    with tifffile.TiffFile(tif_path) as tf:
        return max(1, len(tf.pages) // max(n_channels, 1))


def load_recording_multi(tif_path, dic_ch=1, fluo_ch=0, max_frames=None):
    """Load a multi-channel recording.

    Returns (dic_uint8, fluo_uint8) — both (N, H, W) uint8 stacks
    after channel-appropriate preprocessing:
      * DIC: flat-field σ=80, p1/p99 rescale
      * Fluorescence: p1/p99.5 percentile rescale (no flat-field —
        fluorescence has minimal vignette)
    """
    nch = parse_n_channels(tif_path)
    n = parse_n_frames(tif_path, nch)
    if max_frames is not None:
        n = min(n, max_frames)
    with tifffile.TiffFile(tif_path) as tf:
        h, w = tf.pages[0].shape
        dic = np.empty((n, h, w), dtype=np.uint8)
        fluo = np.empty((n, h, w), dtype=np.uint8)
        for i in range(n):
            dic_raw = tf.pages[i * nch + dic_ch].asarray()
            fluo_raw = tf.pages[i * nch + fluo_ch].asarray()
            dic[i] = to_uint8_dic(dic_raw)
            fluo[i] = to_uint8_fluorescence(fluo_raw)
    return dic, fluo


def _fast_flat_field_bg(f, sigma, downscale=None):
    """Approximate `scipy.ndimage.gaussian_filter(f, sigma)` by
    downsampling first.

    For flat-field correction we only care about the LOW-frequency
    background, so we can blur a downsampled image with a smaller σ
    and upsample. With σ=80 (DIC) the kernel radius is ~320 px on a
    1024² image — extremely slow as a separable filter. Downsampling
    by 4× lets us blur a 256² image with σ=20 (~30× faster) and the
    result agrees with scipy within ~1 intensity unit on the final
    uint8 (well below the noise floor of the underlying image).

    Defaults: downscale=4 for σ ≤ 100, downscale=8 for larger σ.
    """
    import cv2
    h, w = f.shape
    if downscale is None:
        downscale = 4 if sigma <= 100 else 8
    # Floor downscale so the small image is at least 32×32
    downscale = min(downscale, h // 32, w // 32)
    if downscale < 2:
        # Image too small to benefit — just call scipy directly
        return gaussian_filter(f, sigma=sigma)
    small = cv2.resize(f, (w // downscale, h // downscale),
                       interpolation=cv2.INTER_AREA)
    small_sigma = sigma / downscale
    small_bg = cv2.GaussianBlur(small, (0, 0), small_sigma)
    return cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)


def to_uint8_dic(raw_uint16, sigma=80):
    """Flat-field correction + p1/p99 → uint8.

    Uses the fast downsampled background estimate (see
    `_fast_flat_field_bg`). Reduces per-frame cost from ~500 ms
    (scipy) to ~5 ms.
    """
    f = raw_uint16.astype(np.float32)
    bg = _fast_flat_field_bg(f, sigma)
    bg = np.where(bg < 1, 1, bg)
    flat = f / bg - 1.0
    p1, p99 = np.percentile(flat, [1, 99])
    return np.clip((flat - p1) / max(p99 - p1, 1e-6) * 255,
                   0, 255).astype(np.uint8)


def to_uint8_fluorescence(raw_uint16, low=1.0, high=99.5,
                           flat_field_sigma=200):
    """Cy5/fluorescence: optional flat-field + percentile rescale.

    flat_field_sigma — Gaussian-blur σ for vignette estimation.
    Set to None/0 to skip (some scopes have negligible vignette).
    Default 200 (larger than DIC's 80) because fluorescent cell
    clusters span larger spatial scales and we don't want to
    subtract real cell signal as "background".

    high=99.5 (not 99) preserves the bright cell peaks; low=1 clips
    the noise floor.

    Uses fast downsampled bg estimate; ~1400× faster than scipy at
    σ=200.
    """
    f = raw_uint16.astype(np.float32)
    if flat_field_sigma:
        bg = _fast_flat_field_bg(f, flat_field_sigma)
        bg = np.where(bg < 1, 1, bg)
        f = f / bg - 1.0
    p1, p99 = np.percentile(f, [low, high])
    return np.clip((f - p1) / max(p99 - p1, 1e-6) * 255,
                   0, 255).astype(np.uint8)


# ────────────────────────────────────────────────────────────────────
# Cy5 presence scoring
# ────────────────────────────────────────────────────────────────────

def _local_background(cell_mask, image_uint8, expand_px=30):
    """Median of pixels in a dilated annulus around the mask."""
    dilated = binary_dilation(cell_mask, iterations=expand_px)
    ring = dilated & ~cell_mask
    if ring.sum() < 10:
        # Cell at frame edge — fall back to global low-percentile
        return float(np.percentile(image_uint8, 25))
    return float(np.median(image_uint8[ring]))


def cy5_presence_score(cell_mask_bool, cy5_uint8, expand_px=30,
                        max_z=10.0):
    """0-1 score: how strongly is Cy5 present inside this cell mask?

    1.0 = strong, ≥ max_z standard MADs above local background
    0.0 = inside is at-or-below local background
    Linear ramp in between. Robust (median + MAD), so faint cells
    score above zero, debris scores ~zero.
    """
    if not cell_mask_bool.any():
        return 0.0
    inside = cy5_uint8[cell_mask_bool].astype(np.float32)
    bg_med = _local_background(cell_mask_bool, cy5_uint8, expand_px)

    # Robust scale via MAD (median absolute deviation) of background ring
    dilated = binary_dilation(cell_mask_bool, iterations=expand_px)
    ring = dilated & ~cell_mask_bool
    if ring.sum() >= 10:
        bg_pix = cy5_uint8[ring].astype(np.float32)
        bg_mad = np.median(np.abs(bg_pix - bg_med))
    else:
        bg_mad = float(np.percentile(np.abs(
            cy5_uint8.astype(np.float32) - bg_med), 50))
    bg_mad = max(bg_mad, 1.0)  # avoid divide-by-zero

    # Use 75th percentile of inside, not mean — robust to filopodia
    # which have low actin density and would otherwise drag the score.
    inside_p75 = float(np.percentile(inside, 75))
    z = (inside_p75 - bg_med) / bg_mad
    return float(min(1.0, max(0.0, z / max_z)))


def score_all_labels(label_frame_int32, cy5_uint8, expand_px=30):
    """Score every cell ID in a label frame. Returns dict {id: score}."""
    out = {}
    if label_frame_int32.max() == 0:
        return out
    for lab in range(1, int(label_frame_int32.max()) + 1):
        m = label_frame_int32 == lab
        if not m.any():
            continue
        out[lab] = cy5_presence_score(m, cy5_uint8, expand_px)
    return out


def filter_dic_labels_by_cy5(label_frame_int32, cy5_uint8,
                              min_score=0.3, expand_px=30):
    """Drop DIC label IDs with Cy5 score < min_score.

    Returns (filtered_labels_int32, scores_dict, kept_count).
    Re-compacts label IDs to 1..N after filtering.
    """
    scores = score_all_labels(label_frame_int32, cy5_uint8, expand_px)
    kept_ids = sorted(lab for lab, s in scores.items() if s >= min_score)
    out = np.zeros_like(label_frame_int32)
    for new_id, old_id in enumerate(kept_ids, start=1):
        out[label_frame_int32 == old_id] = new_id
    return out, scores, len(kept_ids)


# ────────────────────────────────────────────────────────────────────
# Per-cell Cy5 features (for tracking + analytics)
# ────────────────────────────────────────────────────────────────────


def per_cell_cy5_features(cell_mask_bool, cy5_uint8):
    """Per-cell Cy5 statistics for use in tracking cost / analytics.

    Returns dict with mean, median, p75, p95, total intensity.
    Stable across frames for a given cell — useful as a tracking
    fingerprint (Hungarian cost extension).
    """
    if not cell_mask_bool.any():
        return {"mean": 0.0, "median": 0.0,
                "p75": 0.0, "p95": 0.0, "total": 0.0}
    px = cy5_uint8[cell_mask_bool].astype(np.float32)
    return {
        "mean":   float(px.mean()),
        "median": float(np.median(px)),
        "p75":    float(np.percentile(px, 75)),
        "p95":    float(np.percentile(px, 95)),
        "total":  float(px.sum()),
    }


# ────────────────────────────────────────────────────────────────────
# Extra Cy5 cellularity metrics — used by the multi-metric filter
# ────────────────────────────────────────────────────────────────────

def cy5_inside_outside_ratio(cell_mask_bool, cy5_uint8, ring_px=5):
    """Ratio of median Cy5 inside the mask to median in a tight
    exterior ring. Real cells (with actin cortex) score >1.5;
    debris/uniform-background detections score ~1.0.

    Returns float (clipped to [0, 10]).
    """
    if not cell_mask_bool.any():
        return 0.0
    inside = cy5_uint8[cell_mask_bool].astype(np.float32)
    dilated = binary_dilation(cell_mask_bool, iterations=ring_px)
    ring = dilated & ~cell_mask_bool
    if ring.sum() < 5:
        return 0.0
    outside = cy5_uint8[ring].astype(np.float32)
    out_med = max(float(np.median(outside)), 1.0)
    return float(min(10.0, np.median(inside) / out_med))


def cy5_inside_cv(cell_mask_bool, cy5_uint8):
    """Coefficient of variation of inside-mask Cy5 (std / mean).

    F-actin filaments produce heterogeneous Cy5 signal → high CV.
    Smooth/uniform blobs (debris) → low CV near 0.

    Returns float (clipped to [0, 5]).
    """
    if not cell_mask_bool.any():
        return 0.0
    px = cy5_uint8[cell_mask_bool].astype(np.float32)
    m = float(px.mean())
    if m < 1.0:
        return 0.0
    return float(min(5.0, px.std() / m))


def cy5_fraction_positive(cell_mask_bool, cy5_uint8, k_mad=2.0,
                            ring_px=10):
    """Fraction of inside-mask pixels brighter than (bg_med + k·bg_mad).

    Background reference = median of a 10-px exterior ring (robust
    to vignette gradient). Real cells: 30-80% of pixels positive.
    Debris: <10%.

    Returns float in [0, 1].
    """
    if not cell_mask_bool.any():
        return 0.0
    dilated = binary_dilation(cell_mask_bool, iterations=ring_px)
    ring = dilated & ~cell_mask_bool
    if ring.sum() < 5:
        return 0.0
    ring_px_arr = cy5_uint8[ring].astype(np.float32)
    bg_med = float(np.median(ring_px_arr))
    bg_mad = max(1.0, float(np.median(np.abs(ring_px_arr - bg_med))))
    threshold = bg_med + k_mad * bg_mad
    inside = cy5_uint8[cell_mask_bool].astype(np.float32)
    return float((inside > threshold).mean())


def per_cell_cy5_full_metrics(cell_mask_bool, cy5_uint8):
    """One-shot computation of every Cy5 metric we use for filtering.

    Returns dict with keys:
        mean, median, p75, p95, total      (basic stats)
        io_ratio                            (inside/outside ratio)
        inside_cv                           (texture proxy)
        fraction_positive                   (spatial coverage)
    """
    out = per_cell_cy5_features(cell_mask_bool, cy5_uint8)
    out["io_ratio"] = cy5_inside_outside_ratio(cell_mask_bool, cy5_uint8)
    out["inside_cv"] = cy5_inside_cv(cell_mask_bool, cy5_uint8)
    out["fraction_positive"] = cy5_fraction_positive(cell_mask_bool,
                                                      cy5_uint8)
    return out
