"""Cell detection: Cellpose, optical flow, fused detection.

The recommended detector is `detect_cellpose_flow`, which combines a
fine-tuned Cellpose model with quality-aware optical flow refinement.
"""
import os
import logging
import numpy as np
import cv2
from scipy import ndimage

import config as _cfg
from config import (
    MODEL_DIR,
    FLOW_PYR_SCALE, FLOW_LEVELS, FLOW_WINSIZE, FLOW_ITERATIONS,
    FLOW_POLY_N, FLOW_POLY_SIGMA,
)


def _get_runtime(name):
    """Read a config value at CALL time (sees `_ConfigOverride` patches).

    `from config import X` captures X at import time and won't see
    later monkey-patches; this helper re-reads from the module each
    call so workers' DetectionParams overrides take effect.
    """
    return getattr(_cfg, name)

logging.getLogger("cellpose").setLevel(logging.ERROR)

_model_cache = {}


def get_cellpose_model(gpu=True, model_path=None):
    """Load (and cache) a Cellpose model.

    Args:
        gpu: use GPU if available.
        model_path: path to a specific model file. If None, loads the
            default bundled fine-tuned DIC model.
    """
    if model_path is None:
        path = os.path.join(MODEL_DIR, _get_runtime("CELLPOSE_MODEL_NAME"))
    else:
        path = str(model_path)
    if path in _model_cache:
        return _model_cache[path]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cellpose model not found: {path}")
    from cellpose.models import CellposeModel
    model = CellposeModel(pretrained_model=path, gpu=gpu)
    _model_cache[path] = model
    return model


def detect_cellpose(frames, gpu=True, progress_fn=None, model_path=None,
                    flow_threshold=None, cellprob_threshold=None,
                    augment=False, diameter=None):
    """Run Cellpose on each frame, return the largest cell mask per frame.

    Args:
        model_path: path to a specific model file. If None, uses default.
        flow_threshold: override for CELLPOSE_FLOW_THRESHOLD.
        cellprob_threshold: override for CELLPOSE_CELLPROB_THRESHOLD.
    """
    ft = _get_runtime("CELLPOSE_FLOW_THRESHOLD") if flow_threshold is None else flow_threshold
    ct = (_get_runtime("CELLPOSE_CELLPROB_THRESHOLD")
          if cellprob_threshold is None else cellprob_threshold)
    model = get_cellpose_model(gpu=gpu, model_path=model_path)
    n = len(frames)
    masks = np.zeros(frames.shape, dtype=bool)
    for i in range(n):
        if progress_fn:
            progress_fn(i)
        m, _, _ = model.eval(
            frames[i], diameter=diameter,
            flow_threshold=ft, cellprob_threshold=ct,
            augment=augment,
        )
        if m.max() > 0:
            labels, counts = np.unique(m[m > 0], return_counts=True)
            masks[i] = (m == labels[counts.argmax()])
    return masks


def detect_cellpose_labels(frames, gpu=True, progress_fn=None,
                            model_path=None, flow_threshold=None,
                            cellprob_threshold=None, min_area_px=100):
    """Run Cellpose and return the FULL label stack (all cells kept).

    Unlike `detect_cellpose` which reduces to the single largest
    connected component per frame, this preserves every detected cell
    as a unique integer ID per frame (1-indexed, 0 = background).

    For Phase 16 (full-population tracking).

    Returns:
        (N, H, W) int32 stack. Label values are only unique WITHIN a
        frame — cell #3 in frame 0 is not the same cell as cell #3 in
        frame 1. Identity across frames is assigned by the tracker
        (core.multi_cell.track_all_cells).
    """
    ft = _get_runtime("CELLPOSE_FLOW_THRESHOLD") if flow_threshold is None else flow_threshold
    ct = (_get_runtime("CELLPOSE_CELLPROB_THRESHOLD")
          if cellprob_threshold is None else cellprob_threshold)
    model = get_cellpose_model(gpu=gpu, model_path=model_path)
    n = len(frames)
    out = np.zeros(frames.shape, dtype=np.int32)
    for i in range(n):
        if progress_fn:
            progress_fn(i)
        m, _, _ = model.eval(
            frames[i], diameter=None,
            flow_threshold=ft, cellprob_threshold=ct,
        )
        # Drop tiny components, compact the label IDs
        if m.max() > 0:
            labels, counts = np.unique(m[m > 0], return_counts=True)
            keep = labels[counts >= min_area_px]
            frame_out = np.zeros_like(m, dtype=np.int32)
            for new_id, old_id in enumerate(keep, start=1):
                frame_out[m == old_id] = new_id
            out[i] = frame_out
    return out


def detect_cellpose_tiled(frames, gpu=True, progress_fn=None,
                           model_path=None, flow_threshold=None,
                           cellprob_threshold=None,
                           n_tiles=(3, 3), overlap_px=64):
    """Tiled cellpose detection for large frames.

    Default n_tiles=(3, 3) with overlap=64 chosen by Phase 15a sweep
    on 4 Jesse 1024² recordings:
      - 3x3_o64 detects 10/10 frames on all 4 recordings (including
        pos65_ko where full-frame cellpose got 3/10 and 2x2_o64 got
        only 3/10).
      - ~4× more cells/frame on pos59_ko than 2x2_o64.
      - Runtime ~15s per 10 frames (5× slower than full, but worth
        it for the OOD recall boost).

    Tiles are detected independently; the union of per-tile masks is
    reassembled and the largest connected component is kept per frame
    to match `detect_cellpose`'s contract. (For all-cells output use
    `detect_cellpose_labels`.)

    Args:
        n_tiles: (rows, cols) — how many tiles to split each frame into.
        overlap_px: overlap between adjacent tiles to avoid cells
            straddling tile boundaries being missed.
    """
    ft = _get_runtime("CELLPOSE_FLOW_THRESHOLD") if flow_threshold is None else flow_threshold
    ct = (_get_runtime("CELLPOSE_CELLPROB_THRESHOLD")
          if cellprob_threshold is None else cellprob_threshold)
    model = get_cellpose_model(gpu=gpu, model_path=model_path)
    n = len(frames)
    H, W = frames.shape[1:]
    tr, tc = n_tiles
    tile_h = H // tr
    tile_w = W // tc
    # Generate tile bboxes with overlap
    bboxes = []
    for r in range(tr):
        r0 = max(0, r * tile_h - (overlap_px if r > 0 else 0))
        r1 = min(H, (r + 1) * tile_h + (overlap_px if r < tr - 1 else 0))
        for c in range(tc):
            c0 = max(0, c * tile_w - (overlap_px if c > 0 else 0))
            c1 = min(W, (c + 1) * tile_w + (overlap_px if c < tc - 1 else 0))
            bboxes.append((r0, r1, c0, c1))

    masks = np.zeros(frames.shape, dtype=bool)
    for i in range(n):
        if progress_fn:
            progress_fn(i)
        frame_mask = np.zeros((H, W), dtype=bool)
        for (r0, r1, c0, c1) in bboxes:
            tile = frames[i, r0:r1, c0:c1]
            m, _, _ = model.eval(tile, diameter=None,
                                 flow_threshold=ft, cellprob_threshold=ct)
            if m.max() > 0:
                # Take union of all cells in the tile
                frame_mask[r0:r1, c0:c1] |= (m > 0)
        # Keep largest connected component (match detect_cellpose contract)
        if frame_mask.any():
            lbl, _ = ndimage.label(frame_mask)
            sizes = ndimage.sum(frame_mask, lbl, range(1, lbl.max() + 1))
            if len(sizes) > 0:
                biggest = int(np.argmax(sizes)) + 1
                masks[i] = (lbl == biggest)
    return masks


def detect_with_threshold_retry(frames, gpu=True, progress_fn=None,
                                 model_path=None,
                                 retry_cellprob_thresholds=(-2.0, -4.0),
                                 min_area_px=200,
                                 interpolate_gaps=True):
    """Cellpose detection with threshold-retry fallback on missed frames.

    Strategy:
      1. Run cellpose at default (high-quality boundaries).
      2. For each frame with an empty/tiny mask, retry at progressively
         lower cellprob_threshold values (more permissive).
      3. Optionally fill remaining gaps via temporal interpolation.

    Rationale: the defaults give best boundary quality on successful
    detections; lowering cellprob_threshold recovers missed frames but
    degrades cKO boundaries. This asymmetric approach gets the best of
    both — detected frames keep high quality, missed frames are rescued.

    Args:
        retry_cellprob_thresholds: tuple of ct values to try in order
            on failed frames. Default (-2.0, -4.0).
        min_area_px: minimum mask area to count as a valid detection.
        interpolate_gaps: if True, temporally interpolate any frames
            still empty after all retries.

    Returns:
        masks: (N, H, W) bool
        provenance: list[str] — per-frame source tag. Values:
            "primary" (default thresholds),
            "retry_ct{value}" (after retry at that ct),
            "interpolated" (filled by neighbor interpolation),
            "failed" (still empty after all steps).
        stats: dict with counts per source.
    """
    import logging
    logger = logging.getLogger(__name__)
    n = len(frames)
    provenance = ["" for _ in range(n)]

    # --- Step 1: default thresholds ---
    if progress_fn:
        progress_fn(0, "Detection (primary pass)...")
    masks = detect_cellpose(
        frames, gpu=gpu, model_path=model_path,
        progress_fn=lambda i: progress_fn(i, None) if progress_fn else None,
    )
    for i in range(n):
        if masks[i].sum() >= min_area_px:
            provenance[i] = "primary"
    failed = [i for i in range(n) if not provenance[i]]
    logger.info(f"Primary pass: {n - len(failed)}/{n} detected")

    # --- Step 2: retry each ct in order on remaining failed frames ---
    for ct in retry_cellprob_thresholds:
        if not failed:
            break
        if progress_fn:
            progress_fn(0, f"Retry at cellprob_threshold={ct} "
                            f"on {len(failed)} frames...")
        retry_frames = frames[failed]
        retry_masks = detect_cellpose(
            retry_frames, gpu=gpu, model_path=model_path,
            cellprob_threshold=ct,
            progress_fn=(lambda i: progress_fn(failed[i], None)
                         if progress_fn else None),
        )
        recovered = 0
        still_failed = []
        for j, fi in enumerate(failed):
            if retry_masks[j].sum() >= min_area_px:
                masks[fi] = retry_masks[j]
                provenance[fi] = f"retry_ct{ct:+.1f}"
                recovered += 1
            else:
                still_failed.append(fi)
        failed = still_failed
        logger.info(f"  Retry ct={ct}: recovered {recovered}, "
                    f"{len(failed)} still failed")

    # --- Step 3: temporal interpolation for any remaining gaps ---
    if interpolate_gaps and failed:
        if progress_fn:
            progress_fn(0, f"Interpolating {len(failed)} frames...")
        from core.cascade_detect import _fill_gaps_temporal
        _fill_gaps_temporal(frames, masks, failed, provenance)
        logger.info(f"  Interpolated {sum(1 for p in provenance if p == 'interpolated')} frames")

    # Remaining empty slots get "failed" tag
    for i in range(n):
        if not provenance[i]:
            provenance[i] = "failed"

    stats = {
        "primary": sum(1 for p in provenance if p == "primary"),
        "interpolated": sum(1 for p in provenance if p == "interpolated"),
        "failed": sum(1 for p in provenance if p == "failed"),
        "total": n,
    }
    for ct in retry_cellprob_thresholds:
        key = f"retry_ct{ct:+.1f}"
        stats[key] = sum(1 for p in provenance if p == key)

    return masks, provenance, stats


# --- Optical flow ---

def compute_flow(frame_a, frame_b):
    """Farneback optical flow magnitude."""
    flow = cv2.calcOpticalFlowFarneback(
        frame_a, frame_b, None,
        pyr_scale=FLOW_PYR_SCALE, levels=FLOW_LEVELS,
        winsize=FLOW_WINSIZE, iterations=FLOW_ITERATIONS,
        poly_n=FLOW_POLY_N, poly_sigma=FLOW_POLY_SIGMA, flags=0,
    )
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


def compute_flow_stack(frames):
    """Per-frame flow magnitudes (assigned to the later frame in each pair)."""
    n = len(frames)
    h, w = frames.shape[1], frames.shape[2]
    out = np.zeros((n, h, w), dtype=np.float32)
    for i in range(n - 1):
        out[i + 1] = compute_flow(frames[i], frames[i + 1])
    if n > 1:
        out[0] = compute_flow(frames[1], frames[0])
    return out


def compute_flow_stack_clean(frames, pixel_size_um=None,
                              temporal_method="median",
                              spatial_highpass_sigma=40.0):
    """Compute Farneback flow on background-cleaned frames.

    Applies preprocessing (temporal median subtraction + spatial high-pass)
    BEFORE Farneback. This is the optical_flow project's recommended path
    for the flow component — it suppresses static content and illumination
    drift so flow only fires on real motion.

    The original frames are NOT modified — this only affects what flow sees.
    Cellpose detection and the texture metric still get raw frames.
    """
    from core.preprocess import preprocess_sequence
    cleaned = preprocess_sequence(
        frames,
        temporal_method=temporal_method,
        spatial_highpass_sigma=spatial_highpass_sigma,
        debris_diameter_um=0,  # skip debris removal — too aggressive
        pixel_size_um=pixel_size_um,
    )
    return compute_flow_stack(cleaned)


def compute_frame_diff_stack(frames, pixel_size_um=None,
                              temporal_method="median",
                              spatial_highpass_sigma=40.0,
                              clean_first=True):
    """Per-frame motion magnitude via frame differencing |I(t)-I(t-1)|.

    Frame differencing is the optical_flow project's recommended motion
    method for cells moving > 10 px/frame (Farneback's window limit).
    For cKO at 14-36 px/frame, frame diff localizes the moving cell
    cleanly while Farneback returns a smeared blob.

    Optionally cleans frames first (highly recommended — without it,
    background drift produces strong frame-diff signal).

    Returns:
        magnitudes: (N, H, W) float32
    """
    if clean_first:
        from core.preprocess import preprocess_sequence
        src = preprocess_sequence(
            frames,
            temporal_method=temporal_method,
            spatial_highpass_sigma=spatial_highpass_sigma,
            debris_diameter_um=0,
            pixel_size_um=pixel_size_um,
        )
    else:
        src = frames

    n = len(src)
    h, w = src.shape[1], src.shape[2]
    out = np.zeros((n, h, w), dtype=np.float32)
    for i in range(1, n):
        out[i] = np.abs(
            src[i].astype(np.float32) - src[i - 1].astype(np.float32)
        )
    if n > 1:
        out[0] = out[1]  # mirror first frame
    return out


# --- Fused detection ---

def _cellpose_to_prob(mask, sigma=8.0):
    if not mask.any():
        return np.zeros_like(mask, dtype=float)
    di = ndimage.distance_transform_edt(mask)
    do = ndimage.distance_transform_edt(~mask)
    return 1.0 / (1.0 + np.exp(-(di - do) / sigma))


def _guided_flow_prob(flow_mag, cellpose_mask, spatial_sigma=30.0):
    if not cellpose_mask.any():
        return np.zeros_like(flow_mag, dtype=float)
    di = ndimage.distance_transform_edt(cellpose_mask)
    do = ndimage.distance_transform_edt(~cellpose_mask)
    spatial_w = 1.0 / (1.0 + np.exp(-(di - do) / spatial_sigma))
    smoothed = cv2.GaussianBlur(flow_mag.astype(np.float32), (0, 0), 5.0)
    p99 = np.percentile(smoothed, 99)
    norm = np.clip(smoothed / max(p99, 1e-6), 0, 1)
    return norm * spatial_w


def _assess_flow_quality(flow_mag, cp_mask):
    if not cp_mask.any() or flow_mag.max() == 0:
        return 0.0
    inside = float(np.mean(flow_mag[cp_mask]))
    outside = float(np.mean(flow_mag[~cp_mask])) if (~cp_mask).any() else 0.001
    flow_ratio = inside / max(outside, 0.001)
    threshold = np.percentile(flow_mag, 80)
    hotspot = flow_mag >= threshold
    overlap = float(np.logical_and(hotspot, cp_mask).sum())
    hotspot_area = max(float(hotspot.sum()), 1.0)
    overlap_frac = overlap / hotspot_area
    flow_area = float(hotspot.sum())
    cp_area = max(float(cp_mask.sum()), 1.0)
    area_ratio = flow_area / cp_area
    rs = np.clip((flow_ratio - 0.5) / 1.5, 0, 1)
    os_ = np.clip(overlap_frac / 0.5, 0, 1)
    as_ = 1.0 - np.clip(abs(np.log(area_ratio)) / 1.5, 0, 1)
    return float(rs * 0.3 + os_ * 0.5 + as_ * 0.2)


def _fuse_one(image, cp_mask, flow_mag,
              cp_weight=0.6, flow_weight=0.4, quality_threshold=0.3):
    quality = _assess_flow_quality(flow_mag, cp_mask)
    cp_prob = _cellpose_to_prob(cp_mask)
    if quality < quality_threshold:
        eff_w = 0.0
    else:
        scale = (quality - quality_threshold) / (1.0 - quality_threshold)
        eff_w = flow_weight * scale
    if eff_w > 0:
        flow_prob = _guided_flow_prob(flow_mag, cp_mask)
    else:
        flow_prob = np.zeros_like(cp_prob)
    fused = (cp_prob * cp_weight + flow_prob * eff_w) / (cp_weight + eff_w)
    mask = fused >= 0.5
    mask = ndimage.binary_fill_holes(mask)
    from core.contour import keep_largest_component
    return keep_largest_component(mask), quality


def _fuse_one_smart(image, cp_mask, flow_mag,
                    cp_weight=0.5, max_flow_weight=1.0):
    """Improved fusion using joint quality + per-pixel trust map.

    Inspired by Robitaille et al. 2022:
      - Quality is computed from joint flow + image gradient agreement
        at the cellpose boundary (more discriminating than ratio metric)
      - Per-pixel flow trust map gates the flow contribution locally
      - Wider weight range so high-quality frames matter more
    """
    from core.flow_quality import assess_flow_quality_joint, flow_trust_map
    from core.contour import keep_largest_component

    quality, _ = assess_flow_quality_joint(image, flow_mag, cp_mask)

    cp_prob = _cellpose_to_prob(cp_mask)

    # Effective frame-level weight: smooth scaling 0..max_flow_weight
    # Quality typically lies in [0, 1.5]; saturate at 1.0
    eff_w = max_flow_weight * min(quality, 1.0)

    if eff_w <= 0.05:
        return keep_largest_component(
            ndimage.binary_fill_holes(cp_prob >= 0.5)
        ), quality

    # Per-pixel trust map: gates flow's local contribution
    trust = flow_trust_map(image, flow_mag, cp_mask)
    flow_prob = _guided_flow_prob(flow_mag, cp_mask) * trust

    # Mix per-pixel: where trust is high, flow contributes; where trust
    # is zero, only cellpose is used.
    local_flow_w = eff_w * trust  # (H, W)
    total_w = cp_weight + local_flow_w
    fused = (cp_prob * cp_weight + flow_prob * local_flow_w) / np.maximum(total_w, 1e-6)
    mask = fused >= 0.5
    mask = ndimage.binary_fill_holes(mask)
    return keep_largest_component(mask), quality


def detect_cellpose_flow(frames, gpu=True, progress_fn=None,
                         use_smart_fusion=False, max_flow_weight=1.0,
                         flow_method="farneback", pixel_size_um=None,
                         augment=False):
    """Run cellpose+flow fused detection.

    Args:
        use_smart_fusion: if True (default), uses the joint quality + per-pixel
            trust map fusion (inspired by Robitaille 2022). If False, uses the
            older simpler fusion based on inside/outside flow ratio.
        max_flow_weight: cap on flow's per-pixel contribution (smart fusion only).
            1.0 gives flow significant influence on high-quality frames.

    Returns:
        masks: (N, H, W) bool
        flow_quality: (N,) float — diagnostic
        flow_magnitudes: (N, H, W) float32
    """
    n = len(frames)

    if progress_fn:
        progress_fn(0, "Running Cellpose...")
    cp = detect_cellpose(frames, gpu=gpu, augment=augment)

    if progress_fn:
        progress_fn(0, "Computing optical flow...")
    if flow_method == "framediff":
        flow_mags = compute_frame_diff_stack(frames, pixel_size_um=pixel_size_um)
    elif flow_method == "farneback_clean":
        flow_mags = compute_flow_stack_clean(frames, pixel_size_um=pixel_size_um)
    else:  # "farneback" (default)
        flow_mags = compute_flow_stack(frames)

    if progress_fn:
        progress_fn(0, "Fusing detections...")
    masks = np.zeros(frames.shape, dtype=bool)
    quality = np.zeros(n)
    for i in range(n):
        if progress_fn:
            progress_fn(i, None)
        if use_smart_fusion:
            masks[i], quality[i] = _fuse_one_smart(
                frames[i], cp[i], flow_mags[i],
                cp_weight=0.5, max_flow_weight=max_flow_weight,
            )
        else:
            masks[i], quality[i] = _fuse_one(
                frames[i], cp[i], flow_mags[i]
            )
    return masks, quality, flow_mags
