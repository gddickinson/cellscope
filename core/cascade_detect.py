"""Cascade detection: GT-retrained model → original model → temporal fill.

Strategy:
  1. Run the GT-retrained cellpose model (best boundary quality).
  2. Identify frames where detection failed (empty or tiny mask).
  3. Re-run failed frames with the original cellpose_dic model.
  4. For any remaining gaps, interpolate masks from nearest neighbors
     using optical-flow-guided warping or morphological interpolation.
  5. Tag each frame with its provenance (gt_model / original / interpolated).
"""
import os
import logging
import numpy as np
import cv2
from scipy import ndimage

from config import (
    CELLPOSE_FLOW_THRESHOLD, CELLPOSE_CELLPROB_THRESHOLD,
)

logger = logging.getLogger(__name__)

# --- Default model paths ---

from config import PROJECT_DIR

GT_MODEL_ALL120 = os.path.join(
    PROJECT_DIR, "data", "models", "cellpose_manual_gt_all120",
)
GT_MODEL_25 = None  # not available in consolidated project


def _pick_gt_model():
    """Return the best available GT-retrained model path, or None."""
    for p in [GT_MODEL_ALL120, GT_MODEL_25]:
        if os.path.exists(p):
            return p
    return None


def _is_valid_mask(mask, min_area_px=200):
    """Check whether a mask represents a real detection."""
    return mask.any() and mask.sum() >= min_area_px


def _find_failed_frames(masks, min_area_px=200):
    """Return sorted list of frame indices where detection failed."""
    return [i for i in range(len(masks))
            if not _is_valid_mask(masks[i], min_area_px)]


# --- Temporal interpolation ---

def _interpolate_mask_morphological(mask_before, mask_after, alpha):
    """Blend two masks via distance-transform interpolation.

    alpha=0 → mask_before, alpha=1 → mask_after.
    Uses signed distance fields for smooth morphing.
    """
    def signed_dist(m):
        if not m.any():
            return np.full(m.shape, -100.0)
        di = ndimage.distance_transform_edt(m)
        do = ndimage.distance_transform_edt(~m)
        return di - do

    sd_a = signed_dist(mask_before)
    sd_b = signed_dist(mask_after)
    blended = (1.0 - alpha) * sd_a + alpha * sd_b
    return blended > 0


def _interpolate_mask_flow(frame_prev, frame_curr, mask_prev):
    """Warp a mask forward using optical flow from prev to curr frame."""
    flow = cv2.calcOpticalFlowFarneback(
        frame_prev, frame_curr, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    h, w = frame_curr.shape[:2]
    # Build warp map: where each pixel in curr came from in prev
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = gx - flow[..., 0]
    map_y = gy - flow[..., 1]
    warped = cv2.remap(
        mask_prev.astype(np.uint8) * 255,
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 127


def _fill_gaps_temporal(frames, masks, failed_indices, provenance):
    """Fill failed frames using nearest valid neighbors.

    Uses optical-flow warping from the nearest valid neighbor when
    available, falling back to morphological interpolation between
    the nearest valid before and after.
    """
    n = len(frames)
    valid = set(range(n)) - set(failed_indices)
    if not valid:
        logger.warning("No valid frames to interpolate from")
        return

    for i in failed_indices:
        # Find nearest valid before and after
        before = max((v for v in valid if v < i), default=None)
        after = min((v for v in valid if v > i), default=None)

        if before is not None and after is not None:
            # Have both neighbors — morphological interpolation
            alpha = (i - before) / (after - before)
            masks[i] = _interpolate_mask_morphological(
                masks[before], masks[after], alpha,
            )
        elif before is not None:
            # Only have a previous frame — flow-warp forward
            masks[i] = _interpolate_mask_flow(
                frames[before], frames[i], masks[before],
            )
        elif after is not None:
            # Only have a later frame — flow-warp backward
            masks[i] = _interpolate_mask_flow(
                frames[after], frames[i], masks[after],
            )
        else:
            continue

        # Clean up interpolated mask
        masks[i] = ndimage.binary_fill_holes(masks[i])
        from core.contour import keep_largest_component
        masks[i] = keep_largest_component(masks[i])
        provenance[i] = "interpolated"


# --- Main cascade ---

def detect_cascade(frames, gpu=True, progress_fn=None,
                   gt_model_path=None, min_area_px=200):
    """Cascade detection with GT model → original model → temporal fill.

    Args:
        frames: (N, H, W) uint8 array.
        gpu: use GPU for cellpose.
        progress_fn: callable(frame_idx, message) for progress updates.
        gt_model_path: path to GT-retrained model. If None, auto-detects.
        min_area_px: minimum mask area to count as valid detection.

    Returns:
        masks: (N, H, W) bool — detection masks.
        provenance: list[str] — per-frame source tag:
            "gt_model", "original", "interpolated".
        stats: dict with detection counts per source.
    """
    from core.detection import detect_cellpose, get_cellpose_model

    n = len(frames)
    provenance = ["" for _ in range(n)]

    # --- Step 1: GT-retrained model ---
    if gt_model_path is None:
        gt_model_path = _pick_gt_model()

    if gt_model_path is None:
        logger.warning("No GT-retrained model found, using original only")
        masks = np.zeros(frames.shape, dtype=bool)
        failed_gt = list(range(n))
    else:
        if progress_fn:
            progress_fn(0, "Cascade: GT model detection...")
        logger.info(f"Cascade step 1: GT model ({os.path.basename(gt_model_path)})")
        masks = detect_cellpose(
            frames, gpu=gpu,
            progress_fn=lambda i: progress_fn(i, None) if progress_fn else None,
            model_path=gt_model_path,
        )
        for i in range(n):
            if _is_valid_mask(masks[i], min_area_px):
                provenance[i] = "gt_model"
        failed_gt = _find_failed_frames(masks, min_area_px)
        logger.info(f"  GT model: {n - len(failed_gt)}/{n} frames detected")

    # --- Step 2: Original model on failed frames ---
    if failed_gt:
        if progress_fn:
            progress_fn(0, f"Cascade: original model on {len(failed_gt)} frames...")
        logger.info(f"Cascade step 2: original model on {len(failed_gt)} frames")

        # Run original model on just the failed frames
        failed_frames = frames[failed_gt]
        orig_masks = detect_cellpose(
            failed_frames, gpu=gpu,
            progress_fn=lambda i: progress_fn(failed_gt[i], None) if progress_fn else None,
            model_path=None,  # default model
        )
        filled_count = 0
        for j, fi in enumerate(failed_gt):
            if _is_valid_mask(orig_masks[j], min_area_px):
                masks[fi] = orig_masks[j]
                provenance[fi] = "original"
                filled_count += 1

        failed_after_orig = _find_failed_frames(masks, min_area_px)
        logger.info(f"  Original model filled: {filled_count}/{len(failed_gt)} frames")
    else:
        failed_after_orig = []

    # --- Step 3: Temporal interpolation for remaining gaps ---
    if failed_after_orig:
        if progress_fn:
            progress_fn(0, f"Cascade: interpolating {len(failed_after_orig)} frames...")
        logger.info(f"Cascade step 3: interpolating {len(failed_after_orig)} frames")
        _fill_gaps_temporal(frames, masks, failed_after_orig, provenance)

    # --- Stats ---
    stats = {
        "gt_model": sum(1 for p in provenance if p == "gt_model"),
        "original": sum(1 for p in provenance if p == "original"),
        "interpolated": sum(1 for p in provenance if p == "interpolated"),
        "failed": sum(1 for p in provenance if p == ""),
        "total": n,
    }
    logger.info(
        f"Cascade result: {stats['gt_model']} GT + {stats['original']} orig "
        f"+ {stats['interpolated']} interp + {stats['failed']} failed / {n}"
    )
    return masks, provenance, stats
