"""Crop-and-refine: restrict refinement to a bounding box around the cell.

Refinement operates pixel-by-pixel (RF, snap, Fourier, CRF); running it
on the whole frame wastes most of the work. Restricting to a padded
bounding box around the union of all detected masks is typically 4-16x
faster with identical results (the cell never reaches image edges in
our recordings, so the bbox captures all relevant signal).

Design:
  - Compute a single GLOBAL bbox (union of all frames' masks + padding)
    so the cropped stack is a valid (N, h', w') array for any refine fn.
  - Any frames with empty masks don't contribute to the bbox but are
    still included in the crop (temporal methods need full N frames).
  - After refinement, paste the cropped masks back into the full frame.

Why a global bbox instead of per-frame bboxes:
  - Temporal smoothing steps (polar radii, CRF) need consistent shape
  - Keeps the wrapper agnostic to the inner refine function
  - Cells drift < 100 px in our recordings → union bbox is still small
"""
import numpy as np


def compute_global_bbox(masks, padding_px=50, image_shape=None):
    """Bounding box covering all masks, with padding, clipped to image.

    Args:
        masks: (N, H, W) bool — detected cell masks (some may be empty).
        padding_px: pixels of padding around the union bbox.
        image_shape: (H, W) — required for clipping. If None, inferred
            from masks.

    Returns:
        (row_min, row_max, col_min, col_max) — half-open, i.e.
        cropped = frames[:, row_min:row_max, col_min:col_max].
        Returns None if all masks are empty.
    """
    if image_shape is None:
        h, w = masks.shape[1], masks.shape[2]
    else:
        h, w = image_shape

    # Union across all frames
    union = np.any(masks, axis=0)
    if not union.any():
        return None

    rows = np.where(np.any(union, axis=1))[0]
    cols = np.where(np.any(union, axis=0))[0]
    row_min = max(0, int(rows.min()) - padding_px)
    row_max = min(h, int(rows.max()) + 1 + padding_px)
    col_min = max(0, int(cols.min()) - padding_px)
    col_max = min(w, int(cols.max()) + 1 + padding_px)
    return (row_min, row_max, col_min, col_max)


def crop_stack(array, bbox):
    """Crop an (N, H, W, ...) array to a bbox."""
    r0, r1, c0, c1 = bbox
    return array[:, r0:r1, c0:c1]


def paste_back(cropped_masks, bbox, full_shape):
    """Place cropped masks back into full-size mask array.

    Args:
        cropped_masks: (N, h', w') bool
        bbox: (r0, r1, c0, c1)
        full_shape: (N, H, W)

    Returns:
        (N, H, W) bool — zeros outside the bbox, cropped_masks inside.
    """
    out = np.zeros(full_shape, dtype=bool)
    r0, r1, c0, c1 = bbox
    out[:, r0:r1, c0:c1] = cropped_masks
    return out


def crop_refine(frames, base_masks, refine_fn, padding_px=50):
    """Run a refine function on a cropped bounding box, then paste back.

    Args:
        frames: (N, H, W) uint8 — full-frame images.
        base_masks: (N, H, W) bool — detected masks.
        refine_fn: callable(frames_crop, masks_crop) → refined_masks_crop.
            Must preserve the (N, h', w') shape.
        padding_px: padding around the union bbox.

    Returns:
        refined_masks: (N, H, W) bool — refined masks in full-frame coords.
        bbox: (r0, r1, c0, c1) — the crop used, or None if no masks.

    If no masks exist (bbox=None), falls back to refining on the full
    frame (edge case; usually won't happen with threshold-retry).
    """
    full_shape = base_masks.shape
    bbox = compute_global_bbox(
        base_masks, padding_px=padding_px, image_shape=frames.shape[1:],
    )
    if bbox is None:
        # No masks to localize — refine on full frame
        return refine_fn(frames, base_masks), None

    frames_crop = crop_stack(frames, bbox)
    masks_crop = crop_stack(base_masks, bbox)

    refined_crop = refine_fn(frames_crop, masks_crop)

    return paste_back(refined_crop, bbox, full_shape), bbox


# --- Per-cell cropping (one bbox per frame) ---

def compute_per_frame_bbox(mask, padding_px=30, image_shape=None):
    """Bbox for a single frame's mask, padded and clipped.

    Returns (r0,r1,c0,c1) or None if the mask is empty.
    """
    if not mask.any():
        return None
    h, w = image_shape if image_shape is not None else mask.shape
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    return (
        max(0, int(rows.min()) - padding_px),
        min(h, int(rows.max()) + 1 + padding_px),
        max(0, int(cols.min()) - padding_px),
        min(w, int(cols.max()) + 1 + padding_px),
    )


def per_cell_refine(frames, base_masks, refine_fn, padding_px=30):
    """Refine each frame independently on its own padded cell bbox.

    Unlike crop_refine (which uses a single union bbox across all
    frames), this crops each frame to the bbox of its own cell. Useful
    when cells move a lot (union bbox would be wasteful) and for
    per-frame classical methods (Otsu, GMM, etc.) that don't depend on
    temporal context.

    Args:
        frames: (N, H, W) uint8
        base_masks: (N, H, W) bool
        refine_fn: callable(single_frame, single_mask) → single_refined_mask
            (acts on individual 2-D arrays, not stacks).
        padding_px: padding around each frame's bbox.

    Returns:
        refined_masks: (N, H, W) bool — full-frame coords
        per_frame_bboxes: list of (r0,r1,c0,c1) or None per frame
    """
    full_shape = base_masks.shape
    out = np.zeros(full_shape, dtype=bool)
    bboxes = []
    for i in range(full_shape[0]):
        if not base_masks[i].any():
            bboxes.append(None)
            continue
        bb = compute_per_frame_bbox(
            base_masks[i], padding_px=padding_px,
            image_shape=frames.shape[1:],
        )
        if bb is None:
            bboxes.append(None)
            continue
        r0, r1, c0, c1 = bb
        fcrop = frames[i, r0:r1, c0:c1]
        mcrop = base_masks[i, r0:r1, c0:c1]
        try:
            refined_crop = refine_fn(fcrop, mcrop)
        except Exception:
            refined_crop = mcrop
        out[i, r0:r1, c0:c1] = refined_crop
        bboxes.append(bb)
    return out, bboxes
