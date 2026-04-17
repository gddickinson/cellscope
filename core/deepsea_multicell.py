"""Per-cell DeepSea refinement that preserves multi-cell identity.

Unlike `union_with_deepsea` (which collapses to the single largest
connected component), this module refines each labeled cell
independently within its own expanded bounding box, preventing
cells from merging.
"""
import numpy as np
from scipy import ndimage as ndi


def _refine_single_cell(cell_mask, deepsea_mask, expand_px=20):
    """Refine one cell's mask using the DeepSea prediction, clipped
    to the cell's expanded bbox to prevent merging with neighbors."""
    if not cell_mask.any():
        return cell_mask
    H, W = cell_mask.shape
    ys, xs = np.where(cell_mask)
    r0 = max(0, ys.min() - expand_px)
    r1 = min(H, ys.max() + expand_px + 1)
    c0 = max(0, xs.min() - expand_px)
    c1 = min(W, xs.max() + expand_px + 1)

    crop_cell = cell_mask[r0:r1, c0:c1]
    crop_ds = deepsea_mask[r0:r1, c0:c1]
    union = crop_cell | crop_ds
    union = ndi.binary_fill_holes(union)

    lbl, n = ndi.label(union)
    if n > 1:
        sizes = ndi.sum(union, lbl, range(1, n + 1))
        union = (lbl == int(np.argmax(sizes)) + 1)

    out = np.zeros_like(cell_mask)
    out[r0:r1, c0:c1] = union
    return out


def refine_frame_labels_with_deepsea(image, label_frame, expand_px=20):
    """Refine all cells in one frame using DeepSea, preserving
    per-cell identity.

    Args:
        image: (H, W) uint8 grayscale frame
        label_frame: (H, W) int32 with 0=bg, 1,2,...=cell IDs
        expand_px: bbox padding for per-cell refinement

    Returns:
        (H, W) int32 refined label frame
    """
    from core.medsam_deepsea_union import _deepsea_predict

    if label_frame.max() == 0:
        return label_frame.copy()

    ds_mask = _deepsea_predict(image)
    out = np.zeros_like(label_frame)
    for lab in range(1, int(label_frame.max()) + 1):
        cell = label_frame == lab
        if not cell.any():
            continue
        refined = _refine_single_cell(cell, ds_mask, expand_px)
        # Avoid overwriting already-assigned pixels (first cell wins)
        out[refined & (out == 0)] = lab
    return out


def refine_labels_with_deepsea(frames, label_stack, expand_px=20,
                                progress_fn=None):
    """Refine all frames' cell labels using per-cell DeepSea union.

    Args:
        frames: (N, H, W) uint8
        label_stack: (N, H, W) int32
        expand_px: bbox padding
        progress_fn: optional callback(msg, pct)

    Returns:
        (N, H, W) int32 refined label stack
    """
    n = len(frames)
    out = np.zeros_like(label_stack)
    for i in range(n):
        if progress_fn and (i % 10 == 0 or i == n - 1):
            progress_fn(f"DeepSea per-cell {i+1}/{n}",
                        int(100 * i / max(n - 1, 1)))
        out[i] = refine_frame_labels_with_deepsea(
            frames[i], label_stack[i], expand_px)
    return out
