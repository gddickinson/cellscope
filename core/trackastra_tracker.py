"""Trackastra-backed multi-cell tracker (Phase 17b).

Wraps trackastra.model.Trackastra so it produces the same output
shape as `core.multi_cell.track_all_cells`:

    [{"stack": (N, H, W) bool, "centroid_history": [(cy, cx), ...],
      "first_frame": int}, ...]

Entry point: `track_all_cells_trackastra(frames, labels, ...)`.

Unlike btrack / Hungarian, Trackastra needs both the raw DIC images
AND the per-frame label stack — it uses a learned transformer
embedding of cell appearance for association.
"""
import numpy as np
from scipy import ndimage


def track_all_cells_trackastra(frames, labels, min_area_px=200,
                                mode="greedy", device=None):
    """Trackastra-powered replacement for `track_all_cells`.

    Args:
        frames: (N, H, W) uint8 grayscale DIC stack.
        labels: (N, H, W) int32 label stack (from detect_cellpose_labels).
        min_area_px: drop tracks whose max per-frame area < this.
        mode: "greedy" (with division), "greedy_nodiv" (no division),
              "ilp" (slower, more accurate).
        device: None auto-picks mps/cuda/cpu.

    Returns: list of track dicts.
    """
    import torch
    from trackastra.model import Trackastra

    if device is None:
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")

    model = Trackastra.from_pretrained("general_2d", device=device)
    graph, tracked_masks = model.track(frames, labels, mode=mode)

    N, H, W = tracked_masks.shape
    unique = sorted(set(int(v) for v in np.unique(tracked_masks)) - {0})

    tracks = []
    for tid in unique:
        stack = (tracked_masks == tid)
        present = stack.any(axis=(1, 2))
        if not present.any():
            continue
        # min_area filter
        max_area = int(stack.reshape(N, -1).sum(axis=1).max())
        if max_area < min_area_px:
            continue
        first_frame = int(np.argmax(present))
        centroid_history = []
        for f in np.where(present)[0]:
            cy, cx = ndimage.center_of_mass(stack[f])
            centroid_history.append((float(cy), float(cx)))
        tracks.append({
            "stack": stack,
            "centroid_history": centroid_history,
            "first_frame": first_frame,
        })
    return tracks
