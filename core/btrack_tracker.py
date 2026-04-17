"""BTrack-backed multi-cell tracker.

Wraps btrack's Bayesian tracker (Ulicna et al. 2021) so it produces
the same output shape as `core.multi_cell.track_all_cells`:

    [{"stack": (N, H, W) bool, "centroid_history": [(cy, cx), ...],
      "first_frame": int}, ...]

This lets downstream code swap backends via a `tracking_backend=`
argument without any analytics changes.

Entry point: `track_all_cells_btrack(masks, ...)`.
"""
import numpy as np
from scipy import ndimage


def _build_label_stack(masks, min_area_px=200):
    """Produce (N, H, W) int label stack with per-frame connected
    component IDs starting at 1. Drops components below min_area_px.
    """
    out = np.zeros(masks.shape, dtype=np.int32)
    for i in range(len(masks)):
        m = masks[i]
        if m.dtype == bool:
            lbl, n = ndimage.label(m)
        else:
            lbl = m.astype(np.int32)
            n = int(lbl.max())
        next_id = 1
        frame = np.zeros_like(lbl, dtype=np.int32)
        for j in range(1, n + 1):
            cell = (lbl == j)
            if cell.sum() < min_area_px:
                continue
            frame[cell] = next_id
            next_id += 1
        out[i] = frame
    return out


def track_all_cells_btrack(masks, min_area_px=200, config_file=None):
    """BTrack-powered replacement for `track_all_cells`.

    Args:
        masks: (N, H, W) bool or int label stack.
        min_area_px: minimum area to count as a real cell.
        config_file: optional path to a BTrack config JSON.
            Falls back to the bundled `cell_config.json` from the
            btrack-examples cache.

    Returns: list of track dicts (see module docstring).
    """
    import btrack
    from btrack import datasets
    from btrack.utils import segmentation_to_objects

    label_stack = _build_label_stack(masks, min_area_px=min_area_px)
    objects = segmentation_to_objects(label_stack)
    if not objects:
        return []

    if config_file is None:
        config_file = datasets.cell_config()

    N, H, W = masks.shape
    with btrack.BayesianTracker() as tracker:
        tracker.configure(config_file)
        tracker.max_search_radius = 150
        tracker.append(objects)
        tracker.volume = ((0, W), (0, H))
        tracker.track()
        tracker.optimize()
        bt_tracks = tracker.tracks

    # Build (obj_id → (frame, cy, cx)) lookup for mask retrieval
    obj_lookup = {}
    for obj in objects:
        obj_lookup[int(obj.ID)] = (int(obj.t), float(obj.y), float(obj.x))

    tracks = []
    for bt in bt_tracks:
        frames = [int(t) for t in bt.t]
        if not frames:
            continue
        first_frame = frames[0]
        stack = np.zeros(masks.shape, dtype=bool)
        centroid_history = []
        for i, f in enumerate(frames):
            if f < 0 or f >= N:
                continue
            # Reconstruct mask by matching label at centroid
            cy, cx = float(bt.y[i]), float(bt.x[i])
            ry, rx = int(round(cy)), int(round(cx))
            if 0 <= ry < H and 0 <= rx < W:
                lbl_val = label_stack[f, ry, rx]
                if lbl_val > 0:
                    stack[f] = (label_stack[f] == lbl_val)
            centroid_history.append((cy, cx))
        tracks.append({
            "stack": stack,
            "centroid_history": centroid_history,
            "first_frame": first_frame,
        })
    return tracks
