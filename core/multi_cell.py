"""Multi-cell identity tracking across frames.

Given per-frame cellpose masks (a stack of binary masks with possibly
multiple cells per frame), produce consistent per-cell tracks using
centroid-nearest matching.

This makes the existing "keep largest component" pipeline robust to
scenes with multiple cells — rather than flipping between cells when
the "largest" changes, we lock onto each cell identity in frame 0
and follow it across time.

Two entry points:
  `anchor_first_frame_single_cell(masks, ...)`
      Pick ONE cell (the largest) from frame 0 and follow it. Returns
      a (N, H, W) bool stack containing only that cell's mask per
      frame. Replaces the behaviour of `keep_largest_component` when
      "largest" would otherwise flip.

  `track_all_cells(masks, ...)`
      Return per-cell tracks — a list of (N, H, W) stacks, one per
      cell present in frame 0, each following that cell through time.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist


def _label_cells_per_frame(masks, min_area_px=200):
    """Per-frame list of (centroid, area, binary_mask) tuples.

    Accepts either:
      - (N, H, W) bool stack with each frame already a single component
        (the current "largest mask" output) — we re-label to handle any
        stray components
      - (N, H, W) int/label stack from cellpose — we extract all cells
        via connected-components labelling
    """
    out = []
    for i in range(len(masks)):
        m = masks[i]
        if m.dtype == bool:
            lbl, n = ndimage.label(m)
        else:
            lbl = m
            n = int(lbl.max())
        cells = []
        for j in range(1, n + 1):
            cell = (lbl == j)
            area = int(cell.sum())
            if area < min_area_px:
                continue
            cy, cx = ndimage.center_of_mass(cell)
            cells.append(((cy, cx), area, cell))
        out.append(cells)
    return out


def anchor_first_frame_single_cell(masks, min_area_px=200,
                                    max_hop_px=80):
    """Lock onto the largest cell in frame 0 and follow it across frames.

    For each subsequent frame, pick the mask whose centroid is closest
    to the previous frame's centroid (max_hop_px limit). If no cell is
    within that limit, leave the frame's mask empty for that track.

    Args:
        masks: (N, H, W) bool or label stack.
        min_area_px: minimum area to count as a real cell.
        max_hop_px: maximum centroid jump allowed between adjacent
            frames (cells moving faster than this are likely different
            cells).

    Returns:
        (N, H, W) bool stack — the tracked cell's mask per frame.
    """
    per_frame = _label_cells_per_frame(masks, min_area_px=min_area_px)
    n = len(masks)
    out = np.zeros(masks.shape, dtype=bool)
    # Seed from the largest cell in frame 0
    if not per_frame[0]:
        return out  # no cells to anchor on
    per_frame[0].sort(key=lambda c: c[1], reverse=True)
    prev_cy, prev_cx = per_frame[0][0][0]
    out[0] = per_frame[0][0][2]

    for i in range(1, n):
        cells = per_frame[i]
        if not cells:
            continue
        dists = [
            np.hypot(cy - prev_cy, cx - prev_cx)
            for ((cy, cx), _, _) in cells
        ]
        best = int(np.argmin(dists))
        if dists[best] > max_hop_px:
            # Stop tracking — cell has disappeared or the tracker lost it
            continue
        (prev_cy, prev_cx), _, out[i] = cells[best]
    return out


def track_all_cells(masks, min_area_px=200, max_hop_px=150,
                    seed_frame=None, spawn_new_tracks=False,
                    min_track_length=1):
    """Return one (N, H, W) stack per cell seen in the seed frame.

    Args:
        masks: (N, H, W) bool or label stack.
        min_area_px: minimum area to count as a real cell.
        max_hop_px: maximum centroid jump allowed between adjacent
            frames. Cells moving faster than this are likely different
            cells; the track is marked dead.
        seed_frame: frame index to seed tracks from. If None, uses the
            first frame that has at least one detected cell.
        spawn_new_tracks: if True, create a new track each time a
            detected cell cannot be matched to an existing track
            (covers cells entering the field of view after seed_frame).
            Phase 16 uses True for full-population tracking.
        min_track_length: after tracking, drop tracks shorter than this
            (filters spurious detections that don't persist).

    Returns:
        list of dicts: [{"stack": (N, H, W) bool,
                          "centroid_history": [(cy, cx), ...],
                          "first_frame": int}, ...]
    """
    per_frame = _label_cells_per_frame(masks, min_area_px=min_area_px)
    n = len(masks)

    # Find seed frame (first frame with any cells)
    if seed_frame is None:
        seed_frame = 0
        while seed_frame < n and not per_frame[seed_frame]:
            seed_frame += 1
    if seed_frame >= n or not per_frame[seed_frame]:
        return []

    tracks = []
    for seed in per_frame[seed_frame]:
        stack = np.zeros(masks.shape, dtype=bool)
        stack[seed_frame] = seed[2]
        tracks.append({
            "stack": stack,
            "centroid_history": [seed[0]],
            "alive": True,
            "first_frame": seed_frame,
            "last_seen_frame": seed_frame,
        })

    # For each frame, solve a min-cost assignment between alive tracks
    # and detected cells using the Hungarian algorithm. Tracks that
    # aren't assigned for more than MAX_GAP frames are killed.
    from scipy.optimize import linear_sum_assignment
    MAX_GAP = 10
    LARGE = 1e9

    for i in range(seed_frame + 1, n):
        cells = per_frame[i]
        alive = [t for t in tracks if t["alive"]]
        if not cells or not alive:
            for t in alive:
                if (i - t["last_seen_frame"]) > MAX_GAP:
                    t["alive"] = False
            continue

        prev_cents = np.array(
            [t["centroid_history"][-1] for t in alive])
        cur_cents = np.array([c[0] for c in cells])
        # Cost matrix: centroid distance, with gap-tolerance scaling
        gap = np.array([max(1, i - t["last_seen_frame"])
                        for t in alive])[:, None]
        raw_dists = cdist(prev_cents, cur_cents)
        # Forbid assignments > max_hop_px × gap (scaled for missing frames)
        allowed = max_hop_px * gap
        cost = np.where(raw_dists <= allowed, raw_dists, LARGE)

        # Pad to square matrix so Hungarian handles N_tracks != N_cells
        n_tracks, n_cells = cost.shape
        N = max(n_tracks, n_cells)
        square = np.full((N, N), LARGE)
        square[:n_tracks, :n_cells] = cost
        row_ind, col_ind = linear_sum_assignment(square)

        assigned_tracks = set()
        for r, c in zip(row_ind, col_ind):
            if r >= n_tracks or c >= n_cells:
                continue
            if square[r, c] >= LARGE:
                continue
            t = alive[r]
            (cy, cx), _, mask = cells[c]
            t["stack"][i] = mask
            t["centroid_history"].append((cy, cx))
            t["last_seen_frame"] = i
            assigned_tracks.add(r)

        # Tracks that didn't get a valid cell: kill after MAX_GAP
        for r, t in enumerate(alive):
            if r in assigned_tracks:
                continue
            if (i - t["last_seen_frame"]) > MAX_GAP:
                t["alive"] = False

        # Phase 16: spawn new tracks for unmatched detections
        if spawn_new_tracks:
            assigned_cells = {c for r, c in zip(row_ind, col_ind)
                              if r < n_tracks and c < n_cells
                              and square[r, c] < LARGE}
            for c_idx, cell in enumerate(cells):
                if c_idx in assigned_cells:
                    continue
                (cy, cx), _, cmask = cell
                stack = np.zeros(masks.shape, dtype=bool)
                stack[i] = cmask
                tracks.append({
                    "stack": stack,
                    "centroid_history": [(cy, cx)],
                    "alive": True,
                    "first_frame": i,
                    "last_seen_frame": i,
                })

    # Drop short tracks (often spurious)
    if min_track_length > 1:
        tracks = [t for t in tracks
                  if t["stack"].any(axis=(1, 2)).sum() >= min_track_length]

    # Phase 16b: division-detection heuristic.
    # When a new track spawns at frame F with its first centroid
    # within max_hop_px of a track that ended at frame F-1 or F,
    # AND the new track's area is 30-70% of the parent's last area,
    # record a lineage link (parent_id on the new track).
    for i, t in enumerate(tracks):
        if t.get("parent_id") is not None:
            continue
        f_start = t["first_frame"]
        if f_start == 0:
            continue
        cy0, cx0 = t["centroid_history"][0]
        new_area = int(t["stack"][f_start].sum())
        for j, p in enumerate(tracks):
            if i == j:
                continue
            p_frames = np.where(p["stack"].any(axis=(1, 2)))[0]
            if len(p_frames) == 0:
                continue
            p_last = int(p_frames[-1])
            if p_last not in (f_start - 1, f_start):
                continue
            py, px = p["centroid_history"][-1]
            dist = ((cy0 - py) ** 2 + (cx0 - px) ** 2) ** 0.5
            if dist > max_hop_px:
                continue
            parent_area = int(p["stack"][p_last].sum())
            if parent_area == 0:
                continue
            ratio = new_area / parent_area
            if 0.2 <= ratio <= 0.9:
                t["parent_id"] = j
                break

    # Strip helper flags
    for t in tracks:
        t.pop("alive", None)
        t.pop("last_seen_frame", None)
    return tracks
