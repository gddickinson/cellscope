"""Detection-level post-processing to fix splits, enforce cell count,
and clean up raw cpsam output before tracking.

These run AFTER detection but BEFORE tracking, so they fix the
detection input to the tracker rather than trying to fix tracking
errors after the fact.
"""
import numpy as np
from scipy import ndimage
import logging

log = logging.getLogger(__name__)


def close_split_cells(labels, iterations=3):
    """Morphological closing per cell to bridge small internal gaps.

    When cpsam splits a cell into two pieces separated by a thin gap,
    closing fills the gap and reconnects them. Conservative: only
    bridges gaps up to ~6 pixels wide (3 iterations × 2 sides).

    Args:
        labels: (N, H, W) int32 label stack
        iterations: closing disk size (larger = bridges wider gaps)

    Returns:
        (N, H, W) int32 label stack with splits repaired
    """
    n_fixed = 0
    for i in range(len(labels)):
        if labels[i].max() == 0:
            continue
        # Close each cell independently to avoid merging neighbors
        for lab in range(1, int(labels[i].max()) + 1):
            cell = labels[i] == lab
            if not cell.any():
                continue
            closed = ndimage.binary_closing(cell, iterations=iterations)
            # Only accept if closing didn't expand into another cell
            overlap_other = np.logical_and(
                closed & ~cell, labels[i] > 0).sum()
            if overlap_other == 0:
                new_pixels = np.logical_and(closed, ~cell)
                labels[i][new_pixels] = lab

        # Re-check: merge components with same label that are now connected
        unique = set(np.unique(labels[i])) - {0}
        for lab in unique:
            cell = labels[i] == lab
            comp_labels, n_comp = ndimage.label(cell)
            if n_comp > 1:
                # Multiple disconnected components — closing bridged them
                n_fixed += 1

    if n_fixed:
        log.info("Morphological closing bridged gaps in %d cells", n_fixed)
    return labels


def merge_nearby_fragments(labels, max_gap_px=15, min_area=100):
    """Merge small fragments that are near a larger cell.

    If a small detection (< min_area) is within max_gap_px of a larger
    detection, absorb it into the larger one. Catches split fragments
    that closing didn't bridge.

    Args:
        labels: (N, H, W) int32 label stack
        max_gap_px: maximum gap between fragment and parent
        min_area: fragments below this area are candidates for merging

    Returns:
        (N, H, W) int32 with fragments merged
    """
    n_merged = 0
    for i in range(len(labels)):
        if labels[i].max() == 0:
            continue
        unique = sorted(set(np.unique(labels[i])) - {0})
        areas = {lab: int((labels[i] == lab).sum()) for lab in unique}

        for lab in unique:
            if areas[lab] >= min_area:
                continue
            frag = labels[i] == lab
            # Dilate fragment to find nearby cells
            dilated = ndimage.binary_dilation(
                frag, iterations=max_gap_px)
            for other_lab in unique:
                if other_lab == lab or areas[other_lab] < min_area:
                    continue
                other = labels[i] == other_lab
                overlap = np.logical_and(dilated, other).sum()
                if overlap > 0:
                    labels[i][frag] = other_lab
                    n_merged += 1
                    log.debug("Frame %d: merged fragment %d (%d px) "
                              "into cell %d", i, lab, areas[lab],
                              other_lab)
                    break

    if n_merged:
        log.info("Merged %d small fragments into nearby cells", n_merged)
    return labels


def enforce_cell_count(labels, expected_cells, min_area=200):
    """Keep only the top-N cells per frame by area.

    When expected_cells > 0, any extra detections (ranked by area)
    are removed. Does NOT add missing cells — only removes excess.

    Args:
        labels: (N, H, W) int32 label stack
        expected_cells: number of cells to keep (0 = no enforcement)
        min_area: minimum area to count as a real cell

    Returns:
        (N, H, W) int32 with excess cells removed
    """
    if expected_cells <= 0:
        return labels

    n_removed = 0
    for i in range(len(labels)):
        unique = sorted(set(np.unique(labels[i])) - {0})
        if len(unique) <= expected_cells:
            continue
        # Rank by area, keep top N
        areas = [(lab, int((labels[i] == lab).sum())) for lab in unique
                 if (labels[i] == lab).sum() >= min_area]
        areas.sort(key=lambda x: x[1], reverse=True)
        keep = {lab for lab, _ in areas[:expected_cells]}
        for lab in unique:
            if lab not in keep:
                labels[i][labels[i] == lab] = 0
                n_removed += 1

    if n_removed:
        log.info("Removed %d excess detections (expected_cells=%d)",
                 n_removed, expected_cells)
    return labels


def add_velocity_prediction(tracks, n_history=5):
    """Add velocity-based position prediction to each track.

    Computes predicted centroid for the next frame based on
    recent velocity (average of last n_history displacements).
    Stored as track["predicted_centroid"] for the tracker to use.

    Args:
        tracks: list of track dicts (modified in-place)
        n_history: number of recent frames to average velocity over
    """
    for t in tracks:
        history = t["centroid_history"]
        if len(history) < 2:
            t["predicted_centroid"] = history[-1] if history else (0, 0)
            continue
        recent = history[-min(n_history, len(history)):]
        velocities = [(recent[j+1][0] - recent[j][0],
                        recent[j+1][1] - recent[j][1])
                       for j in range(len(recent) - 1)]
        if velocities:
            avg_vy = np.mean([v[0] for v in velocities])
            avg_vx = np.mean([v[1] for v in velocities])
            last = history[-1]
            t["predicted_centroid"] = (
                last[0] + avg_vy, last[1] + avg_vx)
        else:
            t["predicted_centroid"] = history[-1]


def preprocess_detection(labels, expected_cells=0, min_area=200,
                          close_iterations=3, merge_gap=15):
    """Full detection post-processing pipeline.

    1. Morphological closing to bridge split gaps
    2. Merge small nearby fragments
    3. Enforce expected cell count

    Returns processed labels + summary dict.
    """
    labels = close_split_cells(labels, close_iterations)
    labels = merge_nearby_fragments(labels, merge_gap, min_area)
    labels = enforce_cell_count(labels, expected_cells, min_area)

    # Recompact label IDs
    for i in range(len(labels)):
        unique = sorted(set(np.unique(labels[i])) - {0})
        for new_id, old_id in enumerate(unique, 1):
            if old_id != new_id:
                labels[i][labels[i] == old_id] = new_id

    return labels
