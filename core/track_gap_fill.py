"""Post-tracking gap fill for multi-cell pipelines.

After the Hungarian tracker assigns cross-frame identity, some tracks
may have internal gaps (frames where the cell wasn't detected but it
reappears later). This module fills those gaps by:

  1. Identifying internal gaps per track (between first and last frame)
  2. Interpolating the expected centroid from flanking frames
  3. Searching for a cell near that position using cpsam(augment=True)
  4. If found, inserting the mask into the track

This runs AFTER tracking, so it cannot break track identity — it only
adds masks to existing tracks at frames where they're missing.
"""
import logging
import numpy as np
from scipy import ndimage as ndi

log = logging.getLogger(__name__)


def _find_internal_gaps(track, n_frames):
    """Return list of frame indices where this track has a gap
    between its first and last active frame."""
    active = np.array([track["stack"][i].any() for i in range(n_frames)])
    if active.sum() < 2:
        return []
    first = int(np.argmax(active))
    last = n_frames - 1 - int(np.argmax(active[::-1]))
    gaps = [i for i in range(first, last + 1) if not active[i]]
    return gaps


def _interpolate_centroid(track, frame_idx, n_frames):
    """Interpolate expected centroid at frame_idx from the nearest
    flanking frames that have masks."""
    active = [(i, track["stack"][i]) for i in range(n_frames)
              if track["stack"][i].any()]
    if not active:
        return None

    before = [(i, m) for i, m in active if i < frame_idx]
    after = [(i, m) for i, m in active if i > frame_idx]
    if not before or not after:
        if before:
            m = before[-1][1]
        else:
            m = after[0][1]
        ys, xs = np.where(m)
        return (float(ys.mean()), float(xs.mean()))

    bi, bm = before[-1]
    ai, am = after[0]
    bys, bxs = np.where(bm)
    ays, axs = np.where(am)
    bc = (bys.mean(), bxs.mean())
    ac = (ays.mean(), axs.mean())

    t = (frame_idx - bi) / max(ai - bi, 1)
    cy = bc[0] + t * (ac[0] - bc[0])
    cx = bc[1] + t * (ac[1] - bc[1])
    return (cy, cx)


def _pick_nearest_cell(labels, expected_centroid, search_radius, min_area):
    """From a label image, pick the cell nearest expected_centroid."""
    if labels.max() == 0:
        return None
    ey, ex = expected_centroid
    best_dist = float("inf")
    best_mask = None
    for lab in range(1, int(labels.max()) + 1):
        cell = labels == lab
        if cell.sum() < min_area:
            continue
        ys, xs = np.where(cell)
        cy, cx = ys.mean(), xs.mean()
        dist = ((cy - ey)**2 + (cx - ex)**2)**0.5
        if dist < search_radius and dist < best_dist:
            best_dist = dist
            best_mask = cell
    return best_mask


def _find_cell_near(image, expected_centroid, search_radius=100,
                    min_area=300, project_root=None):
    """Try to find a cell near expected_centroid using a cascade:
      1. cpsam(augment=True)
      2. cellpose_combined_robust + MedSAM + DeepSea (CP3 subprocess)
    Returns bool mask or None."""
    from cellpose import models

    # Attempt 1: cpsam with TTA
    m = models.CellposeModel(gpu=True)
    masks_i, _, _ = m.eval(image, augment=True)
    result = _pick_nearest_cell(masks_i, expected_centroid,
                                search_radius, min_area)
    if result is not None:
        return result

    # Attempt 2: cellpose+MedSAM+DeepSea via CP3 subprocess
    if project_root is None:
        return None
    try:
        from core.hybrid_cpsam import _run_cp3_fallback
        import numpy as np
        fallback = _run_cp3_fallback(
            image[np.newaxis], project_root)
        if fallback.any():
            return fallback[0]
    except Exception as e:
        log.warning("CP3 fallback failed for gap frame: %s", e)

    return None


def fill_track_gaps(tracks, frames, min_area=300,
                    search_radius=100, progress_fn=None,
                    project_root=None):
    """Fill internal gaps in tracks by searching for missing cells.

    Cascade per gap frame:
      1. cpsam(augment=True) — find cell near interpolated centroid
      2. cellpose+MedSAM+DeepSea (CP3 subprocess) — if cpsam fails

    Modifies tracks in-place (adds masks to track["stack"] for gap
    frames). Returns count of filled gaps.

    Args:
        tracks: list of track dicts from track_all_cells
        frames: (N, H, W) uint8 original images
        min_area: minimum cell area to accept a fill
        search_radius: max distance from interpolated centroid
        progress_fn: optional callback(msg, pct)
        project_root: path to project root (enables CP3 fallback)
    """
    n_frames = len(frames)
    total_gaps = 0
    filled = 0

    all_gaps = []
    for tid, track in enumerate(tracks):
        gaps = _find_internal_gaps(track, n_frames)
        for g in gaps:
            all_gaps.append((tid, g))
    total_gaps = len(all_gaps)

    if total_gaps == 0:
        log.info("No internal gaps to fill")
        return 0

    log.info("Found %d internal gaps across %d tracks", total_gaps,
             len(tracks))

    for idx, (tid, frame_idx) in enumerate(all_gaps):
        if progress_fn and (idx % 5 == 0 or idx == len(all_gaps) - 1):
            progress_fn(f"Gap fill {idx+1}/{total_gaps}",
                        int(100 * idx / max(total_gaps - 1, 1)))

        track = tracks[tid]
        centroid = _interpolate_centroid(track, frame_idx, n_frames)
        if centroid is None:
            continue

        cell_mask = _find_cell_near(
            frames[frame_idx], centroid,
            search_radius=search_radius, min_area=min_area,
            project_root=project_root)
        if cell_mask is not None:
            track["stack"][frame_idx] = cell_mask
            filled += 1

    log.info("Filled %d/%d gaps", filled, total_gaps)
    return filled
