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


def _try_primary_cpsam(image, expected_centroid, search_radius, min_area,
                       cpsam_model=None):
    """Attempt 1: cpsam(augment=True). Returns mask or None.

    `cpsam_model` is an optional pre-loaded model, so we don't reload
    the ViT-H weights on every gap frame. Falls back to creating a
    fresh model if not supplied (kept for backwards compat).
    """
    from cellpose import models
    m = cpsam_model if cpsam_model is not None else models.CellposeModel(
        gpu=True)
    masks_i, _, _ = m.eval(image, augment=True)
    return _pick_nearest_cell(masks_i, expected_centroid,
                              search_radius, min_area)


def _try_cp3_fallback_batch(images, project_root):
    """Attempt 2 (batched): single subprocess for all CP3 fallback
    frames. Avoids paying the cellpose-3 cold-load cost per gap.
    Returns (N, H, W) bool mask array (zeros for frames where it fails).
    """
    if project_root is None or len(images) == 0:
        return np.zeros((len(images),) + (
            images[0].shape if len(images) else (1, 1)), dtype=bool)
    try:
        from core.hybrid_cpsam import _run_cp3_fallback
        return _run_cp3_fallback(np.asarray(images), project_root)
    except Exception as e:
        log.warning("CP3 fallback batch failed: %s", e)
        return np.zeros((len(images),) + images[0].shape, dtype=bool)


def fill_track_gaps(tracks, frames, min_area=300,
                    search_radius=100, progress_fn=None,
                    project_root=None, use_cp3_fallback=True,
                    use_sam2_video=True,
                    use_mask_propagation=True):
    """Fill internal gaps in tracks by searching for missing cells.

    Four-phase cascade:
      Phase 1 (in-env): one cpsam(augment=True) eval per gap frame using
        a single shared model — no reload between frames.
      Phase 2 (cross-env, batched): for gaps where Phase 1 found nothing,
        run cellpose+MedSAM+DeepSea via ONE subprocess on the whole batch.
        This avoids the ~30-s cold-load of cellpose 3 that would otherwise
        be paid per gap frame.
      Phase 3 (SAM2 video, NEW): for gaps where Phase 1+2 fail, use the
        SAM2 video predictor to propagate the most recent flanking mask
        forward through the gap. Uses image content, so it works even
        when single-frame detectors lose the cell to retraction /
        dimming / shape change. ~1s per gap frame. Tagged in
        `track["sam2_propagated_frames"]`.
      Phase 4 (mask translation): for gaps where ALL three above fail,
        carry the most recent flanking mask forward into the gap so the
        cell exists in the track stack across the gap. Last-resort
        fallback. Tagged in `track["propagated_frames"]`.

    Modifies tracks in-place. Returns count of filled gaps (across all
    four phases).

    Args:
        tracks: list of track dicts from track_all_cells
        frames: (N, H, W) uint8 original images
        min_area: minimum cell area to accept a fill
        search_radius: max distance from interpolated centroid
        progress_fn: optional callback(msg, pct)
        project_root: path to project root (enables CP3 fallback)
        use_cp3_fallback: if False, skip Phase 2.
        use_sam2_video: if False, skip Phase 3 (SAM2 propagation).
        use_mask_propagation: if False, skip Phase 4 (translation).
    """
    n_frames = len(frames)
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

    # Phase 1: shared cpsam model, one eval per gap (fast — no reloads).
    from cellpose import models
    if progress_fn:
        progress_fn(f"Gap fill phase 1: cpsam(augment=True) on "
                    f"{total_gaps} gaps", 0)
    cpsam_model = models.CellposeModel(gpu=True)

    pending = []   # (tid, frame_idx, expected_centroid) for Phase 2
    for idx, (tid, frame_idx) in enumerate(all_gaps):
        if progress_fn and (idx % 5 == 0 or idx == len(all_gaps) - 1):
            progress_fn(f"Gap fill phase 1: {idx+1}/{total_gaps}",
                        int(50 * idx / max(total_gaps - 1, 1)))

        track = tracks[tid]
        centroid = _interpolate_centroid(track, frame_idx, n_frames)
        if centroid is None:
            continue

        cell_mask = _try_primary_cpsam(
            frames[frame_idx], centroid,
            search_radius=search_radius, min_area=min_area,
            cpsam_model=cpsam_model)
        if cell_mask is not None:
            track["stack"][frame_idx] = cell_mask
            filled += 1
        else:
            pending.append((tid, frame_idx, centroid))

    log.info("Phase 1 filled %d/%d gaps; %d pending for Phase 2",
             filled, total_gaps, len(pending))

    if not pending or not use_cp3_fallback:
        if progress_fn:
            progress_fn(f"Gap fill done: {filled}/{total_gaps}", 100)
        return filled

    # Phase 2: BATCHED CP3 subprocess. One subprocess handles all
    # remaining gap frames at once.
    if progress_fn:
        progress_fn(
            f"Gap fill phase 2: CP3 fallback on {len(pending)} "
            f"frames (one subprocess)", 60)
    pending_frames = np.stack([frames[fi] for _, fi, _ in pending])
    fallback_masks = _try_cp3_fallback_batch(pending_frames, project_root)

    for j, (tid, frame_idx, centroid) in enumerate(pending):
        m = fallback_masks[j]
        if not m.any():
            continue
        # Only accept the fallback if the resulting cell is near
        # where we expected it
        ys, xs = np.where(m)
        cy, cx = ys.mean(), xs.mean()
        ey, ex = centroid
        if ((cy - ey)**2 + (cx - ex)**2) ** 0.5 > search_radius:
            continue
        if int(m.sum()) < min_area:
            continue
        tracks[tid]["stack"][frame_idx] = m
        filled += 1

    log.info("Phase 2 brought total to %d/%d gaps filled",
             filled, total_gaps)

    # Phase 3: SAM2 video propagation. For each remaining gap segment,
    # propagate the most recent flanking mask forward using SAM2's
    # memory attention. Cells that were lost to retraction / dimming /
    # mitosis often get tracked accurately here.
    if use_sam2_video:
        try:
            from core.sam2_video import fill_track_gaps_with_sam2
            if progress_fn:
                progress_fn(
                    f"Phase 3: SAM2 video propagation", 70)
            sam2_filled = fill_track_gaps_with_sam2(
                tracks, frames, min_area=min_area,
                progress_fn=lambda m, p: progress_fn(
                    m, int(70 + 20 * p / 100))
                if progress_fn else None)
            if sam2_filled:
                log.info("Phase 3 SAM2 propagation filled %d gap frames",
                         sam2_filled)
                filled += sam2_filled
        except Exception as e:
            log.warning("Phase 3 SAM2 propagation failed: %s "
                        "(falling through to Phase 4)", e)

    # Phase 4: simple mask translation. Last resort. For any track that
    # still has gaps, carry the most recent flanking mask forward
    # (translated to the interpolated centroid) so the cell exists in
    # the stack across the gap. Less accurate than SAM2 but free.
    if use_mask_propagation:
        propagated = _propagate_masks_into_gaps(
            tracks, n_frames, search_radius=search_radius)
        if propagated:
            log.info("Phase 4 translated masks into %d gap frames",
                     propagated)
            filled += propagated

    if progress_fn:
        progress_fn(f"Gap fill done: {filled}/{total_gaps}", 100)
    return filled


def _propagate_masks_into_gaps(tracks, n_frames, search_radius=150):
    """For each track, fill remaining gap frames by carrying the
    nearest flanking mask, translated to the interpolated centroid.

    Marks every propagated frame in `track["propagated_frames"]` (a
    set of frame indices). Tracks without that key get one created.

    Returns the total count of frames propagated across all tracks.
    """
    n_propagated = 0
    for track in tracks:
        track.setdefault("propagated_frames", set())
        gaps = _find_internal_gaps(track, n_frames)
        for fi in gaps:
            # Find nearest before / after frame with a real mask
            before, after = None, None
            for j in range(fi - 1, -1, -1):
                if track["stack"][j].any() and j not in track[
                        "propagated_frames"]:
                    before = j
                    break
            for j in range(fi + 1, n_frames):
                if track["stack"][j].any() and j not in track[
                        "propagated_frames"]:
                    after = j
                    break
            src_frame = before if before is not None else after
            if src_frame is None:
                continue
            # Interpolated centroid for this frame
            cent = _interpolate_centroid(track, fi, n_frames)
            if cent is None:
                continue
            src_mask = track["stack"][src_frame]
            sys, sxs = np.where(src_mask)
            src_cy, src_cx = sys.mean(), sxs.mean()
            dy = int(round(cent[0] - src_cy))
            dx = int(round(cent[1] - src_cx))
            # Translate the source mask by (dy, dx)
            new_mask = np.zeros_like(src_mask)
            H, W = new_mask.shape
            new_ys = sys + dy
            new_xs = sxs + dx
            valid = ((new_ys >= 0) & (new_ys < H)
                     & (new_xs >= 0) & (new_xs < W))
            new_mask[new_ys[valid], new_xs[valid]] = True
            if new_mask.sum() < 20:    # too tiny / off-screen
                continue
            track["stack"][fi] = new_mask
            track["propagated_frames"].add(fi)
            n_propagated += 1
    return n_propagated
