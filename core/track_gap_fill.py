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


def _pick_nearest_cell(labels, expected_centroid, search_radius,
                        min_area, expected_area=None,
                        area_ratio_low=0.3, area_ratio_high=3.0):
    """From a label image, pick the cell nearest expected_centroid.

    When ``expected_area`` is provided, candidates whose area is
    outside ``[expected_area * area_ratio_low,
    expected_area * area_ratio_high]`` are rejected. Without this
    guard, a phantom track (e.g. 200 px) whose gap-fill candidate
    list contains the real cell nearby (2500 px) would inherit the
    big cell's mask, inflating the phantom's lifetime past the
    Cy5 persistence_guard threshold and leaving a fragment in the
    final output — exactly the test2 frame-16 cell-4 bug.
    """
    if labels.max() == 0:
        return None
    ey, ex = expected_centroid
    best_dist = float("inf")
    best_mask = None
    for lab in range(1, int(labels.max()) + 1):
        cell = labels == lab
        area = int(cell.sum())
        if area < min_area:
            continue
        if expected_area is not None and expected_area > 0:
            lo = expected_area * area_ratio_low
            hi = expected_area * area_ratio_high
            if area < lo or area > hi:
                continue
        ys, xs = np.where(cell)
        cy, cx = ys.mean(), xs.mean()
        dist = ((cy - ey)**2 + (cx - ex)**2)**0.5
        if dist < search_radius and dist < best_dist:
            best_dist = dist
            best_mask = cell
    return best_mask


def _gap_crop_window(centroid, search_radius, expected_area, min_area,
                     H, W):
    """Adaptive crop bounds (y0, y1, x0, x1) centred on the expected
    centroid, sized to contain the ENTIRE Phase-1 acceptance region:
    any cell whose centroid is within `search_radius`, plus that cell's
    own radius + a margin. Because Phase 1 rejects every candidate
    farther than `search_radius` from the centroid, no acceptable cell
    can fall outside this window — so cropping does not change recall.

    Returns None when the crop wouldn't be meaningfully smaller than the
    full frame (then we just segment the whole frame)."""
    cy, cx = centroid
    cell_r = int(np.sqrt(max(float(expected_area or 0.0), float(min_area))
                         / np.pi))
    half = int(search_radius + 2 * cell_r + 32)
    if 2 * half >= int(0.85 * min(H, W)):
        return None
    cyi, cxi = int(round(cy)), int(round(cx))
    return (max(0, cyi - half), min(H, cyi + half),
            max(0, cxi - half), min(W, cxi + half))


def _try_primary_cpsam(image, expected_centroid, search_radius, min_area,
                       cpsam_model=None, expected_area=None,
                       use_crop=False, use_augment=True):
    """Attempt 1: cpsam(augment=use_augment). Returns mask or None.

    `use_augment` controls 4-rotation TTA. augment=True runs 4 forward
    passes (4× cost — the dominant per-gap cost at downsampled
    resolution where cpsam isn't pixel-bound); augment=False is ~4×
    cheaper but may miss harder cells (validated vs reviewed GT in
    scripts/bench_gap_fill.py).

    `cpsam_model` is an optional pre-loaded model, so we don't reload
    the ViT-H weights on every gap frame. Falls back to creating a
    fresh model if not supplied (kept for backwards compat).

    `expected_area` (optional) constrains the chosen mask to be
    within a factor of the track's typical size — see
    `_pick_nearest_cell` docstring.

    When `use_crop` is True, cpsam segments only an adaptive crop around
    the expected centroid (`_gap_crop_window`) instead of the full
    frame. This is the dominant Phase-1 speed-up: same recall (the crop
    contains the whole acceptance region), at ~(frame/crop)² less
    compute. The chosen mask is mapped back to full-frame coordinates
    so every downstream guard (size / collision) is unchanged.
    """
    from cellpose import models
    m = cpsam_model if cpsam_model is not None else models.CellposeModel(
        gpu=True)
    H, W = image.shape[:2]
    win = (_gap_crop_window(expected_centroid, search_radius,
                            expected_area, min_area, H, W)
           if use_crop else None)
    if win is not None:
        y0, y1, x0, x1 = win
        masks_i, _, _ = m.eval(image[y0:y1, x0:x1], augment=use_augment)
        local = _pick_nearest_cell(
            masks_i,
            (expected_centroid[0] - y0, expected_centroid[1] - x0),
            search_radius, min_area, expected_area=expected_area)
        if local is None:
            return None
        full = np.zeros((H, W), dtype=bool)
        full[y0:y1, x0:x1] = local
        return full
    masks_i, _, _ = m.eval(image, augment=use_augment)
    return _pick_nearest_cell(masks_i, expected_centroid,
                              search_radius, min_area,
                              expected_area=expected_area)


def _track_median_area(track, n_frames):
    """Median per-frame area of a track over its active frames.
    Returns 0 if the track has no active frames."""
    areas = []
    stack = track.get("stack")
    if stack is None:
        return 0.0
    for f in range(n_frames):
        if stack[f].any():
            areas.append(float(stack[f].sum()))
    return float(np.median(areas)) if areas else 0.0


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
                    use_mask_propagation=True,
                    use_crop=None, use_augment=None, stats_out=None):
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
        use_crop: Phase-1 crop acceleration (None → DEFAULTS.gap_fill_crop).
        stats_out: optional dict; filled with per-phase {time_s, filled}
            for RUN_METADATA / profiling.
    """
    import time as _time
    from core.pipeline_defaults import DEFAULTS as _PD
    if use_crop is None:
        use_crop = _PD.gap_fill_crop
    if use_augment is None:
        use_augment = _PD.gap_fill_augment

    n_frames = len(frames)
    filled = 0
    stats = {"total_gaps": 0, "crop": bool(use_crop),
             "phase1": {"time_s": 0.0, "filled": 0},
             "phase2": {"time_s": 0.0, "filled": 0},
             "phase3": {"time_s": 0.0, "filled": 0},
             "phase4": {"time_s": 0.0, "filled": 0}}

    def _finish():
        log.info(
            "gap-fill timing (crop=%s): p1=%.1fs(%d) p2=%.1fs(%d) "
            "p3=%.1fs(%d) p4=%.1fs(%d) — %d/%d gaps filled",
            use_crop, stats["phase1"]["time_s"], stats["phase1"]["filled"],
            stats["phase2"]["time_s"], stats["phase2"]["filled"],
            stats["phase3"]["time_s"], stats["phase3"]["filled"],
            stats["phase4"]["time_s"], stats["phase4"]["filled"],
            filled, stats["total_gaps"])
        if stats_out is not None:
            stats_out.update(stats)

    all_gaps = []
    for tid, track in enumerate(tracks):
        gaps = _find_internal_gaps(track, n_frames)
        for g in gaps:
            all_gaps.append((tid, g))
    total_gaps = len(all_gaps)
    stats["total_gaps"] = total_gaps

    if total_gaps == 0:
        log.info("No internal gaps to fill")
        _finish()
        return 0

    log.info("Found %d internal gaps across %d tracks", total_gaps,
             len(tracks))

    # Phase 1: shared cpsam model, one eval per gap (fast — no reloads).
    t_p1 = _time.time()
    from cellpose import models
    if progress_fn:
        progress_fn(f"Gap fill phase 1: cpsam(augment=True"
                    f"{', crop' if use_crop else ''}) on "
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
        # Match candidates against the track's typical size so a
        # phantom track (e.g. 200 px) doesn't get fattened by the
        # real cell next door (e.g. 2500 px).
        expected_area = _track_median_area(track, n_frames)

        def _try(crop, aug):
            return _try_primary_cpsam(
                frames[frame_idx], centroid,
                search_radius=search_radius, min_area=min_area,
                cpsam_model=cpsam_model, expected_area=expected_area,
                use_crop=crop, use_augment=aug)

        # Fast first pass (crop, no-augment by default). On a miss, the
        # cascade widens — augment (4× TTA, recovers harder cells), then
        # full frame — so recall is never worse than the old
        # always-full-frame, always-augment path; only the misses pay
        # the extra cost. GT-validated (scripts/bench_gap_fill.py).
        cell_mask = _try(use_crop, use_augment)
        if cell_mask is None and not use_augment:
            cell_mask = _try(use_crop, True)
            stats["phase1"]["augment_fallbacks"] = (
                stats["phase1"].get("augment_fallbacks", 0) + 1)
        if cell_mask is None and use_crop:
            cell_mask = _try(False, True)
            stats["phase1"]["crop_fallbacks"] = (
                stats["phase1"].get("crop_fallbacks", 0) + 1)
        if cell_mask is not None:
            # Don't add a mask that overlaps/touches an already-tracked
            # cell in this frame — that's the cell-3+cell-4 fragment
            # pattern (cpsam(augment) finds a small detection right
            # next to an existing big cell, gap fill calls it a new
            # cell, persistence_guard keeps it).
            from scipy import ndimage as ndi
            dilated = ndi.binary_dilation(cell_mask, iterations=3)
            collision = False
            for other_idx, other_t in enumerate(tracks):
                if other_idx == tid:
                    continue
                other_stack = other_t.get("stack")
                if (other_stack is not None
                        and other_stack[frame_idx].any()
                        and (dilated & other_stack[frame_idx]).any()):
                    collision = True
                    break
            if collision:
                # This frame is already covered by another track;
                # leave the gap rather than introduce a fragment.
                continue
            track["stack"][frame_idx] = cell_mask
            filled += 1
        else:
            pending.append((tid, frame_idx, centroid, expected_area))

    stats["phase1"] = {"time_s": _time.time() - t_p1, "filled": filled}
    log.info("Phase 1 filled %d/%d gaps; %d pending for Phase 2",
             filled, total_gaps, len(pending))

    if not pending or not use_cp3_fallback:
        if progress_fn:
            progress_fn(f"Gap fill done: {filled}/{total_gaps}", 100)
        _finish()
        return filled

    # Phase 2: BATCHED CP3 subprocess. One subprocess handles all
    # remaining gap frames at once.
    t_p2 = _time.time()
    if progress_fn:
        progress_fn(
            f"Gap fill phase 2: CP3 fallback on {len(pending)} "
            f"frames (one subprocess)", 60)
    pending_frames = np.stack([frames[fi] for _, fi, _, _ in pending])
    fallback_masks = _try_cp3_fallback_batch(pending_frames, project_root)

    for j, (tid, frame_idx, centroid, expected_area) in enumerate(pending):
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
        area = int(m.sum())
        if area < min_area:
            continue
        # Same size-similarity guard as Phase 1 — don't let a
        # different (real) cell next door inflate a phantom track.
        if expected_area > 0 and (area < expected_area * 0.3
                                    or area > expected_area * 3.0):
            continue
        # Same overlap/touch guard as Phase 1 — don't introduce a
        # fragment beside an existing tracked cell.
        from scipy import ndimage as ndi
        dilated = ndi.binary_dilation(m, iterations=3)
        collision = False
        for other_idx, other_t in enumerate(tracks):
            if other_idx == tid:
                continue
            other_stack = other_t.get("stack")
            if (other_stack is not None
                    and other_stack[frame_idx].any()
                    and (dilated & other_stack[frame_idx]).any()):
                collision = True
                break
        if collision:
            continue
        tracks[tid]["stack"][frame_idx] = m
        filled += 1

    stats["phase2"] = {"time_s": _time.time() - t_p2,
                       "filled": filled - stats["phase1"]["filled"]}
    log.info("Phase 2 brought total to %d/%d gaps filled",
             filled, total_gaps)

    # Phase 3: SAM2 video propagation. For each remaining gap segment,
    # propagate the most recent flanking mask forward using SAM2's
    # memory attention. Cells that were lost to retraction / dimming /
    # mitosis often get tracked accurately here.
    t_p3 = _time.time()
    before3 = filled
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
    stats["phase3"] = {"time_s": _time.time() - t_p3,
                       "filled": filled - before3}

    # Phase 4: simple mask translation. Last resort. For any track that
    # still has gaps, carry the most recent flanking mask forward
    # (translated to the interpolated centroid) so the cell exists in
    # the stack across the gap. Less accurate than SAM2 but free.
    t_p4 = _time.time()
    before4 = filled
    if use_mask_propagation:
        propagated = _propagate_masks_into_gaps(
            tracks, n_frames, search_radius=search_radius)
        if propagated:
            log.info("Phase 4 translated masks into %d gap frames",
                     propagated)
            filled += propagated
    stats["phase4"] = {"time_s": _time.time() - t_p4,
                       "filled": filled - before4}

    if progress_fn:
        progress_fn(f"Gap fill done: {filled}/{total_gaps}", 100)
    _finish()
    return filled


def _propagate_masks_into_gaps(tracks, n_frames, search_radius=150):
    """For each track, fill remaining gap frames by carrying the
    nearest flanking mask, translated to the interpolated centroid.

    Marks every propagated frame in `track["propagated_frames"]` (a
    set of frame indices). Tracks without that key get one created.

    Returns the total count of frames propagated across all tracks.
    """
    from scipy import ndimage as ndi
    n_propagated = 0
    for ti, track in enumerate(tracks):
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
            sys_, sxs = np.where(src_mask)
            src_cy, src_cx = sys_.mean(), sxs.mean()
            dy = int(round(cent[0] - src_cy))
            dx = int(round(cent[1] - src_cx))
            # Translate the source mask by (dy, dx)
            new_mask = np.zeros_like(src_mask)
            H, W = new_mask.shape
            new_ys = sys_ + dy
            new_xs = sxs + dx
            valid = ((new_ys >= 0) & (new_ys < H)
                     & (new_xs >= 0) & (new_xs < W))
            new_mask[new_ys[valid], new_xs[valid]] = True
            if new_mask.sum() < 20:    # too tiny / off-screen
                continue
            # Don't propagate onto a region already occupied by
            # another track — that's the fragmentation pattern users
            # see (cell 4's frame-15 mask carried into frame 16
            # where cell 3 took over the location; the visible
            # output then shows track 4 as scattered specks where
            # track 3's mask doesn't quite cover, because
            # _build_tracked_labels resolves overlaps via "first
            # track wins"). Reject if the propagated mask would
            # collide with any other track's mask in the target frame.
            dilated = ndi.binary_dilation(new_mask, iterations=3)
            collision = False
            for other_idx, other_t in enumerate(tracks):
                if other_idx == ti:
                    continue
                other_stack = other_t.get("stack")
                if (other_stack is not None
                        and other_stack[fi].any()
                        and (dilated & other_stack[fi]).any()):
                    collision = True
                    break
            if collision:
                continue
            track["stack"][fi] = new_mask
            track["propagated_frames"].add(fi)
            n_propagated += 1
    return n_propagated
