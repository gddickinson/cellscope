"""Post-processing to fix common multi-cell tracking errors.

1. Merge split tracks — short tracks that overlap spatially with
   a longer track are absorbed into it.
2. Stabilize IDs — correct frame-to-frame ID switches by enforcing
   spatial overlap continuity between consecutive frames of a track.
"""
import numpy as np
import logging
from scipy import ndimage

log = logging.getLogger(__name__)


def merge_split_tracks(tracks, overlap_threshold=0.7,
                        max_area_ratio=0.4):
    """Merge tracks that are spatial fragments of a larger track.

    A track is merged into another ONLY if:
    - Its median area is < max_area_ratio of the target's median area
      (it's a fragment, not a full cell)
    - Its average overlap with the target is > overlap_threshold
      (most of its pixels are inside the target)

    This prevents merging two real touching cells (which only overlap
    at their boundary, typically <20%).

    Modifies tracks in-place. Returns count of merged tracks.
    """
    if len(tracks) < 2:
        return 0

    lengths = [int(t["stack"].any(axis=(1, 2)).sum()) for t in tracks]
    order = np.argsort(lengths)[::-1]

    # Compute median area per track
    median_areas = []
    for t in tracks:
        areas = [float(t["stack"][f].sum())
                 for f in range(t["stack"].shape[0])
                 if t["stack"][f].any()]
        median_areas.append(float(np.median(areas)) if areas else 0)

    merged = set()
    n_merged = 0

    for i in range(len(order)):
        short_idx = order[i]
        if short_idx in merged:
            continue

        short = tracks[short_idx]
        best_target = None
        best_overlap = 0

        for j in range(len(order)):
            long_idx = order[j]
            if long_idx == short_idx or long_idx in merged:
                continue
            if lengths[long_idx] <= lengths[short_idx]:
                continue

            # Check area ratio — fragment should be much smaller
            if median_areas[long_idx] > 0:
                area_ratio = median_areas[short_idx] / \
                    median_areas[long_idx]
                if area_ratio > max_area_ratio:
                    continue

            # Compute AVERAGE overlap across shared frames
            long_t = tracks[long_idx]
            n_frames = short["stack"].shape[0]
            overlaps = []
            for f in range(n_frames):
                if not short["stack"][f].any() or \
                        not long_t["stack"][f].any():
                    continue
                inter = np.logical_and(
                    short["stack"][f], long_t["stack"][f]).sum()
                short_area = short["stack"][f].sum()
                if short_area > 0:
                    overlaps.append(inter / short_area)
            if overlaps:
                avg_ov = np.mean(overlaps)
                if avg_ov > best_overlap:
                    best_overlap = avg_ov
                    best_target = long_idx

        if best_target is not None and best_overlap > overlap_threshold:
            target = tracks[best_target]
            n_frames = short["stack"].shape[0]
            for f in range(n_frames):
                if short["stack"][f].any():
                    target["stack"][f] |= short["stack"][f]
            short["stack"][:] = False
            merged.add(short_idx)
            n_merged += 1
            log.info("Merged track %d into track %d "
                     "(avg overlap %.0f%%, area ratio %.0f%%)",
                     short_idx, best_target,
                     best_overlap * 100,
                     median_areas[short_idx] /
                     max(median_areas[best_target], 1) * 100)

    return n_merged


def smooth_bouncing_ids(tracks, window=5):
    """Detect and fix rapid ID bouncing between two tracks.

    If two tracks swap masks back and forth within a short window
    (e.g., A→B→A→B), lock the assignment to the majority owner
    within the window.

    Works by: for each frame, if a track's mask has higher IoU
    with the OTHER track's recent history than its own, swap them.
    Uses a sliding window vote to avoid ping-ponging.
    """
    n_frames = tracks[0]["stack"].shape[0] if tracks else 0
    n_fixed = 0

    for ti in range(len(tracks)):
        for tj in range(ti + 1, len(tracks)):
            ta, tb = tracks[ti], tracks[tj]

            # Build IoU-to-self timeseries for each track
            # (how well does frame N match frame N-1 of the SAME track?)
            swaps = []
            for f in range(1, n_frames):
                if not (ta["stack"][f].any() and tb["stack"][f].any()
                        and ta["stack"][f-1].any()
                        and tb["stack"][f-1].any()):
                    swaps.append(0)
                    continue

                # Current assignment quality
                same_a = _iou(ta["stack"][f-1], ta["stack"][f])
                same_b = _iou(tb["stack"][f-1], tb["stack"][f])

                # Swapped assignment quality
                cross_a = _iou(ta["stack"][f-1], tb["stack"][f])
                cross_b = _iou(tb["stack"][f-1], ta["stack"][f])

                if (cross_a + cross_b) > (same_a + same_b):
                    swaps.append(1)  # swap would be better
                else:
                    swaps.append(0)

            # Sliding window vote: if majority of window says swap,
            # then swap that entire window
            half = window // 2
            for f in range(1, n_frames):
                start = max(0, f - half)
                end = min(len(swaps), f + half)
                region = swaps[start:end]
                if sum(region) > len(region) / 2:
                    # Majority says swap at frame f
                    if ta["stack"][f].any() and tb["stack"][f].any():
                        ta["stack"][f], tb["stack"][f] = \
                            tb["stack"][f].copy(), ta["stack"][f].copy()
                        n_fixed += 1

    if n_fixed:
        log.info("Smoothed %d bouncing ID assignments", n_fixed)
    return n_fixed


def stabilize_track_ids(tracks, iou_threshold=0.05):
    """Fix ID switches by enforcing spatial continuity.

    For each pair of tracks, check if swapping their masks at
    certain frames would improve frame-to-frame IoU continuity.
    If so, swap those frames.

    Modifies tracks in-place. Returns count of corrected switches.
    """
    n_frames = tracks[0]["stack"].shape[0] if tracks else 0
    n_corrected = 0

    for ti in range(len(tracks)):
        for tj in range(ti + 1, len(tracks)):
            t_a = tracks[ti]
            t_b = tracks[tj]

            for f in range(1, n_frames):
                a_prev = t_a["stack"][f - 1]
                a_curr = t_a["stack"][f]
                b_prev = t_b["stack"][f - 1]
                b_curr = t_b["stack"][f]

                if not (a_prev.any() and a_curr.any() and
                        b_prev.any() and b_curr.any()):
                    continue

                # Current assignment IoU
                iou_aa = _iou(a_prev, a_curr)
                iou_bb = _iou(b_prev, b_curr)
                current_score = iou_aa + iou_bb

                # Swapped assignment IoU
                iou_ab = _iou(a_prev, b_curr)
                iou_ba = _iou(b_prev, a_curr)
                swapped_score = iou_ab + iou_ba

                if swapped_score > current_score + iou_threshold:
                    # Swap masks at frame f
                    t_a["stack"][f], t_b["stack"][f] = \
                        t_b["stack"][f].copy(), t_a["stack"][f].copy()
                    n_corrected += 1

    if n_corrected:
        log.info("Corrected %d ID switches", n_corrected)
    return n_corrected


def remove_empty_tracks(tracks, min_frames=3):
    """Remove tracks with fewer than min_frames active frames."""
    before = len(tracks)
    tracks[:] = [t for t in tracks
                 if t["stack"].any(axis=(1, 2)).sum() >= min_frames]
    removed = before - len(tracks)
    if removed:
        log.info("Removed %d empty/short tracks", removed)
    return removed


def reject_false_positives(tracks, frames=None,
                            max_distance_factor=5.0,
                            min_area_fraction=0.1,
                            min_boundary_confidence=15.0):
    """Remove detections that are likely false positives.

    A detection is rejected if it fails multiple quality checks:
    1. Far from any established track position (> max_distance_factor
       × median cell radius)
    2. Area much smaller than median cell area (< min_area_fraction)
    3. Low boundary confidence (weak image gradient along contour)
    4. Appears in only 1-2 frames (transient)

    Modifies tracks in-place by zeroing rejected frames.
    Returns count of rejected detections.
    """
    if not tracks:
        return 0

    n_frames = tracks[0]["stack"].shape[0]
    n_rejected = 0

    # Compute per-track statistics
    track_stats = []
    for t in tracks:
        areas = []
        centroids = []
        for f in range(n_frames):
            if t["stack"][f].any():
                areas.append(float(t["stack"][f].sum()))
                ys, xs = np.where(t["stack"][f])
                centroids.append((ys.mean(), xs.mean()))
        track_stats.append({
            "median_area": float(np.median(areas)) if areas else 0,
            "mean_area": float(np.mean(areas)) if areas else 0,
            "n_active": len(areas),
            "centroids": centroids,
        })

    if not any(ts["n_active"] > 0 for ts in track_stats):
        return 0

    # Global median area across all tracks
    all_areas = [ts["median_area"] for ts in track_stats
                 if ts["median_area"] > 0]
    global_median_area = float(np.median(all_areas)) if all_areas else 1
    median_radius = np.sqrt(global_median_area / np.pi)

    # Check each track's each frame
    for ti, (t, ts) in enumerate(zip(tracks, track_stats)):
        if ts["n_active"] < 3:
            continue

        for f in range(n_frames):
            if not t["stack"][f].any():
                continue
            area = float(t["stack"][f].sum())

            # Check 1: area too small
            if area < global_median_area * min_area_fraction:
                score = 0
            else:
                score = 1

            # Check 2: distance from this track's interpolated position
            ys, xs = np.where(t["stack"][f])
            cy, cx = ys.mean(), xs.mean()

            # Find nearest active frame in this track (before and after)
            near_frames = []
            for df in range(-5, 6):
                nf = f + df
                if 0 <= nf < n_frames and nf != f and \
                        t["stack"][nf].any():
                    nys, nxs = np.where(t["stack"][nf])
                    near_frames.append((nys.mean(), nxs.mean()))
            if near_frames:
                dists = [np.sqrt((cy - ny)**2 + (cx - nx)**2)
                         for ny, nx in near_frames]
                min_dist = min(dists)
                if min_dist > median_radius * max_distance_factor:
                    score -= 1

            # Check 3: boundary confidence (if frames available)
            if frames is not None and score < 1:
                from core.evaluation import boundary_confidence
                bc = boundary_confidence(frames[f], t["stack"][f])
                if bc < min_boundary_confidence:
                    score -= 1

            # Reject if multiple checks fail
            if score < 0:
                t["stack"][f] = False
                n_rejected += 1
                log.info("Rejected FP: track %d frame %d "
                         "(area=%.0f, score=%d)", ti, f, area, score)

    if n_rejected:
        log.info("Rejected %d false positive detections", n_rejected)
    return n_rejected


def reject_edge_artifact_tracks(tracks, frame_shape,
                                  edge_band_px=10,
                                  max_artifact_area=500,
                                  min_real_frames=3):
    """Drop tracks whose initial detection sits on the image boundary
    AND is small AND fails to grow into a real cell over its lifetime.

    Specifically a track is removed if ALL of these are true:
      - Its first real detection's bbox touches the image edge
        (within `edge_band_px` of any boundary).
      - That first detection's area is < max_artifact_area px.
      - The track never grows past `max_artifact_area` in any frame
        (so it's never been a real cell).

    These are typically mounting-medium reflections or edge-of-FOV
    artifacts that cpsam picks up but should not be tracked. Modifies
    tracks in-place; returns count removed.
    """
    H, W = frame_shape[-2:]
    before = len(tracks)
    surviving = []
    for t in tracks:
        stack = t["stack"]
        # Find first real detection
        first_idx = None
        for fi in range(len(stack)):
            if stack[fi].any():
                first_idx = fi
                break
        if first_idx is None:
            continue
        m0 = stack[first_idx]
        a0 = int(m0.sum())
        ys, xs = np.where(m0)
        on_edge = (ys.min() < edge_band_px or ys.max() >= H - edge_band_px
                   or xs.min() < edge_band_px or xs.max() >= W - edge_band_px)
        if not (on_edge and a0 < max_artifact_area):
            surviving.append(t)
            continue
        # Edge + small at start. Did it ever grow into a real cell?
        max_area_seen = max(int(stack[fi].sum())
                             for fi in range(len(stack))
                             if stack[fi].any())
        if max_area_seen >= max_artifact_area:
            surviving.append(t)
            continue
        # Drop it
        log.info("Dropped edge-artifact track: starts F%d at edge, "
                 "max area %d < %d",
                 first_idx, max_area_seen, max_artifact_area)
    tracks[:] = surviving
    removed = before - len(tracks)
    return removed


def reject_edge_sliver_detections(
        tracks, frame_shape,
        edge_band_px=40,
        min_aspect_ratio=5.0,
        max_sliver_thickness_px=50,
        min_density=0.5,
        per_track_threshold=0.5):
    """Drop per-frame detections that look like FoV-edge vignette
    artifacts — thin near-solid bars hugging an image boundary.

    cpsam will sometimes segment the dark illumination-edge gradient
    (the vignette from the microscope aperture / objective shadow)
    as a single elongated cell. The mask is a narrow vertical or
    horizontal strip pressed against an image edge. The existing
    `reject_edge_artifact_tracks` filter catches only small (<500 px)
    edge specks; vignette bars can be 30,000+ px so they pass through.

    Signature of the artifact:
      - touches an edge of the image (within `edge_band_px`)
      - aspect ratio of the bbox ≥ `min_aspect_ratio` (long / short)
      - the short dimension ≤ `max_sliver_thickness_px`
      - density (mask pixels / bbox area) ≥ `min_density` (≈ a solid
        bar; real elongated cells are spindly with density ≪ 0.5)

    A frame matching all four is zeroed in place. If a track has
    `per_track_threshold` or more of its active frames matching, the
    whole track is dropped. Otherwise only the offending frames are
    zeroed and the rest of the track survives.

    Modifies tracks in-place. Returns a dict
    {tracks_dropped, frames_zeroed}.
    """
    H, W = frame_shape[-2:]
    surviving = []
    frames_zeroed = 0
    for t in tracks:
        stack = t["stack"]
        sliver_frames = []
        active_frames = []
        for fi in range(len(stack)):
            m = stack[fi]
            if not m.any():
                continue
            active_frames.append(fi)
            ys, xs = np.where(m)
            bbox_h = int(ys.max() - ys.min() + 1)
            bbox_w = int(xs.max() - xs.min() + 1)
            short = min(bbox_h, bbox_w)
            long_ = max(bbox_h, bbox_w)
            ar = long_ / max(1, short)
            density = int(m.sum()) / max(1, bbox_h * bbox_w)
            on_edge = (xs.min() < edge_band_px
                       or xs.max() >= W - edge_band_px
                       or ys.min() < edge_band_px
                       or ys.max() >= H - edge_band_px)
            if (on_edge and ar >= min_aspect_ratio
                    and short <= max_sliver_thickness_px
                    and density >= min_density):
                sliver_frames.append(fi)
        if not sliver_frames:
            surviving.append(t)
            continue
        sliver_frac = len(sliver_frames) / max(1, len(active_frames))
        if sliver_frac >= per_track_threshold:
            log.info("Dropped edge-sliver track: %d/%d frames look "
                     "like vignette (AR≥%.1f, short≤%dpx, density≥%.2f)",
                     len(sliver_frames), len(active_frames),
                     min_aspect_ratio, max_sliver_thickness_px,
                     min_density)
            continue
        for fi in sliver_frames:
            stack[fi] = False
        frames_zeroed += len(sliver_frames)
        surviving.append(t)
    tracks_dropped = len(tracks) - len(surviving)
    tracks[:] = surviving
    if frames_zeroed:
        log.info("Zeroed %d edge-sliver frames in surviving tracks",
                 frames_zeroed)
    return {"tracks_dropped": tracks_dropped,
            "frames_zeroed": frames_zeroed}


def reject_static_edge_blobs(tracks, frame_shape,
                              edge_band_px=120,
                              min_edge_frac=0.5,
                              max_velocity_px=3.0,
                              min_shape_iou=0.85,
                              min_area_px=2000):
    """Drop tracks that look like static vignette / illumination
    artifacts hugging a field-of-view edge.

    A track is dropped when ALL hold:
      - Centroid lies within ``edge_band_px`` of an image edge in
        ≥ ``min_edge_frac`` of its active frames.
      - Median per-frame centroid displacement < ``max_velocity_px``
        (real cells move; vignettes don't).
      - Median consecutive-frame mask IoU > ``min_shape_iou``
        (the artifact's shape is stable; real cells deform).
      - Median area ≥ ``min_area_px`` — only catches large blobs;
        small near-edge tracks are already handled by
        ``reject_edge_artifact_tracks``.

    The test2 DMSO_busy cell 16 is the prototype: ~7000-9000 px,
    centroid (250, 108) within 110 px of the left edge of a 1024²
    frame, present 39/40 frames, centroid wandered ±5 px total
    across the recording.

    Modifies tracks in-place. Returns count removed.
    """
    H, W = frame_shape[-2:]
    surviving = []
    n_dropped = 0
    for t in tracks:
        stack = t.get("stack")
        if stack is None:
            surviving.append(t)
            continue
        # Compute per-frame centroid + edge-touch + area
        active = []
        for fi in range(len(stack)):
            m = stack[fi]
            if not m.any():
                continue
            ys, xs = np.where(m)
            cy, cx = float(ys.mean()), float(xs.mean())
            on_edge = (cy < edge_band_px or cy > H - edge_band_px
                        or cx < edge_band_px or cx > W - edge_band_px)
            active.append({
                "fi": fi, "cy": cy, "cx": cx, "on_edge": on_edge,
                "area": int(m.sum()), "mask": m})
        if len(active) < 3:
            surviving.append(t)
            continue
        edge_frac = sum(1 for a in active if a["on_edge"]) / len(active)
        median_area = float(np.median([a["area"] for a in active]))
        if (edge_frac < min_edge_frac
                or median_area < min_area_px):
            surviving.append(t)
            continue
        # Velocity: median consecutive-frame centroid displacement
        velocities = []
        for i in range(1, len(active)):
            if active[i]["fi"] - active[i - 1]["fi"] > 1:
                continue
            dy = active[i]["cy"] - active[i - 1]["cy"]
            dx = active[i]["cx"] - active[i - 1]["cx"]
            velocities.append((dy * dy + dx * dx) ** 0.5)
        if not velocities:
            surviving.append(t)
            continue
        median_vel = float(np.median(velocities))
        if median_vel >= max_velocity_px:
            surviving.append(t)
            continue
        # Shape stability: median consecutive-frame IoU
        ious = []
        for i in range(1, len(active)):
            if active[i]["fi"] - active[i - 1]["fi"] > 1:
                continue
            a, b = active[i - 1]["mask"], active[i]["mask"]
            inter = int((a & b).sum())
            union = int((a | b).sum())
            ious.append(inter / union if union > 0 else 0)
        median_iou = float(np.median(ious)) if ious else 0.0
        if median_iou < min_shape_iou:
            surviving.append(t)
            continue
        # Drop
        log.info("Dropped static-edge blob: median area %.0f, "
                 "edge_frac %.2f, velocity %.2f, shape_iou %.2f",
                 median_area, edge_frac, median_vel, median_iou)
        n_dropped += 1
    tracks[:] = surviving
    return n_dropped


def absorb_touching_fragments(tracks, n_frames, area_ratio=0.3,
                                dilate_px=2):
    """Per-frame, if track A's mask is much smaller than a touching
    track B's mask, absorb A's pixels into B and zero A in that frame.

    Catches cpsam over-segmentations that survive tracking + gap fill:
    one cell gets two track IDs in some frames (a big body and a
    tiny adjacent fragment) but separate identities in other frames.
    Pre-tracking ``merge_touching_splits`` only fixes the per-frame
    label stack at detection time — it can't fix a fragment that
    re-emerges after tracking + gap fill assigned IDs.

    Runs per frame. A small-track frame absorbed here just zeros
    that one frame in the small track's stack; the small track
    survives in other frames where it wasn't a fragment.
    """
    from scipy import ndimage as ndi
    n_absorbed = 0
    for fi in range(n_frames):
        present = [(ti, t) for ti, t in enumerate(tracks)
                    if t.get("stack") is not None
                    and ti < len(tracks) and t["stack"][fi].any()]
        if len(present) < 2:
            continue
        # Iterate smallest-first so chain absorptions resolve cleanly
        present.sort(key=lambda p: int(p[1]["stack"][fi].sum()))
        for small_idx, small_t in present:
            small_mask = small_t["stack"][fi]
            if not small_mask.any():
                continue   # already absorbed this pass
            small_area = int(small_mask.sum())
            dilated = ndi.binary_dilation(
                small_mask, iterations=dilate_px)
            # Find touching neighbours among OTHER present tracks
            best_neighbour = None
            best_area = 0
            for other_idx, other_t in present:
                if other_idx == small_idx:
                    continue
                other_mask = other_t["stack"][fi]
                if not other_mask.any():
                    continue
                if not (dilated & other_mask).any():
                    continue
                other_area = int(other_mask.sum())
                if other_area > best_area:
                    best_area = other_area
                    best_neighbour = other_t
            if best_neighbour is None or best_area == 0:
                continue
            ratio = small_area / best_area
            if ratio < area_ratio:
                # Absorb small into best_neighbour
                best_neighbour["stack"][fi] |= small_mask
                small_t["stack"][fi][:] = False
                n_absorbed += 1
    if n_absorbed:
        log.info("Absorbed %d per-frame touching fragments "
                 "into larger neighbours", n_absorbed)
    return n_absorbed


def postprocess_tracks(tracks, frames=None,
                        overlap_threshold=0.3, iou_threshold=0.05,
                        min_frames=3):
    """Full post-processing pipeline.

    1. Reject false positives (outlier detections)
    2. Drop small edge-artifact tracks (mounting reflections, specks)
    3. Drop edge-vignette sliver detections (FoV illumination edge
       segmented as a thin near-solid bar)
    4. Remove empty tracks

    Modifies tracks in-place. Returns summary dict.
    """
    n_fps = reject_false_positives(tracks, frames)

    n_edge = 0
    sliver = {"tracks_dropped": 0, "frames_zeroed": 0}
    n_static = 0
    if frames is not None and tracks:
        n_edge = reject_edge_artifact_tracks(tracks, frames.shape)
        sliver = reject_edge_sliver_detections(tracks, frames.shape)
        n_static = reject_static_edge_blobs(tracks, frames.shape)

    n_removed = remove_empty_tracks(tracks, min_frames)

    return {
        "fps_rejected": n_fps,
        "edge_artifacts_dropped": n_edge,
        "edge_slivers_dropped": sliver["tracks_dropped"],
        "edge_sliver_frames_zeroed": sliver["frames_zeroed"],
        "static_edge_blobs_dropped": n_static,
        "tracks_removed": n_removed,
        "tracks_remaining": len(tracks),
    }


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0
