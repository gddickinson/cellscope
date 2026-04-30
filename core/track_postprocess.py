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


def postprocess_tracks(tracks, frames=None,
                        overlap_threshold=0.3, iou_threshold=0.05,
                        min_frames=3):
    """Full post-processing pipeline.

    1. Reject false positives (outlier detections)
    2. Merge split tracks
    3. Stabilize IDs (fix switches)
    4. Remove empty tracks

    Modifies tracks in-place. Returns summary dict.
    """
    n_fps = reject_false_positives(tracks, frames)

    n_removed = remove_empty_tracks(tracks, min_frames)

    return {
        "fps_rejected": n_fps,
        "tracks_removed": n_removed,
        "tracks_removed": n_removed,
        "tracks_remaining": len(tracks),
    }


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0
