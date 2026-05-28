"""Tier 4: drop DIC tracks whose Cy5 signal indicates a false positive.

The pipeline detects cells from DIC, then annotates each track with
Cy5 features (mean / p75 / p95 / score) per frame. This module turns
those annotations into a filter that drops tracks unlikely to be
real cells (debris, dust, vignette artefacts that look cell-shaped
to cpsam).

Three strategies, in order of conservativeness:

  1. **conservative** (default): drop only tracks with NO Cy5 signal
     at all (mean < mean_threshold AND p95 < p95_threshold). Designed
     to keep faintly-labelled real cells (e.g. Pos14-KO style where
     all cells are dim).

  2. **threshold**: drop tracks below a fixed mean-score threshold
     (e.g. 0.15). Use when you've inspected the score distribution
     and know a sensible cut.

  3. **adaptive**: per-recording bimodal-aware. If the track-mean
     scores show a clear gap between two clusters (debris vs cells),
     cut at the gap midpoint. If the distribution is unimodal, do
     nothing (avoids dropping real cells in uniformly-faint
     recordings).

All three return the same shape: (kept_tracks, dropped_tracks, info).
"""
from __future__ import annotations

import numpy as np


def _track_score(t):
    """Best summary score for a track. Falls back to recomputing from
    per-frame `cy5_score` if the precomputed mean is absent."""
    s = t.get("cy5_mean_score")
    if s is not None:
        return float(s)
    arr = t.get("cy5_score")
    if arr is None:
        return 0.0
    valid = ~np.isnan(arr)
    return float(np.nanmean(arr)) if valid.any() else 0.0


def _track_p95(t):
    """95th-percentile of per-frame Cy5 score across the track lifetime.
    Captures cells that flicker bright even if dim on average."""
    arr = t.get("cy5_score")
    if arr is None:
        return 0.0
    valid = ~np.isnan(arr)
    return float(np.nanpercentile(arr, 95)) if valid.any() else 0.0


def conservative_filter(tracks, mean_threshold=0.05, p95_threshold=0.10):
    """Drop tracks with neither sustained nor flickering Cy5 signal.

    Returns (kept, dropped, info_dict).
    """
    kept, dropped = [], []
    for t in tracks:
        m = _track_score(t)
        p = _track_p95(t)
        if m < mean_threshold and p < p95_threshold:
            t = dict(t)
            t["drop_reason"] = (
                f"no Cy5 signal (mean={m:.3f}<{mean_threshold}, "
                f"p95={p:.3f}<{p95_threshold})")
            dropped.append(t)
        else:
            kept.append(t)
    return kept, dropped, {
        "mode": "conservative",
        "mean_threshold": mean_threshold,
        "p95_threshold": p95_threshold,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
    }


def threshold_filter(tracks, threshold=0.15):
    """Drop tracks with track_mean_score < threshold.

    Returns (kept, dropped, info_dict).
    """
    kept, dropped = [], []
    for t in tracks:
        m = _track_score(t)
        if m < threshold:
            t = dict(t)
            t["drop_reason"] = f"score {m:.3f} < {threshold}"
            dropped.append(t)
        else:
            kept.append(t)
    return kept, dropped, {
        "mode": "threshold",
        "threshold": threshold,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
    }


def adaptive_filter(tracks, min_gap=0.10, gap_ratio=2.0,
                     hard_floor=0.05):
    """Bimodal-aware filter: cut at the largest gap between sorted
    track scores when the distribution is clearly bimodal.

    Bimodal := largest sorted-score gap > `gap_ratio` × median gap
              AND > `min_gap` absolute.

    If unimodal, falls back to a conservative `hard_floor` (drop
    tracks below this absolute level only).

    Returns (kept, dropped, info_dict).
    """
    if len(tracks) < 5:
        return threshold_filter(tracks, threshold=hard_floor)

    scores = np.array([_track_score(t) for t in tracks])
    order = np.argsort(scores)
    sorted_scores = scores[order]
    gaps = np.diff(sorted_scores)
    if len(gaps) == 0:
        return tracks, [], {"mode": "adaptive", "result": "single track",
                             "n_kept": len(tracks), "n_dropped": 0}

    median_gap = float(np.median(gaps))
    max_gap_idx = int(np.argmax(gaps))
    max_gap = float(gaps[max_gap_idx])

    if max_gap > gap_ratio * max(median_gap, 1e-6) and max_gap > min_gap:
        threshold = float(
            (sorted_scores[max_gap_idx]
             + sorted_scores[max_gap_idx + 1]) / 2)
        info_extra = {
            "result": "bimodal",
            "threshold": threshold,
            "max_gap": max_gap,
            "median_gap": median_gap,
        }
    else:
        threshold = hard_floor
        info_extra = {
            "result": "unimodal — using hard floor",
            "threshold": threshold,
            "max_gap": max_gap,
            "median_gap": median_gap,
        }

    kept, dropped = [], []
    for t in tracks:
        m = _track_score(t)
        if m < threshold:
            t = dict(t)
            t["drop_reason"] = (
                f"score {m:.3f} < {info_extra['result']} "
                f"threshold {threshold:.3f}")
            dropped.append(t)
        else:
            kept.append(t)

    return kept, dropped, {
        "mode": "adaptive",
        **info_extra,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
    }


def conservative_strict_filter(tracks, mean_threshold=0.10,
                                 p95_threshold=0.20):
    """Tighter conservative cut. Drops tracks with weak Cy5 signal
    (mean < 0.10 AND p95 < 0.20). Use when conservative leaves too
    much debris."""
    return conservative_filter(tracks, mean_threshold, p95_threshold)


def adaptive_loose_filter(tracks, min_gap=0.20, gap_ratio=4.0,
                           hard_floor=0.10):
    """Less aggressive adaptive: requires a much clearer bimodal gap
    (4× median, 0.20 absolute) before splitting. Falls back to a
    higher hard floor (0.10) when unimodal."""
    return adaptive_filter(tracks, min_gap=min_gap,
                            gap_ratio=gap_ratio, hard_floor=hard_floor)


def _track_metric(t, key, agg="mean"):
    """Pull a multi-frame metric off a track, aggregating across
    valid frames. Returns 0.0 if the metric is absent or all-NaN."""
    arr = t.get(key)
    if arr is None:
        return 0.0
    valid = ~np.isnan(arr)
    if not valid.any():
        return 0.0
    a = arr[valid]
    if agg == "mean":
        return float(np.mean(a))
    if agg == "median":
        return float(np.median(a))
    if agg == "p75":
        return float(np.percentile(a, 75))
    return float(np.mean(a))


def multi_metric_filter(tracks,
                         score_threshold=0.15,
                         io_ratio_threshold=1.10,
                         frac_pos_threshold=0.15,
                         min_passing=2):
    """Composite filter: a track is REAL if at least `min_passing`
    of these THREE discriminative cellularity criteria hold
    (track-mean values across lifetime):

      * cy5_score        > score_threshold      (z-score brightness)
      * cy5_io_ratio     > io_ratio_threshold   (inside vs exterior ring)
      * cy5_fraction_positive > frac_pos_threshold (spatial coverage)

    NOTE: `cy5_inside_cv` was dropped from this filter — on the IC295
    dataset the entire Cy5 field has texture, so cv is uninformative
    (most tracks score 0.4-0.5 regardless of being a cell or debris).

    Default thresholds derived from the per-track distribution across
    all 19 IC295 recordings (356 tracks): each threshold is set near
    the lower-third of the distribution so genuine debris (io~1.0,
    fp~0.10) fails ≥2 metrics while real cells pass ≥2.

    Returns (kept, dropped, info).
    """
    kept, dropped = [], []
    thresholds = {
        "score": score_threshold,
        "io_ratio": io_ratio_threshold,
        "fraction_positive": frac_pos_threshold,
    }
    for t in tracks:
        scores = {
            "score": _track_score(t),
            "io_ratio": _track_metric(t, "cy5_io_ratio"),
            "fraction_positive": _track_metric(
                t, "cy5_fraction_positive"),
        }
        passed = sum(1 for k, v in scores.items() if v > thresholds[k])
        if passed >= min_passing:
            kept.append(t)
        else:
            t = dict(t)
            t["drop_reason"] = (
                f"only {passed}/3 metrics passed "
                f"(score={scores['score']:.2f}, "
                f"io={scores['io_ratio']:.2f}, "
                f"fp={scores['fraction_positive']:.2f})")
            dropped.append(t)
    return kept, dropped, {
        "mode": "multi_metric",
        "min_passing": min_passing,
        "thresholds": thresholds,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
    }


def composite_score_filter(tracks, threshold=1.0):
    """Continuous composite cellularity score, no hard thresholds.

    cellularity = (score/0.30) + max(0, io_ratio-1)/1.0
                  + (fraction_positive/0.40)

    Each component is normalised so a "typical real cell" contributes
    ~1.0; debris contributes ~0.0. Tracks with composite > `threshold`
    are kept. Default threshold=1.0 keeps anything that looks at least
    half-cell-like by any combination of the three signals.

    More forgiving than multi_metric: a strongly bright cell with
    weak edge contrast and low coverage can still pass if the score
    component alone exceeds the threshold.

    Returns (kept, dropped, info).
    """
    kept, dropped = [], []
    for t in tracks:
        score = _track_score(t)
        io = _track_metric(t, "cy5_io_ratio")
        fp = _track_metric(t, "cy5_fraction_positive")
        composite = score / 0.30 + max(0.0, io - 1.0) / 1.0 + fp / 0.40
        if composite >= threshold:
            kept.append(t)
        else:
            t = dict(t)
            t["drop_reason"] = (
                f"composite {composite:.2f} < {threshold} "
                f"(score={score:.2f}, io={io:.2f}, fp={fp:.2f})")
            dropped.append(t)
    return kept, dropped, {
        "mode": "composite_score",
        "threshold": threshold,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
    }


def temporal_stability_filter(tracks, min_corr=0.20,
                                min_lifetime=5):
    """Drop tracks whose per-frame Cy5 score looks like white noise
    (frame-to-frame autocorrelation below `min_corr`).

    Real cells have a stable cytoskeleton over minute timescales →
    Cy5(t) and Cy5(t+1) are correlated. Random debris detections in
    a noisy fluorescence frame have uncorrelated scores.

    Tracks shorter than `min_lifetime` are kept (insufficient data).
    """
    kept, dropped = [], []
    for t in tracks:
        arr = t.get("cy5_score")
        if arr is None:
            kept.append(t)
            continue
        valid_mask = ~np.isnan(arr)
        if valid_mask.sum() < min_lifetime:
            kept.append(t)
            continue
        a = arr[valid_mask]
        if len(a) < 2:
            kept.append(t)
            continue
        # lag-1 autocorrelation
        a_centered = a - a.mean()
        denom = float((a_centered ** 2).sum())
        if denom < 1e-9:
            corr = 0.0
        else:
            corr = float((a_centered[:-1] * a_centered[1:]).sum() / denom)
        if corr < min_corr:
            t = dict(t)
            t["drop_reason"] = (
                f"temporal noise: lag-1 autocorr {corr:.2f} < {min_corr}")
            dropped.append(t)
        else:
            kept.append(t)
    return kept, dropped, {
        "mode": "temporal_stability",
        "min_corr": min_corr,
        "min_lifetime": min_lifetime,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
    }


def consensus_filter(tracks):
    """Conservative-by-design composite: drop a track only if BOTH
    `multi_metric` and `composite_score` agree it should be dropped.

    Each filter individually drops ~67-68% of IC295 tracks but with
    different logic (3-of-3 voting vs continuous-sum). Disagreements
    between them are usually borderline cases — keeping them avoids
    false negatives. The intersection is the most defensible
    "definite false positive" set.

    Returns (kept, dropped, info).
    """
    # Determine drop status by re-evaluating each track inline rather
    # than relying on object identity (the inner filters return copies
    # of dropped tracks, which breaks id()-based set lookup).
    kept, dropped = [], []
    n_mm_drop = 0
    n_cs_drop = 0
    for t in tracks:
        score = _track_score(t)
        io = _track_metric(t, "cy5_io_ratio")
        fp = _track_metric(t, "cy5_fraction_positive")

        # multi_metric (defaults: ≥2/3 of score>0.15, io>1.10, fp>0.15)
        passed = ((score > 0.15) + (io > 1.10) + (fp > 0.15))
        mm_drop = passed < 2
        if mm_drop:
            n_mm_drop += 1

        # composite_score (default: composite ≥ 1.0)
        composite = score / 0.30 + max(0.0, io - 1.0) / 1.0 + fp / 0.40
        cs_drop = composite < 1.0
        if cs_drop:
            n_cs_drop += 1

        if mm_drop and cs_drop:
            t = dict(t)
            t["drop_reason"] = (
                f"consensus drop: multi_metric ({passed}/3) AND "
                f"composite ({composite:.2f} < 1.0)")
            dropped.append(t)
        else:
            kept.append(t)

    return kept, dropped, {
        "mode": "consensus",
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "n_multi_metric_only_drop": n_mm_drop,
        "n_composite_only_drop": n_cs_drop,
    }


def _track_motion_and_shape(t):
    """Return (mean_velocity_px_per_frame, median_consecutive_iou).

    Velocity is the mean per-frame centroid displacement. Shape IoU is
    the median of consecutive-frame mask overlaps. Used to detect
    static, shape-stable phantom tracks (vignette artefacts / stuck
    debris that survive any lifetime threshold but actively fail Cy5).

    Returns (0.0, 1.0) for tracks with <2 active frames (no motion
    measurable) — treated as static by callers.
    """
    stack = t.get("stack")
    if stack is None:
        return (0.0, 1.0)
    active_frames = []
    centroids = []
    for fi in range(stack.shape[0]):
        m = stack[fi].astype(bool)
        if not m.any():
            continue
        ys, xs = np.where(m)
        centroids.append((fi, float(ys.mean()), float(xs.mean())))
        active_frames.append((fi, m))
    if len(centroids) < 2:
        return (0.0, 1.0)
    ys = np.array([c[1] for c in centroids])
    xs = np.array([c[2] for c in centroids])
    per_step = np.sqrt(np.diff(ys) ** 2 + np.diff(xs) ** 2)
    velocity = float(per_step.mean()) if len(per_step) else 0.0
    ious = []
    for i in range(len(active_frames) - 1):
        fi1, m1 = active_frames[i]
        fi2, m2 = active_frames[i + 1]
        if fi2 - fi1 != 1:
            continue
        inter = int(np.logical_and(m1, m2).sum())
        union = int(np.logical_or(m1, m2).sum())
        if union > 0:
            ious.append(inter / union)
    shape_iou = float(np.median(ious)) if ious else 1.0
    return (velocity, shape_iou)


def persistence_guard_filter(tracks, min_lifetime=35,
                              score_threshold=0.15,
                              io_ratio_threshold=1.10,
                              frac_pos_threshold=0.15,
                              min_passing=2,
                              static_velocity_px=3.0,
                              static_shape_iou=0.85):
    """Three-stage filter to recover weak-Cy5 real cells without
    keeping persistent vignette / debris phantoms:

      1. multi_metric pass (≥ `min_passing` of 3 Cy5 thresholds) →
         KEEP. Real cells with informative SiR-actin signal.
      2. mm-fail AND lifetime < `min_lifetime` → DROP.
         Short transient detection — not enough evidence.
      3. mm-fail AND lifetime ≥ `min_lifetime`:
            - If track is STATIC (mean per-frame centroid displacement
              < `static_velocity_px` AND median consecutive-frame mask
              IoU > `static_shape_iou`) → DROP. Phantom signature:
              vignette artefacts and stuck debris are static AND
              shape-stable. Real cells either move or deform.
            - Otherwise → KEEP. Long-lived moving/deforming cell with
              weak fluorescence (GOF/OT/DMSO conditions).

    Calibration: 13-recording GT audit (989 GT cells). multi_metric
    alone drops 70 real cells (-7.1% recall) — overly strict in
    weak-Cy5 conditions. Pure persistence_guard at ml=35 recovers
    +97 cells but adds +102 phantoms (notably 1 in Pos51_Y1 that
    survives any lifetime threshold). Adding the static-track gate
    drops the phantom without losing any GT-matched real cells with
    measurable motion.

    Returns (kept, dropped, info).
    """
    kept, dropped = [], []
    n_long_kept = 0
    n_short_pass = 0
    n_short_drop = 0
    n_static_drop = 0
    for t in tracks:
        stack = t.get("stack")
        if stack is not None:
            n_valid = int((stack.sum(axis=tuple(range(1, stack.ndim)))
                            > 0).sum())
        else:
            n_valid = 0
        score = _track_score(t)
        io = _track_metric(t, "cy5_io_ratio")
        fp = _track_metric(t, "cy5_fraction_positive")
        passed = ((score > score_threshold)
                   + (io > io_ratio_threshold)
                   + (fp > frac_pos_threshold))
        # Always keep tracks that pass multi-metric Cy5 (real cell
        # signature)
        if passed >= min_passing:
            kept.append(t)
            n_short_pass += (1 if n_valid < min_lifetime else 0)
            n_long_kept += (1 if n_valid >= min_lifetime else 0)
            continue
        # mm < min_passing → need other evidence
        if n_valid < min_lifetime:
            t = dict(t)
            t["drop_reason"] = (
                f"short ({n_valid} frames) and only {passed}/3 metrics: "
                f"score={score:.2f}, io={io:.2f}, fp={fp:.2f}")
            dropped.append(t)
            n_short_drop += 1
            continue
        # Long-lived but mm<min_passing → check motion + shape
        # signature. Phantoms (vignette / stuck debris) have BOTH
        # low velocity AND high frame-to-frame shape stability — real
        # cells either move OR deform.
        velocity, shape_iou = _track_motion_and_shape(t)
        is_static = (velocity < static_velocity_px
                      and shape_iou > static_shape_iou)
        if is_static:
            t = dict(t)
            t["drop_reason"] = (
                f"long ({n_valid} frames) but static (vel={velocity:.2f}"
                f"<{static_velocity_px}, shape_iou={shape_iou:.2f}"
                f">{static_shape_iou}) and only {passed}/3 Cy5 metrics: "
                f"score={score:.2f}, io={io:.2f}, fp={fp:.2f}")
            dropped.append(t)
            n_static_drop += 1
            continue
        # Long-lived AND (moves OR deforms) → real cell, keep
        kept.append(t)
        n_long_kept += 1
    return kept, dropped, {
        "mode": "persistence_guard",
        "min_lifetime": min_lifetime,
        "static_velocity_px": static_velocity_px,
        "static_shape_iou": static_shape_iou,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "n_long_lived_kept": n_long_kept,
        "n_short_passed_multimetric": n_short_pass,
        "n_short_dropped": n_short_drop,
        "n_static_dropped": n_static_drop,
    }


def apply_cy5_filter(tracks, mode="off", threshold=0.15,
                      mean_threshold=0.05, p95_threshold=0.10,
                      pg_min_lifetime=None,
                      pg_static_velocity_px=None,
                      pg_static_shape_iou=None):
    """Dispatch to the requested strategy.

    mode ∈ {"off", "conservative", "conservative_strict",
            "threshold", "adaptive", "adaptive_loose",
            "multi_metric", "temporal_stability",
            "persistence_guard"}.
    Returns (kept, dropped, info). When mode='off' returns
    (tracks, [], {"mode": "off"}).

    For mode='persistence_guard', the three pg_* args override the
    filter's defaults (35, 3.0, 0.85) when provided. None → use the
    defaults baked into persistence_guard_filter.
    """
    if mode == "off" or not tracks:
        return tracks, [], {"mode": "off",
                             "n_kept": len(tracks), "n_dropped": 0}
    if mode == "conservative":
        return conservative_filter(tracks, mean_threshold, p95_threshold)
    if mode == "conservative_strict":
        return conservative_strict_filter(tracks)
    if mode == "threshold":
        return threshold_filter(tracks, threshold)
    if mode == "adaptive":
        return adaptive_filter(tracks)
    if mode == "adaptive_loose":
        return adaptive_loose_filter(tracks)
    if mode == "multi_metric":
        return multi_metric_filter(tracks)
    if mode == "composite_score":
        # composite_score's natural threshold is around 1.0, not the
        # 0.15 used for raw-score thresholding. Use composite-style
        # threshold from the param if explicitly raised, else 1.0.
        comp_thr = threshold if threshold > 0.5 else 1.0
        return composite_score_filter(tracks, threshold=comp_thr)
    if mode == "temporal_stability":
        return temporal_stability_filter(tracks)
    if mode == "consensus":
        return consensus_filter(tracks)
    if mode == "persistence_guard":
        kwargs = {}
        if pg_min_lifetime is not None:
            kwargs["min_lifetime"] = pg_min_lifetime
        if pg_static_velocity_px is not None:
            kwargs["static_velocity_px"] = pg_static_velocity_px
        if pg_static_shape_iou is not None:
            kwargs["static_shape_iou"] = pg_static_shape_iou
        return persistence_guard_filter(tracks, **kwargs)
    raise ValueError(f"Unknown mode {mode!r}")


def rebuild_label_stack(tracks, shape, dtype=np.int32):
    """Re-build (N, H, W) int32 label stack from a list of tracks
    after filtering. Track IDs become 1..len(tracks).

    Defensive against mixed-resolution track stacks: if a track's
    per-frame mask doesn't match the target (H, W) — e.g. a
    downsampled stack that unified_detection's upscale pass missed —
    it's nearest-neighbour resized to fit rather than raising a
    broadcast error.
    """
    H, W = shape[1], shape[2]
    out = np.zeros(shape, dtype=dtype)
    for tid, t in enumerate(tracks, start=1):
        s = t.get("stack")
        if s is None:
            continue
        for i in range(shape[0]):
            si = s[i]
            if si.shape != (H, W):
                import cv2
                si = cv2.resize(
                    si.astype(np.uint8), (W, H),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
            out[i][si & (out[i] == 0)] = tid
    return out
