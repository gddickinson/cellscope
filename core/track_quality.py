"""Per-track quality scoring for the tracking GUI.

Combines three signals into a single 0-1 composite so users can pick
"good" cells without trial and error:

* ``frames_score`` — fraction of recording frames the track is present.
  Short tracks are usually fragments of a longer cell or noise.
* ``area_score`` — 1 - (area_std / area_mean), clipped to [0, 1].
  Wild area swings usually mean intermittent over/under-segmentation.
* ``path_score`` — total_distance_um / ``path_target_um``, clipped to
  [0, 1]. A proxy for "did the cell actually move" — distinguishes
  real migrating cells from debris/noise. Default target 50 µm.

Composite weights (0.5 / 0.3 / 0.2) favour completeness over the
other two; tweak by passing ``weights=`` if a specific use case
needs to weight differently.

`compute_track_quality()` works whether the track has been analysed
(uses precise area/path values) or not (falls back to a frames-only
score). Returned dict is JSON-safe.
"""
from __future__ import annotations

import numpy as np


_DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)


def compute_track_quality(
    track,
    n_total_frames,
    analysis_result=None,
    path_target_um=50.0,
    weights=_DEFAULT_WEIGHTS,
):
    """Compute a 0-1 quality score for one track.

    Args:
      track: dict with at least ``stack`` (T, H, W bool/int).
      n_total_frames: int — total frames in the recording.
      analysis_result: optional dict from analyze_recording(); if
        provided we use real area summary + total_distance, otherwise
        we fall back to area-from-mask and skip the path component.
      path_target_um: distance at which path_score saturates to 1.
      weights: (frames, area, path) — must sum to 1.

    Returns:
      dict with keys frames_score, area_score, path_score, composite,
      label ("good"/"ok"/"poor"). Components missing data return None
      and are excluded from the weighted average (weights renormalised).
    """
    stack = track["stack"]
    n_total = max(int(n_total_frames), 1)
    frames_active = int(stack.any(axis=(1, 2)).sum())
    frames_score = frames_active / n_total

    area_score = None
    path_score = None

    if analysis_result is not None:
        ss = analysis_result.get("shape_summary", {})
        area = ss.get("area_um2") or {}
        mean = area.get("mean")
        std = area.get("std")
        if mean and mean > 0 and std is not None:
            cv = float(std) / float(mean)
            area_score = float(np.clip(1.0 - cv, 0.0, 1.0))
        td = analysis_result.get("total_distance")
        if td is not None:
            path_score = float(np.clip(float(td) / path_target_um,
                                       0.0, 1.0))
    else:
        # Cheap fallback: area std from mask pixel counts.
        per_frame = stack.any(axis=(1, 2))
        if per_frame.sum() >= 2:
            areas = stack[per_frame].sum(axis=(1, 2)).astype(float)
            if areas.mean() > 0:
                cv = float(areas.std() / areas.mean())
                area_score = float(np.clip(1.0 - cv, 0.0, 1.0))

    # Weighted average over whichever components are present.
    parts = [(frames_score, weights[0])]
    if area_score is not None:
        parts.append((area_score, weights[1]))
    if path_score is not None:
        parts.append((path_score, weights[2]))
    total_w = sum(w for _, w in parts)
    composite = sum(s * w for s, w in parts) / total_w if total_w else 0.0

    if composite >= 0.7:
        label = "good"
    elif composite >= 0.4:
        label = "ok"
    else:
        label = "poor"

    return {
        "frames_score": float(frames_score),
        "area_score": area_score,
        "path_score": path_score,
        "composite": float(composite),
        "label": label,
        "frames_active": frames_active,
        "frames_total": n_total,
    }


def quality_color(label):
    """Map a quality label → (r, g, b, a) for QTableWidgetItem.setBackground.

    Uses pale tints so foreground text stays readable.
    """
    return {
        "good": (200, 240, 200, 255),  # pale green
        "ok":   (250, 235, 180, 255),  # pale amber
        "poor": (250, 200, 200, 255),  # pale red
    }.get(label, (255, 255, 255, 255))
