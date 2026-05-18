"""Multi-channel pipeline orchestrator.

Wraps `core.hybrid_cpsam_multi.detect_hybrid_cpsam_multi` (which runs
on DIC frames) with a per-frame Cy5 presence filter. Tracks that have
strong actin signal across most frames are kept; debris tracks (cell-
shaped DIC features without actin) are dropped.

Track-level filtering, not just per-frame:
  * For each frame in a track's timeline, compute the Cy5 presence
    score for that mask.
  * If the **median** across-frame score for a track ≥ `min_track_score`,
    keep the track. Else drop.
  * Single-frame failures (e.g. SiR-actin briefly out-of-focus) don't
    kill an otherwise-good track.

Per-track Cy5 features (mean / median / p75 / p95 / total) are also
returned so downstream analytics + Hungarian-cost extensions can use
them.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def detect_hybrid_multichannel(dic_frames, fluo_frames,
                                min_track_score=0.30,
                                expand_px=30,
                                use_cy5_recovery=False,
                                cy5_recovery_use_tta=True,
                                progress_fn=None,
                                **dic_pipeline_kwargs):
    """Multi-channel detect+track pipeline.

    Args:
      dic_frames:  (N, H, W) uint8 — DIC channel preprocessed
      fluo_frames: (N, H, W) uint8 — Cy5/actin channel preprocessed
      min_track_score: median Cy5 score required to keep a track
      expand_px: Cy5 background ring size for per-mask scoring
      use_cy5_recovery: enable the per-frame "Cy5 fail-safe" — for
        each Cy5+ region not covered by any DIC mask, crop and
        re-run cpsam to recover missed cells. Adds latency
        proportional to the number of missed cells per frame.
      cy5_recovery_use_tta: if recovery is on, use cpsam test-time
        augmentation on the small crop. Cheap on a 200×200 crop
        (~5 s vs ~75 s on the full frame).
      progress_fn: optional callback(msg, pct)
      **dic_pipeline_kwargs: passed to detect_hybrid_cpsam_multi
        (use_tta, use_gap_fill, use_deepsea, etc.)

    Returns dict with keys:
      labels:    (N, H, W) int32 — tracked cell IDs after Cy5 filter
      tracks:    list of kept track dicts (with cy5_features added)
      tracks_dropped: list of dropped track dicts (debris)
      raw_labels: pre-filter (N, H, W) int32 from DIC pipeline
      raw_tracks: pre-filter list (for diagnostic)
      n_cy5_recovered: total recovered cells across all frames
    """
    assert dic_frames.shape == fluo_frames.shape, (
        f"channel shapes don't match: dic={dic_frames.shape} "
        f"fluo={fluo_frames.shape}")

    from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi
    from core.multichannel import (cy5_presence_score,
                                    per_cell_cy5_features)
    from core.cy5_fallbacks import recover_missed_cells_via_dic_crop

    if progress_fn:
        progress_fn("DIC detection (hybrid_cpsam_multi)", 0)
    dic_result = detect_hybrid_cpsam_multi(
        dic_frames, progress_fn=progress_fn,
        **dic_pipeline_kwargs)

    raw_labels = dic_result["labels"]
    raw_tracks = dic_result["tracks"]
    n_frames = len(dic_frames)
    log.info("DIC stage: %d tracks before Cy5 filter", len(raw_tracks))

    # Optional fail-safe: per-frame Cy5 recovery for cells the DIC
    # pipeline missed entirely. This produces NEW masks that aren't
    # tracked yet — they enter the per-track Cy5 filter loop below
    # as orphan single-frame "tracks" with no temporal continuity.
    # Hungarian re-tracking would be cleaner but adds complexity;
    # for now, recovered cells appear as their own short tracks.
    n_cy5_recovered = 0
    if use_cy5_recovery:
        if progress_fn:
            progress_fn("Cy5 fail-safe: scanning for missed cells", 88)
        from cellpose import models
        cpsam_recover = models.CellposeModel(gpu=True)
        for fi in range(n_frames):
            new_labels, n_added, _ = recover_missed_cells_via_dic_crop(
                dic_frames[fi], raw_labels[fi], fluo_frames[fi],
                cpsam_recover, use_tta=cy5_recovery_use_tta)
            if n_added:
                raw_labels[fi] = new_labels
                n_cy5_recovered += n_added
        log.info("Cy5 recovery added %d new cell-frame detections",
                 n_cy5_recovered)
        if n_cy5_recovered:
            # Re-track to incorporate the recovered cells as proper
            # tracks rather than orphan single-frame detections.
            from core.multi_cell import track_all_cells
            if progress_fn:
                progress_fn(
                    f"Re-tracking after Cy5 recovery "
                    f"(+{n_cy5_recovered} detections)", 90)
            raw_tracks = track_all_cells(
                raw_labels,
                min_area_px=200,
                max_hop_px=150,
                spawn_new_tracks=True,
                min_track_length=3,
                frames=dic_frames)
            log.info("After re-track: %d tracks total", len(raw_tracks))

    if progress_fn:
        progress_fn("Cy5 presence scoring per track", 92)

    kept = []
    dropped = []
    for tid, track in enumerate(raw_tracks):
        stack = track["stack"]  # (N, H, W) bool
        per_frame_scores = []
        per_frame_features = []
        for fi in range(n_frames):
            mask = stack[fi]
            if not mask.any():
                continue
            score = cy5_presence_score(mask, fluo_frames[fi],
                                        expand_px=expand_px)
            feats = per_cell_cy5_features(mask, fluo_frames[fi])
            per_frame_scores.append((fi, score))
            per_frame_features.append((fi, feats))
        if not per_frame_scores:
            dropped.append(track)
            continue
        scores = np.array([s for _, s in per_frame_scores])
        median_score = float(np.median(scores))
        track["cy5_median_score"] = median_score
        track["cy5_per_frame_scores"] = per_frame_scores
        track["cy5_per_frame_features"] = per_frame_features
        if median_score >= min_track_score:
            kept.append(track)
        else:
            dropped.append(track)

    log.info("Cy5 filter: kept %d, dropped %d (median score < %.2f)",
             len(kept), len(dropped), min_track_score)
    if progress_fn:
        progress_fn(f"Cy5 filter kept {len(kept)}/{len(raw_tracks)}",
                    98)

    # Build new label stack with only the kept tracks (re-numbered)
    out_labels = np.zeros_like(raw_labels)
    for new_id, track in enumerate(kept, start=1):
        for fi in range(n_frames):
            sl = track["stack"][fi]
            if sl.any():
                out_labels[fi][sl & (out_labels[fi] == 0)] = new_id

    return {
        "labels": out_labels,
        "tracks": kept,
        "tracks_dropped": dropped,
        "raw_labels": raw_labels,
        "raw_tracks": raw_tracks,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "n_cy5_recovered": n_cy5_recovered,
    }
