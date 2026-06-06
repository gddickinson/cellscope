"""Single canonical detection entry point — used by both the GUI and
the GT-evaluation script so they always produce the same output.

The full chain:
  1. (multichannel only) measure + apply DIC↔Cy5 alignment shift
  2. resolve downsample factor from recording shape
  3. downsample DIC + Cy5 stacks if factor > 1
  4. convert physical-unit thresholds (µm², µm/min) to per-recording px
  5. auto-select cpsam_dic vs raw cpsam by sampled cell density
  6. run hybrid_dic_multi or hybrid_cpsam_multi (cpsam Cy5 fusion ON if
     a fluorescence channel is present)
  7. (multichannel only) annotate per-cell Cy5 metrics, then apply the
     Cy5 multi-metric filter
  8. (downsample only) upscale labels + fusion_source_stack + per-track
     stacks back to original resolution

`detect_recording(...)` returns the same result dict shape produced
by the pipeline functions, plus an `auto` sub-dict recording the
decisions made (alignment shift, downsample factor + reason,
pipeline choice + sample counts).

Any caller can override the auto choices by passing explicit kwargs
(`downsample=2`, `pipeline_kind="cpsam_dic"`, etc.).
"""
from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger(__name__)


def detect_recording(dic_frames, cy5_frames=None,
                      um_per_px=None, time_interval_min=None,
                      downsample="auto",
                      downsample_small_px=None,
                      downsample_large_px=None,
                      align_channels=True,
                      pipeline_kind="auto",
                      run_cy5_filter=True,
                      cy5_filter_mode="persistence_guard",
                      cy5_filter_threshold=0.15,
                      cy5_pg_min_lifetime=None,
                      cy5_pg_static_velocity_px=None,
                      cy5_pg_static_shape_iou=None,
                      use_mirror_pad=None,
                      use_deepsea=None,
                      use_gap_fill=None,
                      use_sam2_video_gap_fill=None,
                      gap_fill_crop=None,
                      gap_fill_augment=None,
                      max_gap_frames=None,
                      min_track_length=None,
                      max_hop_px=None,
                      use_tta=None,
                      use_cpsam_cy5_union=None,
                      use_fallback=None,
                      use_preprocess=None,
                      use_retry=None,
                      cy5_fusion_jaccard_thresh=None,
                      cy5_fusion_max_overlap_frac=None,
                      cy5_fusion_augment_cpsam=None,
                      use_bfloat16=None,
                      progress_fn=None):
    """Run the canonical end-to-end detection.

    Args mostly self-explanatory. Returns a dict containing:
      masks, labels, tracks, fusion_source_stack (if Cy5 fusion ran),
      auto: {alignment, downsample_factor, downsample_reason,
             pipeline_kind, pipeline_reason, model_path, ...},
      n_cy5_fusion_added, cy5_filter_kept, cy5_filter_dropped, …
    """
    from core.pipeline_defaults import (
        DEFAULTS, resolve_downsample, select_pipeline_for_recording,
        resolve_dic_model_path,
        DOWNSAMPLE_SMALL_PX, DOWNSAMPLE_LARGE_PX,
    )

    has_cy5 = cy5_frames is not None
    original_shape = dic_frames.shape[1:]
    auto = {
        "alignment": None,
        "downsample_factor": 1,
        "downsample_reason": "not applied",
        "pipeline_kind": pipeline_kind,
        "pipeline_reason": None,
        "model_path": None,
    }

    def _emit(msg, pct):
        if progress_fn:
            progress_fn(msg, pct)

    # ---- 1. Channel alignment ----
    aligned_cy5_orig_scale = None
    if has_cy5 and align_channels:
        from core.channel_alignment import align_cy5_to_dic
        _emit("Measuring DIC↔Cy5 alignment", 2)
        cy5_frames, alignment_info = align_cy5_to_dic(
            dic_frames, cy5_frames, verbose=True)
        auto["alignment"] = alignment_info
        log.info(
            "alignment dy=%+.2f dx=%+.2f px (n=%d, applied=%s)",
            alignment_info.get("median_dy_px", 0),
            alignment_info.get("median_dx_px", 0),
            alignment_info.get("n_pairs", 0),
            alignment_info.get("shift_applied", False))
        # Snapshot aligned cy5 BEFORE downsampling so callers (like
        # the fusion diagnostic renderer) can use it at full res.
        aligned_cy5_orig_scale = cy5_frames
    elif has_cy5:
        aligned_cy5_orig_scale = cy5_frames

    # ---- 2/3. Downsample ----
    small_px = (downsample_small_px if downsample_small_px is not None
                else DOWNSAMPLE_SMALL_PX)
    large_px = (downsample_large_px if downsample_large_px is not None
                else DOWNSAMPLE_LARGE_PX)
    factor, ds_reason = resolve_downsample(
        downsample, original_shape,
        small_px=small_px, large_px=large_px)
    auto["downsample_factor"] = factor
    auto["downsample_reason"] = ds_reason
    log.info("downsample resolved: factor=%d (%s)", factor, ds_reason)

    if factor > 1:
        import cv2
        H, W = original_shape
        new_h, new_w = H // factor, W // factor
        _emit(f"Downsampling {factor}×", 5)
        dic_small = np.empty((len(dic_frames), new_h, new_w),
                              dtype=np.uint8)
        for i in range(len(dic_frames)):
            dic_small[i] = cv2.resize(
                dic_frames[i], (new_w, new_h),
                interpolation=cv2.INTER_AREA)
        dic_frames = dic_small
        if has_cy5:
            cy5_small = np.empty((len(cy5_frames), new_h, new_w),
                                  dtype=np.uint8)
            for i in range(len(cy5_frames)):
                cy5_small[i] = cv2.resize(
                    cy5_frames[i], (new_w, new_h),
                    interpolation=cv2.INTER_AREA)
            cy5_frames = cy5_small

    # ---- 4. Physical-unit → px thresholds (accounting for ds) ----
    effective_um_per_px = (um_per_px or 1.0) * factor
    px_thresholds = DEFAULTS.pixel_thresholds(
        um_per_px=effective_um_per_px,
        time_interval_min=time_interval_min)

    # ---- 5. Auto-select pipeline kind ----
    if pipeline_kind == "auto":
        _emit("Probing for cell density (auto-select pipeline)", 8)
        pipeline_kind, model_path, sel_info = \
            select_pipeline_for_recording(
                dic_frames,
                min_area_px=px_thresholds["min_area_px"],
                verbose=True)
        auto["pipeline_kind"] = pipeline_kind
        auto["pipeline_reason"] = sel_info["reason"]
        auto["pipeline_sample_counts"] = sel_info.get(
            "cell_counts_at_sample")
        auto["model_path"] = model_path
    else:
        if pipeline_kind == "cpsam_dic":
            auto["model_path"] = resolve_dic_model_path()

    # ---- 6. Run detection ----
    use_cy5_fusion = has_cy5
    eff_max_hop = max_hop_px if max_hop_px is not None else 150
    if pipeline_kind == "cpsam_dic":
        from core.hybrid_dic import detect_hybrid_dic_multi
        _emit(f"Detecting with cpsam_dic", 10)
        result = detect_hybrid_dic_multi(
            dic_frames,
            progress_fn=_emit,
            model_path=auto["model_path"],
            min_area_px=px_thresholds["min_area_px"],
            use_preprocess=use_preprocess,
            use_deepsea=use_deepsea,
            use_retry=use_retry,
            use_tta=use_tta,
            min_track_length=min_track_length,
            max_gap_frames=max_gap_frames,
            max_hop_px=eff_max_hop,
            cy5_frames=cy5_frames,
            use_cy5_fusion=use_cy5_fusion,
            cy5_fusion_augment=cy5_fusion_augment_cpsam,
            cy5_fusion_jaccard_thresh=cy5_fusion_jaccard_thresh,
            gap_fill_crop=gap_fill_crop,
            gap_fill_augment=gap_fill_augment,
            use_bfloat16=use_bfloat16)
    else:
        from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi
        _emit("Detecting with raw cpsam", 10)
        # Per-call use_mirror_pad override falls back to DEFAULTS when
        # not specified. Used by run_pipeline_on_gt_recording.py to
        # honor sidecar JSON's "use_mirror_pad" key (e.g. Pos39_OT
        # uses "off" because mirror-padding merges its dividing cell
        # pair into a single mass and the second daughter is lost).
        # Each kwarg falls back to DEFAULTS when None — lets callers
        # (GUI, scripts) override individually without having to know
        # every default.
        effective_mirror_pad = (use_mirror_pad
                                 if use_mirror_pad is not None
                                 else DEFAULTS.use_mirror_pad)
        eff_deepsea = (use_deepsea if use_deepsea is not None
                        else DEFAULTS.use_deepsea)
        eff_gap_fill = (use_gap_fill if use_gap_fill is not None
                         else DEFAULTS.use_gap_fill)
        eff_tta = use_tta if use_tta is not None else DEFAULTS.use_tta
        eff_cpsam_cy5 = (use_cpsam_cy5_union
                          if use_cpsam_cy5_union is not None
                          else DEFAULTS.use_cpsam_cy5_union)
        eff_fallback = (use_fallback if use_fallback is not None
                         else False)
        result = detect_hybrid_cpsam_multi(
            dic_frames,
            progress_fn=_emit,
            min_area_px=px_thresholds["min_area_px"],
            use_fallback=eff_fallback,
            use_deepsea=eff_deepsea,
            use_gap_fill=eff_gap_fill,
            use_sam2_video_gap_fill=use_sam2_video_gap_fill,
            max_gap_frames=max_gap_frames,
            min_track_length=(min_track_length if min_track_length
                               is not None else DEFAULTS.min_track_length),
            max_hop_px=eff_max_hop,
            use_tta=eff_tta,
            use_mirror_pad=effective_mirror_pad,
            use_cpsam_cy5_union=eff_cpsam_cy5,
            cy5_frames=cy5_frames,
            recover_with_cy5=False,
            use_cy5_fusion=use_cy5_fusion,
            gap_fill_crop=gap_fill_crop,
            gap_fill_augment=gap_fill_augment,
            use_bfloat16=use_bfloat16)

    # ---- 7. Cy5 multi-metric filter (multichannel only) ----
    if has_cy5 and run_cy5_filter and result.get("tracks"):
        _emit("Cy5 annotation + multi-metric filter", 92)
        from core.multichannel import (
            per_cell_cy5_features, cy5_presence_score,
            cy5_inside_outside_ratio, cy5_fraction_positive)
        from core.cy5_filter import (apply_cy5_filter,
                                       rebuild_label_stack)
        n = len(cy5_frames)
        for t in result["tracks"]:
            stack = t.get("stack")
            if stack is None:
                continue
            mean = np.full(n, np.nan, dtype=np.float32)
            score = np.full(n, np.nan, dtype=np.float32)
            ior = np.full(n, np.nan, dtype=np.float32)
            fp = np.full(n, np.nan, dtype=np.float32)
            for i in range(n):
                m = stack[i].astype(bool)
                if not m.any():
                    continue
                mean[i] = per_cell_cy5_features(m, cy5_frames[i])["mean"]
                score[i] = cy5_presence_score(m, cy5_frames[i])
                ior[i] = cy5_inside_outside_ratio(m, cy5_frames[i])
                fp[i] = cy5_fraction_positive(m, cy5_frames[i])
            t["cy5_mean"] = mean
            t["cy5_score"] = score
            t["cy5_io_ratio"] = ior
            t["cy5_fraction_positive"] = fp
            valid = ~np.isnan(score)
            t["cy5_mean_score"] = (float(np.nanmean(score))
                                    if valid.any() else 0.0)
        kept, dropped, info = apply_cy5_filter(
            result["tracks"], mode=cy5_filter_mode,
            threshold=cy5_filter_threshold,
            pg_min_lifetime=cy5_pg_min_lifetime,
            pg_static_velocity_px=cy5_pg_static_velocity_px,
            pg_static_shape_iou=cy5_pg_static_shape_iou)
        result["tracks_raw"] = result["tracks"]
        result["tracks"] = kept
        result["tracks_dropped"] = dropped
        result["cy5_filter_info"] = info
        result["labels"] = rebuild_label_stack(
            kept, dic_frames.shape)
        result["masks"] = result["labels"] > 0

    # ---- 8. Upscale labels back to original resolution ----
    if factor > 1:
        import cv2
        H, W = original_shape

        def _upscale(arr_stack):
            if arr_stack is None:
                return None
            out = np.empty((len(arr_stack), H, W),
                            dtype=arr_stack.dtype)
            for i in range(len(arr_stack)):
                out[i] = cv2.resize(
                    arr_stack[i], (W, H),
                    interpolation=cv2.INTER_NEAREST)
            return out

        _emit(f"Upscaling labels back to {original_shape}", 97)
        result["labels"] = _upscale(result["labels"])
        result["masks"] = result["labels"] > 0
        if result.get("fusion_source_stack") is not None:
            result["fusion_source_stack"] = _upscale(
                result["fusion_source_stack"])
        # Upscale every track's stack to the original resolution so
        # downstream code (save_unfiltered_detections, rebuild_label_stack)
        # sees one consistent shape. tracks_raw entries are the
        # ORIGINAL track objects — both kept (also referenced by
        # result["tracks"]) and dropped (the dropped copies in
        # tracks_dropped are shallow dict copies, NOT the same
        # objects, so upscaling them wouldn't affect tracks_raw).
        # The shape-guard prevents double-upscaling shared refs.
        seen = set()
        for t in (list(result.get("tracks_raw", []) or [])
                   + list(result.get("tracks", []) or [])):
            if id(t) in seen:
                continue
            seen.add(id(t))
            if t.get("stack") is not None and t["stack"].shape[-1] != W:
                t["stack"] = _upscale(
                    t["stack"].astype(np.uint8)).astype(bool)

    result["auto"] = auto
    result["px_thresholds"] = px_thresholds
    if aligned_cy5_orig_scale is not None:
        result["aligned_cy5_orig_scale"] = aligned_cy5_orig_scale
    _emit("Detection done", 100)
    return result
