"""Full analysis orchestration: detect → refine → analyze → results dict.

The Result dict produced by `analyze_recording` is the canonical output
format consumed by both the GUI and the output writers.
"""
import time
import numpy as np

from core.detection import detect_cellpose_flow
from core.auto_params import auto_select_refinement
from core.tracking import (
    extract_centroids, centroids_to_um, instantaneous_speed,
    trajectory_origin_normalized, total_distance, net_displacement,
    persistence_ratio, mean_squared_displacement,
    direction_autocorrelation,
)
from core.morphology import shape_timeseries, shape_summary
from core.edge_dynamics import edge_velocity_kymograph, edge_summary
from core.evaluation import (
    boundary_confidence_timeseries, area_stability,
)


def detect(frames, progress_fn=None, mode="default",
           preprocess_temporal_method=None,
           preprocess_spatial_highpass_sigma=None,
           preprocess_debris_diameter_um=None,
           preprocess_pixel_size_um=None,
           roi=None, augment=False, diameter=None,
           cellprob_threshold=None, flow_threshold=None,
           model_path=None):
    """Initial detection step. Returns masks dict.

    Args:
        mode: "default" uses cellpose+flow fusion.
              "cascade" uses GT model → original model → temporal fill.
              "threshold_retry" uses cellpose at defaults, then retries
                failed frames at progressively lower cellprob_threshold
                values, then temporally interpolates any remaining gaps.
        preprocess_temporal_method: if set ('median'/'mean'/'min'/'max'),
            subtract the per-pixel temporal statistic from every frame
            before detection. Suppresses static content + illumination drift.
        preprocess_spatial_highpass_sigma: if > 0, apply a per-frame
            Gaussian high-pass (sigma in pixels) before detection.
        preprocess_debris_diameter_um: if > 0 AND pixel_size_um > 0,
            wipe bright/dark features smaller than this size (ASF filter).
        preprocess_pixel_size_um: pixel size in μm for debris removal.

    Returns:
        {
            "masks": (N, H, W) bool,
            "flow_quality": (N,) — only for default mode,
            "flow_magnitudes": (N, H, W) — only for default mode,
            "provenance": list[str] — for cascade and threshold_retry,
            "cascade_stats": dict — for cascade mode,
            "retry_stats": dict — for threshold_retry mode,
            "elapsed": float,
        }
    """
    t0 = time.time()

    # ROI masking (first — the zeroed-out area shouldn't skew temporal
    # statistics in the other preprocessing steps).
    if roi is not None:
        from core.preprocess import apply_roi_mask
        if progress_fn:
            progress_fn(0, f"Applying ROI mask {tuple(roi)}...")
        frames = apply_roi_mask(frames, roi)

    # Optional sequence preprocessing before detection
    if (preprocess_temporal_method
            or (preprocess_spatial_highpass_sigma or 0) > 0
            or ((preprocess_debris_diameter_um or 0) > 0
                and (preprocess_pixel_size_um or 0) > 0)):
        from core.preprocess import preprocess_sequence
        if progress_fn:
            progress_fn(0, "Preprocessing frames...")
        frames = preprocess_sequence(
            frames,
            temporal_method=preprocess_temporal_method,
            spatial_highpass_sigma=preprocess_spatial_highpass_sigma or 0,
            debris_diameter_um=preprocess_debris_diameter_um or 0,
            pixel_size_um=preprocess_pixel_size_um or 0,
        )

    if mode == "cascade":
        from core.cascade_detect import detect_cascade
        masks, provenance, stats = detect_cascade(
            frames, progress_fn=progress_fn,
        )
        return {
            "masks": masks,
            "flow_quality": np.zeros(len(frames)),
            "flow_magnitudes": np.zeros_like(frames, dtype=np.float32),
            "provenance": provenance,
            "cascade_stats": stats,
            "elapsed": time.time() - t0,
        }
    elif mode == "multicell":
        from core.detection import detect_cellpose_labels
        label_stack = detect_cellpose_labels(
            frames, progress_fn=progress_fn,
        )
        return {
            "masks": label_stack > 0,   # bool for backward-compat plots
            "labels": label_stack,       # full N,H,W int32 label stack
            "flow_quality": np.zeros(len(frames)),
            "flow_magnitudes": np.zeros_like(frames, dtype=np.float32),
            "elapsed": time.time() - t0,
        }
    elif mode == "cpsam":
        # Cellpose-SAM (cellpose 4.1.1+ default model). Uses ViT
        # backbone trained on a curated cell corpus. Requires the
        # `cellpose4` env (cellpose >= 4). At defaults gives 0.915
        # mean IoU on Ignasi 65 GT (vs 0.874 baseline).
        import cellpose
        if not cellpose.version.startswith("4"):
            raise RuntimeError(
                f"mode='cpsam' requires cellpose >=4.1, got "
                f"{cellpose.version}. Run from the cellpose4 env: "
                f"`conda run -n cellpose4 python ...`")
        from cellpose import models
        m = models.CellposeModel(gpu=True)
        all_masks = np.zeros(frames.shape, dtype=bool)
        for i, f in enumerate(frames):
            if progress_fn is not None:
                progress_fn(f"cpsam frame {i+1}/{len(frames)}",
                            int(100 * i / max(len(frames)-1, 1)))
            masks_i, _, _ = m.eval(f)
            all_masks[i] = masks_i > 0
        return {
            "masks": all_masks,
            "flow_quality": np.zeros(len(frames)),
            "flow_magnitudes": np.zeros_like(frames, dtype=np.float32),
            "elapsed": time.time() - t0,
        }
    elif mode == "hybrid_cpsam":
        # cpsam → cellpose+MedSAM+DeepSea fallback for missed frames
        # → DeepSea union on all. Requires cellpose4 env (active) +
        # cellpose env (available for subprocess). Best result on
        # Ignasi: 0.932 IoU with zero missed frames.
        from core.hybrid_cpsam import detect_hybrid_cpsam
        masks, missed = detect_hybrid_cpsam(
            frames, progress_fn=progress_fn)
        return {
            "masks": masks,
            "missed_frames": missed,
            "flow_quality": np.zeros(len(frames)),
            "flow_magnitudes": np.zeros_like(frames, dtype=np.float32),
            "elapsed": time.time() - t0,
        }
    elif mode == "hybrid_cpsam_multi":
        from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi
        result = detect_hybrid_cpsam_multi(
            frames, progress_fn=progress_fn)
        result["flow_quality"] = np.zeros(len(frames))
        result["flow_magnitudes"] = np.zeros_like(
            frames, dtype=np.float32)
        result["elapsed"] = time.time() - t0
        return result
    elif mode == "tiled":
        from core.detection import detect_cellpose_tiled
        masks = detect_cellpose_tiled(frames, progress_fn=progress_fn)
        return {
            "masks": masks,
            "flow_quality": np.zeros(len(frames)),
            "flow_magnitudes": np.zeros_like(frames, dtype=np.float32),
            "elapsed": time.time() - t0,
        }
    elif mode == "threshold_retry":
        from core.detection import detect_with_threshold_retry
        masks, provenance, stats = detect_with_threshold_retry(
            frames, progress_fn=progress_fn,
        )
        return {
            "masks": masks,
            "flow_quality": np.zeros(len(frames)),
            "flow_magnitudes": np.zeros_like(frames, dtype=np.float32),
            "provenance": provenance,
            "retry_stats": stats,
            "elapsed": time.time() - t0,
        }
    else:
        # If user passes diameter / explicit thresholds / model_path,
        # use plain cellpose (skip optical-flow fusion). Phase 14i+
        # tuning showed this is critical for cells outside the
        # default trained scale (e.g. Ignasi IC293 at diameter=50).
        if (diameter not in (None, 0)
                or cellprob_threshold is not None
                or flow_threshold is not None
                or model_path):
            from core.detection import detect_cellpose
            d = None if diameter in (None, 0) else float(diameter)
            masks = detect_cellpose(
                frames, gpu=True, progress_fn=progress_fn,
                model_path=model_path,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                augment=augment, diameter=d,
            )
            return {
                "masks": masks,
                "flow_quality": np.zeros(len(frames)),
                "flow_magnitudes": np.zeros_like(frames, dtype=np.float32),
                "elapsed": time.time() - t0,
            }
        masks, quality, flow_mags = detect_cellpose_flow(
            frames, progress_fn=progress_fn, augment=augment,
        )
        return {
            "masks": masks,
            "flow_quality": quality,
            "flow_magnitudes": flow_mags,
            "elapsed": time.time() - t0,
        }


def refine(frames, base_masks, progress_fn=None, mode="iterative",
           skip_frames=None, use_crop=True, crop_padding_px=50,
           crop_mode=None, per_cell_padding_px=30,
           alt_method=None, rf_filter_bank=None,
           hybrid_method=None, hybrid_rf_threshold=0.6,
           hybrid_ensemble_threshold=0.5,
           hybrid_boundary_dilate=10, hybrid_boundary_erode=5,
           sam_model_type="vit_b", sam_use_mask_prompt=True,
           sam_version="v1"):
    """Refinement step.

    Modes:
      - "iterative" (default): tries 3 full-stack presets (gentle, default,
        aggressive) and picks the best by membrane score × stability.
        ~3x slower than fixed but adapts to per-recording optics.
      - "fixed": runs the proven default full-stack with no iteration.
      - "snap_only": legacy snap-only auto-selector (no RF/CRF). Fastest.
      - "none": pass through base_masks unchanged.
      - "alt_method": apply a classical alt-segmentation method
        (see `core.alt_segmentation.METHODS`). Requires alt_method key.

    Args:
        skip_frames: optional set/list of frame indices to preserve during
            refinement (used with cascade).
        use_crop: deprecated in favor of crop_mode. If crop_mode is None,
            this is mapped: True → "global", False → "none".
        crop_mode: "none" | "global" | "per_cell". global uses one union
            bbox; per_cell crops each frame independently.
        crop_padding_px: padding for global bbox.
        per_cell_padding_px: padding for per-cell bboxes.
        alt_method: key from alt_segmentation.METHODS (used when
            mode="alt_method").

    Returns:
        {
            "masks": (N, H, W) bool,
            "config": dict,
            "score_log": list of dicts,
            "crop_bbox": tuple or None (global mode) or
                         list of tuples (per_cell mode),
            "elapsed": float,
        }
    """
    t0 = time.time()
    scores = []

    # Resolve crop_mode
    if crop_mode is None:
        crop_mode = "global" if use_crop else "none"

    # --- Build refine function closures ---
    def _do_iterative(f, m):
        from core.refinement import iterative_full_stack_refine
        refined, preset, s = iterative_full_stack_refine(
            f, m, progress_fn=progress_fn,
        )
        scores.extend(s)
        _do_iterative.chosen_preset = preset
        return refined

    def _do_fixed(f, m):
        from core.refinement import full_stack_refine
        return full_stack_refine(
            f, m, progress_fn=progress_fn, skip_frames=skip_frames,
            rf_filter_bank=rf_filter_bank,
        )

    def _do_snap_only(f, m):
        refined, chosen, s = auto_select_refinement(
            f, m, progress_fn=progress_fn,
        )
        scores.extend(s)
        _do_snap_only.chosen_config = chosen
        return refined

    def _do_alt_method_stack(f, m):
        from core.alt_segmentation import apply_method_to_stack
        if progress_fn:
            progress_fn(f"Alt method: {alt_method}")
        return apply_method_to_stack(f, m, alt_method)

    def _do_alt_method_single(single_frame, single_mask):
        """Per-cell wrapper: single-frame version of alt method."""
        from core.alt_segmentation import METHODS
        fn, _ = METHODS[alt_method]
        if not single_mask.any():
            return single_mask
        try:
            return fn(single_frame, single_mask)
        except Exception:
            return single_mask

    def _do_hybrid_stack(f, m):
        from core.hybrid_rf import apply_method_to_stack
        if progress_fn:
            progress_fn(f"Hybrid RF: {hybrid_method}")
        return apply_method_to_stack(
            f, m, hybrid_method,
            rf_filter_bank=rf_filter_bank,
            **_hybrid_method_kwargs(hybrid_method),
        )

    def _do_hybrid_single(single_frame, single_mask):
        from core.hybrid_rf import METHODS as H_METHODS
        fn, _ = H_METHODS[hybrid_method]
        if not single_mask.any():
            return single_mask
        try:
            return fn(single_frame, single_mask,
                      rf_filter_bank=rf_filter_bank,
                      **_hybrid_method_kwargs(hybrid_method))
        except Exception:
            return single_mask

    def _hybrid_method_kwargs(key):
        if key == "ensemble":
            return {"thresh": hybrid_ensemble_threshold}
        return {"rf_thresh": hybrid_rf_threshold,
                "dilate": hybrid_boundary_dilate,
                "erode": hybrid_boundary_erode}

    def _do_sam_stack(f, m):
        from core.sam_refine import refine_all_with_sam
        if progress_fn:
            progress_fn(f"SAM ({sam_model_type}) refining all frames...")
        return refine_all_with_sam(
            f, m, model_type=sam_model_type,
            use_mask_prompt=sam_use_mask_prompt,
        )

    def _do_sam_single(single_frame, single_mask):
        from core.sam_refine import refine_with_sam, load_sam
        if not single_mask.any():
            return single_mask
        try:
            predictor = load_sam(sam_model_type)
            return refine_with_sam(single_frame, single_mask,
                                   predictor=predictor,
                                   use_mask_prompt=sam_use_mask_prompt)
        except Exception:
            return single_mask

    def _do_medsam_stack(f, m):
        from core.medsam_refine import refine_all_with_medsam
        if progress_fn:
            progress_fn("MedSAM refining all frames...")
        return refine_all_with_medsam(f, m)

    def _do_medsam_single(single_frame, single_mask):
        from core.medsam_refine import refine_with_medsam
        if not single_mask.any():
            return single_mask
        try:
            return refine_with_medsam(single_frame, single_mask)
        except Exception:
            return single_mask

    def _do_none(f, m):
        return m

    # --- Dispatch ---
    if mode == "iterative":
        refine_fn = _do_iterative
    elif mode == "fixed":
        refine_fn = _do_fixed
    elif mode == "snap_only":
        refine_fn = _do_snap_only
    elif mode == "alt_method":
        if not alt_method:
            raise ValueError(
                "refine(mode='alt_method') requires alt_method key"
            )
        refine_fn = _do_alt_method_stack
    elif mode == "hybrid":
        if not hybrid_method:
            raise ValueError(
                "refine(mode='hybrid') requires hybrid_method key"
            )
        refine_fn = _do_hybrid_stack
    elif mode == "sam":
        refine_fn = _do_sam_stack
    elif mode == "medsam":
        refine_fn = _do_medsam_stack
    elif mode == "none":
        refine_fn = _do_none
    else:
        raise ValueError(f"Unknown refinement mode: {mode}")

    # --- Apply via crop strategy ---
    bbox = None
    if crop_mode == "global" and mode != "none":
        from core.crop_refine import crop_refine
        refined, bbox = crop_refine(
            frames, base_masks, refine_fn, padding_px=crop_padding_px,
        )
    elif crop_mode == "per_cell" and mode == "alt_method":
        # per_cell is only meaningful for per-frame methods
        from core.crop_refine import per_cell_refine
        refined, bbox = per_cell_refine(
            frames, base_masks, _do_alt_method_single,
            padding_px=per_cell_padding_px,
        )
    elif crop_mode == "per_cell" and mode == "hybrid":
        from core.crop_refine import per_cell_refine
        refined, bbox = per_cell_refine(
            frames, base_masks, _do_hybrid_single,
            padding_px=per_cell_padding_px,
        )
    elif crop_mode == "per_cell" and mode == "sam":
        from core.crop_refine import per_cell_refine
        refined, bbox = per_cell_refine(
            frames, base_masks, _do_sam_single,
            padding_px=per_cell_padding_px,
        )
    elif crop_mode == "per_cell" and mode == "medsam":
        from core.crop_refine import per_cell_refine
        refined, bbox = per_cell_refine(
            frames, base_masks, _do_medsam_single,
            padding_px=per_cell_padding_px,
        )
    else:
        refined = refine_fn(frames, base_masks)

    # --- Build config dict ---
    if mode == "iterative":
        preset = _do_iterative.chosen_preset
        chosen = {
            "name": f"full_stack_{preset['name']}",
            "search": preset["search_radius"],
            "max_step": preset["max_step"],
            "smooth": preset["snap_smooth"],
            "fourier": True, "rf": True, "crf": True,
            "temporal_sigma": preset["temporal_sigma"],
        }
    elif mode == "fixed":
        chosen = {
            "name": "full_stack",
            "search": 8, "max_step": 5, "smooth": 2.0,
            "fourier": True, "rf": True, "crf": True,
            "temporal_sigma": 1.5,
        }
        if skip_frames:
            chosen["skip_frames"] = len(set(skip_frames))
    elif mode == "alt_method":
        chosen = {"name": f"alt_method:{alt_method}",
                  "alt_method": alt_method}
    elif mode == "hybrid":
        chosen = {"name": f"hybrid:{hybrid_method}",
                  "hybrid_method": hybrid_method,
                  "hybrid_rf_threshold": hybrid_rf_threshold,
                  "hybrid_ensemble_threshold": hybrid_ensemble_threshold,
                  "hybrid_boundary_dilate": hybrid_boundary_dilate,
                  "hybrid_boundary_erode": hybrid_boundary_erode}
    elif mode == "sam":
        chosen = {"name": f"sam:{sam_model_type}",
                  "sam_model_type": sam_model_type,
                  "sam_use_mask_prompt": sam_use_mask_prompt}
    elif mode == "medsam":
        chosen = {"name": "medsam",
                  "model": "flaviagiammarino/medsam-vit-base"}
    elif mode == "none":
        chosen = {"name": "none"}
    else:
        chosen = _do_snap_only.chosen_config

    chosen["crop_mode"] = crop_mode
    if rf_filter_bank:
        chosen["rf_filter_bank"] = rf_filter_bank
    if crop_mode == "global" and bbox is not None:
        chosen["crop_bbox"] = bbox
        chosen["crop_padding_px"] = crop_padding_px
    elif crop_mode == "per_cell":
        chosen["per_cell_padding_px"] = per_cell_padding_px

    return {
        "masks": refined,
        "config": chosen,
        "score_log": scores,
        "crop_bbox": bbox,
        "elapsed": time.time() - t0,
    }


def analyze_recording(recording, masks, progress_fn=None):
    """Compute all downstream analytics from a refined mask stack.

    Args:
        recording: dict from io.load_recording (must include name,
            frames, um_per_px, time_interval_min)
        masks: (N, H, W) bool — refined segmentation masks
        progress_fn: optional callable(message)

    Returns:
        result: dict with all analytics for this recording
    """
    name = recording["name"]
    frames = recording["frames"]
    um = recording["um_per_px"]
    dt = recording["time_interval_min"]

    if progress_fn:
        progress_fn("Tracking...")
    cpx = extract_centroids(masks)
    cum = centroids_to_um(cpx, um)
    speed = instantaneous_speed(cum, dt)
    traj = trajectory_origin_normalized(cum)
    msd_l, msd_v, msd_s = mean_squared_displacement(cum)
    ac_l, ac_v, ac_s = direction_autocorrelation(cum)

    if progress_fn:
        progress_fn("Shape...")
    shape_ts = shape_timeseries(masks, um)
    shape_sum = shape_summary(shape_ts)

    if progress_fn:
        progress_fn("Edge dynamics...")
    edge_angles, edge_vel = edge_velocity_kymograph(masks, cpx, um, dt)
    edge_sum = edge_summary(edge_vel)

    if progress_fn:
        progress_fn("Evaluation metrics...")
    bnd_conf = boundary_confidence_timeseries(frames, masks)
    stab = area_stability(masks, um)

    result = {
        "name": name,
        "video_path": recording.get("video_path", ""),
        "n_frames": int(len(frames)),
        "um_per_px": float(um),
        "time_interval_min": float(dt),
        "frames": frames,
        "masks": masks,

        # Tracking
        "centroids_px": cpx,
        "centroids_um": cum,
        "trajectory": traj,
        "speed": speed,
        "mean_speed": float(np.nanmean(speed)),
        "total_distance": total_distance(cum),
        "net_displacement": net_displacement(cum),
        "persistence": persistence_ratio(cum),
        "msd": (msd_l, msd_v, msd_s),
        "autocorr": (ac_l, ac_v, ac_s),

        # Morphology
        "shape_timeseries": shape_ts,
        "shape_summary": shape_sum,

        # Edge dynamics
        "edge_angles": edge_angles,
        "edge_velocity": edge_vel,
        "edge_summary": edge_sum,

        # Quality
        "boundary_confidence": bnd_conf,
        "mean_boundary_confidence": float(np.nanmean(bnd_conf)),
        "area_stability": stab,
    }
    return result


def run_full_pipeline(recording, progress_fn=None, detect_mode="default",
                      skip_refine_gt=True):
    """End-to-end: detect → refine → analyze.

    Args:
        detect_mode: "default" or "cascade". Cascade uses the GT-retrained
            model as primary detector with original model as fallback.
        skip_refine_gt: if True and detect_mode=="cascade", skip refinement
            on frames detected by the GT model (they have superior boundaries).
            Temporal smoothing is still applied to all frames.

    Returns the result dict + diagnostics about each step.
    """
    frames = recording["frames"]

    if progress_fn:
        progress_fn("Detecting cells...")
    det = detect(frames, progress_fn=lambda i, msg=None: None,
                 mode=detect_mode)

    # Build skip list for cascade + selective refinement
    skip_frames = None
    if detect_mode == "cascade" and skip_refine_gt:
        provenance = det.get("provenance", [])
        skip_frames = [i for i, p in enumerate(provenance) if p == "gt_model"]

    if progress_fn:
        progress_fn("Auto-refining boundaries...")
    ref = refine(frames, det["masks"], progress_fn=progress_fn,
                 skip_frames=skip_frames)

    if progress_fn:
        progress_fn("Computing analytics...")
    result = analyze_recording(recording, ref["masks"], progress_fn=progress_fn)

    result["detection_elapsed_s"] = det["elapsed"]
    result["refinement_elapsed_s"] = ref["elapsed"]
    result["refinement_config"] = ref["config"]
    result["refinement_score_log"] = ref["score_log"]
    result["refinement_crop_bbox"] = ref.get("crop_bbox")
    result["mean_flow_quality"] = float(np.mean(det["flow_quality"]))
    if "provenance" in det:
        result["detection_provenance"] = det["provenance"]
    if "cascade_stats" in det:
        result["cascade_stats"] = det["cascade_stats"]
    if "retry_stats" in det:
        result["retry_stats"] = det["retry_stats"]
    return result
