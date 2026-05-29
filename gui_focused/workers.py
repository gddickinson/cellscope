"""Pipeline workers for the focused GUI.

All fallback values for params.get(...) come from
core.pipeline_defaults.DEFAULTS so the worker matches the params panel
when a key is missing from the dict.
"""
import time
import logging
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from core.pipeline_defaults import DEFAULTS as _PD

log = logging.getLogger(__name__)


class FocusedDetectWorker(QThread):
    """Run detection: hybrid_cpsam, hybrid_dic, or their multi variants."""

    progress = pyqtSignal(str, int)
    log_event = pyqtSignal(str, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, recording, mode, params):
        super().__init__()
        self.recording = recording
        self.mode = mode
        self.params = params  # dict from ParamsPanel.get_detect_params()

    def _annotate_tracks_with_cy5(self, tracks, cy5_frames):
        """Compute per-track per-frame Cy5 stats and attach to each
        track dict for later display + filtering.

        Includes:
          - cy5_mean, cy5_p75: basic intensity stats
          - cy5_score: z-score brightness vs local background
          - cy5_io_ratio: inside-vs-exterior-ring intensity ratio
          - cy5_fraction_positive: fraction of pixels above local bg
        The latter three are required by the multi_metric filter."""
        from core.multichannel import (
            per_cell_cy5_features, cy5_presence_score,
            cy5_inside_outside_ratio, cy5_fraction_positive)
        T = len(cy5_frames)
        for t in tracks:
            stack = t.get("stack")
            if stack is None:
                continue
            mean = np.full(T, np.nan, dtype=np.float32)
            p75 = np.full(T, np.nan, dtype=np.float32)
            score = np.full(T, np.nan, dtype=np.float32)
            io_r = np.full(T, np.nan, dtype=np.float32)
            fpos = np.full(T, np.nan, dtype=np.float32)
            for i in range(T):
                m = stack[i].astype(bool)
                if not m.any():
                    continue
                f = per_cell_cy5_features(m, cy5_frames[i])
                mean[i] = f["mean"]
                p75[i] = f["p75"]
                score[i] = cy5_presence_score(m, cy5_frames[i])
                io_r[i] = cy5_inside_outside_ratio(m, cy5_frames[i])
                fpos[i] = cy5_fraction_positive(m, cy5_frames[i])
            t["cy5_mean"] = mean
            t["cy5_p75"] = p75
            t["cy5_score"] = score
            t["cy5_io_ratio"] = io_r
            t["cy5_fraction_positive"] = fpos
            valid = ~np.isnan(score)
            t["cy5_mean_score"] = (float(np.nanmean(score))
                                    if valid.any() else 0.0)

    def run(self):
        try:
            t0 = time.time()
            self.log_event.emit("start", f"Detection: mode={self.mode}")
            frames = self.recording["frames"]
            min_area = self.params.get("min_area_px", _PD.min_area_px)

            def cb(msg, pct):
                self.progress.emit(msg, pct)

            use_deepsea = self.params.get("use_deepsea", _PD.use_deepsea)
            use_fallback = self.params.get("use_fallback", _PD.use_fallback)
            use_gap_fill = self.params.get("use_gap_fill", _PD.use_gap_fill)
            use_tiling = self.params.get("use_tiling", _PD.use_tiling)
            tile_grid = self.params.get("tile_grid", _PD.tile_grid)

            dic_model_path = self.params.get("dic_model_path", None)

            use_tta = self.params.get("use_tta", _PD.use_tta)
            use_mirror_pad = self.params.get(
                "use_mirror_pad", _PD.use_mirror_pad)
            use_cpsam_cy5_union = self.params.get(
                "use_cpsam_cy5_union", _PD.use_cpsam_cy5_union)
            if self.mode == "auto":
                # Canonical end-to-end detection used by both the GUI
                # and the evaluation script (run_pipeline_on_gt_recording).
                # Performs alignment + downsample + auto-pipeline-select +
                # cy5 filter + upscale-back in one shot.
                from core.unified_detection import detect_recording
                cy5_frames = self.recording.get("cy5_frames")
                result = detect_recording(
                    frames, cy5_frames=cy5_frames,
                    um_per_px=self.recording.get("um_per_px"),
                    time_interval_min=self.recording.get(
                        "time_interval_min"),
                    downsample=self.params.get("downsample", "auto"),
                    align_channels=self.params.get(
                        "align_channels", True),
                    run_cy5_filter=(
                        self.params.get("cy5_filter_mode", "off")
                        != "off"),
                    cy5_filter_mode=self.params.get(
                        "cy5_filter_mode", _PD.cy5_filter_mode),
                    cy5_filter_threshold=self.params.get(
                        "cy5_filter_threshold",
                        _PD.cy5_filter_threshold),
                    cy5_pg_min_lifetime=self.params.get(
                        "cy5_pg_min_lifetime"),
                    cy5_pg_static_velocity_px=self.params.get(
                        "cy5_pg_static_velocity_px"),
                    cy5_pg_static_shape_iou=self.params.get(
                        "cy5_pg_static_shape_iou"),
                    use_deepsea=self.params.get("use_deepsea"),
                    use_gap_fill=self.params.get("use_gap_fill"),
                    use_sam2_video_gap_fill=self.params.get(
                        "use_sam2_video_gap_fill"),
                    max_gap_frames=self.params.get("max_gap_frames"),
                    min_track_length=self.params.get(
                        "min_track_length"),
                    # The GUI's "Search radius (px)" widget controls
                    # the multi-cell tracker's max centroid hop —
                    # tighten it for slow keratinocytes to suppress
                    # ID switches across nearby cells.
                    max_hop_px=self.params.get("search_radius"),
                    use_tta=self.params.get("use_tta"),
                    use_cpsam_cy5_union=self.params.get(
                        "use_cpsam_cy5_union"),
                    use_fallback=self.params.get("use_fallback"),
                    use_mirror_pad=self.params.get("use_mirror_pad"),
                    use_preprocess=self.params.get("use_preprocess"),
                    use_retry=self.params.get("use_retry"),
                    cy5_fusion_jaccard_thresh=self.params.get(
                        "cy5_fusion_jaccard_thresh"),
                    cy5_fusion_max_overlap_frac=self.params.get(
                        "cy5_fusion_max_overlap_frac"),
                    cy5_fusion_augment_cpsam=self.params.get(
                        "cy5_fusion_augment_cpsam"),
                    progress_fn=cb,
                )
                auto = result.get("auto", {})
                align_info = auto.get('alignment') or {}
                align_applied = align_info.get('shift_applied', 'n/a')
                self.log_event.emit(
                    "info",
                    f"auto: pipeline={auto.get('pipeline_kind')} "
                    f"downsample={auto.get('downsample_factor')}× "
                    f"alignment={align_applied}")
                result["flow_quality"] = np.zeros(len(frames))
                result["flow_magnitudes"] = np.zeros_like(
                    frames, dtype=np.float32)
            elif self.mode == "hybrid_dic":
                from core.hybrid_dic import detect_hybrid_dic
                masks, missed = detect_hybrid_dic(
                    frames, progress_fn=cb,
                    area_threshold=min_area,
                    use_preprocess=True,
                    use_deepsea=use_deepsea,
                    use_retry=True,
                    model_path=dic_model_path,
                    use_tta=use_tta)
                result = {
                    "masks": masks,
                    "missed_frames": missed,
                    "flow_quality": np.zeros(len(frames)),
                    "flow_magnitudes": np.zeros_like(
                        frames, dtype=np.float32),
                }
            elif self.mode == "hybrid_dic_multi":
                from core.hybrid_dic import detect_hybrid_dic_multi
                cy5_frames = self.recording.get("cy5_frames")
                use_cy5_fusion = (
                    self.params.get(
                        "use_cy5_fusion", _PD.use_cy5_fusion)
                    and cy5_frames is not None)
                result = detect_hybrid_dic_multi(
                    frames, progress_fn=cb,
                    min_area_px=min_area,
                    use_preprocess=True,
                    use_deepsea=use_deepsea,
                    use_retry=True,
                    use_gap_fill=use_gap_fill,
                    model_path=dic_model_path,
                    use_tta=use_tta,
                    cy5_frames=cy5_frames,
                    use_cy5_fusion=use_cy5_fusion)
                if use_cy5_fusion:
                    self.log_event.emit(
                        "info",
                        f"Cy5 fusion added "
                        f"{result.get('n_cy5_fusion_added', 0)} cells")
                # Optional Cy5 multi-metric filter (same as cpsam_multi)
                if cy5_frames is not None and result.get("tracks"):
                    from core.multichannel import (
                        per_cell_cy5_features, cy5_presence_score)
                    self._annotate_tracks_with_cy5(
                        result["tracks"], cy5_frames)
                cy5_filter_mode = self.params.get("cy5_filter_mode", "off")
                if (cy5_filter_mode != "off" and cy5_frames is not None
                        and result.get("tracks")):
                    from core.cy5_filter import (
                        apply_cy5_filter, rebuild_label_stack)
                    kept, dropped, info = apply_cy5_filter(
                        result["tracks"],
                        mode=cy5_filter_mode,
                        threshold=self.params.get(
                            "cy5_filter_threshold", 0.15))
                    result["tracks_raw"] = result["tracks"]
                    result["tracks"] = kept
                    result["tracks_dropped"] = dropped
                    result["cy5_filter_info"] = info
                    result["labels"] = rebuild_label_stack(
                        kept, frames.shape)
                    result["masks"] = result["labels"] > 0
                    self.log_event.emit(
                        "info",
                        f"Cy5 filter ({cy5_filter_mode}): kept "
                        f"{len(kept)}, dropped {len(dropped)}")
                result["flow_quality"] = np.zeros(len(frames))
                result["flow_magnitudes"] = np.zeros_like(
                    frames, dtype=np.float32)
            elif use_tiling:
                from core.cpsam_tiled import detect_cpsam_tiled
                self.log_event.emit("info",
                                    f"Tiled detection: {tile_grid}x{tile_grid}")
                labels = detect_cpsam_tiled(
                    frames, n_tiles=(tile_grid, tile_grid),
                    overlap=64, min_area=min_area,
                    progress_fn=cb)
                if use_deepsea:
                    from core.deepsea_multicell import (
                        refine_labels_with_deepsea)
                    labels = refine_labels_with_deepsea(
                        frames, labels, expand_px=20)
                result = {
                    "masks": labels > 0,
                    "labels": labels,
                    "missed_frames": [],
                    "flow_quality": np.zeros(len(frames)),
                    "flow_magnitudes": np.zeros_like(
                        frames, dtype=np.float32),
                }
            elif self.mode == "hybrid_cpsam":
                from core.hybrid_cpsam import detect_hybrid_cpsam
                masks, missed = detect_hybrid_cpsam(
                    frames, progress_fn=cb,
                    area_threshold=min_area,
                    use_fallback=use_fallback,
                    use_deepsea=use_deepsea)
                result = {
                    "masks": masks,
                    "missed_frames": missed,
                    "flow_quality": np.zeros(len(frames)),
                    "flow_magnitudes": np.zeros_like(
                        frames, dtype=np.float32),
                }
            elif self.mode == "hybrid_cpsam_multi":
                from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi
                cy5_frames = self.recording.get("cy5_frames")
                recover = (self.params.get(
                    "use_cy5_recovery", _PD.use_cy5_recovery)
                           and cy5_frames is not None)
                use_cy5_fusion = (
                    self.params.get(
                        "use_cy5_fusion", _PD.use_cy5_fusion)
                    and cy5_frames is not None)
                result = detect_hybrid_cpsam_multi(
                    frames, progress_fn=cb,
                    min_area_px=min_area,
                    use_fallback=use_fallback,
                    use_deepsea=use_deepsea,
                    use_gap_fill=use_gap_fill,
                    use_tta=use_tta,
                    use_mirror_pad=use_mirror_pad,
                    use_cpsam_cy5_union=use_cpsam_cy5_union,
                    cy5_frames=cy5_frames,
                    recover_with_cy5=recover,
                    use_cy5_fusion=use_cy5_fusion)
                if recover:
                    self.log_event.emit(
                        "info",
                        f"Cy5 recovery added "
                        f"{result.get('n_cy5_recovered', 0)} cells")
                if use_cy5_fusion:
                    self.log_event.emit(
                        "info",
                        f"Cy5 fusion added "
                        f"{result.get('n_cy5_fusion_added', 0)} cells")
                # Annotate per-track Cy5 features for downstream display
                if cy5_frames is not None and result.get("tracks"):
                    self._annotate_tracks_with_cy5(
                        result["tracks"], cy5_frames)
                # Optional Tier 4: Cy5-based false-positive filter
                cy5_filter_mode = self.params.get("cy5_filter_mode", "off")
                if (cy5_filter_mode != "off" and cy5_frames is not None
                        and result.get("tracks")):
                    from core.cy5_filter import (
                        apply_cy5_filter, rebuild_label_stack)
                    kept, dropped, info = apply_cy5_filter(
                        result["tracks"],
                        mode=cy5_filter_mode,
                        threshold=self.params.get(
                            "cy5_filter_threshold", 0.15))
                    result["tracks_raw"] = result["tracks"]
                    result["tracks"] = kept
                    result["tracks_dropped"] = dropped
                    result["cy5_filter_info"] = info
                    # Rebuild label stack so dropped tracks no longer
                    # appear in the viewer / analysis
                    result["labels"] = rebuild_label_stack(
                        kept, frames.shape)
                    result["masks"] = result["labels"] > 0
                    self.log_event.emit(
                        "info",
                        f"Cy5 filter ({cy5_filter_mode}): kept "
                        f"{len(kept)}, dropped {len(dropped)}"
                        + (f" @ threshold {info.get('threshold', '?'):.3f}"
                           if "threshold" in info else ""))
                result["flow_quality"] = np.zeros(len(frames))
                result["flow_magnitudes"] = np.zeros_like(
                    frames, dtype=np.float32)
            else:
                from core.pipeline import detect
                result = detect(frames, mode=self.mode, progress_fn=cb)

            result["elapsed"] = time.time() - t0
            n_masks = int(result["masks"].any(axis=(1, 2)).sum())
            self.log_event.emit("done",
                                f"Detection done: {n_masks} frames with cells "
                                f"in {result['elapsed']:.1f}s")
            if "tracks" in result:
                self.log_event.emit("info",
                                    f"Found {len(result['tracks'])} tracks, "
                                    f"max {result.get('cell_count', '?')} "
                                    f"cells/frame")
            self.finished.emit(result)
        except Exception as e:
            log.exception("Detection failed")
            self.error.emit(str(e))


class FocusedGapFillWorker(QThread):
    """Run gap fill on existing tracks."""

    progress = pyqtSignal(str, int)
    log_event = pyqtSignal(str, str)
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, tracks, frames, params, project_root):
        super().__init__()
        self.tracks = tracks
        self.frames = frames
        self.params = params
        self.project_root = project_root

    def run(self):
        try:
            from core.track_gap_fill import fill_track_gaps
            self.log_event.emit("start", "Filling track gaps")

            def cb(msg, pct):
                self.progress.emit(msg, pct)

            # Unified key naming: prefer min_area_px (matches the rest
            # of the codebase) but accept the legacy min_area for
            # backward-compat. Fallback to canonical DEFAULTS.
            min_area = self.params.get(
                "min_area_px",
                self.params.get("min_area", _PD.min_area_px))
            search_radius = self.params.get(
                "search_radius", _PD.search_radius_px)
            n_filled = fill_track_gaps(
                self.tracks, self.frames,
                min_area=min_area,
                search_radius=search_radius,
                project_root=self.project_root,
                progress_fn=cb,
            )
            self.log_event.emit("done", f"Filled {n_filled} gaps")
            self.finished.emit(n_filled)
        except Exception as e:
            log.exception("Gap fill failed")
            self.error.emit(str(e))


class FocusedAnalyzeWorker(QThread):
    """Run analysis on detected masks (single or multi-cell)."""

    progress = pyqtSignal(str, int)
    log_event = pyqtSignal(str, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, recording, detect_result, mode,
                 scale_overrides=None, vampire_params=None,
                 state_params=None):
        super().__init__()
        self.recording = recording
        self.detect_result = detect_result
        self.mode = mode
        self.scale_overrides = scale_overrides or {}
        self.vampire_params = vampire_params or {}
        self.state_params = state_params or {"enabled": False}

    def _annotate_with_state(self, result, masks_or_stack, um_per_px,
                              dt_min):
        """Add state classification + per-state motility to a single
        cell's analysis result. `masks_or_stack` is (N, H, W) bool."""
        from core.cell_state import (
            classify_track_states, state_fraction,
            STATE_BALLED, STATE_ATTACHED, STATE_TRANSITIONAL)
        from core.motility_state import (
            state_speeds, state_msd, state_persistence,
            state_total_displacement)
        from core.tracking import extract_centroids

        thresholds = self.state_params.get("thresholds")
        sd = classify_track_states(
            masks_or_stack.astype(bool), thresholds)
        states = sd["states"]
        cents = extract_centroids(masks_or_stack.astype(bool))
        result["state_per_frame"] = states
        result["state_frac_balled"] = state_fraction(
            states, STATE_BALLED)
        result["state_frac_attached"] = state_fraction(
            states, STATE_ATTACHED)
        result["state_frac_transitional"] = state_fraction(
            states, STATE_TRANSITIONAL)
        for state, prefix in ((STATE_BALLED, "balled"),
                                (STATE_ATTACHED, "attached")):
            sp = state_speeds(cents, states, state, um_per_px, dt_min)
            msd = state_msd(cents, states, state, um_per_px,
                              max_lag=20)
            per = state_persistence(cents, states, state, um_per_px)
            td = state_total_displacement(cents, states, state,
                                            um_per_px)
            result[f"{prefix}_n_speed_samples"] = int(len(sp))
            result[f"{prefix}_mean_speed_um_per_min"] = (
                float(np.mean(sp)) if len(sp) else float("nan"))
            result[f"{prefix}_median_speed_um_per_min"] = (
                float(np.median(sp)) if len(sp) else float("nan"))
            result[f"{prefix}_persistence_lag1"] = per["lag1"]
            result[f"{prefix}_msd_lag5_um2"] = (
                float(msd["msd"][4])
                if len(msd["msd"]) > 4 and not np.isnan(msd["msd"][4])
                else float("nan"))
            result[f"{prefix}_total_displacement_um"] = (
                td["total_displacement_um"])
            result[f"{prefix}_straightness"] = td["straightness"]

    def run(self):
        try:
            from core.pipeline import analyze_recording
            t0 = time.time()
            self.log_event.emit("start", "Analyzing")

            # Apply scale overrides to the recording
            rec = dict(self.recording)
            um = self.scale_overrides.get("um_per_px")
            dt = self.scale_overrides.get("time_interval_min")
            if um and um > 0:
                rec["um_per_px"] = um
            if dt and dt > 0:
                rec["time_interval_min"] = dt

            um_per_px = float(rec.get("um_per_px", 1.0)) or 1.0
            dt_min = float(rec.get("time_interval_min", 1.0)) or 1.0

            if self.mode == "single":
                masks = self.detect_result["masks"]
                result = analyze_recording(rec, masks)
                if self.vampire_params.get("enabled"):
                    self._run_vampire(result, masks)
                if self.state_params.get("enabled"):
                    self._annotate_with_state(result, masks,
                                                um_per_px, dt_min)
                    self.log_event.emit(
                        "info",
                        f"State: balled "
                        f"{100*result['state_frac_balled']:.0f}%, "
                        f"attached "
                        f"{100*result['state_frac_attached']:.0f}%")
                self.log_event.emit("done",
                                    f"Analysis done in {time.time()-t0:.1f}s")
                self.finished.emit(result)
            else:
                tracks = self.detect_result.get("tracks", [])
                # Cell-division annotation: detection pipelines do this
                # inline (hybrid_cpsam_multi / hybrid_dic). When tracks
                # come from a loaded masks.npz WITHOUT a divisions.json
                # sidecar, run it here so analysis surfaces lineage.
                # Signal: detect_result["divisions"] is absent.
                if tracks and "divisions" not in self.detect_result:
                    labels = self.detect_result.get("labels")
                    if labels is not None:
                        try:
                            from core.division_annotator import (
                                annotate_track_lineage)
                            um = (self.recording.get("um_per_px")
                                  if self.recording else 1.0) or 1.0
                            self.progress.emit(
                                "Detecting cell divisions...", 0)
                            cands, n_set = annotate_track_lineage(
                                tracks, labels, um_per_px=um)
                            self.detect_result["divisions"] = cands
                            self.log_event.emit(
                                "info",
                                f"Detected {len(cands)} division "
                                f"candidate(s), {n_set} lineage "
                                f"link(s) set")
                        except Exception as e:
                            self.log_event.emit(
                                "warn",
                                f"Division annotation failed: {e}")
                per_cell = []
                for tid, track in enumerate(tracks):
                    self.progress.emit(
                        f"Analyzing cell {tid+1}/{len(tracks)}",
                        int(100 * tid / max(len(tracks) - 1, 1)))
                    cell_masks = track["stack"]
                    result = analyze_recording(rec, cell_masks)
                    # Run VAMPIRE per-cell when enabled (matches the
                    # single-cell branch — was previously missing here,
                    # which is why the VAMPIRE plots said "enable in
                    # Analysis params" even when it was already on).
                    if self.vampire_params.get("enabled"):
                        self._run_vampire(result, cell_masks)
                    if self.state_params.get("enabled"):
                        self._annotate_with_state(
                            result, cell_masks, um_per_px, dt_min)
                    result["cell_id"] = tid + 1
                    result["track_info"] = {
                        "first_frame": track["first_frame"],
                        "frames_tracked": int(
                            cell_masks.any(axis=(1, 2)).sum()),
                        "parent_id": track.get("parent_id"),
                        "division_frame": track.get("division_frame"),
                        "division_score": track.get("division_score"),
                    }
                    per_cell.append(result)
                self.log_event.emit(
                    "done",
                    f"Analyzed {len(per_cell)} cells in "
                    f"{time.time()-t0:.1f}s")
                self.finished.emit(per_cell)
        except Exception as e:
            log.exception("Analysis failed")
            self.error.emit(str(e))

    def _run_vampire(self, result, masks):
        """Run VAMPIRE shape mode analysis and attach to result."""
        try:
            from core.vampire_analysis import run_vampire_analysis
            n_cl = self.vampire_params.get(
                "n_clusters", _PD.vampire_n_clusters)
            self.log_event.emit("info", f"VAMPIRE: {n_cl} clusters")
            vamp = run_vampire_analysis(masks, n_clusters=n_cl)
            if vamp:
                result["vampire"] = vamp
                h = vamp["heterogeneity"]
                self.log_event.emit(
                    "info",
                    f"VAMPIRE: {vamp['n_contours']} contours, "
                    f"H={h['entropy']:.2f}/{h['max_entropy']:.2f}")
        except Exception as e:
            log.warning("VAMPIRE failed: %s", e)
            self.log_event.emit("warn", f"VAMPIRE failed: {e}")
