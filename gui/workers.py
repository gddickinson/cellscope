"""QThread workers wrapping core pipeline operations.

Workers accept a `RunParams` (from gui.options) describing the full
requested pipeline configuration and emit granular `log` signals so
the shared `RunLogger` can capture exactly what happened.
"""
import os
import time
import traceback
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.io import load_recording, find_recordings
from core.pipeline import detect, refine, analyze_recording
from output.results import write_recording_results, _build_metrics_dict
from output.summary import write_all_summaries

from gui.options.params import RunParams


# --- Helpers ---

def _build_detect_kwargs(params):
    """Translate RunParams.detection into pipeline.detect kwargs.

    pipeline.detect() currently accepts only `mode`. Any sub-mode params
    that differ from config.py are applied by temporarily overriding
    config values for the duration of the run.
    """
    return {"mode": params.detection.mode}


def _apply_overrides(recording, analysis_params):
    """Apply um_per_px / time_interval_min overrides if set."""
    rec = dict(recording)
    if analysis_params.override_um_per_px:
        rec["um_per_px"] = analysis_params.override_um_per_px
    if analysis_params.override_time_interval_min:
        rec["time_interval_min"] = analysis_params.override_time_interval_min
    return rec


class _ConfigOverride:
    """Context manager that temporarily patches config values.

    Used so per-run DetectionParams take effect even when the core
    modules read config constants at import time.
    """
    def __init__(self, patches: dict):
        self.patches = patches
        self._prev = {}

    def __enter__(self):
        import config as cfg
        for k, v in self.patches.items():
            self._prev[k] = getattr(cfg, k)
            setattr(cfg, k, v)
        return self

    def __exit__(self, *a):
        import config as cfg
        for k, v in self._prev.items():
            setattr(cfg, k, v)


def _overrides_from_params(params: RunParams) -> dict:
    d = params.detection
    return {
        "CELLPOSE_FLOW_THRESHOLD": d.flow_threshold,
        "CELLPOSE_CELLPROB_THRESHOLD": d.cellprob_threshold,
        "CELLPOSE_MODEL_NAME": d.model_name,
        "MIN_CELL_AREA_PX": d.min_area_px,
    }


# --- Detection worker ---

class DetectWorker(QThread):
    progress = pyqtSignal(str, int)
    log = pyqtSignal(str, str, dict)   # kind, message, details
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, recording, params: RunParams):
        super().__init__()
        self.recording = recording
        self.params = params

    def run(self):
        try:
            rec = _apply_overrides(self.recording, self.params.analysis)
            n = len(rec["frames"])

            def cb(i, msg=None):
                if msg:
                    self.progress.emit(msg, 0)
                    self.log.emit("info", msg, {})
                else:
                    pct = int((i + 1) / max(n, 1) * 100)
                    self.progress.emit(
                        f"Detecting frame {i + 1}/{n}", pct
                    )

            self.log.emit(
                "start",
                f"Detection ({self.params.detection.mode})",
                {"mode": self.params.detection.mode,
                 "frames": n,
                 "flow_threshold": self.params.detection.flow_threshold,
                 "cellprob_threshold":
                     self.params.detection.cellprob_threshold},
            )

            t0 = time.time()
            d = self.params.detection
            # Pixel size for debris removal: use recording default if override=0
            px_size = (d.preprocess_pixel_size_um
                       if d.preprocess_pixel_size_um > 0
                       else rec.get("um_per_px"))
            with _ConfigOverride(_overrides_from_params(self.params)):
                det = detect(
                    rec["frames"],
                    progress_fn=cb,
                    mode=d.mode,
                    preprocess_temporal_method=d.preprocess_temporal_method,
                    preprocess_spatial_highpass_sigma=(
                        d.preprocess_spatial_highpass_sigma
                    ),
                    preprocess_debris_diameter_um=(
                        d.preprocess_debris_diameter_um
                    ),
                    preprocess_pixel_size_um=px_size,
                    roi=rec.get("roi"),
                    augment=getattr(d, "detect_augment", False),
                    diameter=getattr(d, "diameter", 0),
                )
            elapsed = time.time() - t0
            det["elapsed"] = elapsed
            self.progress.emit("Detection complete", 100)

            # Summarize result for log
            masks = det["masks"]
            n_with = int(masks.any(axis=(1, 2)).sum())
            details = {"elapsed_s": round(elapsed, 2),
                       "frames_with_mask": f"{n_with}/{n}"}
            if "cascade_stats" in det:
                details["cascade_stats"] = det["cascade_stats"]
            if "retry_stats" in det:
                details["retry_stats"] = det["retry_stats"]
            self.log.emit("done", "Detection finished", details)

            self.finished.emit(det)
        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit("error", f"Detection failed: {e}", {"traceback": tb})
            self.error.emit(f"{e}\n{tb}")


# --- Refinement + analysis worker ---

class RefineWorker(QThread):
    progress = pyqtSignal(str, int)
    log = pyqtSignal(str, str, dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, recording, base_masks, params: RunParams,
                 detection_provenance=None):
        super().__init__()
        self.recording = recording
        self.base_masks = base_masks
        self.params = params
        self.detection_provenance = detection_provenance

    def run(self):
        try:
            rec = _apply_overrides(self.recording, self.params.analysis)
            rp = self.params.refinement
            self._step = 0

            def cb(msg):
                self._step += 1
                self.progress.emit(msg, min(95, self._step * 12))
                self.log.emit("info", msg, {})

            # Optional pre-refinement: run MedSAM first so the chosen
            # refinement step receives tight bbox-precise seed masks.
            base_masks = self.base_masks
            if getattr(rp, "pre_refine_medsam", False):
                self.log.emit("start",
                               "Pre-refinement: MedSAM",
                               {"reason": "tighten seed before main refiner"})
                from core.medsam_refine import refine_all_with_medsam
                t_pre = time.time()
                n_pre = len(rec["frames"])

                def medsam_cb(i):
                    if i % 5 == 0 or i == n_pre - 1:
                        self.progress.emit(
                            f"MedSAM pre-refine {i+1}/{n_pre}",
                            min(95, int(100 * i / max(n_pre - 1, 1))))
                base_masks = refine_all_with_medsam(
                    rec["frames"], base_masks, progress_fn=medsam_cb,
                )
                self.log.emit("done",
                               f"Pre-refinement done in "
                               f"{time.time()-t_pre:.1f}s", {})

            # Post-MedSAM: union with DeepSea (recovers under-segmented
            # boundary pixels). Phase Ignasi: +0.025 IoU, 5 more frames
            # above 0.85.
            if getattr(rp, "union_with_deepsea", False):
                self.log.emit("start", "Union with DeepSea", {})
                from core.medsam_deepsea_union import union_with_deepsea_all
                t_du = time.time()
                base_masks = union_with_deepsea_all(
                    rec["frames"], base_masks)
                self.log.emit("done",
                               f"Union done in {time.time()-t_du:.1f}s", {})

            if rp.mode == "none":
                self.log.emit(
                    "info",
                    "Main refinement skipped (mode=none)" +
                    (" — pre-refine MedSAM output retained"
                     if getattr(rp, "pre_refine_medsam", False) else ""),
                    {},
                )
                refined = base_masks
                cfg = {"name": "medsam_only" if getattr(
                    rp, "pre_refine_medsam", False) else "none"}
                scores = []
                bbox = None
                elapsed = 0.0
            else:
                self.log.emit(
                    "start", f"Refinement ({rp.mode})",
                    {"mode": rp.mode,
                     "use_crop": rp.use_crop,
                     "crop_padding_px": rp.crop_padding_px},
                )
                # Skip-cascade frames
                skip = None
                if (self.detection_provenance
                        and rp.skip_cascade_gt_frames):
                    skip = [i for i, p in enumerate(self.detection_provenance)
                            if p == "gt_model"]
                    if skip:
                        self.log.emit(
                            "info",
                            f"Preserving {len(skip)} GT-model frames "
                            f"from refinement", {},
                        )

                t0 = time.time()
                ref = refine(
                    rec["frames"], base_masks,
                    progress_fn=cb, mode=rp.mode,
                    skip_frames=skip,
                    use_crop=rp.use_crop,
                    crop_padding_px=rp.crop_padding_px,
                    crop_mode=rp.crop_mode,
                    per_cell_padding_px=rp.per_cell_padding_px,
                    alt_method=rp.alt_method,
                    rf_filter_bank=rp.rf_filter_bank,
                    hybrid_method=rp.hybrid_method,
                    hybrid_rf_threshold=rp.hybrid_rf_threshold,
                    hybrid_ensemble_threshold=rp.hybrid_ensemble_threshold,
                    hybrid_boundary_dilate=rp.hybrid_boundary_dilate,
                    hybrid_boundary_erode=rp.hybrid_boundary_erode,
                    sam_model_type=rp.sam_model_type,
                    sam_use_mask_prompt=rp.sam_use_mask_prompt,
                )
                elapsed = time.time() - t0
                refined = ref["masks"]
                cfg = ref["config"]
                scores = ref["score_log"]
                bbox = ref.get("crop_bbox")

                done_details = {"elapsed_s": round(elapsed, 2),
                                "chosen": cfg.get("name")}
                if bbox is not None:
                    done_details["crop_bbox"] = list(bbox)
                self.log.emit("done", "Refinement finished", done_details)

            # Analysis
            self.progress.emit("Computing analytics...", 96)
            self.log.emit("start", "Analysis", {})
            t0 = time.time()
            result = analyze_recording(
                rec, refined, progress_fn=lambda m: None,
            )
            ana_elapsed = time.time() - t0
            result["refinement_config"] = cfg
            result["refinement_score_log"] = scores
            result["refinement_elapsed_s"] = elapsed
            result["refinement_crop_bbox"] = bbox
            self.log.emit(
                "done", "Analysis finished",
                {"elapsed_s": round(ana_elapsed, 2),
                 "mean_speed_um_min": round(result.get("mean_speed", 0), 3),
                 "mean_area_um2": round(
                     result.get("area_stability", {}).get(
                         "mean_area_um2", 0), 1),
                 "persistence": round(result.get("persistence", 0), 3)},
            )

            self.progress.emit("Done", 100)
            self.finished.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit("error", f"Refinement failed: {e}",
                          {"traceback": tb})
            self.error.emit(f"{e}\n{tb}")


# --- Batch worker ---

class BatchWorker(QThread):
    progress = pyqtSignal(str, int)
    log = pyqtSignal(str, str, dict)
    file_done = pyqtSignal(str, str, dict)  # group, name, metrics
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, input_dir, output_dir, params: RunParams):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.params = params

    def run(self):
        try:
            groups = find_recordings(self.input_dir)
            if not groups:
                self.error.emit(f"No videos found in {self.input_dir}")
                return
            total = sum(len(v) for v in groups.values())
            self.log.emit(
                "start", f"Batch: {total} recording(s) in "
                         f"{len(groups)} group(s)",
                {"input": self.input_dir, "output": self.output_dir,
                 "preset": self.params.preset_name},
            )

            all_metrics = []
            done = 0

            for group_name, paths in sorted(groups.items()):
                for video_path in paths:
                    base = os.path.splitext(
                        os.path.basename(video_path)
                    )[0]
                    pct = int(done / max(total, 1) * 100)
                    self.progress.emit(
                        f"[{done + 1}/{total}] {group_name}/{base}", pct,
                    )
                    self.log.emit(
                        "info", f"Processing {group_name}/{base}", {},
                    )
                    try:
                        metrics = self._process_one(video_path)
                        all_metrics.append((group_name, metrics))
                        self.file_done.emit(group_name, base, metrics)
                    except Exception as e:
                        self.log.emit(
                            "warn", f"Skipping {base}: {e}", {},
                        )
                    done += 1

            self.progress.emit("Writing summary...", 99)
            summary_paths = write_all_summaries(
                all_metrics, self.output_dir
            )
            self.progress.emit("Done", 100)
            self.log.emit(
                "done", "Batch complete",
                {"summaries": summary_paths, "recordings": done},
            )
            self.finished.emit(summary_paths)
        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit("error", f"Batch failed: {e}",
                          {"traceback": tb})
            self.error.emit(f"{e}\n{tb}")

    def _process_one(self, video_path):
        rec = load_recording(video_path)
        rec = _apply_overrides(rec, self.params.analysis)
        rp = self.params.refinement

        with _ConfigOverride(_overrides_from_params(self.params)):
            det = detect(rec["frames"],
                         mode=self.params.detection.mode,
                         roi=rec.get("roi"),
                         augment=getattr(self.params.detection,
                                         "detect_augment", False),
                         diameter=getattr(self.params.detection,
                                          "diameter", 0))

            if rp.mode == "none":
                result = analyze_recording(rec, det["masks"])
                result["refinement_config"] = {"name": "none"}
                result["refinement_score_log"] = []
                result["refinement_elapsed_s"] = 0.0
                result["refinement_crop_bbox"] = None
            else:
                skip = None
                if (rp.skip_cascade_gt_frames
                        and "provenance" in det):
                    skip = [
                        i for i, p in enumerate(det["provenance"])
                        if p == "gt_model"
                    ]
                ref = refine(
                    rec["frames"], det["masks"], mode=rp.mode,
                    skip_frames=skip,
                    use_crop=rp.use_crop,
                    crop_padding_px=rp.crop_padding_px,
                    crop_mode=rp.crop_mode,
                    per_cell_padding_px=rp.per_cell_padding_px,
                    alt_method=rp.alt_method,
                    rf_filter_bank=rp.rf_filter_bank,
                    hybrid_method=rp.hybrid_method,
                    hybrid_rf_threshold=rp.hybrid_rf_threshold,
                    hybrid_ensemble_threshold=rp.hybrid_ensemble_threshold,
                    hybrid_boundary_dilate=rp.hybrid_boundary_dilate,
                    hybrid_boundary_erode=rp.hybrid_boundary_erode,
                    sam_model_type=rp.sam_model_type,
                    sam_use_mask_prompt=rp.sam_use_mask_prompt,
                )
                result = analyze_recording(rec, ref["masks"])
                result["refinement_config"] = ref["config"]
                result["refinement_score_log"] = ref["score_log"]
                result["refinement_elapsed_s"] = ref["elapsed"]
                result["refinement_crop_bbox"] = ref.get("crop_bbox")

        result["detection_elapsed_s"] = det["elapsed"]
        result["mean_flow_quality"] = float(
            np.mean(det.get("flow_quality", [0]))
        )
        base = os.path.splitext(os.path.basename(video_path))[0]
        group_name = os.path.basename(os.path.dirname(video_path))
        rec_out_dir = os.path.join(self.output_dir, group_name, base)
        write_recording_results(result, rec_out_dir)
        return _build_metrics_dict(result)
