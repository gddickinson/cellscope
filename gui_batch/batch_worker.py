"""QThread worker for batch analysis using hybrid_cpsam pipeline.

All fallback values for params.get(...) come from
core.pipeline_defaults.DEFAULTS so the batch worker matches the rest of
the suite when a key is missing.
"""
import os
import time
import logging
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from core.pipeline_defaults import DEFAULTS as _PD

log = logging.getLogger(__name__)


class BatchAnalysisWorker(QThread):
    """Process multiple recordings sequentially."""

    progress = pyqtSignal(str, int)
    recording_done = pyqtSignal(str, str, dict)  # group, name, metrics
    log_event = pyqtSignal(str, str)
    finished = pyqtSignal(str)   # output directory
    error = pyqtSignal(str)

    def __init__(self, recordings, params, output_dir):
        """
        Args:
            recordings: list of (group, video_path) tuples
            params: dict with detection/analysis settings
            output_dir: base output directory
        """
        super().__init__()
        self.recordings = recordings
        self.params = params
        self.output_dir = output_dir
        self.vampire_params = params.get("vampire", {})
        self._stop = False

    def stop(self):
        self._stop = True

    def _write_state_csv(self, tracks, rec, out_dir, thresholds):
        """Compute state classification + per-state motility for each
        track and write per_cell_state.csv. Returns aggregate state
        composition (mean over cells) for the recording-level metrics
        row."""
        import csv as _csv
        from core.cell_state import (
            classify_track_states, state_fraction,
            STATE_ROUNDED, STATE_SPREAD)
        from core.motility_state import (
            state_speeds, state_msd, state_persistence,
            state_total_displacement)
        from core.tracking import extract_centroids

        um = float(rec.get("um_per_px", 1.0)) or 1.0
        dt = float(rec.get("time_interval_min", 1.0)) or 1.0
        rows = []
        for tid, track in enumerate(tracks):
            stack = track.get("stack")
            if stack is None:
                continue
            sd = classify_track_states(stack.astype(bool), thresholds)
            states = sd["states"]
            cents = extract_centroids(stack.astype(bool))
            row = {"track_id": tid + 1,
                    "lifetime_frames": int(sum(1 for s in states
                                                  if s != "unknown")),
                    "frac_rounded": state_fraction(states, STATE_ROUNDED),
                    "frac_spread": state_fraction(states, STATE_SPREAD)}
            for state, prefix in ((STATE_ROUNDED, "rounded"),
                                    (STATE_SPREAD, "spread")):
                sp = state_speeds(cents, states, state, um, dt)
                msd = state_msd(cents, states, state, um, max_lag=20)
                per = state_persistence(cents, states, state, um)
                td = state_total_displacement(cents, states, state, um)
                row[f"{prefix}_n_speed_samples"] = int(len(sp))
                row[f"{prefix}_mean_speed_um_per_min"] = (
                    float(np.mean(sp)) if len(sp) else float("nan"))
                row[f"{prefix}_persistence_lag1"] = per["lag1"]
                row[f"{prefix}_msd_lag5_um2"] = (
                    float(msd["msd"][4])
                    if len(msd["msd"]) > 4
                    and not np.isnan(msd["msd"][4])
                    else float("nan"))
                row[f"{prefix}_total_displacement_um"] = (
                    td["total_displacement_um"])
                row[f"{prefix}_straightness"] = td["straightness"]
            rows.append(row)
        if not rows:
            return {}
        csv_path = os.path.join(out_dir, "per_cell_state.csv")
        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        # Aggregate: mean across cells in this recording
        return {
            "frac_rounded": float(np.mean(
                [r["frac_rounded"] for r in rows])),
            "frac_spread": float(np.mean(
                [r["frac_spread"] for r in rows])),
            "rounded_speed": float(np.nanmean(
                [r["rounded_mean_speed_um_per_min"] for r in rows])),
            "spread_speed": float(np.nanmean(
                [r["spread_mean_speed_um_per_min"] for r in rows])),
        }

    def _annotate_tracks_with_cy5(self, tracks, cy5_frames):
        """Compute per-track per-frame Cy5 stats and attach to each
        track dict. Required before applying the Tier-4 filter."""
        from core.multichannel import (
            per_cell_cy5_features, cy5_presence_score)
        T = len(cy5_frames)
        for t in tracks:
            stack = t.get("stack")
            if stack is None:
                continue
            mean = np.full(T, np.nan, dtype=np.float32)
            p75 = np.full(T, np.nan, dtype=np.float32)
            score = np.full(T, np.nan, dtype=np.float32)
            for i in range(T):
                m = stack[i].astype(bool)
                if not m.any():
                    continue
                f = per_cell_cy5_features(m, cy5_frames[i])
                mean[i] = f["mean"]
                p75[i] = f["p75"]
                score[i] = cy5_presence_score(m, cy5_frames[i])
            t["cy5_mean"] = mean
            t["cy5_p75"] = p75
            t["cy5_score"] = score
            valid = ~np.isnan(score)
            t["cy5_mean_score"] = (float(np.nanmean(score))
                                    if valid.any() else 0.0)

    def run(self):
        try:
            from core.io import load_recording
            from core.pipeline import analyze_recording
            from output.results import write_recording_results

            mode = self.params.get("mode", "hybrid_cpsam")
            min_area = self.params.get("min_area_px", _PD.min_area_px)
            use_deepsea = self.params.get(
                "use_deepsea", _PD.use_deepsea)
            use_fallback = self.params.get(
                "use_fallback", _PD.use_fallback)
            use_gap_fill = self.params.get(
                "use_gap_fill", _PD.use_gap_fill)
            multichannel = self.params.get("multichannel", False)
            dic_channel = self.params.get("dic_channel", 1)
            fluo_channel = self.params.get("fluo_channel", 0)
            use_cy5_recovery = self.params.get(
                "use_cy5_recovery", _PD.use_cy5_recovery)
            # cy5_filter_mode: "off" fallback is intentional — when the
            # key is missing we don't want to silently apply the
            # canonical "multi_metric" filter (which would change track
            # counts). Caller must opt in explicitly.
            cy5_filter_mode = self.params.get("cy5_filter_mode", "off")
            cy5_filter_threshold = self.params.get(
                "cy5_filter_threshold", _PD.cy5_filter_threshold)
            compute_states = self.params.get(
                "compute_states", _PD.compute_state_classification)
            state_thresholds = self.params.get("state_thresholds")

            total = len(self.recordings)
            all_metrics = []

            for idx, (group, path) in enumerate(self.recordings):
                if self._stop:
                    self.log_event.emit("warn", "Batch stopped by user")
                    break

                name = os.path.splitext(os.path.basename(path))[0]
                self.progress.emit(
                    f"Processing {idx+1}/{total}: {name}",
                    int(100 * idx / max(total - 1, 1)))
                self.log_event.emit("start", f"{group}/{name}")

                try:
                    if multichannel:
                        rec = load_recording(
                            path, dic_channel=dic_channel,
                            fluo_channel=fluo_channel)
                    else:
                        rec = load_recording(path)
                    frames = rec["frames"]
                    cy5_frames = rec.get("cy5_frames")

                    if mode == "hybrid_cpsam":
                        from core.hybrid_cpsam import detect_hybrid_cpsam
                        masks, _ = detect_hybrid_cpsam(
                            frames, area_threshold=min_area,
                            use_deepsea=use_deepsea,
                            use_fallback=use_fallback)
                    else:
                        from core.hybrid_cpsam_multi import (
                            detect_hybrid_cpsam_multi)
                        recover = (use_cy5_recovery
                                    and cy5_frames is not None)
                        det = detect_hybrid_cpsam_multi(
                            frames, min_area_px=min_area,
                            use_deepsea=use_deepsea,
                            use_fallback=use_fallback,
                            use_gap_fill=use_gap_fill,
                            cy5_frames=cy5_frames,
                            recover_with_cy5=recover)
                        masks = det["masks"]
                        if recover:
                            self.log_event.emit(
                                "info",
                                f"  Cy5 recovery: +"
                                f"{det.get('n_cy5_recovered', 0)} cells")
                        # Annotate tracks with per-frame Cy5 features
                        # then optionally apply Tier-4 false-positive
                        # filter to drop debris-like detections.
                        if (cy5_frames is not None
                                and cy5_filter_mode != "off"
                                and det.get("tracks")):
                            self._annotate_tracks_with_cy5(
                                det["tracks"], cy5_frames)
                            from core.cy5_filter import (
                                apply_cy5_filter, rebuild_label_stack)
                            kept, dropped, info = apply_cy5_filter(
                                det["tracks"], mode=cy5_filter_mode,
                                threshold=cy5_filter_threshold)
                            det["tracks_raw"] = det["tracks"]
                            det["tracks"] = kept
                            det["tracks_dropped"] = dropped
                            det["cy5_filter_info"] = info
                            det["labels"] = rebuild_label_stack(
                                kept, frames.shape)
                            masks = det["labels"] > 0
                            det["masks"] = masks
                            self.log_event.emit(
                                "info",
                                f"  Cy5 filter ({cy5_filter_mode}): "
                                f"kept {len(kept)}, "
                                f"dropped {len(dropped)}")

                    result = analyze_recording(rec, masks)

                    if self.vampire_params.get("enabled"):
                        try:
                            from core.vampire_analysis import (
                                run_vampire_analysis)
                            vamp = run_vampire_analysis(
                                masks,
                                n_clusters=self.vampire_params.get(
                                    "n_clusters",
                                    _PD.vampire_n_clusters))
                            if vamp:
                                result["vampire"] = vamp
                        except Exception:
                            pass

                    rec_dir = os.path.join(self.output_dir, group, name)
                    os.makedirs(rec_dir, exist_ok=True)
                    write_recording_results(result, rec_dir)

                    # Pre-Cy5-filter snapshot — only if persistence_guard
                    # (or whichever Cy5 filter) ran on this recording.
                    # Lets the GUI overlay dropped cells later.
                    try:
                        from output.results import (
                            save_unfiltered_detections)
                        unf = save_unfiltered_detections(rec_dir, det)
                        if unf is not None:
                            n_raw, n_kept, n_dropped = unf
                            self.log_event.emit(
                                "info",
                                f"  Unfiltered: {n_raw} raw, "
                                f"{n_kept} kept, {n_dropped} dropped "
                                f"(masks_unfiltered.npz + "
                                f"filter_decisions.json)")
                    except Exception as e:
                        self.log_event.emit(
                            "warn",
                            f"  Unfiltered save failed: {e}")

                    # Division-lineage sidecar (cheap, always write)
                    divisions = det.get("divisions", []) or []
                    tracks = det.get("tracks", []) or []
                    lineage = [{
                        "track_index": i,
                        "parent_track_index": t.get("parent_id"),
                        "division_frame": t.get("division_frame"),
                        "division_score": t.get("division_score"),
                    } for i, t in enumerate(tracks)
                      if t.get("parent_id") is not None]
                    import json as _json
                    with open(os.path.join(rec_dir,
                                            "divisions.json"), "w") as jf:
                        _json.dump({"n_candidates": len(divisions),
                                    "candidates": divisions,
                                    "track_lineage": lineage},
                                   jf, indent=2)
                    if divisions:
                        self.log_event.emit(
                            "info",
                            f"  Divisions: {len(divisions)} "
                            f"detected, {len(lineage)} lineage links")

                    state_summary = {}
                    if compute_states and "tracks" in det:
                        state_summary = self._write_state_csv(
                            det["tracks"], rec, rec_dir,
                            state_thresholds)
                        self.log_event.emit(
                            "info",
                            f"  State: balled "
                            f"{state_summary.get('frac_balled', 0)*100:.0f}%, "
                            f"attached "
                            f"{state_summary.get('frac_attached', 0)*100:.0f}%")

                    metrics = {
                        "group": group, "name": name,
                        "mean_speed": result.get("mean_speed", 0),
                        "persistence": result.get("persistence", 0),
                        "mean_area": result.get("shape_summary", {}).get(
                            "area_um2", {}).get("mean", 0),
                        "boundary_confidence": result.get(
                            "mean_boundary_confidence", 0),
                        **state_summary,
                    }
                    vamp = result.get("vampire")
                    if vamp:
                        h = vamp["heterogeneity"]
                        metrics["shape_entropy"] = h["entropy"]
                        metrics["n_shape_modes"] = vamp["n_clusters"]
                    all_metrics.append(metrics)
                    self.recording_done.emit(group, name, metrics)
                    self.log_event.emit("done", f"{group}/{name} complete")
                except Exception as e:
                    log.exception("Failed: %s/%s", group, name)
                    self.log_event.emit("error", f"{group}/{name}: {e}")

            # Write group summaries
            try:
                from output.summary import write_all_summaries
                if all_metrics:
                    write_all_summaries(
                        {m["group"]: m for m in all_metrics},
                        self.output_dir)
            except Exception:
                pass

            self.progress.emit("Batch complete", 100)
            self.finished.emit(self.output_dir)
        except Exception as e:
            log.exception("Batch failed")
            self.error.emit(str(e))
