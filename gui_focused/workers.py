"""Pipeline workers for the focused GUI."""
import time
import logging
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)


class FocusedDetectWorker(QThread):
    """Run hybrid_cpsam or hybrid_cpsam_multi detection."""

    progress = pyqtSignal(str, int)
    log_event = pyqtSignal(str, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, recording, mode, params):
        super().__init__()
        self.recording = recording
        self.mode = mode
        self.params = params  # dict from ParamsPanel.get_detect_params()

    def run(self):
        try:
            t0 = time.time()
            self.log_event.emit("start", f"Detection: mode={self.mode}")
            frames = self.recording["frames"]
            min_area = self.params.get("min_area_px", 500)

            def cb(msg, pct):
                self.progress.emit(msg, pct)

            use_deepsea = self.params.get("use_deepsea", True)
            use_fallback = self.params.get("use_fallback", True)
            use_gap_fill = self.params.get("use_gap_fill", True)

            if self.mode == "hybrid_cpsam":
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
                result = detect_hybrid_cpsam_multi(
                    frames, progress_fn=cb,
                    min_area_px=min_area,
                    use_fallback=use_fallback,
                    use_deepsea=use_deepsea,
                    use_gap_fill=use_gap_fill)
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

            n_filled = fill_track_gaps(
                self.tracks, self.frames,
                min_area=self.params.get("min_area", 300),
                search_radius=self.params.get("search_radius", 150),
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
                 scale_overrides=None):
        super().__init__()
        self.recording = recording
        self.detect_result = detect_result
        self.mode = mode
        self.scale_overrides = scale_overrides or {}

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

            if self.mode == "single":
                masks = self.detect_result["masks"]
                result = analyze_recording(rec, masks)
                self.log_event.emit("done",
                                    f"Analysis done in {time.time()-t0:.1f}s")
                self.finished.emit(result)
            else:
                tracks = self.detect_result.get("tracks", [])
                per_cell = []
                for tid, track in enumerate(tracks):
                    self.progress.emit(
                        f"Analyzing cell {tid+1}/{len(tracks)}",
                        int(100 * tid / max(len(tracks) - 1, 1)))
                    cell_masks = track["stack"]
                    result = analyze_recording(rec, cell_masks)
                    result["cell_id"] = tid + 1
                    result["track_info"] = {
                        "first_frame": track["first_frame"],
                        "frames_tracked": int(
                            cell_masks.any(axis=(1, 2)).sum()),
                        "parent_id": track.get("parent_id"),
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
