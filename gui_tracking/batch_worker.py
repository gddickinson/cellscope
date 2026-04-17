"""QThread worker for batch tracking and analysis."""
import os
import time
import logging
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)


class TrackingBatchWorker(QThread):
    """Detect + track + analyze multiple recordings."""

    progress = pyqtSignal(str, int)
    recording_done = pyqtSignal(str, str, dict)  # group, name, metrics
    log_event = pyqtSignal(str, str)
    finished = pyqtSignal(dict)   # {group: [metrics_list]}
    error = pyqtSignal(str)

    def __init__(self, recordings, params, output_dir):
        super().__init__()
        self.recordings = recordings
        self.params = params
        self.output_dir = output_dir
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from core.io import load_recording
            from core.pipeline import analyze_recording
            from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi
            from core.hybrid_cpsam import detect_hybrid_cpsam

            mode = self.params.get("mode", "hybrid_cpsam")
            min_area = self.params.get("min_area_px", 500)
            total = len(self.recordings)
            group_results = {}

            for idx, (group, path) in enumerate(self.recordings):
                if self._stop:
                    break
                name = os.path.splitext(os.path.basename(path))[0]
                self.progress.emit(
                    f"{idx+1}/{total}: {name}", int(100*idx/max(total-1,1)))
                self.log_event.emit("start", f"{group}/{name}")

                try:
                    rec = load_recording(path)
                    if mode == "hybrid_cpsam_multi":
                        det = detect_hybrid_cpsam_multi(
                            rec["frames"], min_area_px=min_area)
                        tracks = det.get("tracks", [])
                        per_cell = []
                        for tid, t in enumerate(tracks):
                            r = analyze_recording(rec, t["stack"])
                            r["cell_id"] = tid + 1
                            r["track_info"] = {
                                "first_frame": t["first_frame"],
                                "frames_tracked": int(
                                    t["stack"].any(axis=(1,2)).sum()),
                            }
                            per_cell.append(r)
                        if per_cell:
                            metrics = {
                                "mean_speed": np.mean([
                                    r["mean_speed"] for r in per_cell]),
                                "persistence": np.mean([
                                    r["persistence"] for r in per_cell]),
                                "mean_area": np.mean([
                                    r.get("shape_summary", {}).get(
                                        "area_um2", {}).get("mean", 0)
                                    for r in per_cell]),
                                "n_cells": len(per_cell),
                            }
                        else:
                            metrics = {}
                    else:
                        masks, _ = detect_hybrid_cpsam(
                            rec["frames"], area_threshold=min_area)
                        result = analyze_recording(rec, masks)
                        metrics = {
                            "mean_speed": result.get("mean_speed", 0),
                            "persistence": result.get("persistence", 0),
                            "mean_area": result.get("shape_summary", {}).get(
                                "area_um2", {}).get("mean", 0),
                            "boundary_confidence": result.get(
                                "mean_boundary_confidence", 0),
                        }

                    metrics["group"] = group
                    metrics["name"] = name
                    group_results.setdefault(group, []).append(metrics)
                    self.recording_done.emit(group, name, metrics)
                    self.log_event.emit("done", f"{group}/{name}")

                except Exception as e:
                    log.exception("Failed: %s/%s", group, name)
                    self.log_event.emit("error", f"{group}/{name}: {e}")

            self.progress.emit("Complete", 100)
            self.finished.emit(group_results)
        except Exception as e:
            log.exception("Batch failed")
            self.error.emit(str(e))
