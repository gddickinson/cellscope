"""QThread worker for batch analysis using hybrid_cpsam pipeline."""
import os
import time
import logging
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

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

    def run(self):
        try:
            from core.io import load_recording
            from core.pipeline import analyze_recording
            from output.results import write_recording_results

            mode = self.params.get("mode", "hybrid_cpsam")
            min_area = self.params.get("min_area_px", 500)
            use_deepsea = self.params.get("use_deepsea", True)
            use_fallback = self.params.get("use_fallback", True)
            use_gap_fill = self.params.get("use_gap_fill", True)

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
                    rec = load_recording(path)
                    frames = rec["frames"]

                    if mode == "hybrid_cpsam":
                        from core.hybrid_cpsam import detect_hybrid_cpsam
                        masks, _ = detect_hybrid_cpsam(
                            frames, area_threshold=min_area,
                            use_deepsea=use_deepsea,
                            use_fallback=use_fallback)
                    else:
                        from core.hybrid_cpsam_multi import (
                            detect_hybrid_cpsam_multi)
                        det = detect_hybrid_cpsam_multi(
                            frames, min_area_px=min_area,
                            use_deepsea=use_deepsea,
                            use_fallback=use_fallback,
                            use_gap_fill=use_gap_fill)
                        masks = det["masks"]

                    result = analyze_recording(rec, masks)

                    if self.vampire_params.get("enabled"):
                        try:
                            from core.vampire_analysis import (
                                run_vampire_analysis)
                            vamp = run_vampire_analysis(
                                masks,
                                n_clusters=self.vampire_params.get(
                                    "n_clusters", 5))
                            if vamp:
                                result["vampire"] = vamp
                        except Exception:
                            pass

                    rec_dir = os.path.join(self.output_dir, group, name)
                    os.makedirs(rec_dir, exist_ok=True)
                    write_recording_results(result, rec_dir)

                    metrics = {
                        "group": group, "name": name,
                        "mean_speed": result.get("mean_speed", 0),
                        "persistence": result.get("persistence", 0),
                        "mean_area": result.get("shape_summary", {}).get(
                            "area_um2", {}).get("mean", 0),
                        "boundary_confidence": result.get(
                            "mean_boundary_confidence", 0),
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
