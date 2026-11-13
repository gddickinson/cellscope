"""QThread worker for the focused GUI's 🔬 Test on frame (detect one frame off the GUI thread).

`FocusedMainWindow._on_test_frame` gathers the inputs on the GUI thread and starts this worker;
the modality probe and `core.unified_detection.detect_recording` (20-60 s) run here, and the result
comes back through `finished` / `error` signals (queued to the GUI thread), where
`_on_test_frame_done` reports the cell count and the runtime estimate.
"""
from __future__ import annotations

import time

from PyQt5.QtCore import QThread, pyqtSignal


class TestFrameWorker(QThread):
    """Detect frame `idx` of a recording with the GUI's current parameters."""

    finished = pyqtSignal(dict)      # {"result", "elapsed", "modality", "pipeline_kind", "idx"}
    error = pyqtSignal(str)

    def __init__(self, frames, one_frame, one_cy5, idx, params, mode, um_per_px, time_interval_min):
        super().__init__()
        self.frames, self.one_frame, self.one_cy5 = frames, one_frame, one_cy5
        self.idx, self.params, self.mode = idx, params, mode
        self.um_per_px, self.time_interval_min = um_per_px, time_interval_min

    def run(self):
        try:
            self.finished.emit(self._run())
        except Exception as e:                       # noqa: BLE001 - reported to the user
            self.error.emit(str(e))

    def _run(self):
        params = self.params
        # Honour the modality selector AND the pipeline mode (single / multi) so the test previews
        # what Detect will produce (Detect resolves modality via detect_modality() on the full stack).
        modality = params.get("modality", "auto")
        if modality == "auto":
            from core.modality import detect_modality
            modality = detect_modality(self.frames)
        if self.mode == "single":
            kind = "cpsam_dic" if modality == "dic" else "cpsam"
        else:
            kind = "auto"        # unified_detection's density probe picks cpsam_dic vs raw cpsam
        from core.unified_detection import detect_recording
        t0 = time.time()
        result = detect_recording(
            self.one_frame, cy5_frames=self.one_cy5,
            um_per_px=self.um_per_px, time_interval_min=self.time_interval_min,
            downsample=params.get("downsample", "auto"),
            # Single frame: skip multi-frame stages (alignment, Cy5 filter, gap fill, curation);
            # min_track_length=1 so the single-frame "track" is kept.
            align_channels=False, pipeline_kind=kind, run_cy5_filter=False, use_gap_fill=False,
            single_cell_curation=False, min_track_length=1,
            use_mirror_pad=params.get("use_mirror_pad"), use_deepsea=params.get("use_deepsea"),
            use_tta=params.get("use_tta"), use_cpsam_cy5_union=params.get("use_cpsam_cy5_union"),
            use_fallback=params.get("use_fallback"), use_bfloat16=params.get("use_bfloat16"),
            progress_fn=None)
        return {"result": result, "elapsed": time.time() - t0, "modality": modality,
                "pipeline_kind": kind, "idx": self.idx}
