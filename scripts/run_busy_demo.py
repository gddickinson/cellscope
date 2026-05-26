"""Launch the CellScope focused GUI pre-loaded with the busy
multichannel demo, configured for multi-cell + Cy5 filtering, and
auto-run detection then analysis. After the pipeline finishes, the
GUI stays open for interactive inspection.

Run with:
    conda run -n cellpose4 python /tmp/run_busy_demo.py
"""
import os
import sys
import time

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

# Ensure cellscope is importable AND that relative-path lookups
# (data/models/cellpose_dic, etc.) resolve correctly. The detection
# code looks for `data/models/...` relative to the cwd, so we must
# chdir to the project root before importing anything.
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

DEMO = ("/Users/george/claude_test/cellscope/data/examples/"
        "multichannel_DIC_Cy5_DMSO_busy/"
        "multichannel_DIC_Cy5_DMSO_busy.ome.tif")


def main():
    app = QApplication(sys.argv)

    from gui_focused.main_window import FocusedMainWindow
    from core.io import load_recording

    w = FocusedMainWindow()
    w.resize(1600, 1000)
    w.show()
    w.raise_()              # bring above other windows
    w.activateWindow()      # give it keyboard focus

    # ------------------------------------------------------------------
    # Step 1: Load recording directly (bypass the channel-chooser modal).
    # The demo has ch0 = Cy5 (fluorescence), ch1 = DIC.
    # ------------------------------------------------------------------
    w.logger.log("info", f"Loading demo: {os.path.basename(DEMO)}")
    rec = load_recording(DEMO, dic_channel=1, fluo_channel=0)
    w.recording = rec
    n = len(rec["frames"])
    name = rec.get("name", os.path.basename(DEMO))
    w.logger.log("info",
                 f"Loaded {name}: {n} frames [+ Cy5 channel]")
    w.viewer.set_data(rec["frames"],
                      fluo_frames=rec.get("cy5_frames"))
    w.detect_result = None
    w.analysis_result = None
    w.analysis.clear()
    w.pipeline.reset_all()
    w.pipeline.set_stage_status("load", "done")
    w.pipeline.enable_stage("detect", True)
    w.status.showMessage(f"Loaded: {name} ({n} frames) [+ Cy5 channel]")
    w.params.set_from_recording(rec)
    w.params.set_cy5_available(True)

    # ------------------------------------------------------------------
    # Step 2: Switch to MULTI-CELL pipeline mode
    # ------------------------------------------------------------------
    w.pipeline.set_mode("multi")
    w.mode = "multi"
    w.params.set_context("detect", "multi")

    # ------------------------------------------------------------------
    # Step 3: Configure the best detection settings for this recording
    # ------------------------------------------------------------------
    # Modality: DIC (we know it is)
    idx = w.params.modality.findText("DIC")
    if idx >= 0:
        w.params.modality.setCurrentIndex(idx)

    # DIC model: Auto → resolves to cpsam_dic (the best fine-tune we have)
    for i in range(w.params.dic_model.count()):
        if w.params.dic_model.itemText(i).startswith("Auto"):
            w.params.dic_model.setCurrentIndex(i)
            break

    # Debris filter: min area 500 px (default; fine for this recording)
    w.params.min_area.setValue(500)

    # Expected cells: 0 = Auto (don't cap — there can be 30+ cells here)
    w.params.expected_cells.setValue(0)

    # Refinement: DeepSea on, TTA off (good default), fallback on, gap-fill on
    w.params.use_deepsea.setChecked(True)
    w.params.use_tta.setChecked(False)
    w.params.use_fallback.setChecked(True)
    w.params.use_gap_fill.setChecked(True)

    # Cy5: recovery ON (Tier 2), filter = Multi-metric (best default)
    w.params.use_cy5_recovery.setChecked(True)
    for i in range(w.params.cy5_filter_mode.count()):
        if w.params.cy5_filter_mode.itemText(i).startswith("Multi-metric"):
            w.params.cy5_filter_mode.setCurrentIndex(i)
            break

    # Analysis-tab options that the user has enabled in the Analysis tab
    w.params.compute_vampire.setChecked(True)
    w.params.vampire_clusters.setValue(5)
    w.params.compute_states.setChecked(True)

    w.logger.log("info",
                 "Configured: DIC modality · cpsam_dic Auto · "
                 "multi-cell · Cy5 multi-metric filter · "
                 "VAMPIRE on · state classification on")

    # ------------------------------------------------------------------
    # Step 4: Trigger detection
    # ------------------------------------------------------------------
    # When detection completes, automatically run analysis.
    def _on_detect_complete(result):
        w.logger.log("info", "Detection complete — auto-running analysis")
        # tiny delay so the GUI gets to redraw the post-detect state
        QTimer.singleShot(200, w._on_analyze)

    # _on_detect_done is what the worker.finished signal calls — hook
    # in additionally without replacing it.
    orig_done = w._on_detect_done
    def wrapped_done(result):
        orig_done(result)
        _on_detect_complete(result)
    w._on_detect_done = wrapped_done

    # When analysis completes, log that the user can now poke around
    orig_an_done = w._on_analyze_done
    def wrapped_an_done(result):
        orig_an_done(result)
        n_cells = (len(result) if isinstance(result, list)
                   else 1)
        w.logger.log(
            "info",
            f"All done — {n_cells} cells analysed. Pick a graph from "
            f"the Graphs tab; channel toggle is in the viewer; "
            f"VAMPIRE is now available per cell.")
        w.status.showMessage(
            f"Done · {n_cells} cells · explore the Graphs + Summary "
            f"docks — GUI stays open")
    w._on_analyze_done = wrapped_an_done

    # Start detection now
    w.logger.log("info", "Running detection — this may take a few minutes…")
    QTimer.singleShot(500, w._on_detect)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
