"""Comprehensive headless test of the focused GUI.

Tests all pipeline stages, UI state transitions, mode switching,
plots, export, and takes screenshots at each stage.

Usage:
    conda run -n cellpose4 python scripts/test_focused_gui.py
    QT_QPA_PLATFORM=offscreen conda run -n cellpose4 python scripts/test_focused_gui.py
"""
import os, sys, time, json, warnings, logging
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

app = QApplication.instance() or QApplication(sys.argv)

OUT_DIR = "results/focused_gui_tests"
os.makedirs(OUT_DIR, exist_ok=True)

RECORDING = "data/ignasi/C1-IC293__1_MMStack_Pos0-WT.ome-1cropped.tif"

passed = []
failed = []


def screenshot(widget, name):
    """Save a screenshot of a widget."""
    from PyQt5.QtGui import QPixmap
    pix = widget.grab()
    path = os.path.join(OUT_DIR, f"{name}.png")
    pix.save(path)
    return path


def check(name, condition, detail=""):
    if condition:
        passed.append(name)
        print(f"  PASS: {name}")
    else:
        failed.append((name, detail))
        print(f"  FAIL: {name} — {detail}")


def main():
    from gui_focused.main_window import FocusedMainWindow
    from core.io import load_recording
    from core.pipeline import detect, analyze_recording

    w = FocusedMainWindow()
    w.resize(1400, 900)
    w.show()
    app.processEvents()

    print("\n=== 1. Startup state ===")
    screenshot(w, "01_startup")
    check("window_title", "Focused Pipeline" in w.windowTitle())
    check("6_stages", len(w.pipeline.stages) == 6)
    check("default_mode", w.mode == "single")
    check("expected_cells_default", w.params.expected_cells.value() == 1)
    check("multi_widgets_disabled",
          not w.params.search_radius.isEnabled())
    check("load_enabled", w.pipeline.stages["load"].isEnabled())
    check("detect_disabled", not w.pipeline.stages["detect"].isEnabled())
    check("gap_fill_hidden", not w.pipeline.stages["gap_fill"].isVisible())

    print("\n=== 2. Mode switching ===")
    w.pipeline.set_mode("multi")
    app.processEvents()
    check("multi_mode", w.mode == "multi")
    check("expected_cells_auto", w.params.expected_cells.value() == 0)
    check("search_radius_enabled", w.params.search_radius.isEnabled())
    check("gap_fill_visible", w.pipeline.stages["gap_fill"].isVisible())
    screenshot(w, "02_multi_mode")

    w.pipeline.set_mode("single")
    app.processEvents()
    check("back_to_single", w.mode == "single")
    check("expected_cells_1", w.params.expected_cells.value() == 1)
    check("search_radius_disabled", not w.params.search_radius.isEnabled())

    print("\n=== 3. Load recording ===")
    rec = load_recording(RECORDING)
    w.recording = rec
    n = len(rec["frames"])
    w.viewer.set_data(rec["frames"])
    w.pipeline.set_stage_status("load", "done")
    w.pipeline.enable_stage("detect", True)
    w.status.showMessage(f"Loaded: {n} frames")
    app.processEvents()
    screenshot(w, "03_loaded")
    check("frames_loaded", w.recording is not None)
    check("viewer_has_data", w.viewer.frames is not None)
    check("frame_slider_enabled", w.viewer.frame_slider.isEnabled())
    check("frame_count", len(w.viewer.frames) == n)

    print("\n=== 4. Image viewer controls ===")
    w.viewer._on_frame(50)
    app.processEvents()
    check("frame_nav", w.viewer.current_frame == 50)

    w.viewer._auto_bc()
    app.processEvents()
    screenshot(w, "04_auto_bc")
    bc_changed = (w.viewer.bright_slider.value() != 0
                  or w.viewer.contrast_slider.value() != 100)
    check("auto_bc_changed", bc_changed,
          f"bright={w.viewer.bright_slider.value()} "
          f"contrast={w.viewer.contrast_slider.value()}")

    w.viewer._reset_bc()
    app.processEvents()
    check("reset_bc", w.viewer.bright_slider.value() == 0
          and w.viewer.contrast_slider.value() == 100)

    w.viewer._zoom_in()
    app.processEvents()
    xl0 = w.viewer._xlim
    check("zoom_in", xl0[1] - xl0[0] < rec["frames"][0].shape[1])
    screenshot(w, "04b_zoomed_in")

    w.viewer._zoom_fit()
    app.processEvents()
    check("zoom_fit", w.viewer._xlim == (0, rec["frames"][0].shape[1]))

    w.viewer.show_mask = False
    w.viewer._redraw()
    app.processEvents()
    check("mask_toggle_off", not w.viewer.show_mask)
    w.viewer.show_mask = True

    print("\n=== 5. Detection (single-cell) ===")
    t0 = time.time()
    det = detect(rec["frames"], mode="hybrid_cpsam")
    det_elapsed = time.time() - t0
    w.detect_result = det
    w.viewer.update_masks(det["masks"])
    w.pipeline.set_stage_status("detect", "done")
    w.pipeline.enable_stage("edit", True)
    w.pipeline.enable_stage("analyze", True)
    app.processEvents()
    screenshot(w, "05_detected_single")
    check("detection_complete", det["masks"] is not None)
    check("masks_have_cells", det["masks"].any())
    areas = [int(m.sum()) for m in det["masks"]]
    check("all_frames_detected", all(a > 0 for a in areas),
          f"empty frames: {[i for i,a in enumerate(areas) if a==0]}")
    print(f"    Detection time: {det_elapsed:.1f}s")

    print("\n=== 6. Analysis (single-cell) ===")
    t0 = time.time()
    result = analyze_recording(rec, det["masks"])
    an_elapsed = time.time() - t0
    w.analysis_result = result
    w.analysis.set_result(result, mode="single")
    w.pipeline.set_stage_status("analyze", "done")
    w.pipeline.enable_stage("export", True)
    app.processEvents()
    screenshot(w, "06_analyzed_single")
    check("analysis_has_speed", "mean_speed" in result)
    check("analysis_has_shape", "shape_summary" in result)
    check("analysis_has_edge", "edge_summary" in result)
    check("summary_text", len(w.analysis.summary_text.toPlainText()) > 50)
    print(f"    Analysis time: {an_elapsed:.1f}s")
    print(f"    Mean speed: {result.get('mean_speed', 0):.3f} um/min")

    print("\n=== 7. Graph rendering (single-cell) ===")
    from gui_focused.analysis_plots import GRAPH_REGISTRY
    w.analysis.tabs.setCurrentIndex(1)  # switch to Graphs tab
    app.processEvents()
    single_graphs = [(n, fn) for n, (fn, multi) in GRAPH_REGISTRY.items()
                     if not multi]
    for gname, fn in single_graphs:
        try:
            w.analysis._on_graph_selected(gname)
            app.processEvents()
            safe = gname.lower().replace(" ", "_").replace("(", "").replace(")", "")
            screenshot(w, f"07_graph_{safe}")
            check(f"graph_{safe}", True)
        except Exception as e:
            check(f"graph_{safe}", False, str(e))
    w.analysis.tabs.setCurrentIndex(0)  # back to Summary

    print("\n=== 8. Export ===")
    export_dir = os.path.join(OUT_DIR, "export_single")
    os.makedirs(export_dir, exist_ok=True)
    from gui_focused.export_dialog import ExportDialog
    dlg = ExportDialog(
        result=result,
        multi_results=None,
        recording=rec,
        detect_result=det,
        logger=w.logger,
        parent=w,
    )
    dlg.dir_edit.setText(export_dir)
    dlg._on_export()
    app.processEvents()
    check("export_masks", os.path.exists(os.path.join(export_dir, "masks.npz")))
    check("export_metrics", os.path.exists(os.path.join(export_dir, "metrics.json")))
    n_plots = len([f for f in os.listdir(export_dir) if f.endswith(".png")
                   and f != "masks.npz"])
    check("export_plots", n_plots >= 5, f"got {n_plots} plots")

    print("\n=== 9. Verify results consistency ===")
    # Light verification: check analysis was run on the detected masks
    # (no expensive re-detection needed)
    check("analysis_used_correct_frames",
          result.get("n_frames") == len(det["masks"]))
    check("masks_saved_correctly",
          os.path.exists(os.path.join(export_dir, "masks.npz")))
    saved = np.load(os.path.join(export_dir, "masks.npz"))
    check("saved_masks_match",
          np.array_equal(saved["masks"], det["masks"]))
    with open(os.path.join(export_dir, "metrics.json")) as f:
        metrics = json.load(f)
    check("exported_speed_matches",
          abs(metrics.get("mean_speed", 0) - result.get("mean_speed", 0)) < 0.001,
          f"export={metrics.get('mean_speed',0):.4f} vs result={result.get('mean_speed',0):.4f}")
    check("exported_persistence_matches",
          abs(metrics.get("persistence", 0) - result.get("persistence", 0)) < 0.001)

    print("\n=== 10. Params panel context switching ===")
    for stage in ["load", "detect", "gap_fill", "edit", "analyze", "export"]:
        w.params.set_context(stage, w.mode)
        app.processEvents()
        check(f"context_{stage}",
              w.params.stack.currentIndex() >= 0)

    w.close()

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("\nFailed tests:")
        for name, detail in failed:
            print(f"  {name}: {detail}")
    else:
        print("\nAll tests passed!")
    print(f"\nScreenshots: {OUT_DIR}/")

    report = {
        "passed": len(passed),
        "failed": len(failed),
        "failed_tests": [{"name": n, "detail": d} for n, d in failed],
        "screenshots": sorted(os.listdir(OUT_DIR)),
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return len(failed) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
