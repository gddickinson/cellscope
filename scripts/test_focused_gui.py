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
try:
    os.makedirs(OUT_DIR, exist_ok=True)
except OSError:
    # `results/` can be a dead symlink (e.g. pointing at an unmounted
    # drive). Fall back to a local, always-writable directory so the
    # test still runs and saves its screenshots.
    OUT_DIR = os.path.join("_test_output", "focused_gui_tests")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[note] results/ unwritable; using {OUT_DIR}")

# Test recording — overridable so the test never hard-codes a path that
# can disappear (the original IC293 cropped WT lived on a drive that
# failed; its local pointers are now dead symlinks). Resolution order:
#   1. first positional CLI arg (a path)
#   2. $CELLSCOPE_TEST_RECORDING
#   3. the bundled single-cell crop (generated from a reviewed IC295 mask
#      stack by scripts/make_single_cell_example.py)
#   4. the single-cell phase example
def _resolve_recording():
    cli = (sys.argv[1] if len(sys.argv) > 1
           and not sys.argv[1].startswith("-") else None)
    for c in (cli,
              os.environ.get("CELLSCOPE_TEST_RECORDING"),
              "data/examples/single_cell_crop_wt/single_cell_crop_wt.tif",
              "data/examples/single_cell_phase_WT/single_cell_phase_WT.tif"):
        if c and os.path.exists(c):
            return c
    raise SystemExit(
        "No test recording found. Pass a path, set "
        "CELLSCOPE_TEST_RECORDING=<path>, or generate one with "
        "scripts/make_single_cell_example.py.")


RECORDING = _resolve_recording()

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
    from core.io import load_recording, detect_channels
    from core.pipeline import detect, analyze_recording
    print(f"Test recording: {RECORDING}")

    w = FocusedMainWindow()
    w.resize(1400, 900)
    w.show()
    app.processEvents()

    print("\n=== 1. Startup state ===")
    screenshot(w, "01_startup")
    check("window_title", "Focused Pipeline" in w.windowTitle())
    # Pipeline currently has 5 stages: load, detect, edit, analyze, export.
    # (Gap-fill is now part of the detect stage, not a separate pipeline
    # button.)
    expected_stages = {"load", "detect", "edit", "analyze", "export"}
    check("5_stages",
          set(w.pipeline.stages.keys()) == expected_stages,
          f"got {set(w.pipeline.stages.keys())}")
    check("default_mode", w.mode == "single")
    check("expected_cells_default", w.params.expected_cells.value() == 1)
    check("multi_widgets_disabled",
          not w.params.search_radius.isEnabled())
    check("load_enabled", w.pipeline.stages["load"].isEnabled())
    check("detect_disabled", not w.pipeline.stages["detect"].isEnabled())

    print("\n=== 2. Mode switching ===")
    w.pipeline.set_mode("multi")
    app.processEvents()
    check("multi_mode", w.mode == "multi")
    check("expected_cells_auto", w.params.expected_cells.value() == 0)
    check("search_radius_enabled", w.params.search_radius.isEnabled())
    screenshot(w, "02_multi_mode")

    w.pipeline.set_mode("single")
    app.processEvents()
    check("back_to_single", w.mode == "single")
    check("expected_cells_1", w.params.expected_cells.value() == 1)
    check("search_radius_disabled", not w.params.search_radius.isEnabled())

    print("\n=== 3. Load recording ===")
    # Channel-aware: a multichannel .ome.tif loads DIC=ch1, fluo=ch0;
    # a plain single-channel stack loads as-is.
    _nch = (detect_channels(RECORDING)
            if RECORDING.lower().endswith((".tif", ".tiff")) else 1)
    rec = (load_recording(RECORDING, dic_channel=1, fluo_channel=0)
           if _nch > 1 else load_recording(RECORDING))
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
    nav = min(50, n - 1)  # relative to recording length (was hardcoded 50)
    w.viewer._on_frame(nav)
    app.processEvents()
    check("frame_nav", w.viewer.current_frame == nav)

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
    # AnalysisView no longer has an internal QTabWidget — Summary, Graphs,
    # and Log are now separate dock widgets. Bring Graphs to the front.
    w.dock_graphs.raise_()
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
    w.dock_summary.raise_()  # back to Summary

    print("\n=== 7b. Colour masks by result (viewer + metric_coloring) ===")
    from gui.metric_coloring import metric_names, ID_METRIC
    all_render = True
    for name in metric_names():
        try:
            w.viewer.color_combo.setCurrentText(name)
            app.processEvents()
        except Exception as e:
            all_render = False
            check(f"colour_by[{name}]", False, str(e))
    check("colour_by_all_render", all_render,
          f"{len(metric_names())} options")
    # Continuous metric → colorizer + gradient legend active
    w.viewer.color_combo.setCurrentText("Mean speed")
    app.processEvents()
    screenshot(w, "07b_colour_mean_speed")
    check("colour_by_colorizer_active",
          w.viewer._metric_colorizer is not None)
    check("colour_by_legend_visible",
          not w.viewer.metric_legend.isHidden())
    # Categorical cell-state metric
    w.viewer.color_combo.setCurrentText("Cell state (balled / attached)")
    app.processEvents()
    screenshot(w, "07b_colour_cell_state")
    check("colour_by_state_active",
          w.viewer._metric_colorizer is not None)
    # Back to Cell ID clears the colorizer + hides the legend
    w.viewer.color_combo.setCurrentText(ID_METRIC)
    app.processEvents()
    check("colour_by_id_clears",
          w.viewer._metric_colorizer is None
          and w.viewer.metric_legend.isHidden())

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
    for stage in ["load", "detect", "edit", "analyze", "export"]:
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
