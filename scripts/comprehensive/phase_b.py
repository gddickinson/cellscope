"""Phase B: Multi-cell detection & analysis in the Focused GUI."""
import time
from PyQt5.QtWidgets import QApplication
from ._common import check, shot, trim, MULTI_CELL


def run():
    from gui_focused.main_window import FocusedMainWindow
    from core.io import load_recording
    from core.pipeline import detect, analyze_recording
    from gui_focused.analysis_plots import GRAPH_REGISTRY

    app = QApplication.instance()

    print("\n=== Phase B: Multi-cell detection & analysis ===")
    w = FocusedMainWindow()
    w.resize(1400, 900)
    w.show()
    app.processEvents()

    # Switch to multi-cell mode
    w.pipeline.set_mode("multi")
    app.processEvents()
    check("B", "mode_switch_multi", w.mode == "multi")
    shot(w, "B_01_multi_mode")

    # Load multi-cell recording
    rec = load_recording(MULTI_CELL)
    rec["frames"] = trim(rec["frames"], 20)
    w.recording = rec
    w.viewer.set_data(rec["frames"])
    w.pipeline.set_stage_status("load", "done")
    w.pipeline.enable_stage("detect", True)
    app.processEvents()
    shot(w, "B_02_multi_loaded")
    check("B", "multi_loaded", w.viewer.frames is not None)

    # Multi-cell detection
    print("  running hybrid_cpsam_multi (20 frames)...")
    t0 = time.time()
    det = detect(rec["frames"], mode="hybrid_cpsam_multi")
    elapsed = time.time() - t0
    w.detect_result = det
    masks = det.get("labels", det["masks"])
    w.viewer.update_masks(masks)
    w.pipeline.set_stage_status("detect", "done")
    w.pipeline.enable_stage("analyze", True)
    app.processEvents()
    shot(w, "B_03_multi_detected")
    n_cells = int(masks.max() if masks.dtype != bool else
                  masks.any(axis=(1, 2)).sum())
    check("B", "multi_detection_complete",
          n_cells > 0,
          f"max cell ID = {n_cells} ({elapsed:.1f}s)")

    # Per-cell analytics
    print("  building per-cell analytics...")
    per_cell = []
    tracks = det.get("tracks", [])
    if not tracks:
        from core.multi_cell import track_all_cells
        tracks = track_all_cells(masks, min_area_px=200,
                                 spawn_new_tracks=True,
                                 min_track_length=3)
    for tid, t in enumerate(tracks[:5]):
        stack = t["stack"] if "stack" in t else (masks == tid + 1)
        r = analyze_recording(rec, stack)
        r["cell_id"] = tid + 1
        r["track_info"] = {
            "first_frame": t.get("first_frame", 0),
            "frames_tracked": int(stack.any(axis=(1, 2)).sum()),
            "parent_id": t.get("parent_id"),
        }
        per_cell.append(r)
    check("B", "per_cell_results", len(per_cell) > 0,
          f"got {len(per_cell)} cells")

    w.analysis.set_multi_result(per_cell)
    w.pipeline.set_stage_status("analyze", "done")
    w.dock_summary.raise_()
    app.processEvents()
    shot(w, "B_04_multi_summary")
    check("B", "multi_summary_renders",
          "Multi-cell" in w.analysis.summary_text.toPlainText())

    # Cell selector
    w.dock_graphs.raise_()
    app.processEvents()
    check("B", "cell_combo_visible", w.analysis.cell_combo.isVisible())
    check("B", "cell_combo_populated",
          w.analysis.cell_combo.count() == len(per_cell) + 1,
          f"got {w.analysis.cell_combo.count()} entries")

    # Cycle every graph
    print("  rendering all graphs...")
    n_ok = 0
    for gname, (fn, _) in GRAPH_REGISTRY.items():
        try:
            w.analysis._on_graph_selected(gname)
            app.processEvents()
            safe = (gname.lower().replace(" ", "_")
                    .replace("(", "").replace(")", ""))
            shot(w, f"B_graph_{safe}")
            n_ok += 1
        except Exception as e:
            check("B", f"graph_{gname}", False, str(e))
    check("B", "all_graphs_render", n_ok == len(GRAPH_REGISTRY),
          f"{n_ok}/{len(GRAPH_REGISTRY)} ok")

    # Individual cell selection
    if len(per_cell) >= 2:
        w.analysis.cell_combo.setCurrentIndex(1)
        app.processEvents()
        w.analysis._on_graph_selected("Trajectory")
        app.processEvents()
        shot(w, "B_graph_cell1_trajectory")
        check("B", "individual_cell_select", True)

    w.close()
    app.processEvents()
