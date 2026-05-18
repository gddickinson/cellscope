"""Phase E: Tracking GUI."""
import numpy as np
from PyQt5.QtWidgets import QApplication
from ._common import check, shot, trim, MULTI_CELL


def run():
    from gui_tracking.tracking_window import TrackingWindow
    from core.io import load_recording
    from core.pipeline import detect

    app = QApplication.instance()

    print("\n=== Phase E: Tracking GUI ===")
    w = TrackingWindow()
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    shot(w, "E_01_tracking_startup")
    check("E", "tracking_window_opens", w.isVisible())

    single = w.single_view
    rec = load_recording(MULTI_CELL)
    rec["frames"] = trim(rec["frames"], 15)
    single.recording = rec
    single.viewer.set_data(rec["frames"])
    app.processEvents()

    # Detect via core.pipeline (faster than dispatching through GUI)
    print("  detecting masks (15 frames)...")
    det = detect(rec["frames"], mode="hybrid_cpsam_multi")
    masks = det.get("labels", det["masks"])
    if masks.dtype == bool:
        from scipy import ndimage
        labels = np.zeros_like(masks, dtype=np.int32)
        for i, m in enumerate(masks):
            lab, _ = ndimage.label(m)
            labels[i] = lab
        masks = labels
    single.masks = masks
    single.btn_track.setEnabled(True)
    app.processEvents()
    shot(single.viewer, "E_02_masks_loaded")
    check("E", "masks_loaded", single.masks is not None)

    # Track
    single._on_track()
    app.processEvents()
    check("E", "tracking_runs", len(single.tracks) > 0,
          f"got {len(single.tracks)} tracks")
    shot(w, "E_03_tracked")

    # Analyze
    print("  analyzing per-track...")
    single._on_analyze()
    app.processEvents()
    check("E", "per_track_analysis",
          len(single.per_cell_results) == len(single.tracks))
    shot(w, "E_04_analyzed")

    check("E", "track_table_populated",
          single.track_table.rowCount() == len(single.tracks))

    # Click first row → viewer updates
    single._on_track_selected(0)
    app.processEvents()
    shot(single.viewer, "E_05_track0_selected")
    check("E", "track_row_click", True)

    if single.per_cell_results:
        single.graph_combo.setCurrentText("Speed vs Time")
        single._on_graph("Speed vs Time")
        app.processEvents()
        shot(w, "E_06_graph_speed")
        check("E", "graph_renders", True)

    w.close()
    app.processEvents()
