"""Phase G: Parameter flow verification."""
from PyQt5.QtWidgets import QApplication
from ._common import check, shot, trim, SINGLE_CELL


def run():
    from gui_focused.main_window import FocusedMainWindow
    from core.io import load_recording
    from core.pipeline import detect

    app = QApplication.instance()

    print("\n=== Phase G: Parameter flow ===")
    w = FocusedMainWindow()
    w.resize(1400, 900)
    w.show()
    app.processEvents()

    rec = load_recording(SINGLE_CELL)
    rec["frames"] = trim(rec["frames"], 3)
    w.recording = rec
    w.viewer.set_data(rec["frames"])
    w.params.set_from_recording(rec)
    app.processEvents()

    # Baseline detection (sanity)
    print("  baseline detection (3 frames)...")
    det = detect(rec["frames"], mode="hybrid_cpsam")
    areas = [int(m.sum()) for m in det["masks"]]
    check("G", "baseline_detects", all(a > 0 for a in areas),
          f"areas={areas}")

    # Param plumbing
    p = w.params.get_detect_params()
    for key in ("min_area_px", "use_deepsea", "use_fallback",
                "modality"):
        check("G", f"params_has_{key}", key in p)

    # min_area change
    w.params.min_area.setValue(5000)
    p2 = w.params.get_detect_params()
    check("G", "min_area_propagates", p2["min_area_px"] == 5000,
          f"got {p2['min_area_px']}")
    w.params.min_area.setValue(200)

    # DeepSea toggle
    w.params.use_deepsea.setChecked(False)
    check("G", "deepsea_off_propagates",
          w.params.get_detect_params()["use_deepsea"] is False)
    w.params.use_deepsea.setChecked(True)

    # Fallback toggle
    w.params.use_fallback.setChecked(False)
    check("G", "fallback_off_propagates",
          w.params.get_detect_params()["use_fallback"] is False)
    w.params.use_fallback.setChecked(True)

    # Scale overrides
    w.params.um_per_px.setValue(0.5)
    w.params.time_interval.setValue(5.0)
    so = w.params.get_scale_overrides()
    check("G", "scale_override_um", so.get("um_per_px") == 0.5)
    check("G", "scale_override_t", so.get("time_interval_min") == 5.0)

    # Zero scale → returns 0.0 (the analyzer interprets 0 as "use
    # the recording's metadata value" downstream).
    w.params.um_per_px.setValue(0.0)
    w.params.time_interval.setValue(0.0)
    so2 = w.params.get_scale_overrides()
    check("G", "scale_zero_returns_zero",
          so2.get("um_per_px") == 0.0, f"got {so2}")

    shot(w, "G_01_params_panel")
    w.close()
    app.processEvents()
