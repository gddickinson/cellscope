"""Phase F: Training + Editor GUIs."""
import os, tempfile
import numpy as np
from PyQt5.QtWidgets import QApplication
from ._common import check, shot, SINGLE_CELL


def run():
    app = QApplication.instance()

    print("\n=== Phase F: Training + Editor ===")

    # --- Training GUI ---
    from gui_training.training_window import TrainingWindow
    tw = TrainingWindow()
    tw.resize(1400, 900)
    tw.show()
    app.processEvents()
    shot(tw, "F_01_training_startup")
    check("F", "training_opens", tw.isVisible())
    check("F", "training_has_train_btn", hasattr(tw, "btn_train"))
    check("F", "training_has_stop_btn", hasattr(tw, "btn_stop"))

    # Optional: scan a training directory if one exists
    train_dirs = [
        "/Users/george/claude_test/piezo1_analysis/data/training/vampire",
        "/Users/george/claude_test/piezo1_analysis/data/manual_gt/control",
    ]
    for d in train_dirs:
        if os.path.exists(d):
            tw.data_edit.setText(d)
            try:
                tw._on_scan()
                app.processEvents()
                shot(tw, "F_02_training_scanned")
                check("F", "training_scan_runs", True)
                break
            except Exception as e:
                check("F", "training_scan_runs", False, str(e))
                break
    tw.close()
    app.processEvents()

    # --- Editor GUI ---
    from gui_editor.editor_window import EditorWindow
    from core.io import load_recording, load_video

    rec = load_recording(SINGLE_CELL)
    H, W = rec["frames"][0].shape
    # Mask stack must match the full video length (editor blocks on a
    # QMessageBox warning otherwise — invisible + blocking in offscreen
    # mode).
    full_frames = load_video(SINGLE_CELL)
    n_full = len(full_frames)
    fake = np.zeros((n_full, H, W), dtype=np.int32)
    fake[:, H // 3:2 * H // 3, W // 3:2 * W // 3] = 1
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez_compressed(tmp.name, masks=fake)
    tmp.close()

    ew = EditorWindow(video_path=SINGLE_CELL, mask_path=tmp.name)
    qt = getattr(ew, "editor", ew)
    qt.resize(1400, 900)
    ew.show()
    app.processEvents()
    shot(qt, "F_03_editor_open")
    check("F", "editor_opens", qt.isVisible())
    check("F", "editor_has_dock", hasattr(ew, "_dock"))
    check("F", "editor_has_results_panel",
          hasattr(ew, "results_panel"))

    ew.close()
    app.processEvents()
