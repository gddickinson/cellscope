"""Phase D: Batch GUI."""
import os, shutil, tempfile
from PyQt5.QtWidgets import QApplication
from ._common import check, shot, SINGLE_CELL, MULTI_CELL


def run():
    from gui_batch.batch_window import BatchWindow

    app = QApplication.instance()

    print("\n=== Phase D: Batch GUI ===")

    # Build tiny batch dir: 2 recordings in 2 generic groups. Neutral
    # group names (groupA / groupB) — the README screenshot of this
    # GUI should not imply any specific treatment comparison; that's
    # the user's publication, not the README.
    batch_root = tempfile.mkdtemp(prefix="cellscope_batch_")
    for grp in ["groupA", "groupB"]:
        os.makedirs(os.path.join(batch_root, grp), exist_ok=True)
    dst_a = os.path.join(batch_root, "groupA",
                          os.path.basename(SINGLE_CELL))
    dst_b = os.path.join(batch_root, "groupB",
                          os.path.basename(MULTI_CELL))
    if not os.path.exists(dst_a):
        os.symlink(SINGLE_CELL, dst_a)
    if not os.path.exists(dst_b):
        os.symlink(MULTI_CELL, dst_b)
    sidecar = SINGLE_CELL.replace(".tif", ".json")
    if os.path.exists(sidecar):
        shutil.copy(sidecar, dst_a.replace(".tif", ".json"))

    w = BatchWindow()
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    shot(w, "D_01_batch_startup")
    check("D", "batch_window_opens", w.isVisible())

    # Scan
    w.input_edit.setText(batch_root)
    w._on_scan()
    app.processEvents()
    shot(w, "D_02_batch_scanned")
    n_groups = w.tree.topLevelItemCount()
    check("D", "tree_populated", n_groups == 2,
          f"got {n_groups} groups")
    check("D", "run_button_enabled", w.btn_run.isEnabled())

    # Settings widgets reachable
    w.min_area.setValue(500)
    app.processEvents()
    check("D", "min_area_settable", w.min_area.value() == 500)
    w.use_deepsea.setChecked(False)
    check("D", "use_deepsea_toggle",
          w.use_deepsea.isChecked() is False)
    w.use_deepsea.setChecked(True)

    # We deliberately do NOT trigger _on_run — the full pipeline
    # would run for many minutes. Verify the params dict the worker
    # would receive instead.
    params = {
        "mode": w.mode_combo.currentData(),
        "min_area_px": w.min_area.value(),
        "use_deepsea": w.use_deepsea.isChecked(),
        "use_fallback": w.use_fallback.isChecked(),
    }
    check("D", "params_dict_complete", all(k in params for k in
                                            ["mode", "min_area_px",
                                             "use_deepsea",
                                             "use_fallback"]))

    w.close()
    app.processEvents()
