"""Project save/load + scan handlers extracted from main_window.
Keeps main_window.py under 500 lines."""
import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication
from PyQt5.QtCore import Qt


def on_save_project(win):
    if win.recording is None:
        return
    path, _ = QFileDialog.getSaveFileName(
        win, "Save Project", "",
        "CellScope Project (*.cellscope)")
    if not path:
        return
    if not path.endswith(".cellscope"):
        path += ".cellscope"
    from core.project import save_project
    save_project(
        path, win.recording, win.detect_result,
        win.analysis_result, win.params.get_detect_params(),
        win.mode,
        roi_mask=win.roi.roi_mask if win.roi.has_roi() else None)
    win.logger.log("info", f"Project saved: {path}")
    win.status.showMessage(f"Project saved: {os.path.basename(path)}")


def on_load_project(win):
    path, _ = QFileDialog.getOpenFileName(
        win, "Open Project", "",
        "CellScope Project (*.cellscope)")
    if not path:
        return
    from core.project import load_project
    proj = load_project(path)
    info = proj["recording_info"]
    video = info.get("video_path", "")
    if video and os.path.exists(video):
        win._load_path(video)
    if proj["masks"] is not None:
        win.detect_result = {"masks": proj["masks"]}
        if proj["labels"] is not None:
            win.detect_result["labels"] = proj["labels"]
        masks = proj["labels"] if proj["labels"] is not None \
            else proj["masks"]
        win.viewer.update_masks(masks)
        win.viewer.nav_bar.set_status(proj["masks"])
        win.pipeline.set_stage_status("detect", "done")
        win.pipeline.enable_stage("edit", True)
        win.pipeline.enable_stage("analyze", True)
    win.pipeline.set_mode(proj["mode"])
    win.logger.log("info", f"Project loaded: {path}")
    win.status.showMessage(
        f"Project loaded: {os.path.basename(path)}")


def on_scan_cells(win):
    if win.recording is None:
        QMessageBox.information(win, "Scan", "Load a recording first.")
        return
    try:
        from core.hybrid_cpsam_multi import scan_cell_count
        QApplication.setOverrideCursor(Qt.WaitCursor)
        win.status.showMessage("Scanning for cell count...")
        QApplication.processEvents()
        count = scan_cell_count(
            win.recording["frames"], n_sample=5,
            min_area_px=win.params.min_area.value())
        win.params.expected_cells.setValue(max(1, count))
        win.logger.log("info", f"Scan: {count} cells detected")
        win.status.showMessage(f"Scan complete: {count} cells/frame")
    except Exception as e:
        QMessageBox.warning(win, "Scan Error", str(e))
    finally:
        QApplication.restoreOverrideCursor()
