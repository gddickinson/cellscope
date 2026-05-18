"""Restart the CellScope focused GUI with a saved .cellscope project
pre-loaded AND re-run analysis (so all graphs are immediately
populated). The slow detection step is skipped because the masks are
already in the project.

Edit PROJECT below to point at your saved project file.

Run:
    cd /Users/george/claude_test/cellscope
    conda run -n cellpose4 python scripts/restart_with_saved.py
"""
import os
import sys

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
PROJECT = "/Users/george/Desktop/test.cellscope"

os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication


def main():
    if not os.path.exists(PROJECT):
        print(f"ERROR: project file not found: {PROJECT}")
        sys.exit(1)

    app = QApplication(sys.argv)
    from gui_focused.main_window import FocusedMainWindow
    from core.project import load_project

    w = FocusedMainWindow()
    w.resize(1600, 1000)
    w.show()
    w.raise_()
    w.activateWindow()

    # Bypass the file-picker dialog — call the loader directly with our path
    proj = load_project(PROJECT)
    info = proj["recording_info"]
    video = info.get("video_path", "")
    if video and os.path.exists(video):
        w._load_path(video)
    else:
        print(f"ERROR: recording referenced by project is missing: {video}")
        return app.exec_()

    if proj["masks"] is not None:
        w.detect_result = {"masks": proj["masks"]}
        if proj.get("labels") is not None:
            w.detect_result["labels"] = proj["labels"]
        masks = (proj["labels"] if proj.get("labels") is not None
                 else proj["masks"])
        w.viewer.update_masks(masks)
        w.viewer.nav_bar.set_status(proj["masks"])
        w.pipeline.set_stage_status("detect", "done")
        w.pipeline.enable_stage("edit", True)
        w.pipeline.enable_stage("analyze", True)
    if proj.get("mode"):
        w.pipeline.set_mode(proj["mode"])
        w.mode = proj["mode"]
    w.logger.log("info", f"Project loaded: {PROJECT}")

    # Re-run analysis automatically so graphs are populated
    w.params.compute_vampire.setChecked(True)
    w.params.compute_states.setChecked(True)
    QTimer.singleShot(300, w._on_analyze)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
