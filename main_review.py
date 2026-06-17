"""Rapid single-cell review GUI.

Step through each cell's frames and flag problematic detections fast:
  ←/→            previous / next frame
  PageUp/Down    previous / next recording   (also n / p)
  scroll         zoom in/out at the cursor;  Home resets to fit the cell
  F  /  Space    flag / unflag this cell as problematic (auto-saves)

Launch (from the cellpose4 env so single-channel DIC loads):
    conda run -n cellpose4 python main_review.py
    conda run -n cellpose4 python main_review.py <by_condition_dir> <flags.csv>

Defaults to the IC293 analysis set; flags persist to
ic293_analysis/review/review_flags.csv.
"""
import os
import sys

from PyQt5.QtWidgets import QApplication

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    app = QApplication(sys.argv)
    root = (sys.argv[1] if len(sys.argv) > 1 else
            os.path.join(PROJECT_ROOT, "ic293_analysis", "by_condition"))
    flags = (sys.argv[2] if len(sys.argv) > 2 else
             os.path.join(PROJECT_ROOT, "ic293_analysis", "review",
                          "review_flags.csv"))
    try:
        from scripts.ic293_common import ANALYSIS_EXCLUDE
        exclude = ANALYSIS_EXCLUDE
    except Exception:
        exclude = frozenset()

    from gui_review.review_window import ReviewWindow
    win = ReviewWindow(root, flags, exclude=exclude)
    try:
        from gui_focused.remote_control import attach_minimal
        attach_minimal(win, gui_type="review", default_port=8773)
    except Exception:
        pass
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
