"""Single-cell candidate-review GUI.

Step through each difficult cell's frames and PICK the best outline candidate
(or keep Original), fast:
  ←/→            previous / next frame
  PageUp/Down    previous / next recording   (also n / p)
  1 / 2 / 3 / 4  pick Original / Moderate / Conservative / SAM2  (auto-saves)
  Tab            cycle candidates
  F              flag this cell for manual brushing later
  scroll         zoom in/out at the cursor;  Home resets to fit the cell

Launch (from the cellpose4 env so single-channel DIC loads):
    conda run -n cellpose4 python main_review.py
    conda run -n cellpose4 python main_review.py <by_condition_dir> <choices.csv>

Defaults to the IC293 set; picks persist to
ic293_analysis/review/review_choices.csv and are applied by
scripts/ic293_apply_choices.py. Only cells whose candidates differ are shown.
"""
import os
import sys

from PyQt5.QtWidgets import QApplication

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    app = QApplication(sys.argv)
    root = (sys.argv[1] if len(sys.argv) > 1 else
            os.path.join(PROJECT_ROOT, "ic293_analysis", "by_condition"))
    review_dir = os.path.join(PROJECT_ROOT, "ic293_analysis", "review")
    choices = (sys.argv[2] if len(sys.argv) > 2 else
               os.path.join(review_dir, "review_choices.csv"))
    manifest = os.path.join(review_dir, "review_candidates.json")
    try:
        from scripts.ic293_common import ANALYSIS_EXCLUDE
        exclude = ANALYSIS_EXCLUDE
    except Exception:
        exclude = frozenset()

    from gui_review.review_window import ReviewWindow
    win = ReviewWindow(root, choices, manifest, exclude=exclude)
    try:
        from gui_focused.remote_control import attach_minimal
        attach_minimal(win, gui_type="review", default_port=8773)
    except Exception:
        pass
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
