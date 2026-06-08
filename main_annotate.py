"""Standalone annotation GUI — label cell-frames into classes for training.

Launch:
    conda run -n cellpose4 python main_annotate.py
    conda run -n cellpose4 python main_annotate.py path/to/labels.csv

With no argument it opens ic295_analysis/state_labels/labels.csv if present.
"""
import os
import sys
from PyQt5.QtWidgets import QApplication
from gui_annotate.annotate_window import AnnotateWindow


def main():
    app = QApplication(sys.argv)
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    if csv_path is None:
        default = os.path.join("ic295_analysis", "state_labels", "labels.csv")
        if os.path.exists(default):
            csv_path = default
    win = AnnotateWindow(csv_path=csv_path)
    from gui_focused.remote_control import attach_minimal
    attach_minimal(win, gui_type="annotate", default_port=8772)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
