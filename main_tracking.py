"""Tracking & Analysis GUI with batch comparison.

Launch:
    conda run -n cellpose4 python main_tracking.py
"""
import sys
from PyQt5.QtWidgets import QApplication
from gui_tracking.tracking_window import TrackingWindow


def main():
    app = QApplication(sys.argv)
    win = TrackingWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
