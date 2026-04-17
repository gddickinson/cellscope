"""Focused pipeline GUI — streamlined for cpsam analysis.

Launch:
    conda run -n cellpose4 python main_focused.py
"""
import sys
from PyQt5.QtWidgets import QApplication
from gui_focused.main_window import FocusedMainWindow


def main():
    app = QApplication(sys.argv)
    win = FocusedMainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
