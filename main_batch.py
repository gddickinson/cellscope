"""Batch analysis GUI — process multiple recordings.

Launch:
    conda run -n cellpose4 python main_batch.py
"""
import sys
from PyQt5.QtWidgets import QApplication
from gui_batch.batch_window import BatchWindow


def main():
    app = QApplication(sys.argv)
    win = BatchWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
