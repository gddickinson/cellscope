"""Cellpose fine-tuning GUI.

Launch:
    conda run -n cellpose4 python main_training.py
"""
import sys
from PyQt5.QtWidgets import QApplication
from gui_training.training_window import TrainingWindow


def main():
    app = QApplication(sys.argv)
    win = TrainingWindow()
    from gui_focused.remote_control import attach_minimal
    attach_minimal(win, gui_type="training", default_port=8769)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
