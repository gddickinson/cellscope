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
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
