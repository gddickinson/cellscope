"""Standalone mask editor with results viewer.

Launch:
    conda run -n cellpose4 python main_editor.py
    conda run -n cellpose4 python main_editor.py path/to/video.tif
"""
import sys
from PyQt5.QtWidgets import QApplication
from gui_editor.editor_window import EditorWindow


def main():
    app = QApplication(sys.argv)
    video = sys.argv[1] if len(sys.argv) > 1 else None
    win = EditorWindow(video_path=video)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
