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
    from gui_focused.remote_control import attach_minimal
    # EditorWindow is a plain wrapper; attach to the actual QMainWindow
    # (win.editor) so /save_screenshot grabs the right widget too.
    attach_minimal(win.editor, gui_type="editor", default_port=8767)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
