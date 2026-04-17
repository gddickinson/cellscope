"""Enhanced mask editor window with dockable results panel.

Wraps the existing MaskEditor (QMainWindow) and adds a results
viewer as a QDockWidget — no modifications to the original editor.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDockWidget

from gui.mask_editor import MaskEditor
from gui_editor.results_panel import ResultsPanel


class EditorWindow:
    """Launches MaskEditor with an attached results panel dock."""

    def __init__(self, video_path=None, mask_path=None):
        self.editor = MaskEditor(video_path=video_path,
                                  mask_path=mask_path)
        self.editor.setWindowTitle(
            "CellScope — Mask Editor")
        self.results_panel = ResultsPanel()
        self._dock = QDockWidget("Analysis Results", self.editor)
        self._dock.setWidget(self.results_panel)
        self._dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable)
        self.editor.addDockWidget(Qt.RightDockWidgetArea, self._dock)

    def show(self):
        self.editor.show()

    def close(self):
        self.editor.close()
