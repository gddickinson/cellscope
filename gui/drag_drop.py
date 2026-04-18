"""Reusable drag-and-drop mixin for loading recordings/files.

Usage: call `setup_drag_drop(window, callback)` after __init__.
The callback receives the dropped file path as a string.
"""

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".tif", ".tiff")
MASK_EXTS = (".npz",)
ALL_EXTS = VIDEO_EXTS + MASK_EXTS + (".png",)


def setup_drag_drop(widget, on_drop, extensions=VIDEO_EXTS):
    """Enable drag-and-drop on a QWidget.

    Args:
        widget: QMainWindow or QWidget to accept drops
        on_drop: callback(path: str) called when a valid file is dropped
        extensions: tuple of accepted file extensions
    """
    widget.setAcceptDrops(True)
    widget._dd_callback = on_drop
    widget._dd_extensions = extensions

    original_drag = getattr(widget.__class__, "dragEnterEvent",
                            lambda s, e: None)
    original_drop = getattr(widget.__class__, "dropEvent",
                            lambda s, e: None)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile().lower()
                if any(path.endswith(ext) for ext in self._dd_extensions):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if any(path.lower().endswith(ext)
                   for ext in self._dd_extensions):
                self._dd_callback(path)
                return

    widget.__class__.dragEnterEvent = dragEnterEvent
    widget.__class__.dropEvent = dropEvent
