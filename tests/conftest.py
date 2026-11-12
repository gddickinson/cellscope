"""Shared test setup for tests/.

The GUIs are PyQt5. A shared venv may also hold PyQt6 / PySide6, so pin the Qt binding
for pyqtgraph, matplotlib and pytest-qt before anything imports Qt, and run headless.
GUI tests must not leave windows to random garbage collection (tearing down Qt widgets
while a later test processes events can abort the interpreter), so this autouse fixture
closes every top-level widget the test created, flushes events and collects garbage at
the end of the test that made it.
"""
import gc
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_QT_API", "pyqt5")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
os.environ.setdefault("QT_API", "pyqt5")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _destroy_qt_widgets():
    yield
    QtWidgets = sys.modules.get("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() if QtWidgets else None
    if app is None:
        return
    from PyQt5 import QtCore
    for w in QtWidgets.QApplication.topLevelWidgets():
        for th in w.findChildren(QtCore.QThread):
            if th.isRunning():
                th.quit()
                th.wait(10000)
        w.close()
    w = None
    app.processEvents()
    gc.collect()
    app.processEvents()
