"""Smoke-test all 5 cellscope GUIs offscreen.

For each GUI we:
  1. instantiate the main window class
  2. show it (forces layout)
  3. capture a screenshot to results/gui_smoke/
  4. close it

Exits non-zero if any GUI fails to import or instantiate.

Usage:
  conda run -n cellpose python scripts/test_all_guis.py
"""
import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

from PyQt5.QtWidgets import QApplication, QMainWindow

OUT_DIR = "results/gui_smoke"
os.makedirs(OUT_DIR, exist_ok=True)

app = QApplication.instance() or QApplication(sys.argv)


def screenshot(widget, name):
    pix = widget.grab()
    path = os.path.join(OUT_DIR, f"{name}.png")
    pix.save(path)
    return path


GUIS = [
    # (label, callable returning the top-level window)
    ("focused",
     lambda: __import__(
         "gui_focused.main_window", fromlist=["FocusedMainWindow"]
     ).FocusedMainWindow()),
    ("batch",
     lambda: __import__(
         "gui_batch.batch_window", fromlist=["BatchWindow"]
     ).BatchWindow()),
    ("tracking",
     lambda: __import__(
         "gui_tracking.tracking_window", fromlist=["TrackingWindow"]
     ).TrackingWindow()),
    ("editor",
     lambda: __import__(
         "gui_editor.editor_window", fromlist=["EditorWindow"]
     ).EditorWindow()),
    ("training",
     lambda: __import__(
         "gui_training.training_window", fromlist=["TrainingWindow"]
     ).TrainingWindow()),
]


def main():
    results = []
    for label, factory in GUIS:
        print(f"\n=== {label} ===")
        try:
            w = factory()
            # EditorWindow is a wrapper, the real Qt window is .editor
            qt_widget = getattr(w, "editor", w)
            qt_widget.resize(1400, 900)
            w.show()
            app.processEvents()
            shot = screenshot(qt_widget, label)
            print(f"  ✓ launched + screenshot saved: {shot}")
            results.append((label, "OK", shot))
            w.close()
            app.processEvents()
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ✗ FAILED")
            print(tb)
            results.append((label, f"FAIL: {e}", None))

    # Suite launcher (tkinter, not Qt — different harness)
    print("\n=== suite (tkinter) ===")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "main_suite", "main_suite.py")
        mod = importlib.util.module_from_spec(spec)
        # Don't actually run mainloop — just import and inspect
        spec.loader.exec_module(mod)
        print(f"  ✓ main_suite.py imports cleanly")
        results.append(("suite", "OK (import only)", None))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  ✗ FAILED: {e}")
        print(tb)
        results.append(("suite", f"FAIL: {e}", None))

    print("\n=== summary ===")
    n_ok = sum(1 for _, status, _ in results if status.startswith("OK"))
    for label, status, shot in results:
        flag = "✓" if status.startswith("OK") else "✗"
        print(f"  {flag} {label:10s} {status}")
    print(f"\n{n_ok}/{len(results)} OK")

    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
