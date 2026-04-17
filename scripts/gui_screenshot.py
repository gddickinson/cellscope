"""Headless Qt GUI screenshot harness.

Launches a MainWindow using the Qt offscreen platform (no display needed)
and provides helpers to switch tabs, apply presets, and snapshot widgets
to PNG files.

Typical use (from another script):

    from scripts.gui_screenshot import HeadlessGUI
    with HeadlessGUI("v2") as gui:
        gui.apply_preset("Best for Jesse")
        gui.snapshot("out/v2_best_jesse.png")

Also usable standalone:

    python scripts/gui_screenshot.py --flavor v2 --out v2_main.png
    python scripts/gui_screenshot.py --flavor classic --preset "Cascade (no refine)" --out classic_cascade.png

Force offscreen with `QT_QPA_PLATFORM=offscreen` — set automatically
unless already defined.
"""
import os
import sys
import time
import argparse

# Force offscreen BEFORE importing Qt
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication, QWidget  # noqa: E402
from PyQt5.QtCore import Qt, QSize  # noqa: E402
from PyQt5.QtGui import QPixmap  # noqa: E402


_app_singleton = None


def get_app():
    """Return (and lazily create) the QApplication singleton."""
    global _app_singleton
    if _app_singleton is None:
        _app_singleton = QApplication.instance() or QApplication(sys.argv)
    return _app_singleton


def save_widget_pixmap(widget: QWidget, out_path: str,
                       min_size: QSize = QSize(1400, 900)) -> str:
    """Grab a widget to a PNG. Ensures a minimum size for layout."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Adopt a size large enough for the full panel layout
    size = widget.sizeHint()
    w = max(size.width(), min_size.width())
    h = max(size.height(), min_size.height())
    widget.resize(w, h)
    widget.show()
    get_app().processEvents()
    # Ensure layout settles
    for _ in range(3):
        get_app().processEvents()
    pix: QPixmap = widget.grab()
    pix.save(out_path, "PNG")
    return out_path


class HeadlessGUI:
    """Context manager that builds a MainWindow and provides helpers.

    Args:
        flavor: "v2" (gui.main_window) or "classic" (gui_classic.gui_main).
        min_size: minimum window size for screenshots.
    """

    def __init__(self, flavor: str = "v2",
                 min_size: QSize = QSize(1500, 950)):
        self.flavor = flavor
        self.min_size = min_size
        self.app = get_app()
        self.window = None

    def __enter__(self):
        if self.flavor == "v2":
            from gui.main_window import MainWindow
        elif self.flavor == "classic":
            from gui_classic.gui_main import MainWindow
        else:
            raise ValueError(f"Unknown flavor: {self.flavor}")
        self.window = MainWindow()
        self.window.resize(self.min_size.width(), self.min_size.height())
        self.window.show()
        self.app.processEvents()
        for _ in range(5):
            self.app.processEvents()
        return self

    def __exit__(self, *a):
        if self.window is not None:
            self.window.close()
            self.app.processEvents()

    # ---- Navigation ---------------------------------------------------
    def tabs(self):
        """Return the QTabWidget of the main window (v2 uses centralWidget,
        classic uses .tabs).
        """
        if self.flavor == "v2":
            return self.window.centralWidget()  # QTabWidget
        return self.window.tabs

    def set_main_tab(self, index: int):
        self.tabs().setCurrentIndex(index)
        self.app.processEvents()

    def single_view(self):
        """v2 SingleView widget (tab 0)."""
        assert self.flavor == "v2"
        return self.tabs().widget(0)

    def batch_view(self):
        """v2 BatchView widget (tab 1)."""
        assert self.flavor == "v2"
        return self.tabs().widget(1)

    def config_tab(self):
        """Classic ConfigurationTab (tab 0)."""
        assert self.flavor == "classic"
        return self.window.config_tab

    def options_panel(self):
        """Get the shared OptionsPanel regardless of flavor."""
        if self.flavor == "v2":
            return self.single_view().options
        return self.config_tab().options

    def left_tabs(self):
        """v2 SingleView left-side tab widget (Image/Log/Analytics)."""
        return self.single_view().left_tabs

    def set_options_sub_tab(self, name: str):
        """Select a tab in the OptionsPanel: 'Detection' / 'Refinement'
        / 'Analysis'.
        """
        op = self.options_panel()
        for i in range(op.tabs.count()):
            if op.tabs.tabText(i).lower().startswith(name.lower()):
                op.tabs.setCurrentIndex(i)
                self.app.processEvents()
                return
        raise ValueError(f"No options sub-tab matching '{name}'")

    # ---- Presets / params --------------------------------------------
    def apply_preset(self, name: str):
        from gui.options import presets
        params = presets.load_preset(name)
        self.options_panel().set_params(params)
        self.app.processEvents()
        return params

    def get_params(self):
        return self.options_panel().get_params()

    def set_params(self, params):
        self.options_panel().set_params(params)
        self.app.processEvents()

    # ---- Screenshots --------------------------------------------------
    def snapshot(self, out_path: str, widget: QWidget = None) -> str:
        """Save a PNG of `widget` (default = main window)."""
        target = widget if widget is not None else self.window
        return save_widget_pixmap(target, out_path, self.min_size)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", choices=["v2", "classic"],
                        default="v2")
    parser.add_argument("--preset", default=None,
                        help="Built-in preset name to apply before snap")
    parser.add_argument("--options-tab", default=None,
                        help="Options sub-tab to select "
                             "(Detection|Refinement|Analysis)")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()
    with HeadlessGUI(args.flavor) as gui:
        if args.preset:
            gui.apply_preset(args.preset)
        if args.options_tab:
            gui.set_options_sub_tab(args.options_tab)
        gui.snapshot(args.out)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    _cli()
