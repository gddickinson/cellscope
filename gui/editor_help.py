"""Help menu + keyboard-shortcuts popup + guide dialog for the mask editor.

Kept separate from `gui/mask_editor.py` (already large) so the help UI is
self-contained and reusable. `install_help_menu(window)` adds a Help menu;
`show_shortcuts(parent)` / `show_guide(parent)` open the dialogs.

`SHORTCUTS` is the single human-readable reference shown in the popup. Keep
it in sync with the `QShortcut` registrations in `MaskEditor._build_*` —
the popup is only as accurate as this list.
"""
from __future__ import annotations

import os

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTextBrowser, QLabel)

REPO_URL = "https://github.com/gddickinson/cellscope"
DOC_URL = f"{REPO_URL}/blob/main/docs/mask_editor_guide.md"
_GUIDE_REL = os.path.join("docs", "mask_editor_guide.md")

# (group, [(keys, action), ...]) — the reference shown in the popup.
SHORTCUTS = [
    ("Navigation", [
        ("←  /  →", "Previous / next frame"),
        ("Shift+←  /  Shift+→", "Previous / next target frame (GT review)"),
        ("Ctrl+0", "Fit view to window"),
    ]),
    ("Tools", [
        ("B", "Brush — paint into the active cell"),
        ("E", "Eraser — remove from the active cell"),
        ("P", "Polygon — outline a region to add"),
        ("F", "Fill — flood-fill a region"),
        ("R", "Relabel — reassign pixels to another cell"),
        ("X  /  Delete", "Delete tool — click a track to remove it"),
        ("N", "New cell — start the next free cell ID"),
    ]),
    ("Cell selection", [
        ("1 – 9,  0", "Select cell 1 – 10"),
        ("Shift+1 – 9,  Shift+0", "Select cell 11 – 20"),
        ("I", "Toggle on-image cell-ID labels"),
    ]),
    ("Cleanup", [
        ("Ctrl+Shift+F", "Filter Cells… — bulk-remove tracks by criteria"),
        ("Ctrl+T", "Trim Edges… — strip a border band across frames"),
        ("Ctrl+K", "Clean masks — fill holes / keep largest component"),
    ]),
    ("Save & undo", [
        ("Ctrl+S", "Save masks (in place)"),
        ("Ctrl+Shift+S", "Save & advance one frame"),
        ("Ctrl+Z  /  Ctrl+Shift+Z", "Undo / redo"),
        ("Ctrl+G  /  Ctrl+Shift+G", "Save GT for this frame / all GT frames"),
    ]),
    ("Help", [
        ("F1  /  ?", "Show this keyboard-shortcuts list"),
    ]),
]


def _shortcuts_html():
    rows = [
        "<style>td{padding:3px 14px 3px 0;} h3{margin:14px 0 4px;}"
        "kbd{background:#eee;border:1px solid #bbb;border-radius:4px;"
        "padding:1px 6px;font-family:monospace;}</style>"]
    for group, items in SHORTCUTS:
        rows.append(f"<h3>{group}</h3><table>")
        for keys, action in items:
            rows.append(
                f"<tr><td><kbd>{keys}</kbd></td><td>{action}</td></tr>")
        rows.append("</table>")
    return "\n".join(rows)


class ShortcutsDialog(QDialog):
    """Read-only popup listing every editor keyboard shortcut."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mask Editor — Keyboard Shortcuts")
        self.resize(540, 620)
        lyt = QVBoxLayout(self)
        view = QTextBrowser()
        view.setHtml(_shortcuts_html())
        view.setOpenExternalLinks(True)
        lyt.addWidget(view)
        row = QHBoxLayout()
        guide = QPushButton("Open Full Guide…")
        guide.clicked.connect(lambda: show_guide(self))
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(guide)
        row.addStretch()
        row.addWidget(close)
        lyt.addLayout(row)


class GuideDialog(QDialog):
    """Renders docs/mask_editor_guide.md; falls back to the online link."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mask Editor — User Guide")
        self.resize(700, 720)
        lyt = QVBoxLayout(self)
        view = QTextBrowser()
        view.setOpenExternalLinks(True)
        md = self._load_guide()
        if md is not None and hasattr(view, "setMarkdown"):
            view.setMarkdown(md)
        elif md is not None:
            view.setPlainText(md)
        else:
            view.setHtml(
                "<p>Guide file not found.</p>"
                f'<p><a href="{DOC_URL}">Open the online guide ↗</a></p>')
        lyt.addWidget(view)
        row = QHBoxLayout()
        online = QPushButton("Open Online (GitHub) ↗")
        online.clicked.connect(open_online_docs)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(online)
        row.addStretch()
        row.addWidget(close)
        lyt.addLayout(row)

    @staticmethod
    def _load_guide():
        # gui/editor_help.py → repo root is one level up.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, _GUIDE_REL)
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None


def show_shortcuts(parent=None):
    ShortcutsDialog(parent).exec_()


def show_guide(parent=None):
    GuideDialog(parent).exec_()


def open_online_docs():
    QDesktopServices.openUrl(QUrl(DOC_URL))


def show_about(parent=None):
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.about(
        parent, "About — CellScope Mask Editor",
        "<b>CellScope Mask Editor</b><br>"
        "Review &amp; correct per-cell segmentation masks, then save "
        "<code>masks.npz</code> in place.<br><br>"
        f'<a href="{REPO_URL}">{REPO_URL}</a>')


def install_help_menu(window):
    """Add a Help menu to `window` (a QMainWindow). Idempotent."""
    if getattr(window, "_help_menu_installed", False):
        return
    menu = window.menuBar().addMenu("Help")
    a_sc = menu.addAction("Keyboard Shortcuts…")
    a_sc.setShortcut("F1")
    a_sc.triggered.connect(lambda: show_shortcuts(window))
    a_guide = menu.addAction("Mask Editor Guide…")
    a_guide.triggered.connect(lambda: show_guide(window))
    a_online = menu.addAction("Online Documentation ↗")
    a_online.triggered.connect(open_online_docs)
    menu.addSeparator()
    a_about = menu.addAction("About CellScope Editor")
    a_about.triggered.connect(lambda: show_about(window))
    window._help_menu_installed = True
