"""Mask-editor Help configuration — thin wrapper over `gui.help_menu`.

Holds the editor's keyboard-shortcut reference (`SHORTCUTS`) + guide path,
and exposes `install_help_menu(window)` / `show_shortcuts(parent)` that
`gui.mask_editor` imports. The generic menu/dialog code lives in
`gui.help_menu` (shared with the other CellScope GUIs).
"""
from __future__ import annotations

import os

from gui import help_menu

APP_NAME = "CellScope Mask Editor"
DOC_URL = f"{help_menu.REPO_URL}/blob/main/docs/mask_editor_guide.md"
# gui/editor_help.py → repo root is one level up → docs/mask_editor_guide.md
GUIDE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "mask_editor_guide.md")

# Source of truth for the shortcuts popup — keep in sync with the
# QShortcut registrations in gui.mask_editor.
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


def install_help_menu(window):
    help_menu.install_help_menu(window, APP_NAME, SHORTCUTS,
                                guide_path=GUIDE_PATH, doc_url=DOC_URL)


def show_shortcuts(parent=None):
    help_menu.show_shortcuts(APP_NAME, SHORTCUTS, parent,
                             guide_path=GUIDE_PATH, doc_url=DOC_URL)


def show_guide(parent=None):
    help_menu.show_guide(APP_NAME, GUIDE_PATH, DOC_URL, parent)


def open_online_docs():
    help_menu.open_url(DOC_URL)
