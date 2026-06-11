"""Reusable Help menu for CellScope Qt GUIs.

`install_help_menu(window, app_name, shortcuts=..., guide_path=..., doc_url=...)`
adds (or extends) a **Help** menu on any QMainWindow with:
  - Keyboard Shortcuts…  (F1)  — popup built from `shortcuts`
  - <app> User Guide…          — in-app markdown render of `guide_path` (if given)
  - Online Documentation ↗     — opens `doc_url` (default: repo README)
  - About <app>

It creates the menu bar if the window doesn't have one yet, appends to an
existing "Help" menu rather than duplicating it, and skips items the menu
already has (e.g. a pre-existing About). Idempotent per window.

`shortcuts` is `[(group_title, [(keys, description), ...]), ...]` — the
same structure the popup renders. Each GUI passes its own list.
"""
from __future__ import annotations

import os

from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QDesktopServices, QKeySequence
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTextBrowser, QMessageBox,
    QShortcut)

REPO_URL = "https://github.com/gddickinson/cellscope"
DEFAULT_DOC_URL = f"{REPO_URL}/blob/main/README.md"


def shortcuts_html(shortcuts):
    rows = [
        "<style>td{padding:3px 14px 3px 0;} h3{margin:14px 0 4px;}"
        "kbd{background:#eee;border:1px solid #bbb;border-radius:4px;"
        "padding:1px 6px;font-family:monospace;}</style>"]
    if not shortcuts:
        rows.append("<p>This window has no dedicated keyboard shortcuts.</p>")
    for group, items in shortcuts or []:
        rows.append(f"<h3>{group}</h3><table>")
        for keys, action in items:
            rows.append(
                f"<tr><td><kbd>{keys}</kbd></td><td>{action}</td></tr>")
        rows.append("</table>")
    return "\n".join(rows)


def _load_markdown(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


class ShortcutsDialog(QDialog):
    """Read-only popup listing a GUI's keyboard shortcuts."""

    def __init__(self, title, shortcuts, parent=None,
                 guide_path=None, doc_url=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title} — Keyboard Shortcuts")
        self.resize(540, 600)
        lyt = QVBoxLayout(self)
        view = QTextBrowser()
        view.setHtml(shortcuts_html(shortcuts))
        view.setOpenExternalLinks(True)
        lyt.addWidget(view)
        row = QHBoxLayout()
        if guide_path:
            g = QPushButton("Open Full Guide…")
            g.clicked.connect(lambda: show_guide(title, guide_path,
                                                 doc_url, self))
            row.addWidget(g)
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lyt.addLayout(row)


class GuideDialog(QDialog):
    """Renders a markdown guide file; falls back to the online link."""

    def __init__(self, title, guide_path, doc_url=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title} — User Guide")
        self.resize(700, 720)
        lyt = QVBoxLayout(self)
        view = QTextBrowser()
        view.setOpenExternalLinks(True)
        md = _load_markdown(guide_path) if guide_path else None
        if md is not None and hasattr(view, "setMarkdown"):
            view.setMarkdown(md)
        elif md is not None:
            view.setPlainText(md)
        else:
            url = doc_url or DEFAULT_DOC_URL
            view.setHtml(f'<p>Guide not found.</p>'
                         f'<p><a href="{url}">Open online ↗</a></p>')
        lyt.addWidget(view)
        row = QHBoxLayout()
        online = QPushButton("Open Online (GitHub) ↗")
        online.clicked.connect(lambda: open_url(doc_url or DEFAULT_DOC_URL))
        row.addWidget(online)
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lyt.addLayout(row)


def show_shortcuts(title, shortcuts, parent=None,
                   guide_path=None, doc_url=None):
    ShortcutsDialog(title, shortcuts, parent, guide_path, doc_url).exec_()


def show_guide(title, guide_path, doc_url=None, parent=None):
    GuideDialog(title, guide_path, doc_url, parent).exec_()


def open_url(url):
    QDesktopServices.openUrl(QUrl(url or DEFAULT_DOC_URL))


def show_about(parent, app_name):
    QMessageBox.about(
        parent, f"About — {app_name}",
        f"<b>{app_name}</b><br>part of the CellScope suite.<br><br>"
        f'<a href="{REPO_URL}">{REPO_URL}</a>')


def _find_menu(window, title):
    for act in window.menuBar().actions():
        if act.menu() is not None and act.text().replace("&", "") == title:
            return act.menu()
    return None


def install_help_menu(window, app_name, shortcuts=None,
                      guide_path=None, doc_url=None):
    """Add / extend a Help menu on `window` (a QMainWindow). Idempotent.

    Appends to an existing "Help" menu if present (skipping items it
    already has), else creates one. Calling `window.menuBar()` creates a
    native menu bar for GUIs that don't have one yet.
    """
    if getattr(window, "_help_menu_installed", False):
        return
    doc_url = doc_url or DEFAULT_DOC_URL
    menu = _find_menu(window, "Help") or window.menuBar().addMenu("Help")
    existing = {a.text().replace("&", "") for a in menu.actions()}
    if menu.actions():
        menu.addSeparator()
    if not any("Shortcut" in t for t in existing):
        a = menu.addAction("Keyboard Shortcuts… (F1)")
        a.triggered.connect(
            lambda: show_shortcuts(app_name, shortcuts, window,
                                   guide_path, doc_url))
        # F1 / ? via ApplicationShortcut so they fire even when a child
        # widget (canvas, table) holds keyboard focus. Registered here as
        # the single source — GUIs must not also bind F1/? themselves.
        for keys in ("F1", "?"):
            sc = QShortcut(QKeySequence(keys), window,
                           activated=lambda: show_shortcuts(
                               app_name, shortcuts, window,
                               guide_path, doc_url))
            sc.setContext(Qt.ApplicationShortcut)
    if guide_path:
        a = menu.addAction("User Guide…")
        a.triggered.connect(
            lambda: show_guide(app_name, guide_path, doc_url, window))
    if not any(("Documentation" in t or "Online" in t) for t in existing):
        a = menu.addAction("Online Documentation ↗")
        a.triggered.connect(lambda: open_url(doc_url))
    if not any("About" in t for t in existing):
        a = menu.addAction(f"About {app_name}")
        a.triggered.connect(lambda: show_about(window, app_name))
    window._help_menu_installed = True
