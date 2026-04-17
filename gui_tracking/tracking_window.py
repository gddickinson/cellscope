"""Tracking & Analysis main window with Single/Batch tabs."""
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QStatusBar, QAction, QMessageBox,
)
from gui.run_log import RunLogger
from gui_tracking.single_view import SingleTrackingView
from gui_tracking.batch_view import BatchTrackingView


class TrackingWindow(QMainWindow):
    """Dedicated cell tracking and analysis application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "CellScope — Tracking & Analysis")
        self.resize(1400, 900)
        self.logger = RunLogger()
        self._build_ui()
        self._build_menu()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()
        self.single_view = SingleTrackingView(logger=self.logger)
        self.tabs.addTab(self.single_view, "Single Recording")
        self.batch_view = BatchTrackingView(logger=self.logger)
        self.tabs.addTab(self.batch_view, "Batch Comparison")
        layout.addWidget(self.tabs)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    def _build_menu(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("File")
        act_open = QAction("Open Recording...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.single_view._on_load)
        file_menu.addAction(act_open)
        act_masks = QAction("Load Masks...", self)
        act_masks.setShortcut("Ctrl+M")
        act_masks.triggered.connect(self.single_view._on_load_masks)
        file_menu.addAction(act_masks)
        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = mb.addMenu("View")
        act_single = QAction("Single Recording", self)
        act_single.setShortcut("Ctrl+1")
        act_single.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        view_menu.addAction(act_single)
        act_batch = QAction("Batch Comparison", self)
        act_batch.setShortcut("Ctrl+2")
        act_batch.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        view_menu.addAction(act_batch)

        help_menu = mb.addMenu("Help")
        act_about = QAction("About...", self)
        act_about.triggered.connect(lambda: QMessageBox.about(
            self, "About",
            "CellScope — Tracking & Analysis\n\n"
            "Single Recording: load masks, track, analyze per-cell\n"
            "Batch: process multiple recordings, compare groups\n"
            "with t-test, Mann-Whitney, ANOVA + significance plots"))
        help_menu.addAction(act_about)
