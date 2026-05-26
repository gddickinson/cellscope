"""Pipeline stage buttons with status indicators + mode selector."""
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QLabel,
)


STAGES = [
    ("load", "Load"),
    ("detect", "Detect"),
    ("edit", "Edit Masks"),
    ("analyze", "Analyze"),
    ("export", "Export"),
]

STATUS_ICONS = {
    "idle": "",
    "running": " ...",
    "done": " \u2713",
    "error": " \u2717",
}


class StageButton(QPushButton):
    """Pipeline stage button with status indicator."""

    def __init__(self, label):
        super().__init__(label)
        self._base_label = label
        self._status = "idle"
        self.setMinimumWidth(70)
        self.setMinimumHeight(36)
        self._update_style()

    def set_status(self, status):
        self._status = status
        self.setText(self._base_label + STATUS_ICONS.get(status, ""))
        self._update_style()

    def _update_style(self):
        colors = {
            "idle": "",
            "running": "background-color: #FFD700; font-weight: bold;",
            "done": "background-color: #90EE90; font-weight: bold;",
            "error": "background-color: #FF6B6B; font-weight: bold;",
        }
        self.setStyleSheet(colors.get(self._status, ""))


class PipelinePanel(QWidget):
    """Horizontal row of 6 pipeline stage buttons + mode selector."""

    load_clicked = pyqtSignal()
    detect_clicked = pyqtSignal()
    edit_clicked = pyqtSignal()
    analyze_clicked = pyqtSignal()
    export_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()
    undo_clicked = pyqtSignal()
    clear_all_clicked = pyqtSignal()
    test_frame_clicked = pyqtSignal()
    mode_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.stages = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Pipeline mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Single Cell (hybrid_cpsam)", "single")
        self.mode_combo.addItem("Multi Cell (hybrid_cpsam_multi)", "multi")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        self.mode_label = QLabel("")
        mode_row.addWidget(self.mode_label)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        btn_row = QHBoxLayout()
        signals = {
            "load": self.load_clicked,
            "detect": self.detect_clicked,
            "edit": self.edit_clicked,
            "analyze": self.analyze_clicked,
            "export": self.export_clicked,
        }
        for key, label in STAGES:
            btn = StageButton(label)
            btn.clicked.connect(signals[key].emit)
            btn.setEnabled(key == "load")
            self.stages[key] = btn
            btn_row.addWidget(btn)
        btn_row.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setToolTip("Cancel running detection")
        self.btn_cancel.clicked.connect(self.cancel_clicked.emit)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(
            "QPushButton:enabled { color: #c22; font-weight: bold; }")
        btn_row.addWidget(self.btn_cancel)

        self.btn_test_frame = QPushButton("🔬 Test on frame")
        self.btn_test_frame.setToolTip(
            "Run detection on the CURRENT displayed frame only with\n"
            "the current GUI parameter settings. Times the run and\n"
            "extrapolates a total runtime estimate for the full\n"
            "recording. Skips multi-frame steps (tracking, gap fill,\n"
            "Cy5 filter). Use this to preview how parameter changes\n"
            "affect detection before committing to a full run.")
        self.btn_test_frame.clicked.connect(self.test_frame_clicked.emit)
        self.btn_test_frame.setEnabled(False)
        btn_row.addWidget(self.btn_test_frame)

        btn_undo = QPushButton("Undo Detect")
        btn_undo.setToolTip("Revert to previous detection result")
        btn_undo.clicked.connect(self.undo_clicked.emit)
        btn_row.addWidget(btn_undo)

        btn_clear = QPushButton("Clear All")
        btn_clear.setToolTip("Clear all results, keep recording loaded")
        btn_clear.clicked.connect(self.clear_all_clicked.emit)
        btn_row.addWidget(btn_clear)

        layout.addLayout(btn_row)

    def _on_mode_changed(self, _idx):
        mode = self.mode_combo.currentData()
        self.mode_changed.emit(mode)

    def current_mode(self):
        return self.mode_combo.currentData()

    def set_mode(self, mode):
        idx = self.mode_combo.findData(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

    def set_mode_label(self, text):
        self.mode_label.setText(text)

    def set_stage_status(self, name, status):
        if name in self.stages:
            self.stages[name].set_status(status)

    def enable_through(self, stage_name):
        """Enable stages up to and including stage_name, disable rest."""
        found = False
        for key, _ in STAGES:
            if key == stage_name:
                found = True
                self.stages[key].setEnabled(True)
            elif found:
                self.stages[key].setEnabled(False)
            else:
                self.stages[key].setEnabled(True)
        # Test-on-frame is available once detection is reachable
        self.btn_test_frame.setEnabled(
            self.stages.get("detect") is not None
            and self.stages["detect"].isEnabled())

    def enable_stage(self, name, enabled=True):
        if name in self.stages:
            self.stages[name].setEnabled(enabled)
            if name == "detect":
                self.btn_test_frame.setEnabled(enabled)

    def reset_all(self):
        for key in self.stages:
            self.stages[key].set_status("idle")
            self.stages[key].setEnabled(key == "load")
        self.btn_test_frame.setEnabled(False)
