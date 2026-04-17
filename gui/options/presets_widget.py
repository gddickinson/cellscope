"""Preset selector widget: dropdown + Save/Delete/Reset buttons."""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QComboBox, QPushButton, QInputDialog, QMessageBox,
)

from gui.options.params import RunParams
from gui.options import presets as preset_module


class PresetSelector(QWidget):
    """Small widget: [preset ▼] [Apply] [Save as…] [Delete]."""

    # Emitted when user picks a preset and clicks Apply. Payload = RunParams.
    preset_applied = pyqtSignal(object)

    # Emitted when the user clicks "Save as…": parent should collect the
    # current panel values and pass them back via save_current(name, params).
    save_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.combo = QComboBox()
        self.combo.setMinimumWidth(260)
        self.combo.setToolTip(
            "Built-in presets based on SESSION_LOG findings + any "
            "user-saved presets from data/presets/."
        )
        layout.addWidget(self.combo, 1)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setToolTip(
            "Load the selected preset into all panels."
        )
        self.btn_apply.clicked.connect(self._on_apply)
        layout.addWidget(self.btn_apply)

        self.btn_save = QPushButton("Save as…")
        self.btn_save.setToolTip(
            "Save the current option values as a named user preset."
        )
        self.btn_save.clicked.connect(self._on_save)
        layout.addWidget(self.btn_save)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setToolTip(
            "Delete the selected user preset (built-ins are protected)."
        )
        self.btn_delete.clicked.connect(self._on_delete)
        layout.addWidget(self.btn_delete)

    def refresh(self):
        """Reload the preset list from disk."""
        current = self.combo.currentText()
        self.combo.blockSignals(True)
        self.combo.clear()
        for name in preset_module.all_preset_names():
            self.combo.addItem(name)
        # Try to restore previous selection, else first item
        idx = self.combo.findText(current)
        self.combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo.blockSignals(False)

    def current_name(self) -> str:
        return self.combo.currentText()

    def _on_apply(self):
        name = self.current_name()
        if not name:
            return
        try:
            params = preset_module.load_preset(name)
        except Exception as e:
            QMessageBox.warning(self, "Preset load failed", str(e))
            return
        self.preset_applied.emit(params)

    def _on_save(self):
        name, ok = QInputDialog.getText(
            self, "Save preset",
            "Name for this preset (will save to data/presets/<name>.json):"
        )
        if not ok or not name.strip():
            return
        # Don't overwrite built-ins
        if name in preset_module.BUILTIN_PRESETS:
            QMessageBox.warning(
                self, "Name reserved",
                f"'{name}' is a built-in preset. Choose a different name."
            )
            return
        self.save_requested.emit(name.strip())

    def save_current(self, name: str, params: RunParams):
        """Call this from the host after save_requested to persist."""
        try:
            path = preset_module.save_user_preset(name, params)
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        QMessageBox.information(
            self, "Preset saved", f"Saved to {path}"
        )
        self.refresh()
        idx = self.combo.findText(name)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def _on_delete(self):
        name = self.current_name()
        if not name:
            return
        if name in preset_module.BUILTIN_PRESETS:
            QMessageBox.warning(
                self, "Protected",
                f"'{name}' is a built-in preset and cannot be deleted."
            )
            return
        confirm = QMessageBox.question(
            self, "Confirm", f"Delete user preset '{name}'?"
        )
        if confirm != QMessageBox.Yes:
            return
        if preset_module.delete_user_preset(name):
            self.refresh()
