"""Training GUI — fine-tune cellpose on user-provided GT masks."""
import os
import numpy as np
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QDoubleSpinBox, QFileDialog, QProgressBar, QStatusBar,
    QGroupBox, QFormLayout, QMessageBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gui.run_log import RunLogger, RunLogWidget
from gui_training.data_preview import DataPreview
from gui_training.training_worker import TrainingWorker


def _list_models():
    model_dir = "data/models"
    if not os.path.isdir(model_dir):
        return ["(cellpose default)"]
    out = ["(cellpose default)"]
    for f in sorted(os.listdir(model_dir)):
        if f.startswith("cellpose") or f.startswith("cpsam"):
            out.append(f)
    return out


class TrainingWindow(QMainWindow):
    """Fine-tune cellpose on user-provided training data."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CellScope — Model Training")
        self.resize(1200, 800)
        self.logger = RunLogger()
        self._worker = None
        self._losses = []
        self._build_ui()
        from gui.drag_drop import setup_drag_drop
        setup_drag_drop(self, self._on_drop_folder,
                        (".png", ".tif", ".tiff", ".npz"))

    def _on_drop_folder(self, path):
        import os
        d = os.path.dirname(path) if os.path.isfile(path) else path
        self.data_edit.setText(d)
        self._on_scan()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Vertical)

        # Top: data + config
        top = QWidget()
        tl = QHBoxLayout(top)

        # Left: data selection + preview
        data_panel = QWidget()
        dl = QVBoxLayout(data_panel)
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Training data:"))
        self.data_edit = QLineEdit()
        dir_row.addWidget(self.data_edit, stretch=1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._pick_data)
        dir_row.addWidget(btn_browse)
        btn_scan = QPushButton("Scan")
        btn_scan.clicked.connect(self._on_scan)
        dir_row.addWidget(btn_scan)
        dl.addLayout(dir_row)
        self.preview = DataPreview()
        dl.addWidget(self.preview, stretch=1)
        tl.addWidget(data_panel, stretch=2)

        # Right: config
        config_panel = QWidget()
        cl = QVBoxLayout(config_panel)
        cg = QGroupBox("Training Configuration")
        form = QFormLayout(cg)
        self.base_model = QComboBox()
        for m in _list_models():
            self.base_model.addItem(m)
        form.addRow("Base model:", self.base_model)
        self.model_name = QLineEdit("cellpose_finetuned")
        form.addRow("Output name:", self.model_name)
        self.n_epochs = QSpinBox()
        self.n_epochs.setRange(1, 500)
        self.n_epochs.setValue(80)
        form.addRow("Epochs:", self.n_epochs)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(1e-7, 1e-2)
        self.learning_rate.setDecimals(7)
        self.learning_rate.setValue(5e-6)
        form.addRow("Learning rate:", self.learning_rate)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 32)
        self.batch_size.setValue(8)
        form.addRow("Batch size:", self.batch_size)
        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setRange(0, 0.1)
        self.weight_decay.setDecimals(5)
        self.weight_decay.setValue(1e-5)
        form.addRow("Weight decay:", self.weight_decay)
        self.use_sgd = QCheckBox("SGD (default: Adam)")
        form.addRow("Optimizer:", self.use_sgd)
        self.augment = QCheckBox()
        self.augment.setChecked(True)
        self.augment.setToolTip("Apply noise/gamma/flip augmentations")
        form.addRow("Augment:", self.augment)
        cl.addWidget(cg)

        btn_row = QHBoxLayout()
        self.btn_train = QPushButton("Train")
        self.btn_train.clicked.connect(self._on_train)
        btn_row.addWidget(self.btn_train)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        cl.addLayout(btn_row)
        cl.addStretch()
        tl.addWidget(config_panel, stretch=1)

        splitter.addWidget(top)

        # Bottom: loss curve + log
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        self.loss_fig = Figure(figsize=(5, 3), dpi=100)
        self.loss_canvas = FigureCanvasQTAgg(self.loss_fig)
        bl.addWidget(self.loss_canvas, stretch=2)
        log_widget = RunLogWidget(self.logger)
        bl.addWidget(log_widget, stretch=1)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Select training data directory")

    def _pick_data(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select training data folder")
        if path:
            self.data_edit.setText(path)

    def _on_scan(self):
        path = self.data_edit.text()
        if not path or not os.path.isdir(path):
            QMessageBox.warning(self, "Scan", "Select a valid directory")
            return
        self.preview.scan(path)
        self.status.showMessage(
            f"Found {len(self.preview.get_pairs())} training pairs")

    def _on_train(self):
        pairs = self.preview.get_pairs()
        if not pairs:
            QMessageBox.warning(self, "Train", "Scan data first")
            return

        images, masks = [], []
        for img_p, mask_p in pairs:
            img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
            if img is None or mask is None:
                continue
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            images.append(img)
            masks.append(mask.astype(np.uint16))

        if not images:
            QMessageBox.warning(self, "Train", "No valid pairs found")
            return

        base = self.base_model.currentText()
        base_path = "" if base == "(cellpose default)" else \
            os.path.join("data/models", base)

        config = {
            "base_model": base_path,
            "model_name": self.model_name.text() or "cellpose_finetuned",
            "n_epochs": self.n_epochs.value(),
            "learning_rate": self.learning_rate.value(),
            "batch_size": self.batch_size.value(),
            "weight_decay": self.weight_decay.value(),
            "sgd": self.use_sgd.isChecked(),
            "augment": self.augment.isChecked(),
        }

        self._losses = []
        self._worker = TrainingWorker(images, masks, config)
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.log_event.connect(
            lambda k, m: self.logger.log(k, m))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(
            lambda e: (QMessageBox.critical(self, "Error", e),
                       self._reset_buttons()))
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
            self._reset_buttons()

    def _on_epoch(self, epoch, loss):
        self._losses.append((epoch, loss))
        self.progress.setValue(
            int(100 * epoch / max(self.n_epochs.value(), 1)))
        self.status.showMessage(
            f"Epoch {epoch}/{self.n_epochs.value()} loss={loss:.4f}")
        self._update_loss_plot()

    def _update_loss_plot(self):
        if not self._losses:
            return
        self.loss_fig.clear()
        ax = self.loss_fig.add_subplot(111)
        epochs = [e for e, _ in self._losses]
        losses = [l for _, l in self._losses]
        ax.plot(epochs, losses, "b-", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(alpha=0.3)
        self.loss_fig.tight_layout()
        self.loss_canvas.draw_idle()

    def _on_finished(self, model_path):
        self._reset_buttons()
        self.status.showMessage(f"Training complete: {model_path}")
        self.logger.log("done", f"Model saved: {model_path}")

    def _reset_buttons(self):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._worker = None
