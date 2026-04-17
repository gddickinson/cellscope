"""Training data preview — thumbnail grid of image+mask pairs."""
import os
import re
import numpy as np
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QGridLayout, QLabel,
)

THUMB_SIZE = 128


def _find_pairs(data_dir):
    """Find image+mask pairs in a directory."""
    pairs = []
    files = sorted(os.listdir(data_dir))
    mask_files = {f for f in files if "_masks" in f.lower()
                  or "_mask" in f.lower()}
    for mf in sorted(mask_files):
        base = mf.replace("_masks", "").replace("_mask", "")
        if base in files:
            pairs.append((
                os.path.join(data_dir, base),
                os.path.join(data_dir, mf)))
    if not pairs:
        imgs = [f for f in files if f.endswith((".png", ".tif", ".tiff"))
                and "mask" not in f.lower()]
        for img_f in imgs:
            nums = re.findall(r"\d+", img_f)
            if not nums:
                continue
            idx = nums[-1]
            candidates = [f for f in mask_files if idx in f]
            if candidates:
                pairs.append((
                    os.path.join(data_dir, img_f),
                    os.path.join(data_dir, candidates[0])))
    return pairs


def _make_thumbnail(img_path, mask_path, size=THUMB_SIZE):
    """Create a thumbnail with mask overlay."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None
    h, w = img.shape
    scale = size / max(h, w)
    img_s = cv2.resize(img, (int(w * scale), int(h * scale)))
    mask_s = cv2.resize(mask, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(img_s, cv2.COLOR_GRAY2RGB)
    m = mask_s > 0
    rgb[m] = (rgb[m] * 0.6 + np.array([0, 180, 0]) * 0.4).astype(np.uint8)
    contours, _ = cv2.findContours(
        m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)
    return rgb


class DataPreview(QWidget):
    """Scrollable grid of training image+mask thumbnails."""

    def __init__(self):
        super().__init__()
        self.pairs = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.info_label = QLabel("No data loaded")
        layout.addWidget(self.info_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(4)
        scroll.setWidget(self.grid_widget)
        layout.addWidget(scroll, stretch=1)

    def scan(self, data_dir):
        self.pairs = _find_pairs(data_dir)
        self.info_label.setText(f"{len(self.pairs)} image+mask pairs")
        self._populate_grid()

    def _populate_grid(self):
        while self.grid_layout.count():
            w = self.grid_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        cols = 6
        for i, (img_p, mask_p) in enumerate(self.pairs[:60]):
            thumb = _make_thumbnail(img_p, mask_p)
            if thumb is None:
                continue
            h, w, _ = thumb.shape
            qimg = QImage(thumb.data, w, h, 3 * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setToolTip(os.path.basename(img_p))
            self.grid_layout.addWidget(lbl, i // cols, i % cols)

    def get_pairs(self):
        return self.pairs
