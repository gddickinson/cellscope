"""Interactive Cy5 filter threshold tuner.

Standalone Qt app. Pick a recording + frame, slide thresholds,
see kept/dropped counts and a live overlay update.

Usage:
    conda run -n cellpose python scripts/cy5_filter_tuner.py \\
        --cache-dir results/ic295_full_v2

If the cache dir has multiple Pos*.npz files, a dropdown lets
you switch between them.
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("QT_API", "pyqt5")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports
setup_imports()

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFormLayout, QComboBox, QSlider, QLabel, QPushButton, QDoubleSpinBox,
    QGroupBox, QSpinBox,
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure


class TunerWindow(QMainWindow):
    def __init__(self, npz_paths):
        super().__init__()
        self.setWindowTitle("CellScope — Cy5 Filter Tuner")
        self.resize(1500, 850)
        self.npz_paths = {os.path.basename(p).replace(".npz", ""): p
                          for p in npz_paths}
        self.current_npz = None
        self.frames = None
        self.cy5_frames = None
        self.tracks = []

        self._build_ui()
        if self.npz_paths:
            self.recording_combo.setCurrentIndex(0)
            self._load_recording()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: matplotlib canvas
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, stretch=3)

        # Right: controls
        side = QWidget()
        sl = QVBoxLayout(side)

        # Recording picker
        rg = QGroupBox("Recording")
        rgl = QFormLayout(rg)
        self.recording_combo = QComboBox()
        self.recording_combo.addItems(sorted(self.npz_paths.keys()))
        self.recording_combo.currentTextChanged.connect(self._load_recording)
        rgl.addRow("File:", self.recording_combo)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(96)
        self.frame_slider.setValue(48)
        self.frame_slider.valueChanged.connect(self._refresh)
        self.frame_label = QLabel("frame 48")
        rgl.addRow("Frame:", self.frame_slider)
        rgl.addRow("", self.frame_label)
        sl.addWidget(rg)

        # Filter mode
        fg = QGroupBox("Filter mode")
        fgl = QFormLayout(fg)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "off", "conservative", "conservative_strict",
            "adaptive", "adaptive_loose",
            "multi_metric", "composite_score", "consensus",
            "temporal_stability"])
        self.mode_combo.setCurrentText("multi_metric")
        self.mode_combo.currentTextChanged.connect(self._refresh)
        fgl.addRow("Mode:", self.mode_combo)
        sl.addWidget(fg)

        # Multi-metric thresholds
        mg = QGroupBox("Multi-metric thresholds")
        mgl = QFormLayout(mg)
        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(0.0, 1.0); self.score_spin.setSingleStep(0.01)
        self.score_spin.setDecimals(2); self.score_spin.setValue(0.15)
        self.score_spin.valueChanged.connect(self._refresh)
        mgl.addRow("score >", self.score_spin)
        self.io_spin = QDoubleSpinBox()
        self.io_spin.setRange(0.5, 5.0); self.io_spin.setSingleStep(0.05)
        self.io_spin.setDecimals(2); self.io_spin.setValue(1.10)
        self.io_spin.valueChanged.connect(self._refresh)
        mgl.addRow("io_ratio >", self.io_spin)
        self.fp_spin = QDoubleSpinBox()
        self.fp_spin.setRange(0.0, 1.0); self.fp_spin.setSingleStep(0.05)
        self.fp_spin.setDecimals(2); self.fp_spin.setValue(0.15)
        self.fp_spin.valueChanged.connect(self._refresh)
        mgl.addRow("frac_positive >", self.fp_spin)
        self.min_pass_spin = QSpinBox()
        self.min_pass_spin.setRange(1, 3); self.min_pass_spin.setValue(2)
        self.min_pass_spin.valueChanged.connect(self._refresh)
        mgl.addRow("min passing:", self.min_pass_spin)
        sl.addWidget(mg)

        # Composite threshold
        cg = QGroupBox("Composite score threshold")
        cgl = QFormLayout(cg)
        self.composite_spin = QDoubleSpinBox()
        self.composite_spin.setRange(0.0, 5.0)
        self.composite_spin.setSingleStep(0.1)
        self.composite_spin.setDecimals(2); self.composite_spin.setValue(1.0)
        self.composite_spin.valueChanged.connect(self._refresh)
        cgl.addRow("composite ≥", self.composite_spin)
        sl.addWidget(cg)

        # Stats display
        self.stats_label = QLabel("Loading…")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet(
            "QLabel { background: #f0f0f0; padding: 8px; "
            "font-family: monospace; }")
        sl.addWidget(self.stats_label)

        # Apply-to-all button
        self.apply_btn = QPushButton(
            "Apply current settings to ALL recordings (CLI)")
        self.apply_btn.clicked.connect(self._show_cli)
        sl.addWidget(self.apply_btn)

        sl.addStretch()
        root.addWidget(side, stretch=1)

    def _load_recording(self):
        from scripts.apply_cy5_filter_to_results import (
            load_tracks_from_npz)
        name = self.recording_combo.currentText()
        if not name:
            return
        path = self.npz_paths[name]
        z = np.load(path, allow_pickle=False)
        self.frames = z["frames"]
        self.cy5_frames = z["cy5_frames"]
        self.tracks = load_tracks_from_npz(z)
        self.frame_slider.setMaximum(len(self.frames) - 1)
        self.frame_slider.setValue(min(48, len(self.frames) - 1))
        self._refresh()

    def _filter_now(self):
        """Apply current filter with current threshold spinbox values.
        Patches multi_metric / composite thresholds at runtime."""
        from core import cy5_filter as cf
        mode = self.mode_combo.currentText()
        if mode == "multi_metric":
            return cf.multi_metric_filter(
                self.tracks,
                score_threshold=self.score_spin.value(),
                io_ratio_threshold=self.io_spin.value(),
                frac_pos_threshold=self.fp_spin.value(),
                min_passing=self.min_pass_spin.value())
        if mode == "composite_score":
            return cf.composite_score_filter(
                self.tracks, threshold=self.composite_spin.value())
        return cf.apply_cy5_filter(self.tracks, mode=mode)

    def _refresh(self):
        if self.frames is None or not self.tracks:
            return
        fi = self.frame_slider.value()
        self.frame_label.setText(f"frame {fi}")
        kept, dropped, info = self._filter_now()

        self.ax.clear()
        # DIC + Cy5 composite
        rgb = np.stack([self.frames[fi]] * 3, axis=-1).astype(np.float32)
        rgb[..., 0] = np.maximum(rgb[..., 0],
                                  self.cy5_frames[fi].astype(np.float32))
        self.ax.imshow(rgb.clip(0, 255).astype(np.uint8))
        # kept = green
        for t in kept:
            m = t["stack"][fi].astype(bool)
            if m.any():
                self.ax.contour(m, levels=[0.5], colors=["lime"],
                                  linewidths=1.0)
        # dropped = dashed red
        for t in dropped:
            m = t["stack"][fi].astype(bool)
            if m.any():
                self.ax.contour(m, levels=[0.5], colors=["red"],
                                  linewidths=0.7, linestyles="--")
        self.ax.set_title(
            f"{self.recording_combo.currentText()} f{fi} — "
            f"{self.mode_combo.currentText()}: "
            f"kept {len(kept)} (green) dropped {len(dropped)} (red dashed)")
        self.ax.axis("off")
        self.canvas.draw()

        # Stats
        info_str = ", ".join(f"{k}={v}" for k, v in info.items()
                              if k not in ("mode", "thresholds"))
        thresh = info.get("thresholds", info.get("threshold", ""))
        self.stats_label.setText(
            f"Mode: {info.get('mode', '?')}\n"
            f"Tracks: kept {len(kept)} / dropped {len(dropped)} "
            f"of {len(self.tracks)}\n"
            f"Drop rate: {100*len(dropped)/max(len(self.tracks), 1):.0f}%\n"
            f"Thresholds: {thresh}\n"
            f"Info: {info_str}")

    def _show_cli(self):
        from PyQt5.QtWidgets import QMessageBox
        mode = self.mode_combo.currentText()
        cli = (
            f"conda run -n cellpose python \\\n"
            f"  scripts/apply_cy5_filter_to_results.py \\\n"
            f"  --filter-mode {mode}")
        if mode == "multi_metric":
            cli += (
                f"\n\nNote: tuner's multi_metric thresholds "
                f"(score>{self.score_spin.value():.2f}, "
                f"io>{self.io_spin.value():.2f}, "
                f"frac_pos>{self.fp_spin.value():.2f}, "
                f"min_passing={self.min_pass_spin.value()})\n"
                f"are NOT exposed by the CLI — defaults will be used.\n"
                f"Edit core/cy5_filter.py:multi_metric_filter to lock\n"
                f"in these tuner values.")
        QMessageBox.information(self, "CLI command", cli)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="results/ic295_full_v2")
    args = ap.parse_args()
    npz_paths = sorted(glob.glob(
        os.path.join(args.cache_dir, "Pos*.npz")))
    if not npz_paths:
        print(f"No NPZs found in {args.cache_dir}")
        sys.exit(1)

    app = QApplication(sys.argv)
    w = TunerWindow(npz_paths)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
