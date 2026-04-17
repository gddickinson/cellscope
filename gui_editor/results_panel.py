"""Results viewer panel for the standalone mask editor.

Loads metrics.json from a results directory and displays scalar
metrics + analysis plots via GRAPH_REGISTRY.
"""
import os
import json
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QFileDialog,
    QHeaderView,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ResultsPanel(QWidget):
    """Displays analysis metrics and plots from a results directory."""

    def __init__(self):
        super().__init__()
        self.metrics = None
        self.results_dir = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        btn_load = QPushButton("Load Results")
        btn_load.clicked.connect(self._on_load)
        top.addWidget(btn_load)
        self.path_label = QLabel("No results loaded")
        top.addWidget(self.path_label, stretch=1)
        layout.addLayout(top)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(200)
        layout.addWidget(self.table)

        plot_row = QHBoxLayout()
        plot_row.addWidget(QLabel("Plot:"))
        self.plot_combo = QComboBox()
        self.plot_combo.currentTextChanged.connect(self._on_plot_changed)
        plot_row.addWidget(self.plot_combo, stretch=1)
        layout.addLayout(plot_row)

        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas, stretch=1)

    def _on_load(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select results directory")
        if path:
            self.load_from_dir(path)

    def load_from_dir(self, results_dir):
        metrics_path = os.path.join(results_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            self.path_label.setText(f"No metrics.json in {results_dir}")
            return
        with open(metrics_path) as f:
            self.metrics = json.load(f)
        self.results_dir = results_dir
        self.path_label.setText(os.path.basename(results_dir))
        self._populate_table()
        self._populate_plots()

    def _populate_table(self):
        if not self.metrics:
            return
        display = [
            ("Recording", self.metrics.get("name", "?")),
            ("Frames", self.metrics.get("n_frames", "?")),
            ("um/px", self.metrics.get("um_per_px", "?")),
            ("Time interval (min)", self.metrics.get("time_interval_min", "?")),
            ("Mean speed (um/min)", f"{self.metrics.get('mean_speed', 0):.3f}"),
            ("Total distance (um)", f"{self.metrics.get('total_distance', 0):.1f}"),
            ("Net displacement (um)", f"{self.metrics.get('net_displacement', 0):.1f}"),
            ("Persistence", f"{self.metrics.get('persistence', 0):.3f}"),
            ("Boundary confidence", f"{self.metrics.get('mean_boundary_confidence', 0):.1f}"),
        ]
        ss = self.metrics.get("shape_summary", {})
        if "area_um2" in ss:
            a = ss["area_um2"]
            display.append(("Mean area (um^2)", f"{a.get('mean', 0):.0f}"))
        es = self.metrics.get("edge_summary", {})
        if es:
            display.append(("Protrusion vel", f"{es.get('mean_protrusion_velocity', 0):.3f}"))
            display.append(("Retraction vel", f"{es.get('mean_retraction_velocity', 0):.3f}"))

        self.table.setRowCount(len(display))
        for i, (key, val) in enumerate(display):
            self.table.setItem(i, 0, QTableWidgetItem(str(key)))
            self.table.setItem(i, 1, QTableWidgetItem(str(val)))

    def _populate_plots(self):
        self.plot_combo.clear()
        if not self.results_dir:
            return
        for fname in sorted(os.listdir(self.results_dir)):
            if fname.endswith((".png", ".svg", ".pdf")) and fname != "metrics.json":
                self.plot_combo.addItem(fname)

    def _on_plot_changed(self, name):
        if not name or not self.results_dir:
            return
        path = os.path.join(self.results_dir, name)
        if not os.path.exists(path):
            return
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        try:
            from matplotlib.image import imread
            img = imread(path)
            ax.imshow(img)
            ax.axis("off")
        except Exception:
            ax.text(0.5, 0.5, f"Cannot display {name}",
                    ha="center", va="center", transform=ax.transAxes)
        self.fig.tight_layout()
        self.canvas.draw_idle()
