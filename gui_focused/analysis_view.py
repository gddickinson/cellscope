"""Analysis results view: Summary / Graphs / Log tabs."""
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel,
    QComboBox, QPushButton, QPlainTextEdit, QFileDialog,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gui_focused.analysis_plots import GRAPH_REGISTRY


class AnalysisView(QWidget):
    """Three-tab widget: Summary text, interactive Graphs, Run Log."""

    def __init__(self, logger=None):
        super().__init__()
        self.result = None         # single-cell result dict
        self.multi_results = None  # list of per-cell result dicts
        self.mode = "single"
        self._build_ui(logger)

    def _build_ui(self, logger):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Summary
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Run Analyze to see results...")
        self.tabs.addTab(self.summary_text, "Summary")

        # Tab 2: Graphs
        graphs_widget = QWidget()
        gl = QVBoxLayout(graphs_widget)
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Graph:"))
        self.graph_combo = QComboBox()
        self.graph_combo.currentTextChanged.connect(self._on_graph_selected)
        ctrl_row.addWidget(self.graph_combo, stretch=1)
        self.cell_combo = QComboBox()
        self.cell_combo.addItem("All Cells")
        self.cell_combo.currentIndexChanged.connect(self._on_cell_selected)
        self.cell_combo.setVisible(False)
        ctrl_row.addWidget(QLabel("Cell:"))
        ctrl_row.addWidget(self.cell_combo)
        self.btn_save_plot = QPushButton("Save Plot")
        self.btn_save_plot.clicked.connect(self._on_save_plot)
        self.btn_save_plot.setEnabled(False)
        ctrl_row.addWidget(self.btn_save_plot)
        gl.addLayout(ctrl_row)
        self.plot_fig = Figure(figsize=(5, 4), dpi=100)
        self.plot_canvas = FigureCanvasQTAgg(self.plot_fig)
        gl.addWidget(self.plot_canvas, stretch=1)
        self.tabs.addTab(graphs_widget, "Graphs")

        # Tab 3: Log
        if logger:
            from gui.run_log import RunLogWidget
            self.log_widget = RunLogWidget(logger)
            self.tabs.addTab(self.log_widget, "Log")

    def set_result(self, result, mode="single"):
        """Set analysis result and populate Summary + Graphs."""
        self.result = result
        self.mode = mode
        self.multi_results = None
        self._populate_summary_single(result)
        self._populate_graph_combo(is_multi=False)
        self.cell_combo.setVisible(False)
        self.btn_save_plot.setEnabled(True)

    def set_multi_result(self, results):
        """Set per-cell results for multi-cell mode."""
        self.multi_results = results
        self.mode = "multi"
        self.result = results[0] if results else None
        self._populate_summary_multi(results)
        self._populate_graph_combo(is_multi=True)
        self.cell_combo.setVisible(True)
        self.cell_combo.clear()
        self.cell_combo.addItem("All Cells", None)
        for r in results:
            cid = r.get("cell_id", "?")
            self.cell_combo.addItem(f"Cell {cid}", cid)
        self.btn_save_plot.setEnabled(True)

    def clear(self):
        self.result = None
        self.multi_results = None
        self.summary_text.clear()
        self.graph_combo.clear()
        self.cell_combo.setVisible(False)
        self.plot_fig.clear()
        self.plot_canvas.draw_idle()
        self.btn_save_plot.setEnabled(False)

    def _populate_summary_single(self, r):
        lines = [f"Recording: {r.get('name', '?')}",
                 f"Frames: {r.get('n_frames', '?')}",
                 ""]
        if "mean_speed" in r:
            lines.append(f"Mean speed: {r['mean_speed']:.3f} um/min")
            lines.append(f"Total distance: {r.get('total_distance', 0):.1f} um")
            lines.append(f"Persistence: {r.get('persistence', 0):.3f}")
        ss = r.get("shape_summary", {})
        if "area_um2" in ss:
            a = ss["area_um2"]
            lines.append(f"\nMean area: {a.get('mean', 0):.0f} um^2")
        es = r.get("edge_summary", {})
        if es:
            lines.append(f"\nProtrusion vel: "
                         f"{es.get('mean_protrusion_velocity', 0):.3f} um/min")
            lines.append(f"Retraction vel: "
                         f"{es.get('mean_retraction_velocity', 0):.3f} um/min")
        bc = r.get("mean_boundary_confidence")
        if bc is not None:
            lines.append(f"\nBoundary confidence: {bc:.3f}")
        self.summary_text.setPlainText("\n".join(lines))

    def _populate_summary_multi(self, results):
        lines = [f"Multi-cell analysis: {len(results)} cells\n"]
        for r in results:
            cid = r.get("cell_id", "?")
            ti = r.get("track_info", {})
            lines.append(f"--- Cell {cid} ---")
            lines.append(f"  Frames tracked: {ti.get('frames_tracked', '?')}")
            lines.append(f"  First frame: {ti.get('first_frame', '?')}")
            if ti.get("parent_id"):
                lines.append(f"  Parent: Cell {ti['parent_id']} (division)")
            if "mean_speed" in r:
                lines.append(f"  Speed: {r['mean_speed']:.3f} um/min")
            ss = r.get("shape_summary", {})
            if "area_um2" in ss:
                lines.append(f"  Area: {ss['area_um2'].get('mean', 0):.0f} um^2")
            lines.append(f"  Persistence: {r.get('persistence', 0):.3f}")
            lines.append("")
        self.summary_text.setPlainText("\n".join(lines))

    def _populate_graph_combo(self, is_multi):
        self.graph_combo.clear()
        for name, (fn, requires_multi) in GRAPH_REGISTRY.items():
            if requires_multi and not is_multi:
                continue
            self.graph_combo.addItem(name)

    def _get_current_result(self):
        """Get the result for the currently selected cell."""
        if self.mode == "single" or self.multi_results is None:
            return self.result
        cell_id = self.cell_combo.currentData()
        if cell_id is None:
            return self.multi_results[0] if self.multi_results else None
        for r in self.multi_results:
            if r.get("cell_id") == cell_id:
                return r
        return self.result

    def _on_graph_selected(self, name):
        if not name or name not in GRAPH_REGISTRY:
            return
        fn, requires_multi = GRAPH_REGISTRY[name]
        all_selected = (self.cell_combo.currentData() is None
                        and self.multi_results is not None)
        if requires_multi and self.multi_results:
            fn(self.plot_fig, self.multi_results)
        elif all_selected and not requires_multi and self.multi_results:
            self._plot_all_cells_overlaid(fn, name)
        else:
            r = self._get_current_result()
            if r:
                fn(self.plot_fig, r)
        self.plot_canvas.draw_idle()

    def _plot_all_cells_overlaid(self, fn, name):
        """For non-comparison graphs with 'All Cells' selected,
        plot each cell on the same axes with different colors."""
        from gui.mask_editor_multicell import CELL_COLORS
        self.plot_fig.clear()
        ax = self.plot_fig.add_subplot(111)
        for i, r in enumerate(self.multi_results):
            cid = r.get("cell_id", i + 1)
            color = tuple(c / 255.0 for c in
                          CELL_COLORS[i % len(CELL_COLORS)])
            if "speed" in name.lower() and "speed" in r:
                ax.plot(r["speed"], color=color, lw=0.8,
                        alpha=0.8, label=f"Cell {cid}")
            elif "area" in name.lower():
                ts = r.get("shape_timeseries", {})
                area = ts.get("area_um2")
                if area is not None:
                    ax.plot(area, color=color, lw=0.8,
                            alpha=0.8, label=f"Cell {cid}")
            elif "trajectory" in name.lower():
                traj = r.get("trajectory")
                if traj is not None:
                    valid = ~np.isnan(traj[:, 0])
                    ax.plot(traj[valid, 1], traj[valid, 0],
                            color=color, lw=1.0, alpha=0.8,
                            label=f"Cell {cid}")
            elif "boundary" in name.lower():
                bc = r.get("boundary_confidence_per_frame")
                if bc is not None:
                    ax.plot(bc, color=color, lw=0.8,
                            alpha=0.8, label=f"Cell {cid}")
            else:
                fn(self.plot_fig, r)
                return
        ax.legend(fontsize=8)
        ax.set_title(f"{name} — All Cells")
        ax.grid(alpha=0.3)
        self.plot_fig.tight_layout()

    def _on_cell_selected(self, _idx):
        name = self.graph_combo.currentText()
        if name:
            self._on_graph_selected(name)

    def _on_save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "plot.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            self.plot_fig.savefig(path, dpi=150, bbox_inches="tight")
