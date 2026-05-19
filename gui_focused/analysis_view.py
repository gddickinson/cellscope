"""Analysis results view: Summary / Graphs / Log panels.

This module builds three independent panel widgets (summary_panel,
graphs_panel, log_panel) and exposes them publicly so the main window
can place each one in its own QDockWidget. The class also offers a
back-compat `embed_in_tabs()` helper for callers that just want a
single tabbed widget.
"""
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
    """Builds three independent panel widgets — summary_panel,
    graphs_panel, log_panel — that can be embedded anywhere (tabs,
    dock widgets, splitters). State + populating methods (set_result,
    clear, etc.) remain on this class regardless of how the panels
    are laid out."""

    def __init__(self, logger=None):
        super().__init__()
        self.result = None         # single-cell result dict
        self.multi_results = None  # list of per-cell result dicts
        self.mode = "single"
        self._build_panels(logger)

    # ------------------------------------------------------------------
    # Panel construction (each panel is a standalone QWidget so the
    # main window can put it in a dock widget).
    # ------------------------------------------------------------------
    def _build_panels(self, logger):
        # --- Summary panel ---
        self.summary_panel = QWidget()
        sp_layout = QVBoxLayout(self.summary_panel)
        sp_layout.setContentsMargins(2, 2, 2, 2)
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Run Analyze to see results...")
        sp_layout.addWidget(self.summary_text)

        # --- Graphs panel ---
        self.graphs_panel = QWidget()
        gl = QVBoxLayout(self.graphs_panel)
        gl.setContentsMargins(2, 2, 2, 2)
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
        ctrl_row.addWidget(QLabel("Gap fill:"))
        self.gap_combo = QComboBox()
        self.gap_combo.setToolTip(
            "Linearly interpolate short NaN gaps in timeseries plots.\n"
            "Filled samples are drawn dotted so they're never confused\n"
            "with measured points. Off by default.")
        for label, val in [("off", 0), ("≤1", 1), ("≤2", 2),
                           ("≤3", 3), ("≤5", 5)]:
            self.gap_combo.addItem(label, val)
        self.gap_combo.currentIndexChanged.connect(
            lambda _: self._on_graph_selected(self.graph_combo.currentText()))
        ctrl_row.addWidget(self.gap_combo)
        self.btn_save_plot = QPushButton("Save Plot")
        self.btn_save_plot.clicked.connect(self._on_save_plot)
        self.btn_save_plot.setEnabled(False)
        ctrl_row.addWidget(self.btn_save_plot)
        gl.addLayout(ctrl_row)
        self.plot_fig = Figure(figsize=(5, 4), dpi=100)
        self.plot_canvas = FigureCanvasQTAgg(self.plot_fig)
        gl.addWidget(self.plot_canvas, stretch=1)

        # --- Log panel ---
        if logger:
            from gui.run_log import RunLogWidget
            self.log_widget = RunLogWidget(logger)
            self.log_panel = self.log_widget  # log widget is its own panel
        else:
            self.log_widget = None
            placeholder = QPlainTextEdit()
            placeholder.setReadOnly(True)
            placeholder.setPlaceholderText("(no logger provided)")
            self.log_panel = placeholder

    def embed_in_tabs(self):
        """Back-compat helper: returns a QTabWidget containing all
        three panels. Used when AnalysisView is shown without dock
        widgets (kept for older entry points)."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.tabs.addTab(self.summary_panel, "Summary")
        self.tabs.addTab(self.graphs_panel, "Graphs")
        if self.log_widget is not None:
            self.tabs.addTab(self.log_panel, "Log")
        return self.tabs

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
        n_divisions = sum(1 for r in results
                          if r.get("track_info", {}).get("parent_id")
                          is not None)
        div_txt = (f", {n_divisions} division(s) detected"
                   if n_divisions else "")
        lines = [f"Multi-cell analysis: {len(results)} cells"
                  f"{div_txt}\n"]
        for r in results:
            cid = r.get("cell_id", "?")
            ti = r.get("track_info", {})
            lines.append(f"--- Cell {cid} ---")
            lines.append(f"  Frames tracked: {ti.get('frames_tracked', '?')}")
            lines.append(f"  First frame: {ti.get('first_frame', '?')}")
            if ti.get("parent_id") is not None:
                # parent_id is 0-based list index; daughter labels are
                # 1-based — show parent as Cell (parent_id + 1)
                pid = ti["parent_id"]
                div_frame = ti.get("division_frame")
                div_score = ti.get("division_score")
                extra = ""
                if div_frame is not None:
                    extra = f" @F{div_frame}"
                    if div_score is not None:
                        extra += f", score {div_score:.2f}"
                lines.append(
                    f"  Parent: Cell {pid + 1} (division{extra})")
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
        gap_max = self.gap_combo.currentData() or 0
        all_selected = (self.cell_combo.currentData() is None
                        and self.multi_results is not None)
        if requires_multi and self.multi_results:
            fn(self.plot_fig, self.multi_results, gap_interp_max=gap_max)
        elif all_selected and not requires_multi and self.multi_results:
            self._plot_all_cells_overlaid(fn, name, gap_max)
        else:
            r = self._get_current_result()
            if r:
                fn(self.plot_fig, r, gap_interp_max=gap_max)
        self.plot_canvas.draw_idle()

    # ------------------------------------------------------------------
    # All-Cells aggregation
    # ------------------------------------------------------------------
    # When the user picks 'All Cells', each graph aggregates across the
    # tracked cells differently:
    #   - Trajectory          → per-cell paths overlaid (mean has no meaning)
    #   - 1D timeseries       → mean curve + ±SD shaded band (n cells)
    #   - MSD / Autocorrelation → mean curve + ±SD band, x = lag
    #   - Shape panel         → 6 subplots, each mean ± SD curve
    #   - Edge kymograph      → mean of per-cell 2D kymographs (imshow)
    #   - Edge summary bar    → bar chart, height=mean, error bar=SD
    #   - Frame quality       → mean curve + ±SD band
    #   - VAMPIRE plots       → small multiples (per-cell PCA, can't pool)
    #
    # Error bars are STANDARD DEVIATION across cells, not standard error,
    # since each cell is the experimental unit.

    def _plot_all_cells_overlaid(self, fn, name, gap_interp_max=0):
        """Dispatch to the right aggregator for this graph."""
        self.plot_fig.clear()
        lname = name.lower()
        n_cells = len(self.multi_results)

        if "trajectory" in lname:
            self._plot_trajectories_overlay(name, n_cells)
            return

        # Shape panel = 6 metrics, each on its own subplot with mean ± SD
        if "shape panel" in lname:
            self._plot_shape_panel_mean_sd(name, n_cells)
            return

        # Edge kymograph = 2D image, mean of per-cell kymographs
        if "kymograph" in lname:
            self._plot_kymograph_mean(name, n_cells)
            return

        # Edge summary bar = mean bar height + SD error bars per angular sector
        if "edge summary" in lname:
            self._plot_edge_summary_mean_sd(name, n_cells)
            return

        # MSD and direction autocorrelation = lag-indexed curves
        if lname.startswith("msd") or "autocorrelation" in lname:
            self._plot_lag_curve_mean_sd(name, n_cells, lname,
                                          gap_interp_max=gap_interp_max)
            return

        # 1D timeseries graphs (speed, area, circularity, boundary, IoU,
        # frame quality) — mean curve + ±SD band
        ts_keys = {
            "speed":             ("Speed (µm/min)",
                                  lambda r: r.get("speed")),
            "area":              ("Area (µm²)",
                                  lambda r: (r.get("shape_timeseries") or {})
                                               .get("area_um2")),
            "circularity":       ("Circularity",
                                  lambda r: (r.get("shape_timeseries") or {})
                                               .get("circularity")),
            "boundary":          ("Boundary confidence",
                                  lambda r: r.get("boundary_confidence_per_frame")
                                  if r.get("boundary_confidence_per_frame") is not None
                                  else r.get("boundary_confidence")),
            "consecutive iou":   ("Consecutive IoU",
                                  lambda r: r.get("consecutive_iou")),
            "frame quality":     ("Quality flag",
                                  lambda r: r.get("quality_per_frame")),
        }
        for key, (ylabel, extractor) in ts_keys.items():
            if key in lname:
                self._plot_timeseries_mean_sd(name, n_cells, extractor,
                                                ylabel,
                                                xlabel="Frame")
                return

        # VAMPIRE plots — aggregate across cells as mean ± SD instead
        # of per-cell small multiples.
        if "vampire" in lname:
            self._plot_vampire_aggregate(name, n_cells)
            return

        # Fall back to small multiples for anything else we didn't
        # explicitly handle
        self._plot_small_multiples(fn, name, n_cells, gap_interp_max)

    # --- Aggregation helpers (one per chart family) -------------------

    @staticmethod
    def _stack_ragged(arrays):
        """Stack a list of 1D arrays of possibly different lengths into
        a (n_cells, max_len) array, padding shorter ones with NaN."""
        arrays = [np.asarray(a, dtype=float) for a in arrays
                  if a is not None and np.asarray(a).size > 0]
        if not arrays:
            return None
        max_len = max(a.size for a in arrays)
        stack = np.full((len(arrays), max_len), np.nan)
        for i, a in enumerate(arrays):
            stack[i, :a.size] = a
        return stack

    @staticmethod
    def _mean_sd(stack):
        """Return (mean, sd, n_valid) across the first axis,
        ignoring NaNs. SD uses ddof=1 (sample SD)."""
        with np.errstate(invalid="ignore"):
            n_valid = np.sum(~np.isnan(stack), axis=0)
            mean = np.nanmean(stack, axis=0)
            sd = np.nanstd(stack, axis=0, ddof=1)
        return mean, sd, n_valid

    @staticmethod
    def _draw_mean_sd(ax, x, mean, sd, n_valid, n_cells,
                      label="Mean", color="C0", band_alpha=0.25):
        """Standard mean-curve-with-SD-band drawing on a given Axes."""
        valid = n_valid > 0
        ax.plot(x[valid], mean[valid], color=color, lw=2.0,
                label=f"{label} (n={n_cells})", zorder=5)
        band_valid = n_valid > 1
        ax.fill_between(x[band_valid],
                        (mean - sd)[band_valid],
                        (mean + sd)[band_valid],
                        color=color, alpha=band_alpha, zorder=4,
                        linewidth=0, label="± SD")

    def _no_data_panel(self, name, msg=None):
        ax = self.plot_fig.add_subplot(111)
        ax.text(0.5, 0.5,
                msg or f"No '{name}' data available across cells",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12)
        ax.axis("off")
        self.plot_fig.tight_layout()

    # --- Per-chart aggregators ---------------------------------------

    def _plot_trajectories_overlay(self, name, n_cells):
        """Trajectory: overlay per-cell paths. No aggregation."""
        from gui.mask_editor_multicell import CELL_COLORS
        ax = self.plot_fig.add_subplot(111)
        any_drawn = False
        for i, r in enumerate(self.multi_results):
            cid = r.get("cell_id", i + 1)
            color = tuple(c / 255.0 for c in
                          CELL_COLORS[i % len(CELL_COLORS)])
            traj = r.get("trajectory")
            if traj is None:
                continue
            traj = np.asarray(traj)
            valid = ~np.isnan(traj[:, 0])
            ax.plot(traj[valid, 1], traj[valid, 0],
                    color=color, lw=1.2, alpha=0.85,
                    label=f"Cell {cid}")
            any_drawn = True
        if not any_drawn:
            self._no_data_panel(name)
            return
        ax.legend(fontsize=8, ncol=min(n_cells, 3))
        ax.set_title(f"{name} — All Cells (n={n_cells})")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        self.plot_fig.tight_layout()

    def _plot_timeseries_mean_sd(self, name, n_cells, extractor,
                                   ylabel, xlabel="Frame"):
        arrays = [extractor(r) for r in self.multi_results]
        stack = self._stack_ragged(arrays)
        if stack is None:
            self._no_data_panel(name)
            return
        mean, sd, n_valid = self._mean_sd(stack)
        x = np.arange(stack.shape[1])
        ax = self.plot_fig.add_subplot(111)
        self._draw_mean_sd(ax, x, mean, sd, n_valid, n_cells)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{name} — All Cells (n={n_cells}, mean ± SD)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        self.plot_fig.tight_layout()

    def _plot_lag_curve_mean_sd(self, name, n_cells, lname,
                                 gap_interp_max=0):
        """MSD / autocorrelation: lag on x-axis, mean ± SD on y."""
        key = "msd" if lname.startswith("msd") else "autocorr"
        x_vals = None
        ys = []
        for r in self.multi_results:
            tup = r.get(key)
            if tup is None or len(tup) < 2:
                continue
            lags, vals = np.asarray(tup[0]), np.asarray(tup[1])
            if lags.size == 0:
                continue
            if x_vals is None or x_vals.size < lags.size:
                x_vals = lags
            ys.append(np.asarray(vals, dtype=float))
        stack = self._stack_ragged(ys)
        if stack is None or x_vals is None:
            self._no_data_panel(name)
            return
        mean, sd, n_valid = self._mean_sd(stack)
        # Trim x to match stack length if needed
        x = np.asarray(x_vals[:stack.shape[1]])
        ax = self.plot_fig.add_subplot(111)
        self._draw_mean_sd(ax, x, mean, sd, n_valid, n_cells)
        ax.set_xlabel("Lag (frames)")
        if key == "msd":
            ax.set_ylabel("MSD (µm²)")
            if "fit" in lname:
                # Overlay a power-law fit to the MEAN curve
                self._overlay_msd_fit(ax, x, mean)
        else:
            ax.set_ylabel("Direction autocorrelation")
        ax.set_title(f"{name} — All Cells (n={n_cells}, mean ± SD)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        self.plot_fig.tight_layout()

    @staticmethod
    def _overlay_msd_fit(ax, x, mean):
        """Fit MSD ≈ D·τ^α to the mean curve and overlay it."""
        try:
            from scipy.optimize import curve_fit
            valid = (x > 0) & ~np.isnan(mean) & (mean > 0)
            if valid.sum() < 4:
                return
            def model(t, D, alpha):
                return D * t ** alpha
            popt, _ = curve_fit(model, x[valid], mean[valid],
                                 p0=[1.0, 1.0], maxfev=5000)
            D, alpha = popt
            xs = np.linspace(x[valid].min(), x[valid].max(), 200)
            ax.plot(xs, model(xs, D, alpha), "--",
                    color="C3", lw=1.8,
                    label=f"Fit: D={D:.3g}, α={alpha:.2f}",
                    zorder=6)
        except Exception:
            pass

    def _plot_shape_panel_mean_sd(self, name, n_cells):
        """Six metrics laid out in a 2×3 grid, each with mean ± SD curve."""
        metrics = [
            ("area_um2",     "Area (µm²)"),
            ("perimeter_um", "Perimeter (µm)"),
            ("circularity",  "Circularity"),
            ("solidity",     "Solidity"),
            ("aspect_ratio", "Aspect ratio"),
            ("eccentricity", "Eccentricity"),
        ]
        for idx, (key, label) in enumerate(metrics):
            ax = self.plot_fig.add_subplot(2, 3, idx + 1)
            arrays = [(r.get("shape_timeseries") or {}).get(key)
                      for r in self.multi_results]
            stack = self._stack_ragged(arrays)
            if stack is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_title(label, fontsize=10)
                ax.axis("off")
                continue
            mean, sd, n_valid = self._mean_sd(stack)
            x = np.arange(stack.shape[1])
            self._draw_mean_sd(ax, x, mean, sd, n_valid, n_cells)
            ax.set_xlabel("Frame", fontsize=8)
            ax.set_title(label, fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3)
        self.plot_fig.suptitle(
            f"{name} — All Cells (n={n_cells}, mean ± SD)",
            fontsize=12, y=1.0)
        self.plot_fig.tight_layout()

    def _plot_kymograph_mean(self, name, n_cells):
        """Edge kymograph: each cell has a 2D (frames × angles) array.
        Stack them and show the mean kymograph."""
        stacks = []
        for r in self.multi_results:
            ev = r.get("edge_velocity")
            if ev is None:
                continue
            stacks.append(np.asarray(ev, dtype=float))
        if not stacks:
            self._no_data_panel(name)
            return
        # Pad to common (max_frames, n_angles)
        max_t = max(s.shape[0] for s in stacks)
        n_ang = stacks[0].shape[1]
        big = np.full((len(stacks), max_t, n_ang), np.nan)
        for i, s in enumerate(stacks):
            big[i, :s.shape[0], :s.shape[1]] = s
        with np.errstate(invalid="ignore"):
            mean_kymo = np.nanmean(big, axis=0)
        ax = self.plot_fig.add_subplot(111)
        vmax = np.nanmax(np.abs(mean_kymo))
        if vmax == 0 or np.isnan(vmax):
            vmax = 1.0
        im = ax.imshow(mean_kymo.T, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, origin="lower",
                        interpolation="nearest")
        cb = self.plot_fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Mean edge velocity (µm/min)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angular sector")
        ax.set_title(f"{name} — All Cells (n={n_cells}, mean across cells)")
        self.plot_fig.tight_layout()

    def _plot_edge_summary_mean_sd(self, name, n_cells):
        """Edge summary bar chart: per-sector protrusion-vs-retraction
        proportions. Show mean ± SD bars across cells."""
        # Use per-cell `edge_velocity` (n_frames, n_angles). Take per-cell
        # per-sector mean across frames, then mean ± SD across cells.
        per_cell_sector_means = []
        for r in self.multi_results:
            ev = r.get("edge_velocity")
            if ev is None:
                continue
            ev = np.asarray(ev, dtype=float)
            if ev.ndim != 2:
                continue
            with np.errstate(invalid="ignore"):
                per_cell_sector_means.append(np.nanmean(ev, axis=0))
        if not per_cell_sector_means:
            self._no_data_panel(name)
            return
        stack = np.vstack(per_cell_sector_means)   # (n_cells, n_angles)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stack, axis=0)
            sd = np.nanstd(stack, axis=0, ddof=1)
        ax = self.plot_fig.add_subplot(111)
        x = np.arange(mean.size)
        colors = ["#c0392b" if m < 0 else "#2980b9" for m in mean]
        ax.bar(x, mean, yerr=sd, capsize=4, color=colors,
                edgecolor="black", linewidth=0.6,
                error_kw={"elinewidth": 1.0, "ecolor": "black"})
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Angular sector")
        ax.set_ylabel("Mean edge velocity (µm/min)")
        ax.set_title(f"{name} — All Cells (n={n_cells}, mean ± SD)")
        ax.grid(alpha=0.3, axis="y")
        self.plot_fig.tight_layout()

    # ------------------------------------------------------------------
    # VAMPIRE aggregators (per-cell PCA can't be pooled directly, so we
    # aggregate at the "mode fractions" / "explained variance" /
    # "heterogeneity" level — all of which have a consistent index
    # across cells. Mode indices themselves are NOT directly comparable
    # between cells (each cell has its own PCA basis), and the plot
    # title says so.)
    # ------------------------------------------------------------------

    def _plot_vampire_aggregate(self, name, n_cells):
        """Aggregate VAMPIRE plots across cells using mean ± SD."""
        lname = name.lower()
        vamps = [r.get("vampire") for r in self.multi_results
                 if r.get("vampire") is not None]
        if not vamps:
            self._no_data_panel(name)
            return
        if "eigenshape" in lname:
            self._plot_vampire_explained_variance(vamps, name)
            return
        if "over time" in lname:
            self._plot_vampire_entropy_per_cell(vamps, name)
            return
        # "Shape Modes" and "Mode Distribution" both collapse to the
        # same all-cells view: mean mode-fraction per index ± SD.
        self._plot_vampire_mode_fractions(vamps, name)

    def _plot_vampire_mode_fractions(self, vamps, name):
        """Bar chart of mean mode fraction across cells, ±SD per mode."""
        # Each cell has its own mode_fractions vector (length = its
        # n_clusters). Stack ragged, pad with NaN.
        fracs = [
            np.asarray(v.get("heterogeneity", {}).get("mode_fractions", []),
                       dtype=float)
            for v in vamps]
        stack = self._stack_ragged(fracs)
        if stack is None:
            self._no_data_panel(name)
            return
        mean, sd, n_valid = self._mean_sd(stack)
        ax = self.plot_fig.add_subplot(111)
        x = np.arange(1, mean.size + 1)
        ax.bar(x, mean, yerr=sd, capsize=4, color="#4CAF50",
                edgecolor="black", linewidth=0.6,
                error_kw={"elinewidth": 1.0, "ecolor": "black"})
        ax.set_xlabel("Shape mode index")
        ax.set_ylabel("Fraction of frames")
        ax.set_title(
            f"{name} — All Cells (mean ± SD across n={len(vamps)} cells)\n"
            f"Mode indices are per-cell labels — not directly comparable "
            f"across cells.")
        ax.set_xticks(x)
        ax.grid(alpha=0.3, axis="y")
        self.plot_fig.tight_layout()

    def _plot_vampire_explained_variance(self, vamps, name):
        """Bar chart of mean explained variance per PC ±SD."""
        evs = [
            np.asarray(v.get("modes", {}).get("explained_variance", []),
                       dtype=float)
            for v in vamps]
        stack = self._stack_ragged(evs)
        if stack is None:
            self._no_data_panel(name)
            return
        mean, sd, n_valid = self._mean_sd(stack)
        ax = self.plot_fig.add_subplot(111)
        x = np.arange(1, mean.size + 1)
        ax.bar(x, mean * 100, yerr=sd * 100, capsize=4,
                color="#2196F3", edgecolor="black", linewidth=0.6,
                error_kw={"elinewidth": 1.0, "ecolor": "black"})
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance (%)")
        ax.set_title(
            f"{name} — All Cells (mean ± SD across n={len(vamps)} cells)")
        ax.set_xticks(x)
        ax.grid(alpha=0.3, axis="y")
        self.plot_fig.tight_layout()

    def _plot_vampire_entropy_per_cell(self, vamps, name):
        """Per-cell normalised entropy (heterogeneity) as a scatter, with
        the across-cell mean ± SD drawn as a horizontal band.

        Entropy is bounded by max_entropy = log2(n_clusters), so we
        plot the normalised value entropy / max_entropy ∈ [0, 1] to
        keep cells with different n_clusters comparable."""
        ratios = []
        for v in vamps:
            h = v.get("heterogeneity", {})
            ent = float(h.get("entropy", 0.0))
            mx = float(h.get("max_entropy", 0.0)) or 1.0
            ratios.append(ent / mx)
        if not ratios:
            self._no_data_panel(name)
            return
        arr = np.asarray(ratios)
        mean = float(np.nanmean(arr))
        sd = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
        ax = self.plot_fig.add_subplot(111)
        cells = np.arange(1, arr.size + 1)
        ax.scatter(cells, arr, s=50, color="#FF9800",
                    edgecolor="black", linewidth=0.6, zorder=3)
        ax.axhline(mean, color="black", lw=1.0, label=f"mean = {mean:.2f}")
        ax.axhspan(mean - sd, mean + sd, color="gray", alpha=0.2,
                    label=f"±SD = {sd:.2f}")
        ax.set_xlabel("Cell index")
        ax.set_ylabel("Normalised shape entropy (entropy / max_entropy)")
        ax.set_xticks(cells)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"{name} — All Cells (per-cell heterogeneity, "
            f"n={arr.size} cells)\n"
            f"1.0 = all modes equally occupied; 0.0 = single dominant mode")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3, axis="y")
        self.plot_fig.tight_layout()

    def _plot_small_multiples(self, fn, name, n_cells, gap_interp_max=0):
        """Fallback for VAMPIRE plots and other unaggregatable views —
        one mini panel per cell, in a 3-column grid."""
        import math
        from matplotlib.figure import Figure as _Fig
        from matplotlib.backends.backend_agg import (
            FigureCanvasAgg as _Canvas)
        ncols = min(3, max(1, n_cells))
        nrows = math.ceil(n_cells / ncols)
        for i, r in enumerate(self.multi_results):
            cid = r.get("cell_id", i + 1)
            sub_fig = _Fig(figsize=(4, 3), dpi=90)
            _Canvas(sub_fig)
            try:
                fn(sub_fig, r, gap_interp_max=gap_interp_max)
            except Exception as e:
                sub_fig.clear()
                sax = sub_fig.add_subplot(111)
                sax.text(0.5, 0.5, f"(plot failed: {e})",
                         ha="center", va="center",
                         transform=sax.transAxes, fontsize=8)
                sax.axis("off")
            sub_fig.canvas.draw()
            buf = np.asarray(sub_fig.canvas.buffer_rgba())[..., :3].copy()
            grid_ax = self.plot_fig.add_subplot(nrows, ncols, i + 1)
            grid_ax.imshow(buf)
            grid_ax.set_title(f"Cell {cid}", fontsize=10)
            grid_ax.axis("off")
        self.plot_fig.suptitle(
            f"{name} — All Cells (small multiples, n={n_cells})",
            fontsize=12, y=0.98)
        self.plot_fig.tight_layout(rect=[0, 0, 1, 0.96])

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
