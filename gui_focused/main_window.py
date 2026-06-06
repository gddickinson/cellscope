"""Focused pipeline GUI main window."""
import os
import time
import logging
import numpy as np
import tempfile

from PyQt5.QtCore import Qt, QByteArray
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QStatusBar, QProgressBar, QMenuBar,
    QAction, QLabel, QDockWidget,
)

from gui.run_log import RunLogger
from gui_focused.image_viewer import ImageViewer
from gui_focused.pipeline_panel import PipelinePanel
from gui_focused.params_panel import ParamsPanel
from gui_focused.analysis_view import AnalysisView
from gui_focused.dialogs import (
    detect_gpu, show_system_info, show_recording_info,
    show_shortcuts, show_about, open_doc, channel_chooser,
)
from gui_focused.roi_selector import ROISelector

log = logging.getLogger(__name__)


class FocusedMainWindow(QMainWindow):
    """Streamlined GUI for cpsam single/multi-cell analysis."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CellScope — Focused Pipeline")
        self.resize(1400, 900)
        self.setAcceptDrops(True)

        self.recording = None
        self.detect_result = None
        self.analysis_result = None
        self._prev_detect_result = None   # for undo
        # Tracks unsaved detection / analysis / edit results so we can
        # warn the user on close (or before they overwrite state by
        # loading another recording / clearing).
        self._dirty = False
        self.mode = "single"
        self.logger = RunLogger()
        self._worker = None

        self._build_ui()
        self.roi = ROISelector(self.viewer)
        self.viewer._roi_selector = self.roi
        self.roi.on_roi_drawn = self._on_roi_drawn
        self._build_menu()
        self._connect_signals()
        self.params.set_context("load", self.mode)
        self._maybe_start_remote_control()

    def _maybe_start_remote_control(self):
        """Start the HTTP control server if CELLSCOPE_REMOTE is set."""
        from gui_focused.remote_control import attach
        handlers = {
            "/status":              lambda d: self._remote_status(),
            "/params":              lambda d: self.params.get_detect_params(),
            "/log":                 lambda d: {"lines": self._remote_log(
                                        int(d.get("limit", 50)))},
            "/load_recording":      self._remote_load_recording,
            "/load_pipeline_results": self._remote_load_pipeline_results,
            "/load_project":        self._remote_load_project,
            "/clear_all":           self._remote_clear_all,
            "/set_param":           self._remote_set_param,
            "/set_frame":           self._remote_set_frame,
            "/set_view":            self._remote_set_view,
            "/set_mode":            self._remote_set_mode,
            "/detect":              self._remote_detect,
            "/test_frame":          self._remote_test_frame,
            "/analyze":             self._remote_analyze,
            "/save_screenshot":     self._remote_save_screenshot,
            "/save_project":        self._remote_save_project,
            "/export":              self._remote_export,
        }
        self._remote = None
        attach(self, gui_type="focused", handlers=handlers,
                default_port=8765, status_logger=self.logger.log)

    # --- Remote control handlers ---

    def _remote_status(self):
        rec = self.recording or {}
        s = {
            "recording_loaded": self.recording is not None,
            "recording_name": rec.get("name"),
            "recording_path": rec.get("video_path"),
            "n_frames": (len(rec.get("frames"))
                          if self.recording else None),
            "has_cy5": rec.get("cy5_frames") is not None,
            "current_frame": (self.viewer.current_frame
                               if self.viewer.frames is not None
                               else None),
            "mode": self.mode,
            "detect_result_present": self.detect_result is not None,
            "n_tracks_kept": (len(self.detect_result.get("tracks",
                                                          []) or [])
                                if self.detect_result else 0),
            "dirty": bool(self._dirty),
        }
        return s

    def _remote_log(self, limit):
        events = getattr(self.logger, "events", [])
        recent = events[-int(limit):]
        out = []
        for ev in recent:
            if hasattr(ev, "to_dict"):
                out.append(ev.to_dict())
            elif isinstance(ev, dict):
                out.append(ev)
            else:
                out.append({"kind": getattr(ev, "kind", "?"),
                             "msg": getattr(ev, "message",
                                             str(ev)),
                             "ts": getattr(ev, "ts", None)})
        return out

    def _remote_load_recording(self, data):
        path = data.get("path")
        if not path or not os.path.exists(path):
            raise ValueError(f"path missing or not found: {path!r}")
        # Bypass the channel-chooser dialog if dic/fluo specified.
        dic_ch = data.get("dic_channel")
        fluo_ch = data.get("fluo_channel")
        from core.io import load_recording, detect_channels
        n_ch = (detect_channels(path)
                 if path.lower().endswith((".tif", ".tiff")) else 1)
        if n_ch > 1 and dic_ch is None:
            # Default DIC=ch1, fluo=ch0 (matches cellscope convention)
            dic_ch, fluo_ch = 1, 0
        self.recording = load_recording(
            path, dic_channel=dic_ch, fluo_channel=fluo_ch)
        n = len(self.recording["frames"])
        name = self.recording.get("name", os.path.basename(path))
        self.logger.log(
            "info", f"Remote loaded {name}: {n} frames"
            + (" [+ Cy5]" if self.recording.get("cy5_frames")
                is not None else ""))
        self.viewer.set_data(
            self.recording["frames"],
            fluo_frames=self.recording.get("cy5_frames"),
            um_per_px=self.recording.get("um_per_px"),
            dt_min=self.recording.get("time_interval_min"))
        self.detect_result = None
        self.analysis_result = None
        self.analysis.clear()
        self.pipeline.reset_all()
        self.pipeline.set_stage_status("load", "done")
        self.pipeline.enable_stage("detect", True)
        self.params.set_from_recording(self.recording)
        if hasattr(self.params, "set_cy5_available"):
            self.params.set_cy5_available(
                self.recording.get("cy5_frames") is not None)
        self.params.set_context("detect", self.mode)
        self._dirty = False
        return {"ok": True, "n_frames": n, "name": name}

    def _remote_load_pipeline_results(self, data):
        path = data.get("path")
        if not path or not os.path.isdir(path):
            raise ValueError(f"folder missing: {path!r}")
        from gui_focused.project_handlers import (
            on_load_pipeline_results)
        on_load_pipeline_results(self, path=path)
        return {"ok": True}

    def _remote_load_project(self, data):
        path = data.get("path")
        if not path or not os.path.exists(path):
            raise ValueError(f"project missing: {path!r}")
        from gui_focused.project_handlers import on_load_project
        on_load_project(self, path=path)
        return {"ok": True}

    def _remote_clear_all(self, data):
        # Bypass the confirm-discard prompt for scripted use:
        self._dirty = False
        self._on_clear_all()
        return {"ok": True}

    def _remote_set_param(self, data):
        name = data.get("name")
        value = data.get("value")
        widget = getattr(self.params, name, None)
        if widget is None:
            raise ValueError(f"no params widget named {name!r}")
        # Try the common setter signatures
        for setter in ("setValue", "setChecked", "setCurrentText"):
            fn = getattr(widget, setter, None)
            if fn is None:
                continue
            try:
                if setter == "setCurrentText":
                    fn(str(value))
                elif setter == "setChecked":
                    fn(bool(value))
                else:
                    fn(value)
                return {"ok": True, "set": setter, "value": value}
            except Exception:
                continue
        raise ValueError(f"don't know how to set {name!r}")

    def _remote_set_frame(self, data):
        idx = int(data.get("index", 0))
        n = (len(self.viewer.frames)
              if self.viewer.frames is not None else 0)
        if idx < 0 or idx >= n:
            raise ValueError(f"frame {idx} out of range [0,{n})")
        self.viewer.frame_slider.setValue(idx)
        return {"ok": True, "index": idx}

    def _remote_set_view(self, data):
        control = data.get("control")
        checked = data.get("checked")
        mapping = {
            "mask": self.viewer.chk_mask,
            "contour": self.viewer.chk_contour,
            "ids": self.viewer.chk_ids,
            "tracks": self.viewer.chk_tracks,
            "source": self.viewer.chk_source,
            "dropped": self.viewer.chk_dropped,
        }
        if control == "channel":
            val = str(checked).lower()
            if val == "fluo" and self.viewer.radio_fluo.isVisible():
                self.viewer.radio_fluo.setChecked(True)
            elif val == "dic":
                self.viewer.radio_dic.setChecked(True)
            else:
                raise ValueError(
                    f"channel must be 'dic' or 'fluo', got {val!r}")
            return {"ok": True, "channel": val}
        widget = mapping.get(control)
        if widget is None:
            raise ValueError(
                f"unknown view control {control!r}; "
                f"valid: {list(mapping)} + 'channel'")
        widget.setChecked(bool(checked))
        return {"ok": True, "control": control,
                 "checked": bool(checked)}

    def _remote_detect(self, data):
        self._on_detect()
        return {"ok": True, "status": "detection started"}

    def _remote_test_frame(self, data):
        self._on_test_frame()
        return {"ok": True}

    def _remote_save_screenshot(self, data):
        path = data.get("path")
        if not path:
            raise ValueError("path required")
        viewer_only = bool(data.get("viewer_only", False))
        if viewer_only:
            self.viewer.fig.savefig(
                path, dpi=int(data.get("dpi", 200)),
                bbox_inches="tight",
                facecolor=self.viewer.fig.get_facecolor())
        else:
            pix = self.grab()
            if not pix.save(path, "PNG"):
                raise RuntimeError("QPixmap.save returned False")
        return {"ok": True, "path": path}

    def _remote_set_mode(self, data):
        mode = data.get("mode")
        if mode not in ("single", "multi"):
            raise ValueError(
                f"mode must be 'single' or 'multi', got {mode!r}")
        self.pipeline.set_mode(mode)
        return {"ok": True, "mode": mode}

    def _remote_analyze(self, data):
        if self.detect_result is None:
            raise ValueError("no detection result — call /detect first")
        self._on_analyze()
        return {"ok": True, "status": "analyze started"}

    def _remote_export(self, data):
        """Export to a directory non-interactively. Bypasses the
        export dialog by writing directly via the same machinery
        the dialog uses on OK."""
        out_dir = data.get("out_dir")
        if not out_dir:
            raise ValueError("out_dir required")
        if self.detect_result is None:
            raise ValueError("no detection result — call /detect first")
        os.makedirs(out_dir, exist_ok=True)
        # Write masks.npz
        import numpy as np
        labels = self.detect_result.get("labels")
        masks = self.detect_result.get("masks")
        np.savez_compressed(
            os.path.join(out_dir, "masks.npz"),
            labels=labels if labels is not None else None,
            masks=masks)
        # Write a small RUN_METADATA.json
        import json as _json
        meta = {
            "recording": (self.recording.get("name")
                           if self.recording else None),
            "n_frames": (len(self.recording["frames"])
                          if self.recording else 0),
            "n_tracks": int(labels.max()) if labels is not None else 0,
            "params": self.params.get_detect_params(),
        }
        with open(os.path.join(out_dir, "RUN_METADATA.json"),
                   "w") as f:
            _json.dump(meta, f, indent=2, default=str)
        return {"ok": True, "out_dir": out_dir,
                 "files": ["masks.npz", "RUN_METADATA.json"]}

    def _remote_save_project(self, data):
        path = data.get("path")
        if not path:
            raise ValueError("path required")
        from core.project import save_project
        save_project(
            path, self.recording, self.detect_result,
            self.analysis_result, self.params.get_detect_params(),
            self.mode,
            roi_mask=self.roi.roi_mask if self.roi.has_roi()
                else None)
        self._dirty = False
        return {"ok": True, "path": path}

    def _build_ui(self):
        """Dock-widget layout — every panel is detachable + resizable.

        Default layout:
          Central:   ImageViewer (always docked, takes most of the window)
          Right top: Pipeline dock
          Right mid: Parameters dock
          Right bot: Summary / Graphs / Log docks (tabified)

        Users can drag any dock out to its own window, drop it into a
        different area, hide it (close button), or tabify it with another
        dock. View > Reset Layout returns everything to this default.
        """
        # Allow nested + animated docking, with tab moves
        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.AllowNestedDocks
            | QMainWindow.AllowTabbedDocks
            | QMainWindow.AnimatedDocks)

        # --- Central widget: the image viewer ---
        self.viewer = ImageViewer()
        self.setCentralWidget(self.viewer)

        # --- Create the AnalysisView controller (builds 3 child panels
        #     internally; we put each into its own dock below) ---
        self.analysis = AnalysisView(logger=self.logger)

        # --- Build each dock ---
        self.pipeline = PipelinePanel()
        self.params = ParamsPanel()

        self.dock_pipeline = self._make_dock("Pipeline", self.pipeline)
        self.dock_params   = self._make_dock("Parameters", self.params)
        self.dock_summary  = self._make_dock("Summary",
                                              self.analysis.summary_panel)
        self.dock_graphs   = self._make_dock("Graphs",
                                              self.analysis.graphs_panel)
        self.dock_log      = self._make_dock("Log",
                                              self.analysis.log_panel)

        # Set reasonable starting sizes
        self.dock_pipeline.setMinimumWidth(300)
        self.dock_params.setMinimumWidth(300)
        self.dock_summary.setMinimumWidth(360)
        self.dock_graphs.setMinimumWidth(360)
        self.dock_log.setMinimumWidth(360)

        # --- Status bar ---
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status.addPermanentWidget(self.progress_bar)
        self.status.showMessage("Ready - load a recording to begin")

        # --- Apply default layout ---
        self._apply_default_layout()
        # Save state immediately so Reset Layout can restore exactly this.
        self._default_state = self.saveState()
        self._default_geometry = self.saveGeometry()

    def _make_dock(self, title, widget):
        """Wrap a widget in a QDockWidget with full move/float/close
        features enabled."""
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.lower().replace(' ', '_')}")
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        dock.setFeatures(QDockWidget.DockWidgetMovable
                         | QDockWidget.DockWidgetFloatable
                         | QDockWidget.DockWidgetClosable)
        dock.setWidget(widget)
        return dock

    def _apply_default_layout(self):
        """Place each dock in its default area. Called by both initial
        setup and View > Reset Layout."""
        # Remove anything currently docked, in case we're resetting
        for dock in [self.dock_pipeline, self.dock_params,
                     self.dock_summary, self.dock_graphs, self.dock_log]:
            self.removeDockWidget(dock)
            dock.setFloating(False)
            dock.setVisible(True)

        # Right column, top-to-bottom: Pipeline, Parameters
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_pipeline)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_params)
        # Split to put Pipeline above Params
        self.splitDockWidget(self.dock_pipeline, self.dock_params,
                             Qt.Vertical)

        # Bottom of right column: Summary / Graphs / Log tabified together
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_summary)
        self.splitDockWidget(self.dock_params, self.dock_summary,
                             Qt.Vertical)
        self.tabifyDockWidget(self.dock_summary, self.dock_graphs)
        self.tabifyDockWidget(self.dock_summary, self.dock_log)
        # Show Summary on top by default
        self.dock_summary.raise_()

        # Ensure visibility on each
        for d in [self.dock_pipeline, self.dock_params,
                  self.dock_summary, self.dock_graphs, self.dock_log]:
            d.setVisible(True)

    def reset_layout(self):
        """Restore the layout to its initial state (called from View
        menu). Re-shows all docks and returns them to default areas."""
        if hasattr(self, "_default_state") and self._default_state is not None:
            # Make sure all docks are visible before restoring (closed
            # docks aren't recreated by restoreState)
            for d in [self.dock_pipeline, self.dock_params,
                      self.dock_summary, self.dock_graphs, self.dock_log]:
                d.setFloating(False)
                d.setVisible(True)
            self.restoreState(self._default_state)
        else:
            self._apply_default_layout()

    def _build_menu(self):
        mb = self.menuBar()

        # --- File ---
        file_menu = mb.addMenu("File")
        act_open = QAction("Open Recording...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_load)
        file_menu.addAction(act_open)
        file_menu.addSeparator()
        act_save_proj = QAction("Save Project...", self)
        act_save_proj.setShortcut("Ctrl+S")
        act_save_proj.triggered.connect(self._on_save_project)
        file_menu.addAction(act_save_proj)
        act_load_proj = QAction("Open Project...", self)
        act_load_proj.setShortcut("Ctrl+Shift+O")
        act_load_proj.triggered.connect(self._on_load_project)
        file_menu.addAction(act_load_proj)
        act_load_pr = QAction("Open Pipeline Results...", self)
        act_load_pr.setShortcut("Ctrl+Shift+R")
        act_load_pr.setStatusTip(
            "Open a pipeline_results/ folder produced by a batch run "
            "— loads masks, source map, and (if present) the "
            "pre-Cy5-filter dropped-cell overlay.")
        act_load_pr.triggered.connect(self._on_load_pipeline_results)
        file_menu.addAction(act_load_pr)
        act_load_masks = QAction("Open Masks File...", self)
        act_load_masks.setStatusTip(
            "Open a masks.npz file directly — same loader as Open "
            "Pipeline Results, but picks the file instead of its "
            "containing folder. Equivalent to dragging the .npz in.")
        act_load_masks.triggered.connect(self._on_load_masks_file)
        file_menu.addAction(act_load_masks)
        file_menu.addSeparator()
        act_export = QAction("Export Results...", self)
        act_export.setShortcut("Ctrl+Shift+S")
        act_export.triggered.connect(self._on_export)
        file_menu.addAction(act_export)
        act_screenshot = QAction("Save Screenshot...", self)
        act_screenshot.setShortcut("Ctrl+Shift+P")
        act_screenshot.setStatusTip(
            "Save a PNG of the current window (with overlays, "
            "controls, log) — useful for sharing detection results.")
        act_screenshot.triggered.connect(self._on_save_screenshot)
        file_menu.addAction(act_screenshot)
        act_screenshot_view = QAction("Save Viewer Screenshot...", self)
        act_screenshot_view.setShortcut("Ctrl+Alt+P")
        act_screenshot_view.setStatusTip(
            "Save a PNG of just the image viewer (current frame + "
            "overlays), without the parameter panel or log.")
        act_screenshot_view.triggered.connect(
            self._on_save_viewer_screenshot)
        file_menu.addAction(act_screenshot_view)
        act_share = QAction("Export Shareable Image…", self)
        act_share.setShortcut("Ctrl+Shift+I")
        act_share.setStatusTip(
            "Export a small, shareable PNG / JPEG / GIF / MP4 / montage "
            "of the current frame or whole recording, with selectable "
            "overlays (mask, contour, IDs, tracks, timestamp, scale bar).")
        act_share.triggered.connect(self._on_share_image)
        file_menu.addAction(act_share)
        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # --- Edit ---
        edit_menu = mb.addMenu("Edit")
        act_edit = QAction("Edit Masks...", self)
        act_edit.setShortcut("Ctrl+E")
        act_edit.triggered.connect(self._on_edit)
        edit_menu.addAction(act_edit)
        edit_menu.addSeparator()
        roi_menu = edit_menu.addMenu("Select ROI")
        for shape, label in [("rectangle", "Rectangle ROI"),
                              ("ellipse", "Ellipse ROI"),
                              ("polygon", "Polygon ROI (right-click to close)")]:
            act = QAction(label, self)
            act.triggered.connect(
                lambda checked, s=shape: self.roi.start(s))
            roi_menu.addAction(act)
        act_clear_roi = QAction("Clear ROI", self)
        act_clear_roi.triggered.connect(self._on_clear_roi)
        edit_menu.addAction(act_clear_roi)
        edit_menu.addSeparator()
        act_undo_det = QAction("Undo Detection", self)
        act_undo_det.setShortcut("Ctrl+Z")
        act_undo_det.triggered.connect(self._on_undo_detect)
        edit_menu.addAction(act_undo_det)
        edit_menu.addSeparator()
        act_clear = QAction("Clear All Results", self)
        act_clear.setShortcut("Ctrl+Shift+C")
        act_clear.triggered.connect(self._on_clear_all)
        edit_menu.addAction(act_clear)

        # --- View ---
        view_menu = mb.addMenu("View")
        act_zin = QAction("Zoom In", self)
        act_zin.setShortcut("Ctrl+=")
        act_zin.triggered.connect(self.viewer._zoom_in)
        view_menu.addAction(act_zin)
        act_zout = QAction("Zoom Out", self)
        act_zout.setShortcut("Ctrl+-")
        act_zout.triggered.connect(self.viewer._zoom_out)
        view_menu.addAction(act_zout)
        act_zfit = QAction("Zoom to Fit", self)
        act_zfit.setShortcut("Ctrl+0")
        act_zfit.triggered.connect(self.viewer._zoom_fit)
        view_menu.addAction(act_zfit)
        view_menu.addSeparator()
        act_abc = QAction("Auto Brightness/Contrast", self)
        act_abc.triggered.connect(self.viewer._auto_bc)
        view_menu.addAction(act_abc)
        act_rbc = QAction("Reset Brightness/Contrast", self)
        act_rbc.triggered.connect(self.viewer._reset_bc)
        view_menu.addAction(act_rbc)
        view_menu.addSeparator()
        act_overlay = QAction("Overlay Options…", self)
        act_overlay.triggered.connect(self._on_overlay_settings)
        view_menu.addAction(act_overlay)
        view_menu.addSeparator()

        # --- Panels submenu: show/hide each dock individually ---
        panels_menu = view_menu.addMenu("Panels")
        for dock, label in [
            (self.dock_pipeline, "Pipeline"),
            (self.dock_params,   "Parameters"),
            (self.dock_summary,  "Summary"),
            (self.dock_graphs,   "Graphs"),
            (self.dock_log,      "Log"),
        ]:
            act = dock.toggleViewAction()
            act.setText(label)
            panels_menu.addAction(act)

        act_reset = QAction("Reset Layout", self)
        act_reset.setShortcut("Ctrl+Shift+R")
        act_reset.setToolTip(
            "Return all panels (Pipeline, Parameters, Summary, Graphs, "
            "Log) to their default docked positions.")
        act_reset.triggered.connect(self.reset_layout)
        view_menu.addAction(act_reset)
        view_menu.addSeparator()

        act_info = QAction("Recording Info...", self)
        act_info.setShortcut("Ctrl+I")
        act_info.triggered.connect(self._show_recording_info)
        view_menu.addAction(act_info)

        # --- Settings ---
        settings_menu = mb.addMenu("Settings")
        self.act_gpu = QAction("Use GPU acceleration", self)
        self.act_gpu.setCheckable(True)
        self.act_gpu.setChecked(detect_gpu())
        self.act_gpu.setToolTip(
            "Enable GPU (CUDA/MPS) for detection. Disable for "
            "CPU-only systems (slower but always works).")
        settings_menu.addAction(self.act_gpu)
        settings_menu.addSeparator()
        act_sysinfo = QAction("System Info...", self)
        act_sysinfo.triggered.connect(self._show_system_info)
        settings_menu.addAction(act_sysinfo)

        # --- Help ---
        help_menu = mb.addMenu("Help")
        act_guide = QAction("Quick Start Guide", self)
        act_guide.triggered.connect(lambda: open_doc("README.md"))
        help_menu.addAction(act_guide)
        act_roadmap = QAction("Development Roadmap", self)
        act_roadmap.triggered.connect(lambda: open_doc("ROADMAP.md"))
        help_menu.addAction(act_roadmap)
        act_interface = QAction("Interface Map (modules)", self)
        act_interface.triggered.connect(
            lambda: open_doc("INTERFACE.md"))
        help_menu.addAction(act_interface)
        act_session = QAction("Session Log (experiment history)", self)
        act_session.triggered.connect(
            lambda: open_doc("SESSION_LOG.md"))
        help_menu.addAction(act_session)
        act_methods = QAction("Detection Methods Report", self)
        act_methods.triggered.connect(
            lambda: open_doc("DETECTION_METHODS_REPORT.md"))
        help_menu.addAction(act_methods)
        help_menu.addSeparator()
        act_shortcuts = QAction("Keyboard Shortcuts...", self)
        act_shortcuts.triggered.connect(self._show_shortcuts)
        help_menu.addAction(act_shortcuts)
        help_menu.addSeparator()
        act_about = QAction("About...", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _connect_signals(self):
        self.pipeline.load_clicked.connect(self._on_load)
        self.pipeline.detect_clicked.connect(self._on_detect)
        self.pipeline.edit_clicked.connect(self._on_edit)
        self.pipeline.analyze_clicked.connect(self._on_analyze)
        self.pipeline.export_clicked.connect(self._on_export)
        self.pipeline.mode_changed.connect(self._on_mode_changed)
        self.pipeline.cancel_clicked.connect(self._on_cancel)
        self.pipeline.undo_clicked.connect(self._on_undo_detect)
        self.pipeline.clear_all_clicked.connect(self._on_clear_all)
        self.pipeline.test_frame_clicked.connect(self._on_test_frame)
        self.params.btn_scan.clicked.connect(self._on_scan_cells)
        self.params.use_roi.toggled.connect(self._on_roi_toggled)

    def _on_mode_changed(self, mode):
        self.mode = mode
        self.params.set_context("detect", mode)

    def _on_load(self):
        if not self._confirm_discard(
                "Loading a new recording will discard them."):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Recording", "",
            "Video/Image (*.mp4 *.avi *.mov *.tif *.tiff)")
        if path:
            self._load_path(path)

    def _load_path(self, path):
        try:
            from core.io import load_recording, detect_channels
            n_ch = (detect_channels(path)
                    if path.lower().endswith((".tif", ".tiff")) else 1)
            dic_ch = fluo_ch = None
            if n_ch > 1:
                dic_ch, fluo_ch = channel_chooser(self, n_ch)
                if dic_ch is None:
                    self.logger.log(
                        "info",
                        f"Multichannel ({n_ch} ch): single-channel mode")
                else:
                    self.logger.log(
                        "info",
                        f"Multichannel ({n_ch} ch): DIC=ch{dic_ch}, "
                        f"Fluo={'ch'+str(fluo_ch) if fluo_ch is not None else 'off'}")
            self.recording = load_recording(
                path, dic_channel=dic_ch, fluo_channel=fluo_ch)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return
        n = len(self.recording["frames"])
        name = self.recording.get("name", os.path.basename(path))
        cy5_note = ""
        if self.recording.get("cy5_frames") is not None:
            cy5_note = " [+ Cy5 channel]"
        self.logger.log("info", f"Loaded {name}: {n} frames{cy5_note}")
        self.viewer.set_data(
            self.recording["frames"],
            fluo_frames=self.recording.get("cy5_frames"),
            um_per_px=self.recording.get("um_per_px"),
            dt_min=self.recording.get("time_interval_min"))
        self.detect_result = None
        self.analysis_result = None
        self.analysis.clear()
        self.pipeline.reset_all()
        self.pipeline.set_stage_status("load", "done")
        self.pipeline.enable_stage("detect", True)
        self.status.showMessage(f"Loaded: {name} ({n} frames){cy5_note}")
        self.params.set_from_recording(self.recording)
        # Fresh recording = clean slate (no in-memory results yet).
        self._dirty = False
        # Enable Cy5 recovery toggle if recording has fluo channel
        has_cy5 = self.recording.get("cy5_frames") is not None
        if hasattr(self.params, "set_cy5_available"):
            self.params.set_cy5_available(has_cy5)
        self.params.set_context("detect", self.mode)

    # Recordings AND .cellscope project files are accepted by drag-drop.
    _DROP_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".tif", ".tiff")
    _DROP_PROJECT_EXTS = (".cellscope",)
    _DROP_MASKS_EXTS = (".npz",)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile().lower()
                if path.endswith(self._DROP_VIDEO_EXTS
                                 + self._DROP_PROJECT_EXTS
                                 + self._DROP_MASKS_EXTS):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            low = path.lower()
            if low.endswith(self._DROP_PROJECT_EXTS):
                from gui_focused.project_handlers import on_load_project
                on_load_project(self, path=path)
                return
            if low.endswith(self._DROP_VIDEO_EXTS):
                self._load_path(path)
                return
            if low.endswith(self._DROP_MASKS_EXTS):
                from gui_focused.project_handlers import (
                    on_load_pipeline_results)
                # Pass the file path directly — loader handles both
                # file (loads exactly that .npz) and folder (looks for
                # masks.npz inside). File-mode lets review/compare
                # workflows drop masks_original.npz etc.
                on_load_pipeline_results(self, path=path)
                return

    def _on_detect(self):
        if self.recording is None:
            return
        from gui_focused.workers import FocusedDetectWorker
        params = self.params.get_detect_params()
        # Apply ROI if active
        det_rec = dict(self.recording)
        if self.roi.active and self.roi.roi_mask is not None:
            det_rec["frames"] = self.roi.apply_to_frames(
                self.recording["frames"])
            self.logger.log("info", "ROI applied to detection")
        # Resolve modality
        modality = params.get("modality", "auto")
        if modality == "auto":
            from core.modality import detect_modality
            modality = detect_modality(det_rec["frames"])
            self.logger.log("info", f"Auto-detected modality: {modality}")
            self.statusBar().showMessage(
                f"Modality: {modality}", 5000)
        # Choose detect mode based on modality and single/multi.
        # For DIC multi-cell, default to the canonical "auto" pipeline
        # — same code path as run_pipeline_on_gt_recording.py (auto
        # downsample + DIC↔Cy5 alignment + auto cpsam_dic-vs-cpsam
        # selection + multi-metric Cy5 filter). User can override
        # back to the legacy explicit modes via params if needed.
        if modality == "dic":
            if self.mode == "multi" and not params.get(
                    "force_legacy_mode", False):
                detect_mode = "auto"
            else:
                detect_mode = ("hybrid_dic" if self.mode == "single"
                                else "hybrid_dic_multi")
        else:
            detect_mode = ("hybrid_cpsam" if self.mode == "single"
                            else "hybrid_cpsam_multi")
        self.logger.log("info",
                        f"Pipeline: {detect_mode} ({modality})")
        self._prev_detect_result = self.detect_result
        self._worker = FocusedDetectWorker(
            det_rec, detect_mode, params)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_event.connect(
            lambda k, m: self.logger.log(k, m))
        self._worker.finished.connect(self._on_detect_done)
        self._worker.error.connect(self._on_error)
        self.pipeline.set_stage_status("detect", "running")
        self.pipeline.enable_stage("detect", False)
        self.pipeline.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self._detect_t0 = time.time()
        self._worker.start()

    def _on_detect_done(self, result):
        self.detect_result = result
        # Record which pipeline function produced this result so
        # RUN_METADATA.md can describe it accurately on export.
        result.setdefault("pipeline_function", self.mode)
        masks = result.get("labels") if "labels" in result else result["masks"]
        self.viewer.update_masks(masks)
        # Push fusion source map to the viewer if available — enables
        # the "Source" colour toggle in the viewer control row.
        if hasattr(self.viewer, "set_source_stack"):
            self.viewer.set_source_stack(
                result.get("fusion_source_stack"))
        # Push dropped-cell labels (from Cy5 persistence_guard) to
        # the viewer — enables the "Dropped" overlay toggle so users
        # can see what the filter cut. Only meaningful when the Cy5
        # filter ran (multichannel recordings).
        if hasattr(self.viewer, "set_dropped_labels"):
            dropped_tracks = result.get("tracks_dropped") or []
            if dropped_tracks and result.get("labels") is not None:
                from core.cy5_filter import rebuild_label_stack
                self.viewer.set_dropped_labels(
                    rebuild_label_stack(
                        dropped_tracks, result["labels"].shape))
            else:
                self.viewer.set_dropped_labels(None)
        self.viewer.nav_bar.set_status(
            result["masks"], result.get("missed_frames"))
        elapsed = time.time() - getattr(self, "_detect_t0", time.time())
        self.pipeline.set_stage_status("detect", "done")
        self.pipeline.stages["detect"].setText(
            f"Detect \u2713 ({elapsed:.0f}s)")
        self.pipeline.btn_cancel.setEnabled(False)
        self.pipeline.enable_stage("edit", True)
        self.pipeline.enable_stage("analyze", True)
        self._dirty = True
        # Surface the Analysis tab now so users can configure VAMPIRE,
        # state classification, and per-metric toggles BEFORE clicking
        # Analyze. The tab is always visible, but auto-switching gives
        # a workflow hint.
        self.params.set_context("analyze", self.mode)
        self.progress_bar.setVisible(False)
        n_det = int(result["masks"].any(axis=(1, 2)).sum())
        n_total = len(result["masks"])
        self.status.showMessage(
            f"Detection: {n_det}/{n_total} frames, {elapsed:.1f}s | "
            f"{self.recording.get('name', '')} | "
            f"{self.recording.get('um_per_px', '?')} um/px")
        # Mode-vs-density mismatch warning (mirrors _on_test_frame).
        # Look at the max distinct cell count across all frames in
        # the labels stack — catches recordings the single-cell
        # pipeline collapsed but where Test on frame would have
        # warned.
        labels = result.get("labels")
        if (self.mode == "single" and labels is not None
                and labels.ndim == 3):
            max_cells = int(max(
                (np.unique(labels[i][labels[i] > 0]).size
                 for i in range(len(labels))), default=0))
            if max_cells >= 3:
                self.logger.log(
                    "warn",
                    f"Pipeline mode is 'Single Cell' but detection "
                    f"found up to {max_cells} distinct cells per "
                    f"frame. The single-cell pipeline likely "
                    f"under-detected — switch Pipeline mode to "
                    f"'Multi Cell (hybrid_cpsam_multi)' and re-run "
                    f"Detect for proper multi-cell results.")
        self._worker = None

    def _on_test_frame(self):
        """Run detection on the CURRENT displayed frame only, with
        all current GUI params. Time the run and extrapolate a total
        runtime estimate for the full recording. Skips multi-frame
        steps (tracking, gap fill, Cy5 multi-metric filter) since
        they're not meaningful on a single frame."""
        if self.recording is None:
            return
        import time as _time
        import numpy as _np
        idx = int(self.viewer.current_frame)
        frames = self.recording["frames"]
        n_total = len(frames)
        if idx < 0 or idx >= n_total:
            return
        # Slice to a 1-frame stack (detect_recording requires N≥1)
        one_frame = frames[idx:idx + 1]
        cy5 = self.recording.get("cy5_frames")
        one_cy5 = cy5[idx:idx + 1] if cy5 is not None else None
        # Apply ROI if active
        if self.roi.active and self.roi.roi_mask is not None:
            one_frame = self.roi.apply_to_frames(one_frame)
        params = self.params.get_detect_params()
        # Honor the modality selector AND the pipeline mode (single /
        # multi cell) so the test faithfully previews what Detect
        # will produce. Previously test passed pipeline_kind="auto"
        # for modality=Auto, which probes via raw cpsam (multi-cell
        # capable). Detect, in contrast, resolves modality via
        # detect_modality() first and — for single-cell + dic — locks
        # to the legacy single-cell hybrid_dic path. That divergence
        # let test report many cells while Detect produced garbage on
        # a multi-cell scene the user had configured as single-cell.
        modality = params.get("modality", "auto")
        if modality == "auto":
            from core.modality import detect_modality
            # Match _on_detect's resolution exactly (uses full stack,
            # not one frame, so the call is identical).
            modality = detect_modality(frames)
            self.logger.log(
                "info", f"Auto-detected modality: {modality}")
        if self.mode == "single":
            # Single-cell paths: cpsam_dic for DIC, raw cpsam for the
            # rest. Both branches in detect_recording match what
            # _on_detect's hybrid_{dic,cpsam} legacy paths produce
            # closely enough that the cell count is predictive.
            test_pipeline_kind = ("cpsam_dic" if modality == "dic"
                                   else "cpsam")
        else:
            # Multi-cell: let unified_detection's auto-probe choose
            # between cpsam_dic and raw cpsam based on scene density.
            # Same path _on_detect uses for multi mode.
            test_pipeline_kind = "auto"
        self.logger.log(
            "info",
            f"Test detect on frame {idx} (1/{n_total}); "
            f"mode={self.mode} modality={modality} "
            f"pipeline_kind={test_pipeline_kind}…")
        self.status.showMessage(
            f"Testing frame {idx}…", 0)
        # Time the run
        from core.unified_detection import detect_recording
        from core.pipeline_defaults import DEFAULTS as _PD
        t0 = _time.time()
        try:
            result = detect_recording(
                one_frame, cy5_frames=one_cy5,
                um_per_px=self.recording.get("um_per_px"),
                time_interval_min=self.recording.get(
                    "time_interval_min"),
                downsample=params.get("downsample", "auto"),
                # Single-frame: skip multi-frame stages
                align_channels=False,
                pipeline_kind=test_pipeline_kind,
                run_cy5_filter=False,
                use_gap_fill=False,
                # Tracking needs min_track_length=1 for a 1-frame run
                # (default 3 would drop the single-frame "track").
                min_track_length=1,
                use_mirror_pad=params.get("use_mirror_pad"),
                use_deepsea=params.get("use_deepsea"),
                use_tta=params.get("use_tta"),
                use_cpsam_cy5_union=params.get(
                    "use_cpsam_cy5_union"),
                use_fallback=params.get("use_fallback"),
                use_bfloat16=params.get("use_bfloat16"),
                progress_fn=None,
            )
        except Exception as e:
            self.status.showMessage(
                f"Test failed on frame {idx}: {e}", 8000)
            self.logger.log("error", f"Test detect error: {e}")
            return
        elapsed = _time.time() - t0
        labels = result.get("labels")
        if labels is None:
            labels = result["masks"]
        # Count cells in the frame
        if labels.ndim == 3:
            lab0 = labels[0]
        else:
            lab0 = labels
        n_cells = int(_np.unique(lab0[lab0 > 0]).size)
        # Extrapolate: detection-only (precise, linear in N) and an
        # estimated full-pipeline runtime that accounts for the
        # multi-frame post-processing stages (tracking + gap fill +
        # Cy5 fusion + filter). The post-processing multiplier scales
        # with cell density — denser scenes have more gap candidates
        # and more track-pair comparisons, both of which dominate the
        # gap-fill cascade.
        #   sparse (<5 cells/frame):    full ≈ 1.5 × detection
        #   medium (5-10 cells/frame):  full ≈ 2.0 × detection
        #   dense (≥10 cells/frame):    full ≈ 2.5 × detection
        # Calibrated against Pos10_WT (4 cells, canonical ~1.5×
        # extrapolated detection) and Pos68_DMSO (17 cells, canonical
        # ~2.3× extrapolated detection).
        if n_cells < 5:
            mult, density = 1.5, "sparse"
        elif n_cells < 10:
            mult, density = 2.0, "medium"
        else:
            mult, density = 2.5, "dense"
        est_detect_only = elapsed * n_total
        est_full = est_detect_only * mult

        def _fmt(secs):
            return (f"{secs:.0f}s" if secs < 60
                    else f"{secs / 60:.1f} min" if secs < 3600
                    else f"{secs / 3600:.1f} h")

        # Update viewer with the test labels (zero-pad to full stack
        # so the viewer's indexing matches; only frame `idx` carries
        # the test result, others are empty).
        test_stack = _np.zeros(
            (n_total, lab0.shape[0], lab0.shape[1]), dtype=lab0.dtype)
        test_stack[idx] = lab0
        self.viewer.update_masks(test_stack)
        # Report: short status-bar form + full info in log
        msg = (f"Frame {idx}: {n_cells} cell(s) in {elapsed:.2f}s "
               f"→ est. full run ({n_total} frames): "
               f"~{_fmt(est_full)} "
               f"(detect ~{_fmt(est_detect_only)} × {mult:.1f} "
               f"{density} post-proc)")
        self.status.showMessage(msg, 15000)
        self.logger.log("info", msg)
        # Mode-vs-density mismatch warning. Single-cell mode + many
        # cells means Detect will run a single-cell pipeline that
        # collapses or rejects the extras and produces garbage on a
        # multi-cell scene. Surface this loudly while the user is
        # still iterating.
        if self.mode == "single" and n_cells >= 3:
            self.logger.log(
                "warn",
                f"Pipeline mode is 'Single Cell' but {n_cells} cells "
                f"were detected on this frame. The single-cell "
                f"pipeline will likely produce poor results on this "
                f"recording — switch Pipeline mode to "
                f"'Multi Cell (hybrid_cpsam_multi)' for multi-cell "
                f"detection + tracking.")

    def _on_edit(self):
        if self.detect_result is None:
            return
        from gui.mask_editor import MaskEditor
        masks = self.detect_result.get("labels")
        if masks is None:
            masks = self.detect_result["masks"]
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        np.savez_compressed(tmp.name, masks=masks)
        tmp.close()
        self._editor = MaskEditor(
            video_path=self.recording.get("video_path"),
            mask_path=tmp.name)
        self._editor.masks_sent.connect(self._on_masks_received)
        self._editor.show()
        self.pipeline.set_stage_status("edit", "running")
        self.logger.log("info", "Mask editor opened — "
                        "click 'Send to GUI' to apply edits")

    def _on_masks_received(self, edited_masks):
        """Called when user clicks 'Send to GUI' in the mask editor."""
        edited = np.asarray(edited_masks)
        if "labels" in self.detect_result:
            self.detect_result["labels"] = edited.astype(np.int32)
            self.detect_result["masks"] = edited > 0
        else:
            self.detect_result["masks"] = edited > 0
        self.viewer.update_masks(
            self.detect_result.get("labels",
                                   self.detect_result["masks"]))
        self.pipeline.set_stage_status("edit", "done")
        n_cells = int(edited.max())
        n_frames = int((edited > 0).any(axis=(1, 2)).sum())
        self.logger.log("info",
                        f"Masks received: {n_frames} frames, "
                        f"{n_cells} cell IDs")
        self.status.showMessage("Edited masks applied")
        self._dirty = True

    def _on_analyze(self):
        if self.detect_result is None:
            return
        from gui_focused.workers import FocusedAnalyzeWorker
        scale = self.params.get_scale_overrides()
        vampire_params = self.params.get_vampire_params()
        state_params = self.params.get_state_params()
        division_params = self.params.get_division_params()
        self._worker = FocusedAnalyzeWorker(
            self.recording, self.detect_result, self.mode,
            scale_overrides=scale, vampire_params=vampire_params,
            state_params=state_params, division_params=division_params)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_event.connect(
            lambda k, m: self.logger.log(k, m))
        self._worker.finished.connect(self._on_analyze_done)
        self._worker.error.connect(self._on_error)
        self.pipeline.set_stage_status("analyze", "running")
        self.progress_bar.setVisible(True)
        self._worker.start()

    def _on_analyze_done(self, result):
        self.analysis_result = result
        if isinstance(result, list):
            self.analysis.set_multi_result(result)
        else:
            self.analysis.set_result(result, mode=self.mode)
        self.pipeline.set_stage_status("analyze", "done")
        self.pipeline.enable_stage("export", True)
        self.progress_bar.setVisible(False)
        self.status.showMessage("Analysis complete")
        self._dirty = True
        self._worker = None

    # --- Unsaved-results tracking ---

    def _mark_dirty(self):
        self._dirty = True

    def _mark_saved(self):
        self._dirty = False

    def _confirm_discard(self, action_desc):
        """Prompt the user when an action is about to discard unsaved
        results. Returns True if they confirmed (proceed); False if
        they cancelled."""
        if not self._dirty:
            return True
        from PyQt5.QtWidgets import QMessageBox
        choice = QMessageBox.question(
            self, "Unsaved results",
            f"You have unsaved detection / analysis results.\n\n"
            f"{action_desc}\n\n"
            f"Save them as a project first?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save)
        if choice == QMessageBox.Cancel:
            return False
        if choice == QMessageBox.Save:
            self._on_save_project()
            # If user cancelled the save dialog, _dirty is still True
            if self._dirty:
                return False
        return True

    def closeEvent(self, event):
        if not self._confirm_discard(
                "Closing CellScope will lose them."):
            event.ignore()
            return
        event.accept()

    def _on_export(self):
        from gui_focused.export_dialog import ExportDialog
        result = None
        multi = None
        if isinstance(self.analysis_result, list):
            multi = self.analysis_result
            result = multi[0] if multi else {}
        else:
            result = self.analysis_result
        dlg = ExportDialog(
            result=result,
            multi_results=multi,
            recording=self.recording,
            detect_result=self.detect_result,
            logger=self.logger,
            parent=self,
        )
        # Stash the params used for the last detection so RUN_METADATA
        # can record them. self.params is the live panel — values may
        # have been edited since the run, but capturing the live state
        # is still useful and matches what re-running would do.
        try:
            dlg.detect_params_used = self.params.get_detect_params()
        except Exception:
            dlg.detect_params_used = {}
        dlg.exec_()
        self.pipeline.set_stage_status("export", "done")
        # ExportDialog returns whether anything was actually written —
        # but for simplicity treat any close of the dialog as a save
        # checkpoint. Users who cancel will not re-trigger this since
        # they can just re-click Export.
        self._dirty = False

    def _on_scan_cells(self):
        from gui_focused.project_handlers import on_scan_cells
        on_scan_cells(self)

    def _on_roi_drawn(self):
        self.params.use_roi.setChecked(True)
        self.status.showMessage("ROI drawn and active")

    def _on_clear_roi(self):
        self.roi.clear()
        self.params.use_roi.setChecked(False)
        self.status.showMessage("ROI cleared")

    def _on_overlay_settings(self):
        """View → Overlay Options… handler. Edits the viewer's
        overlay settings dict in place and refreshes the canvas."""
        from gui.overlays import OverlaySettingsDialog
        if OverlaySettingsDialog.show(self, self.viewer.overlay_settings):
            self.viewer._redraw()
            self.status.showMessage("Overlay settings updated", 4000)

    def _on_roi_toggled(self, checked):
        if checked and not self.roi.has_roi():
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.information(
                    self, "ROI",
                    "No ROI drawn yet.\n\n"
                    "Draw one first via Edit > Select ROI\n"
                    "(Rectangle, Ellipse, or Polygon).")
            self.params.use_roi.setChecked(False)
            return
        self.roi.active = checked
        self.viewer._redraw()
        if checked:
            self.status.showMessage("ROI active — detection will be "
                                    "restricted to the ROI region")
        else:
            self.status.showMessage("ROI inactive")

    def _on_undo_detect(self):
        if self._prev_detect_result is None:
            self.status.showMessage("Nothing to undo")
            return
        self.detect_result = self._prev_detect_result
        self._prev_detect_result = None
        self.analysis_result = None
        self.analysis.clear()
        masks = self.detect_result.get("labels",
                                       self.detect_result["masks"])
        self.viewer.update_masks(masks)
        # Restore dropped overlay + source attribution for the
        # previous detection (if it had them).
        if hasattr(self.viewer, "set_dropped_labels"):
            prev_dropped = self.detect_result.get("tracks_dropped") or []
            if prev_dropped and self.detect_result.get("labels") is not None:
                from core.cy5_filter import rebuild_label_stack
                self.viewer.set_dropped_labels(
                    rebuild_label_stack(
                        prev_dropped,
                        self.detect_result["labels"].shape))
            else:
                self.viewer.set_dropped_labels(None)
        if hasattr(self.viewer, "set_source_stack"):
            self.viewer.set_source_stack(
                self.detect_result.get("fusion_source_stack"))
        self.viewer.nav_bar.set_status(
            self.detect_result["masks"],
            self.detect_result.get("missed_frames"))
        self.pipeline.set_stage_status("detect", "done")
        self.pipeline.set_stage_status("analyze", "idle")
        self.logger.log("info", "Detection undone — reverted to previous")
        self.status.showMessage("Detection undone")

    def _on_clear_all(self):
        if not self._confirm_discard(
                "Clear All will discard the current detection / "
                "analysis state."):
            return
        self.detect_result = None
        self._prev_detect_result = None
        self.analysis_result = None
        self.viewer.update_masks(None)
        if hasattr(self.viewer, "set_dropped_labels"):
            self.viewer.set_dropped_labels(None)
        if hasattr(self.viewer, "set_source_stack"):
            self.viewer.set_source_stack(None)
        self.viewer.nav_bar.clear()
        self.analysis.clear()
        self.roi.clear()
        self.params.use_roi.setChecked(False)
        self.pipeline.reset_all()
        if self.recording:
            self.pipeline.set_stage_status("load", "done")
            self.pipeline.enable_stage("detect", True)
        self.status.showMessage("All results cleared — ready to re-detect")
        self._dirty = False

    def _on_save_project(self):
        from gui_focused.project_handlers import on_save_project
        on_save_project(self)

    def _on_load_project(self):
        from gui_focused.project_handlers import on_load_project
        on_load_project(self)

    def _on_load_pipeline_results(self):
        from gui_focused.project_handlers import on_load_pipeline_results
        on_load_pipeline_results(self)

    def _on_load_masks_file(self):
        from gui_focused.project_handlers import on_load_pipeline_results
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Masks File", "",
            "Masks (*.npz);;All files (*)")
        if not path:
            return
        # Pass the file path so the loader uses THIS .npz, not the
        # canonical masks.npz in the same dir (needed for comparing
        # masks_original.npz / masks_unfiltered.npz).
        on_load_pipeline_results(self, path=path)

    def _on_save_screenshot(self):
        """Save a PNG of the full window (window contents incl.
        toolbars, parameter panel, log)."""
        self._save_screenshot_of(self, default_suffix="window")

    def _on_save_viewer_screenshot(self):
        """Save a PNG of just the image viewer (current frame with
        overlays — no controls). Re-renders to a higher-DPI offscreen
        canvas so the saved figure is publication-quality, not
        constrained to the on-screen pixel size."""
        rec_name = (self.recording or {}).get("name", "viewer") \
            if self.recording else "viewer"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Viewer Screenshot",
            f"{rec_name}_frame{self.viewer.current_frame}.png",
            "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        # Save the matplotlib figure directly — much higher quality
        # than grabbing the on-screen canvas.
        try:
            self.viewer.fig.savefig(
                path, dpi=200, bbox_inches="tight",
                facecolor=self.viewer.fig.get_facecolor())
        except Exception as e:
            QMessageBox.warning(self, "Screenshot Error", str(e))
            return
        self.logger.log("info", f"Saved viewer screenshot: {path}")
        self.status.showMessage(f"Saved: {os.path.basename(path)}",
                                 5000)

    def _on_share_image(self):
        """File → Export Shareable Image… — compact PNG/JPEG/GIF/MP4
        with selectable overlays, for easily sharing results."""
        if self.recording is None or self.viewer.frames is None:
            QMessageBox.information(
                self, "Export Shareable Image",
                "Load a recording first.")
            return
        from gui_focused.share_export import ShareImageDialog
        ShareImageDialog(self.viewer, self.recording,
                         self.detect_result, self).exec_()

    def _save_screenshot_of(self, widget, default_suffix="screenshot"):
        rec_name = (self.recording or {}).get("name", default_suffix) \
            if self.recording else default_suffix
        idx = self.viewer.current_frame if self.viewer.frames is not None \
            else 0
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot",
            f"{rec_name}_{default_suffix}_f{idx}.png",
            "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            pixmap = widget.grab()
            if not pixmap.save(path, "PNG"):
                raise RuntimeError("QPixmap.save returned False")
        except Exception as e:
            QMessageBox.warning(self, "Screenshot Error", str(e))
            return
        self.logger.log("info", f"Saved screenshot: {path}")
        self.status.showMessage(f"Saved: {os.path.basename(path)}",
                                 5000)

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
            self._worker = None
            self.pipeline.btn_cancel.setEnabled(False)
            self.pipeline.set_stage_status("detect", "idle")
            self.pipeline.enable_stage("detect", True)
            self.progress_bar.setVisible(False)
            self.status.showMessage("Detection cancelled")
            self.logger.log("warn", "Detection cancelled by user")

    def _on_progress(self, msg, pct):
        self.progress_bar.setValue(pct)
        self.status.showMessage(msg)

    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.progress_bar.setVisible(False)
        for key in ["detect", "analyze"]:
            if self.pipeline.stages[key]._status == "running":
                self.pipeline.set_stage_status(key, "error")
        self._worker = None

    # --- Settings / System / Help (delegated to gui_focused.dialogs) ---
    def use_gpu(self):
        return self.act_gpu.isChecked()

    def _show_system_info(self):
        show_system_info(self)

    def _show_recording_info(self):
        show_recording_info(self, self.recording, self.mode,
                            self.detect_result)

    def _show_shortcuts(self):
        show_shortcuts(self)

    def _show_about(self):
        show_about(self)
