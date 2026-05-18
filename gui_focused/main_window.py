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
        file_menu.addSeparator()
        act_export = QAction("Export Results...", self)
        act_export.setShortcut("Ctrl+Shift+S")
        act_export.triggered.connect(self._on_export)
        file_menu.addAction(act_export)
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
        self.params.btn_scan.clicked.connect(self._on_scan_cells)
        self.params.use_roi.toggled.connect(self._on_roi_toggled)

    def _on_mode_changed(self, mode):
        self.mode = mode
        self.params.set_context("detect", mode)

    def _on_load(self):
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
            fluo_frames=self.recording.get("cy5_frames"))
        self.detect_result = None
        self.analysis_result = None
        self.analysis.clear()
        self.pipeline.reset_all()
        self.pipeline.set_stage_status("load", "done")
        self.pipeline.enable_stage("detect", True)
        self.status.showMessage(f"Loaded: {name} ({n} frames){cy5_note}")
        self.params.set_from_recording(self.recording)
        # Enable Cy5 recovery toggle if recording has fluo channel
        has_cy5 = self.recording.get("cy5_frames") is not None
        if hasattr(self.params, "set_cy5_available"):
            self.params.set_cy5_available(has_cy5)
        self.params.set_context("detect", self.mode)

    # Recordings AND .cellscope project files are accepted by drag-drop.
    _DROP_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".tif", ".tiff")
    _DROP_PROJECT_EXTS = (".cellscope",)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile().lower()
                if path.endswith(self._DROP_VIDEO_EXTS
                                 + self._DROP_PROJECT_EXTS):
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
        self.viewer.nav_bar.set_status(
            result["masks"], result.get("missed_frames"))
        elapsed = time.time() - getattr(self, "_detect_t0", time.time())
        self.pipeline.set_stage_status("detect", "done")
        self.pipeline.stages["detect"].setText(
            f"Detect \u2713 ({elapsed:.0f}s)")
        self.pipeline.btn_cancel.setEnabled(False)
        self.pipeline.enable_stage("edit", True)
        self.pipeline.enable_stage("analyze", True)
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
        self._worker = None

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

    def _on_analyze(self):
        if self.detect_result is None:
            return
        from gui_focused.workers import FocusedAnalyzeWorker
        scale = self.params.get_scale_overrides()
        vampire_params = self.params.get_vampire_params()
        state_params = self.params.get_state_params()
        self._worker = FocusedAnalyzeWorker(
            self.recording, self.detect_result, self.mode,
            scale_overrides=scale, vampire_params=vampire_params,
            state_params=state_params)
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
        self._worker = None

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
        self.viewer.nav_bar.set_status(
            self.detect_result["masks"],
            self.detect_result.get("missed_frames"))
        self.pipeline.set_stage_status("detect", "done")
        self.pipeline.set_stage_status("analyze", "idle")
        self.logger.log("info", "Detection undone — reverted to previous")
        self.status.showMessage("Detection undone")

    def _on_clear_all(self):
        self.detect_result = None
        self._prev_detect_result = None
        self.analysis_result = None
        self.viewer.update_masks(None)
        self.viewer.nav_bar.clear()
        self.analysis.clear()
        self.roi.clear()
        self.params.use_roi.setChecked(False)
        self.pipeline.reset_all()
        if self.recording:
            self.pipeline.set_stage_status("load", "done")
            self.pipeline.enable_stage("detect", True)
        self.status.showMessage("All results cleared — ready to re-detect")

    def _on_save_project(self):
        from gui_focused.project_handlers import on_save_project
        on_save_project(self)

    def _on_load_project(self):
        from gui_focused.project_handlers import on_load_project
        on_load_project(self)

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
