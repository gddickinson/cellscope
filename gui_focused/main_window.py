"""Focused pipeline GUI main window."""
import os
import logging
import numpy as np
import tempfile

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QStatusBar, QProgressBar, QMenuBar,
    QAction, QLabel,
)

from gui.run_log import RunLogger
from gui_focused.image_viewer import ImageViewer
from gui_focused.pipeline_panel import PipelinePanel
from gui_focused.params_panel import ParamsPanel
from gui_focused.analysis_view import AnalysisView
from gui_focused.dialogs import (
    detect_gpu, show_system_info, show_recording_info,
    show_shortcuts, show_about, open_doc,
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
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        self.viewer = ImageViewer()
        splitter.addWidget(self.viewer)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self.pipeline = PipelinePanel()
        rl.addWidget(self.pipeline)

        self.params = ParamsPanel()
        rl.addWidget(self.params)

        self.analysis = AnalysisView(logger=self.logger)
        rl.addWidget(self.analysis, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status.addPermanentWidget(self.progress_bar)
        self.status.showMessage("Ready - load a recording to begin")

    def _build_menu(self):
        mb = self.menuBar()

        # --- File ---
        file_menu = mb.addMenu("File")
        act_open = QAction("Open Recording...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_load)
        file_menu.addAction(act_open)
        file_menu.addSeparator()
        act_export = QAction("Export Results...", self)
        act_export.setShortcut("Ctrl+S")
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
            from core.io import load_recording
            self.recording = load_recording(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return
        n = len(self.recording["frames"])
        name = self.recording.get("name", os.path.basename(path))
        self.logger.log("info", f"Loaded {name}: {n} frames")
        self.viewer.set_data(self.recording["frames"])
        self.detect_result = None
        self.analysis_result = None
        self.analysis.clear()
        self.pipeline.reset_all()
        self.pipeline.set_stage_status("load", "done")
        self.pipeline.enable_stage("detect", True)
        self.status.showMessage(f"Loaded: {name} ({n} frames)")
        self.params.set_from_recording(self.recording)
        self.params.set_context("detect", self.mode)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile().lower()
                if path.endswith((".mp4", ".avi", ".mov",
                                  ".tif", ".tiff")):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".mp4", ".avi", ".mov",
                                      ".tif", ".tiff")):
                self._load_path(path)
                return

    def _on_detect(self):
        if self.recording is None:
            return
        from gui_focused.workers import FocusedDetectWorker
        detect_mode = ("hybrid_cpsam" if self.mode == "single"
                        else "hybrid_cpsam_multi")
        params = self.params.get_detect_params()
        # Apply ROI if active
        det_rec = dict(self.recording)
        if self.roi.active and self.roi.roi_mask is not None:
            det_rec["frames"] = self.roi.apply_to_frames(
                self.recording["frames"])
            self.logger.log("info", "ROI applied to detection")
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
        self.progress_bar.setVisible(True)
        self._worker.start()

    def _on_detect_done(self, result):
        self.detect_result = result
        masks = result.get("labels") if "labels" in result else result["masks"]
        self.viewer.update_masks(masks)
        self.viewer.nav_bar.set_status(
            result["masks"], result.get("missed_frames"))
        self.pipeline.set_stage_status("detect", "done")
        self.pipeline.enable_stage("edit", True)
        self.pipeline.enable_stage("analyze", True)
        self.progress_bar.setVisible(False)
        elapsed = result.get("elapsed", 0)
        self.status.showMessage(f"Detection done in {elapsed:.1f}s")
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
        self._worker = FocusedAnalyzeWorker(
            self.recording, self.detect_result, self.mode,
            scale_overrides=scale)
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
        dlg.exec_()
        self.pipeline.set_stage_status("export", "done")

    def _on_scan_cells(self):
        if self.recording is None:
            QMessageBox.information(self, "Scan", "Load a recording first.")
            return
        try:
            from core.hybrid_cpsam_multi import scan_cell_count
            from PyQt5.QtWidgets import QApplication
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.status.showMessage("Scanning for cell count...")
            QApplication.processEvents()
            count = scan_cell_count(
                self.recording["frames"], n_sample=5,
                min_area_px=self.params.min_area.value())
            self.params.expected_cells.setValue(max(1, count))
            self.logger.log("info", f"Scan: {count} cells detected")
            self.status.showMessage(f"Scan complete: {count} cells/frame")
        except Exception as e:
            QMessageBox.warning(self, "Scan Error", str(e))
        finally:
            from PyQt5.QtWidgets import QApplication
            QApplication.restoreOverrideCursor()

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
