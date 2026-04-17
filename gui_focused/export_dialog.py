"""Export dialog for saving analysis results."""
import os
import json
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QGroupBox, QLineEdit, QFileDialog, QSpinBox,
    QRadioButton, QButtonGroup, QMessageBox, QProgressBar,
)
from matplotlib.figure import Figure

from gui_focused.analysis_plots import GRAPH_REGISTRY


class ExportDialog(QDialog):
    """Modal dialog for configuring and executing export."""

    def __init__(self, result, multi_results, recording, detect_result,
                 logger=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setMinimumWidth(450)
        self.result = result
        self.multi_results = multi_results
        self.recording = recording
        self.detect_result = detect_result
        self.logger = logger
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Output directory
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output:"))
        self.dir_edit = QLineEdit()
        name = self.recording.get("name", "recording").replace(" ", "_")
        self.dir_edit.setText(os.path.join("results", name))
        dir_row.addWidget(self.dir_edit, stretch=1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._on_browse)
        dir_row.addWidget(btn_browse)
        layout.addLayout(dir_row)

        # Data group
        data_grp = QGroupBox("Data")
        dl = QVBoxLayout(data_grp)
        self.chk_masks = QCheckBox("Masks (.npz)")
        self.chk_masks.setChecked(True)
        dl.addWidget(self.chk_masks)
        self.chk_metrics = QCheckBox("Metrics (.json)")
        self.chk_metrics.setChecked(True)
        dl.addWidget(self.chk_metrics)
        self.chk_log = QCheckBox("Run log (.md + .json)")
        self.chk_log.setChecked(True)
        dl.addWidget(self.chk_log)
        layout.addWidget(data_grp)

        # Figures group
        fig_grp = QGroupBox("Figures")
        fl = QVBoxLayout(fig_grp)
        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(lambda: self._set_all_plots(True))
        btn_row.addWidget(btn_all)
        btn_none = QPushButton("Deselect All")
        btn_none.clicked.connect(lambda: self._set_all_plots(False))
        btn_row.addWidget(btn_none)
        btn_row.addStretch()
        fl.addLayout(btn_row)
        self.plot_checks = {}
        is_multi = self.multi_results is not None
        for name, (fn, requires_multi) in GRAPH_REGISTRY.items():
            if requires_multi and not is_multi:
                continue
            chk = QCheckBox(name)
            chk.setChecked(True)
            fl.addWidget(chk)
            self.plot_checks[name] = chk
        layout.addWidget(fig_grp)

        # Overlay group
        ovr_grp = QGroupBox("Overlays")
        ol = QVBoxLayout(ovr_grp)
        self.chk_overlay_tif = QCheckBox("Overlay TIFF stack (contours on frames)")
        self.chk_overlay_tif.setChecked(False)
        ol.addWidget(self.chk_overlay_tif)
        layout.addWidget(ovr_grp)

        # Format options
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Plot format:"))
        self.fmt_group = QButtonGroup(self)
        for fmt in ["PNG", "SVG", "PDF"]:
            rb = QRadioButton(fmt)
            self.fmt_group.addButton(rb)
            fmt_row.addWidget(rb)
            if fmt == "PNG":
                rb.setChecked(True)
        fmt_row.addWidget(QLabel("  DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 300)
        self.dpi_spin.setValue(150)
        fmt_row.addWidget(self.dpi_spin)
        fmt_row.addStretch()
        layout.addLayout(fmt_row)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Buttons
        btn_row2 = QHBoxLayout()
        btn_row2.addStretch()
        btn_export = QPushButton("Export")
        btn_export.clicked.connect(self._on_export)
        btn_row2.addWidget(btn_export)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row2.addWidget(btn_cancel)
        layout.addLayout(btn_row2)

    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if path:
            self.dir_edit.setText(path)

    def _set_all_plots(self, checked):
        for chk in self.plot_checks.values():
            chk.setChecked(checked)

    def _on_export(self):
        out_dir = self.dir_edit.text()
        os.makedirs(out_dir, exist_ok=True)
        self.progress.setVisible(True)
        steps = 0
        total = sum([self.chk_masks.isChecked(),
                     self.chk_metrics.isChecked(),
                     self.chk_log.isChecked(),
                     self.chk_overlay_tif.isChecked(),
                     sum(c.isChecked() for c in self.plot_checks.values())])
        self.progress.setMaximum(max(total, 1))

        try:
            if self.chk_masks.isChecked():
                masks = self.detect_result.get("masks", np.array([]))
                labels = self.detect_result.get("labels")
                save_dict = {"masks": masks}
                if labels is not None:
                    save_dict["labels"] = labels
                np.savez_compressed(os.path.join(out_dir, "masks.npz"),
                                    **save_dict)
                steps += 1; self.progress.setValue(steps)

            if self.chk_metrics.isChecked():
                r = self.result or {}
                with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                    json.dump(self._build_metrics(r), f, indent=2,
                              default=str)
                if self.multi_results:
                    for mr in self.multi_results:
                        cid = mr.get("cell_id", 0)
                        with open(os.path.join(out_dir,
                                               f"metrics_cell{cid}.json"), "w") as f:
                            json.dump(self._build_metrics(mr), f, indent=2,
                                      default=str)
                steps += 1; self.progress.setValue(steps)

            if self.chk_log.isChecked() and self.logger:
                self.logger.save(out_dir)
                steps += 1; self.progress.setValue(steps)

            fmt = self.fmt_group.checkedButton().text().lower()
            dpi = self.dpi_spin.value()
            is_multi = self.multi_results is not None
            for name, chk in self.plot_checks.items():
                if not chk.isChecked():
                    continue
                try:
                    fn, requires_multi = GRAPH_REGISTRY[name]
                    fig = Figure(figsize=(8, 5), dpi=dpi)
                    if requires_multi and is_multi:
                        fn(fig, self.multi_results)
                    else:
                        r = self.result or {}
                        fn(fig, r)
                    safe_name = name.lower().replace(" ", "_").replace(
                        "(", "").replace(")", "")
                    fig.savefig(os.path.join(out_dir, f"{safe_name}.{fmt}"),
                                dpi=dpi, bbox_inches="tight")
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Plot '%s' failed: %s", name, e)
                steps += 1; self.progress.setValue(steps)

            if self.chk_overlay_tif.isChecked():
                self._save_overlay_tif(out_dir)
                steps += 1; self.progress.setValue(steps)

            self.export_count = steps
            self.export_dir = out_dir
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.information(self, "Export Complete",
                                        f"Saved {steps} items to:\n{out_dir}")
            self.accept()

        except Exception as e:
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.critical(self, "Export Error", str(e))
            else:
                raise

    def _build_metrics(self, r):
        out = {}
        for k in ["name", "n_frames", "um_per_px", "time_interval_min",
                   "mean_speed", "total_distance", "net_displacement",
                   "persistence", "mean_boundary_confidence", "cell_id"]:
            if k in r:
                v = r[k]
                out[k] = float(v) if isinstance(v, (np.floating,)) else v
        for k in ["shape_summary", "edge_summary", "area_stability",
                   "track_info"]:
            if k in r:
                out[k] = r[k]
        return out

    def _save_overlay_tif(self, out_dir):
        import cv2
        import tifffile
        frames = self.recording["frames"]
        masks = self.detect_result.get("masks")
        labels = self.detect_result.get("labels")
        n, H, W = frames.shape
        overlay = np.zeros((n, H, W, 3), dtype=np.uint8)
        for i in range(n):
            rgb = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2RGB)
            if labels is not None and labels[i].max() > 1:
                from gui.mask_editor_multicell import render_label_overlay
                rgb = render_label_overlay(frames[i], labels[i], opacity=0.3)
            elif masks is not None and masks[i].any():
                m = masks[i] > 0
                contours, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)
            overlay[i] = rgb
        tifffile.imwrite(os.path.join(out_dir, "overlay.tif"),
                         overlay, photometric="rgb", compression="zlib")
