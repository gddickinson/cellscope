"""Export small, shareable image files from the focused GUI.

The existing Export dialog writes a full-resolution overlay TIFF/MP4 +
data — great for archiving, large for sharing. This module produces
*compact* outputs for dropping into a slide / chat / email:

  - Current frame → PNG or JPEG (with quality control)
  - All frames    → MP4 (mp4v) or animated GIF
  - All frames    → a single Montage PNG/JPEG (evenly-sampled grid)

with each overlay independently switchable (mask fill, contours, cell
IDs, tracks, timestamp, scale bar) and a max-dimension downscale so
file sizes stay small.

The renderer reuses `gui.mask_editor_multicell.render_label_overlay`
(mask/contour/ID drawing) and `gui.overlays.draw_overlays`
(timestamp + scale bar baked into the RGB) so the look matches the
viewer exactly.
"""
import os

import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QFileDialog, QComboBox, QSpinBox, QSlider,
    QCheckBox, QRadioButton, QButtonGroup, QMessageBox, QProgressBar,
)
from PyQt5.QtCore import Qt

OVERLAY_KEYS = ("mask", "contour", "ids", "tracks", "timestamp", "scalebar")


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def render_share_frame(frame, labels_frame, opts, frame_idx,
                       um_per_px, dt_min, overlay_settings,
                       mask_opacity, centroids=None):
    """Bake the selected overlays onto a single grayscale frame → RGB."""
    import cv2
    from gui.mask_editor_multicell import render_label_overlay, cell_color

    has_lab = labels_frame is not None and labels_frame.any()
    want_cells = opts["mask"] or opts["contour"] or opts["ids"]
    if has_lab and want_cells and labels_frame.dtype != bool \
            and int(labels_frame.max()) > 1:
        rgb = render_label_overlay(
            frame, labels_frame.astype(np.int32),
            opacity=(mask_opacity if opts["mask"] else 0.0),
            show_ids=opts["ids"], draw_contours=opts["contour"])
    else:
        rgb = np.ascontiguousarray(
            np.stack([frame, frame, frame], axis=-1).astype(np.uint8))
        if has_lab and want_cells:
            m = labels_frame > 0
            if opts["mask"]:
                rgb[m] = (rgb[m] * (1 - mask_opacity)
                          + np.array([0, 255, 0]) * mask_opacity
                          ).astype(np.uint8)
            if opts["contour"]:
                cnts, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, cnts, -1, (0, 255, 0), 1)

    if opts["tracks"] and centroids is not None:
        for ci in range(centroids.shape[0]):
            trail = centroids[ci, :frame_idx + 1]
            valid = ~np.isnan(trail[:, 0])
            if valid.sum() >= 2:
                pts = np.stack([trail[valid, 1], trail[valid, 0]],
                               axis=1).astype(np.int32)
                cv2.polylines(rgb, [pts], False, cell_color(ci + 1), 1,
                              cv2.LINE_AA)

    if opts["timestamp"] or opts["scalebar"]:
        from gui.overlays import draw_overlays
        s = dict(overlay_settings or {})
        s["show_timestamp"] = bool(opts["timestamp"])
        s["show_scale_bar"] = bool(opts["scalebar"])
        draw_overlays(rgb, frame_idx, um_per_px, dt_min, s)
    return rgb


def compute_centroids(labels):
    """(n_cells, n_frames, 2) [y, x] centroids; NaN where a cell is
    absent. Used for the optional Tracks overlay."""
    if labels is None or labels.dtype == bool or labels.max() == 0:
        return None
    n_frames = len(labels)
    n_cells = int(labels.max())
    cents = np.full((n_cells, n_frames, 2), np.nan, dtype=np.float32)
    for fi in range(n_frames):
        lf = labels[fi]
        if lf.max() == 0:
            continue
        for lab in range(1, n_cells + 1):
            ys, xs = np.where(lf == lab)
            if ys.size:
                cents[lab - 1, fi] = (ys.mean(), xs.mean())
    return cents


# ----------------------------------------------------------------------
# Saving (downscale + compress)
# ----------------------------------------------------------------------
def _resize(rgb, max_px, even=False):
    import cv2
    h, w = rgb.shape[:2]
    m = max(h, w)
    if max_px and m > max_px:
        sc = max_px / float(m)
        w, h = int(round(w * sc)), int(round(h * sc))
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    if even:  # many MP4 codecs require even dimensions
        h2, w2 = (rgb.shape[0] // 2) * 2, (rgb.shape[1] // 2) * 2
        rgb = rgb[:h2, :w2]
    return rgb


def save_static(rgb, path, fmt, quality, max_px):
    import cv2
    rgb = _resize(rgb, max_px)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if fmt == "JPEG":
        cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    else:
        cv2.imwrite(path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def save_mp4(frames_rgb, path, fps, max_px):
    import cv2
    first = _resize(frames_rgb[0], max_px, even=True)
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                             max(1, fps), (w, h))
    for f in frames_rgb:
        fr = _resize(f, max_px, even=True)
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h))
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


def save_gif(frames_rgb, path, fps, max_px):
    from PIL import Image
    imgs = [Image.fromarray(_resize(f, max_px)) for f in frames_rgb]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(1, fps)), loop=0, optimize=True)


def make_montage(frames_rgb, n_cols, max_frames, max_px):
    """Evenly sample up to max_frames frames into an n_cols grid PNG."""
    import cv2
    n = len(frames_rgb)
    k = min(n, max_frames)
    idxs = np.linspace(0, n - 1, k).round().astype(int)
    tiles = [frames_rgb[i] for i in idxs]
    th, tw = tiles[0].shape[:2]
    n_cols = max(1, n_cols)
    n_rows = int(np.ceil(k / n_cols))
    grid = np.zeros((n_rows * th, n_cols * tw, 3), dtype=np.uint8)
    for j, t in enumerate(tiles):
        if t.shape[:2] != (th, tw):
            t = cv2.resize(t, (tw, th))
        r, c = divmod(j, n_cols)
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return _resize(grid, max_px)


# ----------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------
class ShareImageDialog(QDialog):
    """Configure + export a compact shareable image / clip."""

    def __init__(self, viewer, recording, detect_result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Shareable Image")
        self.setMinimumWidth(440)
        self.viewer = viewer
        self.frames = viewer.frames
        self.labels = viewer.masks
        self.um_per_px = viewer.um_per_px
        self.dt_min = viewer.dt_min
        self.overlay_settings = viewer.overlay_settings
        self.mask_opacity = viewer.mask_opacity
        self.cur_frame = viewer.current_frame
        rec_name = (recording or {}).get("name", "recording")
        self._stem = str(rec_name).replace(" ", "_")
        self._build_ui()

    # --- UI ---
    def _build_ui(self):
        root = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Output:"))
        self.path_edit = QLineEdit(os.path.join(
            "results", f"{self._stem}_frame{self.cur_frame}.png"))
        row.addWidget(self.path_edit, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(self._browse)
        row.addWidget(btn)
        root.addLayout(row)

        scope_box = QGroupBox("What to export")
        sf = QFormLayout(scope_box)
        self.scope_group = QButtonGroup(self)
        self.rb_current = QRadioButton("Current frame")
        self.rb_all = QRadioButton("All frames")
        self.rb_current.setChecked(True)
        self.scope_group.addButton(self.rb_current)
        self.scope_group.addButton(self.rb_all)
        srow = QHBoxLayout()
        srow.addWidget(self.rb_current)
        srow.addWidget(self.rb_all)
        srow.addStretch()
        sf.addRow(srow)
        self.fmt = QComboBox()
        sf.addRow("Format:", self.fmt)
        self.rb_current.toggled.connect(self._refresh_formats)
        self.fmt.currentTextChanged.connect(self._sync_enable)
        root.addWidget(scope_box)

        opt_box = QGroupBox("Size + quality")
        of = QFormLayout(opt_box)
        self.max_px = QSpinBox()
        self.max_px.setRange(0, 8000)
        self.max_px.setSingleStep(128)
        self.max_px.setValue(1024)
        self.max_px.setSpecialValueText("original")
        self.max_px.setSuffix(" px")
        self.max_px.setToolTip("Downscale so the longest side ≤ this "
                               "(0 = keep original). Smaller = smaller file.")
        of.addRow("Max size:", self.max_px)
        qrow = QHBoxLayout()
        self.quality = QSlider(Qt.Horizontal)
        self.quality.setRange(10, 100)
        self.quality.setValue(85)
        self.q_lbl = QLabel("85")
        self.quality.valueChanged.connect(
            lambda v: self.q_lbl.setText(str(v)))
        qrow.addWidget(self.quality)
        qrow.addWidget(self.q_lbl)
        of.addRow("JPEG quality:", qrow)
        self.fps = QSpinBox()
        self.fps.setRange(1, 30)
        self.fps.setValue(8)
        of.addRow("Animation fps:", self.fps)
        self.cols = QSpinBox()
        self.cols.setRange(1, 12)
        self.cols.setValue(5)
        of.addRow("Montage columns:", self.cols)
        self.montage_n = QSpinBox()
        self.montage_n.setRange(2, 60)
        self.montage_n.setValue(15)
        of.addRow("Montage frames:", self.montage_n)
        root.addWidget(opt_box)

        ov_box = QGroupBox("Overlays")
        ol = QVBoxLayout(ov_box)
        v = self.viewer
        defaults = {
            "mask": getattr(v, "show_mask", True),
            "contour": getattr(v, "show_contour", True),
            "ids": getattr(v, "show_ids", False),
            "tracks": getattr(v, "show_tracks", False),
            "timestamp": bool((self.overlay_settings or {}).get(
                "show_timestamp")),
            "scalebar": bool((self.overlay_settings or {}).get(
                "show_scale_bar")),
        }
        labels = {"mask": "Mask fill", "contour": "Contours",
                  "ids": "Cell IDs", "tracks": "Tracks (trails)",
                  "timestamp": "Timestamp", "scalebar": "Scale bar"}
        self.ov_checks = {}
        for k in OVERLAY_KEYS:
            cb = QCheckBox(labels[k])
            cb.setChecked(bool(defaults[k]))
            if k == "timestamp" and not self.dt_min:
                cb.setChecked(False)
                cb.setEnabled(False)
                cb.setToolTip("Needs a frame interval (time) — unknown "
                              "for this recording.")
            if k == "scalebar" and not self.um_per_px:
                cb.setChecked(False)
                cb.setEnabled(False)
                cb.setToolTip("Needs a pixel size (µm/px) — unknown for "
                              "this recording.")
            ol.addWidget(cb)
            self.ov_checks[k] = cb
        root.addWidget(ov_box)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        brow = QHBoxLayout()
        brow.addStretch()
        ok = QPushButton("Export")
        ok.setDefault(True)
        ok.clicked.connect(self._export)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        brow.addWidget(ok)
        brow.addWidget(cancel)
        root.addLayout(brow)

        self._refresh_formats()

    def _refresh_formats(self):
        self.fmt.blockSignals(True)
        self.fmt.clear()
        if self.rb_current.isChecked():
            self.fmt.addItems(["PNG", "JPEG"])
        else:
            self.fmt.addItems(["MP4", "Animated GIF", "Montage PNG",
                               "Montage JPEG"])
        self.fmt.blockSignals(False)
        self._sync_path_ext()
        self._sync_enable()

    def _sync_enable(self, *_):
        f = self.fmt.currentText()
        self.quality.setEnabled("JPEG" in f)
        self.fps.setEnabled(f in ("MP4", "Animated GIF"))
        is_montage = f.startswith("Montage")
        self.cols.setEnabled(is_montage)
        self.montage_n.setEnabled(is_montage)
        self._sync_path_ext()

    def _ext(self):
        return {"PNG": ".png", "JPEG": ".jpg", "MP4": ".mp4",
                "Animated GIF": ".gif", "Montage PNG": ".png",
                "Montage JPEG": ".jpg"}.get(self.fmt.currentText(), ".png")

    def _sync_path_ext(self):
        p = self.path_edit.text()
        base = os.path.splitext(p)[0] if p else os.path.join(
            "results", self._stem)
        if self.rb_all.isChecked() and "_frame" in base:
            base = base.split("_frame")[0]
        elif self.rb_current.isChecked() and "_frame" not in base:
            base = f"{base}_frame{self.cur_frame}"
        self.path_edit.setText(base + self._ext())

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Save shareable file", self.path_edit.text(),
            "All files (*)")
        if p:
            self.path_edit.setText(p)

    # --- Export ---
    def _opts(self):
        return {k: self.ov_checks[k].isChecked() for k in OVERLAY_KEYS}

    def _export(self):
        if self.frames is None:
            QMessageBox.warning(self, "Export", "No image loaded.")
            return
        path = self.path_edit.text().strip()
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        opts = self._opts()
        fmt = self.fmt.currentText()
        max_px = self.max_px.value()
        cents = (compute_centroids(self.labels)
                 if opts["tracks"] else None)

        def render(i):
            lab = (self.labels[i] if self.labels is not None
                   and i < len(self.labels) else None)
            return render_share_frame(
                self.frames[i], lab, opts, i, self.um_per_px,
                self.dt_min, self.overlay_settings, self.mask_opacity,
                cents)

        try:
            if self.rb_current.isChecked():
                save_static(render(self.cur_frame), path, fmt,
                            self.quality.value(), max_px)
            else:
                self.progress.setVisible(True)
                self.progress.setMaximum(len(self.frames))
                frames_rgb = []
                for i in range(len(self.frames)):
                    frames_rgb.append(render(i))
                    self.progress.setValue(i + 1)
                    if i % 5 == 0:
                        from PyQt5.QtWidgets import QApplication
                        QApplication.processEvents()
                if fmt == "MP4":
                    save_mp4(frames_rgb, path, self.fps.value(), max_px)
                elif fmt == "Animated GIF":
                    save_gif(frames_rgb, path, self.fps.value(), max_px)
                else:  # Montage PNG / JPEG
                    grid = make_montage(frames_rgb, self.cols.value(),
                                        self.montage_n.value(), max_px)
                    save_static(grid, path,
                                "JPEG" if "JPEG" in fmt else "PNG",
                                self.quality.value(), 0)
        except Exception as e:
            self.progress.setVisible(False)
            QMessageBox.critical(self, "Export failed", str(e))
            return
        size_kb = (os.path.getsize(path) / 1024.0
                   if os.path.exists(path) else 0)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QMessageBox.information(
                self, "Exported",
                f"Saved {os.path.basename(path)}  ({size_kb:.0f} KB)")
        self.saved_path = path
        self.accept()
