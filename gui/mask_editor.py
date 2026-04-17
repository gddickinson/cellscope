"""Manual mask editor: draw/correct cell boundaries on DIC time-lapse.

Launch as:
    python -m gui.mask_editor data/examples/control/control.mp4
    python -m gui.mask_editor data/examples/control/control.mp4 --masks saved_masks.npz

Features:
  - Load .mp4 with optional existing mask stack (.npz or PNG folder)
  - Frame slider + arrow-key navigation
  - Brush tool (paint cell) with variable size
  - Eraser (remove cell pixels)
  - Polygon tool (click vertices, close to fill)
  - Fill tool (flood fill from a point)
  - Undo/redo per frame
  - Opacity slider for mask overlay
  - Save corrected masks as cellpose-compatible PNG stack or NPZ

Output goes to data/manual_gt/<recording_name>/ by default.
"""
import os
import sys
import argparse
import json
import numpy as np
import cv2
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QButtonGroup,
    QRadioButton, QSpinBox, QMessageBox, QShortcut, QStatusBar,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.io import load_video, load_metadata
from gui.mask_editor_multicell import (
    render_label_overlay, cell_color, bool_to_labels, next_cell_id,
)


class MaskCanvas(QGraphicsView):
    """Interactive canvas showing frame + mask overlay with drawing tools."""

    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.scene_obj = QGraphicsScene()
        self.setScene(self.scene_obj)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene_obj.addItem(self.pixmap_item)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self.drawing = False
        self.polygon_points = []
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def update_pixmap(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.setSceneRect(self.pixmap_item.boundingRect())

    def _scene_pos(self, event):
        sp = self.mapToScene(event.pos())
        return int(round(sp.x())), int(round(sp.y()))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x, y = self._scene_pos(event)
            tool = self.editor.current_tool()
            if tool in ("brush", "eraser"):
                self.drawing = True
                self.editor.begin_stroke()
                self.editor.paint_at(x, y, tool)
            elif tool == "polygon":
                self.polygon_points.append((x, y))
                self.editor.draw_polygon_preview(self.polygon_points)
            elif tool == "fill":
                self.editor.flood_fill(x, y)
        elif event.button() == Qt.RightButton:
            tool = self.editor.current_tool()
            if tool == "polygon" and len(self.polygon_points) >= 3:
                self.editor.commit_polygon(self.polygon_points)
                self.polygon_points = []
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        x, y = self._scene_pos(event)
        self.editor.show_coords(x, y)
        if self.drawing:
            tool = self.editor.current_tool()
            self.editor.paint_at(x, y, tool)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            self.editor.end_stroke()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.editor.current_tool() == "polygon" and len(self.polygon_points) >= 3:
            self.editor.commit_polygon(self.polygon_points)
            self.polygon_points = []

    def wheelEvent(self, event):
        # Zoom with mouse wheel
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)


class MaskEditor(QMainWindow):
    masks_sent = pyqtSignal(object)

    def __init__(self, video_path=None, mask_path=None):
        super().__init__()
        self.setWindowTitle("Mask Editor — CellScope")
        self.resize(1200, 900)

        self.frames = None          # (N, H, W) uint8
        self.masks = None           # (N, H, W) int32 — 0=bg, 1,2,...=cell IDs
        self.name = "recording"
        self.current_frame = 0
        self.brush_size = 10
        self.mask_opacity = 0.4
        self.tool = "brush"         # brush / eraser / polygon / fill / none
        self.active_cell = 1        # which cell ID to paint with

        # Undo stacks: per-frame list of mask snapshots (deque-like)
        self.undo_stacks = {}   # frame_idx -> list of mask snapshots
        self.redo_stacks = {}
        self.max_undo = 30
        self._stroke_snapshot = None

        self._build_ui()

        if video_path:
            self.load_video(video_path, mask_path)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top toolbar
        tb = QHBoxLayout()
        btn_load = QPushButton("Load Video")
        btn_load.clicked.connect(self._on_load)
        tb.addWidget(btn_load)

        btn_load_masks = QPushButton("Load Mask File (.npz)")
        btn_load_masks.clicked.connect(self._on_load_masks)
        tb.addWidget(btn_load_masks)

        btn_load_folder = QPushButton("Load Mask Folder")
        btn_load_folder.clicked.connect(self._on_load_mask_folder)
        tb.addWidget(btn_load_folder)

        btn_detect = QPushButton("Init from Cellpose+Flow")
        btn_detect.clicked.connect(self._on_detect)
        tb.addWidget(btn_detect)

        btn_save = QPushButton("Save Masks")
        btn_save.clicked.connect(self._on_save)
        tb.addWidget(btn_save)

        btn_send = QPushButton("Send to GUI")
        btn_send.setToolTip(
            "Push current masks back to the main GUI window.\n"
            "The analysis will use these edited masks.")
        btn_send.clicked.connect(self._on_send_to_gui)
        tb.addWidget(btn_send)

        tb.addStretch()
        layout.addLayout(tb)

        # Tool palette
        tools = QHBoxLayout()
        tools.addWidget(QLabel("Tool:"))
        self.tool_group = QButtonGroup(self)
        for name in ["brush", "eraser", "polygon", "fill"]:
            rb = QRadioButton(name)
            rb.toggled.connect(lambda checked, n=name: self._on_tool(n, checked))
            tools.addWidget(rb)
            self.tool_group.addButton(rb)
            if name == "brush":
                rb.setChecked(True)
        tools.addWidget(QLabel("  Brush size:"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 100)
        self.brush_spin.setValue(self.brush_size)
        self.brush_spin.valueChanged.connect(self._on_brush_size)
        tools.addWidget(self.brush_spin)
        tools.addWidget(QLabel("  Overlay α:"))
        self.op_slider = QSlider(Qt.Horizontal)
        self.op_slider.setRange(0, 100)
        self.op_slider.setValue(int(self.mask_opacity * 100))
        self.op_slider.valueChanged.connect(self._on_opacity)
        self.op_slider.setMaximumWidth(150)
        tools.addWidget(self.op_slider)
        btn_undo = QPushButton("Undo")
        btn_undo.clicked.connect(self.undo)
        tools.addWidget(btn_undo)
        btn_redo = QPushButton("Redo")
        btn_redo.clicked.connect(self.redo)
        tools.addWidget(btn_redo)
        btn_clear = QPushButton("Clear Frame")
        btn_clear.clicked.connect(self.clear_current)
        tools.addWidget(btn_clear)
        tools.addWidget(QLabel("  Cell:"))
        self.cell_spin = QSpinBox()
        self.cell_spin.setRange(1, 9)
        self.cell_spin.setValue(1)
        self.cell_spin.setToolTip("Active cell ID (1-9). Paint with this ID.")
        self.cell_spin.valueChanged.connect(self._on_cell_id)
        tools.addWidget(self.cell_spin)
        self.cell_color_label = QLabel("  ●")
        self._update_cell_color_label()
        tools.addWidget(self.cell_color_label)
        btn_new_cell = QPushButton("New Cell")
        btn_new_cell.setToolTip("Add a new cell ID (next unused)")
        btn_new_cell.clicked.connect(self._on_new_cell)
        tools.addWidget(btn_new_cell)
        tools.addStretch()
        layout.addLayout(tools)

        # Canvas
        self.canvas = MaskCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # Frame slider
        sl = QHBoxLayout()
        sl.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame)
        sl.addWidget(self.frame_slider)
        self.frame_label = QLabel("0 / 0")
        sl.addWidget(self.frame_label)
        layout.addLayout(sl)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — load a video to begin")

        # Keyboard shortcuts
        QShortcut(QKeySequence("Left"), self, activated=self.prev_frame)
        QShortcut(QKeySequence("Right"), self, activated=self.next_frame)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.redo)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._on_save)
        QShortcut(QKeySequence("B"), self, activated=lambda: self._select_tool("brush"))
        QShortcut(QKeySequence("E"), self, activated=lambda: self._select_tool("eraser"))
        QShortcut(QKeySequence("P"), self, activated=lambda: self._select_tool("polygon"))
        QShortcut(QKeySequence("F"), self, activated=lambda: self._select_tool("fill"))
        for k in range(1, 10):
            QShortcut(QKeySequence(str(k)), self,
                      activated=lambda v=k: self._set_active_cell(v))

    # ------------------------------------------------------------------
    def load_video(self, video_path, mask_path=None):
        try:
            self.frames = load_video(video_path)
            meta = load_metadata(video_path)
            self.name = meta.get("name", os.path.basename(video_path))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")
            return

        self.video_path = video_path
        n, h, w = self.frames.shape
        self.masks = np.zeros((n, h, w), dtype=np.int32)
        if mask_path and os.path.exists(mask_path):
            self._load_masks_from(mask_path)

        self.undo_stacks.clear()
        self.redo_stacks.clear()
        self.current_frame = 0
        self.frame_slider.setEnabled(True)
        self.frame_slider.setRange(0, n - 1)
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"0 / {n - 1}")
        self.status.showMessage(
            f"Loaded {self.name}: {n} frames, {w}×{h}"
        )
        self._redraw()

    def _load_masks_from(self, path):
        """Load a mask stack from .npz or a folder of PNGs."""
        if path.endswith(".npz"):
            data = np.load(path)
            key = "masks" if "masks" in data else list(data.keys())[0]
            arr = data[key]
            if arr.shape != self.masks.shape:
                QMessageBox.warning(
                    self, "Shape mismatch",
                    f"Mask shape {arr.shape} doesn't match frames {self.masks.shape}"
                )
                return
            if arr.dtype == bool or arr.max() <= 1:
                self.masks = bool_to_labels(arr)
            else:
                self.masks = arr.astype(np.int32)
        elif os.path.isdir(path):
            from skimage import io as skio
            import re
            n_loaded = 0
            for fname in sorted(os.listdir(path)):
                if not fname.lower().endswith((".png", ".tif", ".tiff")):
                    continue
                # Support both "frame_NNNN.png" and "frame_NNNN_masks.png"
                # (and other variants that contain a single integer).
                nums = re.findall(r"\d+", fname)
                if not nums:
                    continue
                # Use the LAST number in the filename (typically the
                # frame index — e.g. "frame_0042_masks.png" → 42).
                # If the filename is "frame_0042.png" → 42 too.
                idx = int(nums[-1] if "masks" not in fname.lower()
                            else nums[-2] if len(nums) >= 2 else nums[-1])
                if 0 <= idx < len(self.masks):
                    m = skio.imread(os.path.join(path, fname))
                    if m.ndim == 3:
                        m = m[..., 0]
                    if m.shape == self.masks[idx].shape:
                        if m.max() <= 1:
                            self.masks[idx] = bool_to_labels(m > 0)
                        else:
                            self.masks[idx] = m.astype(np.int32)
                        n_loaded += 1
            self.status.showMessage(
                f"Loaded {n_loaded} mask PNGs from folder {path}")

    # ------------------------------------------------------------------
    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.tif *.tiff)"
        )
        if path:
            self.load_video(path)

    def _on_load_masks(self):
        if self.frames is None:
            return
        # Accept either an .npz file OR a directory of frame_NNNN_masks.png.
        # Use a custom dialog mode that allows BOTH.
        dlg = QFileDialog(self, "Load Masks (.npz file OR folder)")
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setNameFilters([
            "Mask file (*.npz)", "Folder (any)", "All files (*)"])
        if dlg.exec_() != QFileDialog.Accepted:
            return
        path = dlg.selectedFiles()[0]
        self._load_masks_from(path)
        self._redraw()

    def _on_load_mask_folder(self):
        """Explicit 'Load Mask Folder' for folders of frame_NNNN[_masks].png"""
        if self.frames is None:
            return
        path = QFileDialog.getExistingDirectory(
            self, "Select folder containing frame_NNNN[_masks].png")
        if path:
            self._load_masks_from(path)
            self._redraw()

    def _on_detect(self):
        if self.frames is None:
            return
        self.status.showMessage("Running cellpose+flow...")
        QApplication.processEvents()
        try:
            from core.detection import detect_cellpose_flow
            masks, _, _ = detect_cellpose_flow(self.frames, gpu=True)
            self.masks = bool_to_labels(masks > 0)
            self.undo_stacks.clear()
            self.redo_stacks.clear()
            self.status.showMessage("Cellpose+flow detection done")
            self._redraw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed: {e}")
            self.status.showMessage("Detection failed")

    def _on_send_to_gui(self):
        if self.masks is None:
            QMessageBox.information(self, "Send to GUI",
                                    "No masks to send.")
            return
        self.masks_sent.emit(self.masks.copy())
        n_cells = int(self.masks.max())
        n_frames = int((self.masks > 0).any(axis=(1, 2)).sum())
        self.status.showMessage(
            f"Sent {n_frames} frames ({n_cells} cell IDs) to main GUI")

    def _on_save(self):
        if self.frames is None or self.masks is None:
            return
        default_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..",
            "data", "manual_gt", self.name.replace(" ", "_"),
        )
        default_dir = os.path.normpath(default_dir)
        out_dir = QFileDialog.getExistingDirectory(
            self, "Choose output directory", default_dir,
        )
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        # Save as PNG stack (cellpose format) + NPZ
        from skimage import io as skio
        n_saved = 0
        for i, m in enumerate(self.masks):
            if m.any():
                skio.imsave(
                    os.path.join(out_dir, f"frame_{i:04d}_masks.png"),
                    m.astype(np.uint16),  # int32→uint16, preserves cell IDs
                    check_contrast=False,
                )
                n_saved += 1
        np.savez_compressed(
            os.path.join(out_dir, "masks.npz"),
            masks=self.masks,
        )
        meta = {
            "recording": self.name,
            "n_frames": int(len(self.masks)),
            "n_edited": int(n_saved),
            "video_path": getattr(self, "video_path", ""),
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.status.showMessage(f"Saved {n_saved} frames to {out_dir}")

    # ------------------------------------------------------------------
    def _on_tool(self, name, checked):
        if checked:
            self.tool = name

    def _select_tool(self, name):
        for btn in self.tool_group.buttons():
            if btn.text() == name:
                btn.setChecked(True)
                break

    def current_tool(self):
        return self.tool

    def _on_brush_size(self, val):
        self.brush_size = val

    def _on_opacity(self, val):
        self.mask_opacity = val / 100.0
        self._redraw()

    def _on_frame(self, idx):
        self.current_frame = idx
        self.frame_label.setText(f"{idx} / {self.frame_slider.maximum()}")
        self._redraw()

    def prev_frame(self):
        if self.frame_slider.value() > 0:
            self.frame_slider.setValue(self.frame_slider.value() - 1)

    def next_frame(self):
        if self.frame_slider.value() < self.frame_slider.maximum():
            self.frame_slider.setValue(self.frame_slider.value() + 1)

    def show_coords(self, x, y):
        self.status.showMessage(
            f"tool={self.tool}  cell={self.active_cell}  "
            f"({x}, {y})  brush={self.brush_size}")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def begin_stroke(self):
        """Snapshot the current mask for undo."""
        idx = self.current_frame
        self._stroke_snapshot = self.masks[idx].copy()

    def end_stroke(self):
        idx = self.current_frame
        if self._stroke_snapshot is None:
            return
        # Only push to undo if something actually changed
        if not np.array_equal(self._stroke_snapshot, self.masks[idx]):
            self.undo_stacks.setdefault(idx, []).append(self._stroke_snapshot)
            if len(self.undo_stacks[idx]) > self.max_undo:
                self.undo_stacks[idx].pop(0)
            self.redo_stacks.setdefault(idx, []).clear()
        self._stroke_snapshot = None

    def paint_at(self, x, y, tool):
        if self.masks is None:
            return
        idx = self.current_frame
        m = self.masks[idx]
        h, w = m.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        r = max(1, self.brush_size // 2)
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xx * xx + yy * yy) <= r * r
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        if y1 <= y0 or x1 <= x0:
            return
        py0 = y0 - (y - r)
        py1 = py0 + (y1 - y0)
        px0 = x0 - (x - r)
        px1 = px0 + (x1 - x0)
        patch = disk[py0:py1, px0:px1]
        if tool == "brush":
            m[y0:y1, x0:x1] = np.where(patch, self.active_cell, m[y0:y1, x0:x1])
        else:  # eraser
            m[y0:y1, x0:x1] = np.where(patch, 0, m[y0:y1, x0:x1])
        self._redraw()

    def draw_polygon_preview(self, points):
        """Draw a preview line along clicked polygon points."""
        self._redraw(polygon_preview=points)

    def commit_polygon(self, points):
        if self.masks is None or len(points) < 3:
            return
        idx = self.current_frame
        self.undo_stacks.setdefault(idx, []).append(self.masks[idx].copy())
        if len(self.undo_stacks[idx]) > self.max_undo:
            self.undo_stacks[idx].pop(0)
        pts = np.array(points, dtype=np.int32)
        tmp = np.zeros_like(self.masks[idx], dtype=np.uint8)
        cv2.fillPoly(tmp, [pts], 1)
        self.masks[idx] = np.where(tmp > 0, self.active_cell, self.masks[idx])
        self._redraw()

    def flood_fill(self, x, y):
        if self.masks is None:
            return
        idx = self.current_frame
        m = self.masks[idx]
        h, w = m.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        self.undo_stacks.setdefault(idx, []).append(m.copy())
        if len(self.undo_stacks[idx]) > self.max_undo:
            self.undo_stacks[idx].pop(0)
        from scipy import ndimage
        if m[y, x] == 0:
            labeled, _ = ndimage.label(m == 0)
            comp = labeled == labeled[y, x]
            m[comp] = self.active_cell
        self._redraw()

    def clear_current(self):
        if self.masks is None:
            return
        idx = self.current_frame
        self.undo_stacks.setdefault(idx, []).append(self.masks[idx].copy())
        self.masks[idx] = 0
        self._redraw()

    def undo(self):
        idx = self.current_frame
        if idx not in self.undo_stacks or not self.undo_stacks[idx]:
            return
        self.redo_stacks.setdefault(idx, []).append(self.masks[idx].copy())
        self.masks[idx] = self.undo_stacks[idx].pop()
        self._redraw()

    def redo(self):
        idx = self.current_frame
        if idx not in self.redo_stacks or not self.redo_stacks[idx]:
            return
        self.undo_stacks.setdefault(idx, []).append(self.masks[idx].copy())
        self.masks[idx] = self.redo_stacks[idx].pop()
        self._redraw()

    # ------------------------------------------------------------------
    def _on_cell_id(self, val):
        self.active_cell = val
        self._update_cell_color_label()
        self._redraw()

    def _set_active_cell(self, val):
        self.cell_spin.setValue(val)

    def _on_new_cell(self):
        if self.masks is None:
            return
        new_id = next_cell_id(self.masks[self.current_frame])
        if new_id > 9:
            new_id = 9
        self.cell_spin.setValue(new_id)

    def _update_cell_color_label(self):
        r, g, b = cell_color(self.active_cell)
        self.cell_color_label.setStyleSheet(
            f"color: rgb({r},{g},{b}); font-size: 18px; font-weight: bold;")
        self.cell_color_label.setText(f" ● Cell {self.active_cell}")

    def _redraw(self, polygon_preview=None):
        if self.frames is None:
            return
        idx = self.current_frame
        img = self.frames[idx]
        h, w = img.shape

        if self.masks is not None and self.masks[idx].any():
            rgb = render_label_overlay(
                img, self.masks[idx], opacity=self.mask_opacity,
                active_cell=self.active_cell,
                polygon_preview=polygon_preview)
        else:
            rgb = np.stack([img, img, img], axis=-1).copy()
            if polygon_preview and len(polygon_preview) >= 2:
                pts = np.array(polygon_preview, dtype=np.int32)
                cv2.polylines(rgb, [pts], False, (255, 255, 255), 1)

        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.canvas.update_pixmap(pix)


def main():
    parser = argparse.ArgumentParser(description="Manual mask editor")
    parser.add_argument("video", nargs="?", help="Video file to load")
    parser.add_argument("--masks", help="Existing mask stack (.npz) or folder")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    editor = MaskEditor(video_path=args.video, mask_path=args.masks)
    editor.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
