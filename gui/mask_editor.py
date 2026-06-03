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
from PyQt5.QtCore import Qt, QPointF, QRect, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QButtonGroup,
    QRadioButton, QSpinBox, QMessageBox, QShortcut, QStatusBar,
    QCheckBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QDialog, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QInputDialog, QTabWidget, QDesktopWidget, QRubberBand,
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
        # Right-button drag panning state
        self.panning = False
        self._pan_start = None
        # SAM2 box-prompt drag state. Rubber band is created lazily
        # on first sam2-box drag (so unused tool doesn't cost a widget).
        self.rubber_band = None
        self.sam2_box_origin = None       # viewport coords at drag start
        self.sam2_box_dragging = False
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Accept drops here too — QGraphicsView fills most of the
        # window, so without this the QMainWindow's drag handlers
        # only fire over the title bar / toolbar margins.
        self.setAcceptDrops(True)

    # --- Drag-drop on the canvas → forward to the editor ---
    def dragEnterEvent(self, event):
        if self.editor._dd_accepts(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self.editor._dd_accepts(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.editor._dd_handle(event.mimeData())
        event.acceptProposedAction()

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
            elif tool == "relabel":
                self.editor.relabel_cell(x, y)
            elif tool == "delete":
                if event.modifiers() & Qt.ShiftModifier:
                    # Shift+Click = delete this cell's track across
                    # ALL frames in one go.
                    m = self.editor.masks
                    if (m is not None
                            and 0 <= y < m.shape[1]
                            and 0 <= x < m.shape[2]):
                        cid = int(m[self.editor.current_frame, y, x])
                        if cid > 0:
                            self.editor.delete_cell_track(cid)
                else:
                    self.editor.delete_cell(x, y)
            elif tool == "sam2":
                self.editor.run_sam2_at(x, y)
            elif tool == "sam2-box":
                # Start a rubber-band drag in viewport coordinates.
                if self.rubber_band is None:
                    self.rubber_band = QRubberBand(
                        QRubberBand.Rectangle, self.viewport())
                self.sam2_box_origin = event.pos()
                self.sam2_box_dragging = True
                self.rubber_band.setGeometry(
                    QRect(self.sam2_box_origin, QSize(0, 0)))
                self.rubber_band.show()
        elif event.button() == Qt.RightButton:
            # Shift + Right-click starts a drag-pan. Plain right-click
            # is reserved for the polygon tool (commits the polygon
            # when ≥3 points are queued). This keeps the gestures
            # cleanly separated and avoids accidental pans while
            # drawing polygons.
            if event.modifiers() & Qt.ShiftModifier:
                self.panning = True
                self._pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
            else:
                tool = self.editor.current_tool()
                if tool == "polygon" and len(self.polygon_points) >= 3:
                    self.editor.commit_polygon(self.polygon_points)
                    self.polygon_points = []
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        x, y = self._scene_pos(event)
        self.editor.show_coords(x, y)
        if self.sam2_box_dragging and self.rubber_band is not None:
            self.rubber_band.setGeometry(
                QRect(self.sam2_box_origin, event.pos()).normalized())
            return
        if self.panning and self._pan_start is not None:
            # Translate scrollbars by the cursor delta so the image
            # follows the mouse. Note: y axis uses inverse delta because
            # scrollbar.value() grows downward (same as cursor.y).
            delta = event.pos() - self._pan_start
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            self._pan_start = event.pos()
            return
        if self.drawing:
            tool = self.editor.current_tool()
            self.editor.paint_at(x, y, tool)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            self.editor.end_stroke()
        if (self.sam2_box_dragging
                and event.button() == Qt.LeftButton):
            self.sam2_box_dragging = False
            if self.rubber_band is not None:
                self.rubber_band.hide()
            # All of this is wrapped in try/except: any unhandled
            # Python exception bubbling out of a Qt event override
            # (here mouseReleaseEvent on QGraphicsView) is converted
            # by PyQt to qFatal() and aborts the process. Catching
            # the exception keeps the editor alive and lets us
            # surface the error in the status bar + stderr.
            try:
                # Map viewport corners → scene (image) coordinates.
                # Both corners must go through mapToScene so zoom +
                # pan are honoured.
                p0_sc = self.mapToScene(self.sam2_box_origin)
                x0 = int(round(p0_sc.x()))
                y0 = int(round(p0_sc.y()))
                x1, y1 = self._scene_pos(event)
                self.editor.run_sam2_box_at(x0, y0, x1, y1)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.editor.status.showMessage(
                    f"SAM2-box release error: {e!r}", 12000)
        if self.panning and event.button() == Qt.RightButton:
            self.panning = False
            self._pan_start = None
            self.unsetCursor()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.editor.current_tool() == "polygon" and len(self.polygon_points) >= 3:
            self.editor.commit_polygon(self.polygon_points)
            self.polygon_points = []

    def wheelEvent(self, event):
        # Zoom with mouse wheel
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)


class FilterCellsDialog(QDialog):
    """Bulk-filter cell instances by per-frame and per-track criteria.

    Per-frame filters (min/max area, min edge distance) remove the
    individual mask instance on each frame where it matches —
    preserving the cell ID on other frames where it's valid. This
    matches the typical user-edit pattern where the same tracked cell
    has a normal-sized mask on most frames and an artifact mask
    (vignette blob, debris merge) on a few.

    Per-track filters (min lifetime, max mean velocity for static
    phantoms) remove the entire cell ID from every frame.

    Preview before Apply. Apply pushes per-frame undo entries so
    Ctrl+Z reverses each affected frame.
    """

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.setWindowTitle("Filter cells")
        self.setMinimumSize(620, 540)
        self._candidates = None

        layout = QVBoxLayout(self)
        intro = QLabel(
            "<b>Filter cell mask instances across all frames.</b><br>"
            "<i>Per-frame</i> filters remove individual instances "
            "(other frames where the cell is valid keep it). "
            "<i>Per-track</i> filters remove the entire cell ID from "
            "every frame.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(QLabel(
            "<b>Per-frame filters</b> — remove individual instances"))
        # Safe defaults from cross-recording filter-learning analysis on
        # 4 reviewed recordings (71 cells: 16 REMOVED, 11 TRIMMED, 44 kept):
        # min=947  → smallest real cell observed; never removes a real cell
        # max=15000 → above largest real cell (12006) but below the only
        # observed huge artefact (79855 px). Catches ~56% of artefacts
        # while preserving all kept cells. For more aggressive cleanup,
        # the dialog's Preview lets you raise min and visually deselect
        # any real cells flagged.
        self.chk_min_area, self.spin_min_area = self._row(
            "Min area (px)", 1, 1000000, 947, layout)
        self.chk_max_area, self.spin_max_area = self._row(
            "Max area (px)", 1, 100000000, 15000, layout)
        self.chk_edge, self.spin_edge = self._row(
            "Min distance to frame edge (px)", 0, 5000, 30, layout)

        layout.addWidget(QLabel(
            "<b>Per-track filters</b> — remove cell from all frames"))
        self.chk_lifetime, self.spin_lifetime = self._row(
            "Min lifetime (frames)", 1, 10000, 5, layout)
        self.chk_vel, self.spin_vel = self._row(
            "Static cells: max mean velocity (px/frame)",
            0, 100, 2, layout, is_float=True)

        self.preview_label = QLabel(
            "Click <b>Preview</b> to see what would be removed.")
        layout.addWidget(self.preview_label)
        self.preview_list = QListWidget()
        layout.addWidget(self.preview_list, 1)

        buttons = QHBoxLayout()
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.clicked.connect(self._preview)
        buttons.addWidget(self.btn_preview)
        self.btn_apply = QPushButton("Apply (remove)")
        self.btn_apply.clicked.connect(self._apply)
        self.btn_apply.setEnabled(False)
        buttons.addWidget(self.btn_apply)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        buttons.addWidget(btn_close)
        layout.addLayout(buttons)

    def _row(self, label, lo, hi, default, layout, is_float=False):
        row = QHBoxLayout()
        chk = QCheckBox(label)
        spin = QDoubleSpinBox() if is_float else QSpinBox()
        spin.setRange(lo, hi); spin.setValue(default)
        if is_float:
            spin.setSingleStep(0.5)
        row.addWidget(chk); row.addWidget(spin); row.addStretch()
        layout.addLayout(row)
        return chk, spin

    def _compute_track_stats(self):
        m = self.editor.masks
        n_frames = m.shape[0]
        ids = sorted(set(np.unique(m).tolist()) - {0})
        out = {}
        for cid in ids:
            mask = (m == cid)
            present = mask.any(axis=(1, 2))
            lifetime = int(present.sum())
            if lifetime == 0:
                continue
            cents = []
            for i in range(n_frames):
                if present[i]:
                    ys, xs = np.where(mask[i])
                    cents.append((float(ys.mean()), float(xs.mean())))
            vels = [((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
                    for a, b in zip(cents, cents[1:])]
            mean_vel = sum(vels) / max(len(vels), 1)
            out[cid] = (lifetime, mean_vel)
        return out

    def _candidates_for_removal(self):
        m = self.editor.masks
        n_frames, H, W = m.shape
        ts = self._compute_track_stats()
        whole = set()
        if self.chk_lifetime.isChecked():
            thr = self.spin_lifetime.value()
            for cid, (lt, _) in ts.items():
                if lt < thr:
                    whole.add(cid)
        if self.chk_vel.isChecked():
            thr = self.spin_vel.value()
            for cid, (_, mv) in ts.items():
                if mv < thr:
                    whole.add(cid)
        per_frame = []
        for cid in ts:
            if cid in whole:
                continue
            mask = (m == cid)
            for i in range(n_frames):
                if not mask[i].any():
                    continue
                area = int(mask[i].sum())
                ys, xs = np.where(mask[i])
                cy, cx = float(ys.mean()), float(xs.mean())
                edge_d = min(cy, cx, H - cy, W - cx)
                reasons = []
                if self.chk_min_area.isChecked() and area < self.spin_min_area.value():
                    reasons.append(f"area={area} <{self.spin_min_area.value()}")
                if self.chk_max_area.isChecked() and area > self.spin_max_area.value():
                    reasons.append(f"area={area} >{self.spin_max_area.value()}")
                if self.chk_edge.isChecked() and edge_d < self.spin_edge.value():
                    reasons.append(f"edge={edge_d:.0f} <{self.spin_edge.value()}")
                if reasons:
                    per_frame.append((int(cid), int(i),
                                       ", ".join(reasons)))
        return whole, per_frame, ts

    def _preview(self):
        whole, per_frame, ts = self._candidates_for_removal()
        self.preview_list.clear()
        for cid in sorted(whole):
            lt, mv = ts[cid]
            self.preview_list.addItem(
                f"Cell {cid} ▸ WHOLE TRACK   lifetime={lt}  "
                f"mean_vel={mv:.2f}")
        by_cell = {}
        for cid, fr, _ in per_frame:
            by_cell.setdefault(cid, []).append(fr)
        for cid in sorted(by_cell):
            frs = sorted(by_cell[cid])
            self.preview_list.addItem(
                f"Cell {cid} ▸ {len(frs)} frames "
                f"[{frs[0]}…{frs[-1]}]")
        self.preview_label.setText(
            f"<b>{len(whole)}</b> whole tracks + "
            f"<b>{len(per_frame)}</b> per-frame instances would be "
            f"removed.")
        self.btn_apply.setEnabled(bool(whole or per_frame))
        self._candidates = (whole, per_frame)

    def _apply(self):
        if self._candidates is None:
            return
        whole, per_frame = self._candidates
        # Build the set of frames we're about to mutate, so we can
        # snapshot exactly those for per-frame undo. Whole-track
        # removals affect every frame the cell appears in.
        affected = set([f for _, f, _ in per_frame])
        for cid in whole:
            for i in range(self.editor.masks.shape[0]):
                if (self.editor.masks[i] == cid).any():
                    affected.add(i)
        for i in sorted(affected):
            self.editor.undo_stacks.setdefault(i, []).append(
                self.editor.masks[i].copy())
            if len(self.editor.undo_stacks[i]) > self.editor.max_undo:
                self.editor.undo_stacks[i].pop(0)
            self.editor.redo_stacks.pop(i, None)
            self.editor._dirty_frames.add(i)
        for cid in whole:
            self.editor.masks[self.editor.masks == cid] = 0
        for cid, i, _ in per_frame:
            self.editor.masks[i][self.editor.masks[i] == cid] = 0
        self.editor._redraw()
        msg = (f"Removed {len(whole)} whole tracks + "
                f"{len(per_frame)} per-frame instances across "
                f"{len(affected)} frame(s)")
        self.editor.status.showMessage(msg, 8000)
        print(f"[mask_editor] filter applied: {msg}", flush=True)
        self.close()


class MaskEditor(QMainWindow):
    masks_sent = pyqtSignal(object)

    def __init__(self, video_path=None, mask_path=None):
        super().__init__()
        self.setWindowTitle("Mask Editor — CellScope")
        # Size to fit the screen — many laptop displays are 1440×900
        # (or smaller after the menu bar), so 1200×900 was clipping the
        # bottom of the window. Use ~85% of the available screen
        # geometry, capped at the previous default.
        try:
            screen = QDesktopWidget().availableGeometry()
            w = min(1200, max(900, int(screen.width() * 0.85)))
            h = min(820, max(600, int(screen.height() * 0.85)))
        except Exception:
            w, h = 1100, 750
        self.resize(w, h)
        self.setMinimumSize(900, 600)

        self.frames = None          # (N, H, W) uint8
        self.masks = None           # (N, H, W) int32 — 0=bg, 1,2,...=cell IDs
        self.name = "recording"
        # SAM2 tool settings (mutated by the SAM2 → Options dialog;
        # threaded into predict_at_point / predict_at_box at click time)
        from gui.mask_editor_sam2_settings import default_settings
        self.sam2_settings = default_settings()
        # Overlay settings (timestamp + scale bar). View → Overlay
        # Options… opens the dialog; defaults off until enabled.
        from gui.overlays import default_overlay_settings
        self.overlay_settings = default_overlay_settings()
        # Recording metadata for overlay rendering. Populated by
        # load_video; pass-throughs when no recording is loaded.
        self.um_per_px = None
        self.dt_min = None
        # Canonical path the editor was launched against (passed to
        # `__init__`). Used by Save-In-Place (Ctrl+S) so reviewer
        # workflows can write back to the source `masks.npz`. Drag-drop
        # loads (e.g. dragging `masks_original.npz` in to compare) do
        # NOT update this — so saving always lands at the original
        # canonical target.
        self.canonical_mask_path = None
        self._canonical_mask_key = None
        self.current_frame = 0
        self.brush_size = 10
        self.mask_opacity = 0.4
        self.show_ids = False       # draw cell ID numbers on the overlay
        self.tool = "brush"         # brush / eraser / polygon / fill / none
        self.active_cell = 1        # which cell ID to paint with
        # Ground-truth labeling helpers — populated by load_video() when
        # the recording folder has a GT_FRAMES.txt + gt_masks/ layout
        # (the format produced by scripts/setup_ic295_gt_labeling.py).
        self.gt_frames = None       # list[int] target frames to label
        self.gt_jump_size = 10      # stride when no GT_FRAMES.txt
        self.gt_masks_dir = None    # where labelled masks live
        # Multichannel support — fluo_frames is set when the recording
        # has a fluorescence channel. The active_channel toggle swaps
        # between them for display; masks live in a single shared stack
        # (cells are the same regardless of which channel you view).
        self.dic_frames = None
        self.fluo_frames = None
        self.active_channel = "dic"
        # Per-frame dirty flag for unsaved-edit warning + the toggle
        # that controls whether the warning ever fires.
        self._dirty_frames = set()
        self._warn_on_unsaved = True
        # Inter-target playback timer
        self._play_timer = None
        self._play_target = None

        # Undo stacks: per-frame list of mask snapshots (deque-like)
        self.undo_stacks = {}   # frame_idx -> list of mask snapshots
        self.redo_stacks = {}
        self.max_undo = 30
        self._stroke_snapshot = None

        self._build_ui()
        # Enable drops on the main window. The canvas (QGraphicsView)
        # also accepts drops and forwards to us; both routes call
        # `_dd_handle` so a drop anywhere in the editor works.
        self.setAcceptDrops(True)

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
        btn_save.setToolTip(
            "Save edits.\n"
            "If launched against a specific masks.npz "
            "(e.g. via ic295_review.py), Ctrl+S overwrites that file "
            "in place — preserving its key + dtype.\n"
            "Otherwise prompts for a ground-truth output folder.")
        btn_save.clicked.connect(self._on_save_default)
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
        for name in ["brush", "eraser", "polygon", "fill", "relabel",
                     "delete", "sam2", "sam2-box"]:
            rb = QRadioButton(name)
            if name == "sam2":
                rb.setToolTip(
                    "SAM2 point-click: left-click on an unlabelled "
                    "cell to detect and add it to this frame using "
                    "the active cell ID from the spinner. Runs on a "
                    "512×512 crop centred at the click (~100-200 ms). "
                    "Best for cells with clear boundaries.")
            elif name == "sam2-box":
                rb.setToolTip(
                    "SAM2 box-drag: click-drag a rectangle around a "
                    "single cell. The box hints at the cell's size, "
                    "so SAM2 finds the boundary even when contrast "
                    "is low (DIC membrane edges, shadowed cells). "
                    "Use this when the point-click tool undershoots.")
            rb.toggled.connect(lambda checked, n=name: self._on_tool(n, checked))
            tools.addWidget(rb)
            self.tool_group.addButton(rb)
            if name == "brush":
                rb.setChecked(True)
        tools.addWidget(QLabel("  Brush:"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 100)
        self.brush_spin.setValue(self.brush_size)
        self.brush_spin.setToolTip("Brush size (pixels)")
        self.brush_spin.valueChanged.connect(self._on_brush_size)
        tools.addWidget(self.brush_spin)
        tools.addWidget(QLabel("Eraser:"))
        self.eraser_spin = QSpinBox()
        self.eraser_spin.setRange(1, 100)
        self.eraser_spin.setValue(20)
        self.eraser_spin.setToolTip("Eraser size (pixels)")
        self.eraser_spin.valueChanged.connect(self._on_eraser_size)
        tools.addWidget(self.eraser_spin)
        self.eraser_size = 20
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
        btn_fit = QPushButton("Fit View")
        btn_fit.setToolTip("Reset zoom and center the image")
        btn_fit.clicked.connect(self._fit_view)
        tools.addWidget(btn_fit)
        tools.addWidget(QLabel("  Cell:"))
        self.cell_spin = QSpinBox()
        self.cell_spin.setRange(1, 30)
        self.cell_spin.setValue(1)
        self.cell_spin.setToolTip(
            "Active cell ID (1-30). Paint with this ID.\n"
            "Keyboard: 1-9 select cells 1-9, 0 selects cell 10,\n"
            "Shift+1..Shift+9 select cells 11-19, Shift+0 selects 20.\n"
            "Cells 21-30 are picked via the spinbox or New Cell.")
        self.cell_spin.valueChanged.connect(self._on_cell_id)
        tools.addWidget(self.cell_spin)
        self.cell_color_label = QLabel("  ●")
        self._update_cell_color_label()
        tools.addWidget(self.cell_color_label)
        btn_new_cell = QPushButton("New Cell")
        btn_new_cell.setToolTip(
            "Set the active cell ID to the lowest unused ID across "
            "the WHOLE recording. Reuses IDs freed by deleted tracks. "
            "Use this before adding a brand-new cell (e.g. with the "
            "SAM2 point tool) so the new mask starts its own track "
            "instead of accidentally extending an existing one.")
        btn_new_cell.clicked.connect(self._on_new_cell)
        tools.addWidget(btn_new_cell)
        self.chk_show_ids = QCheckBox("Cell IDs")
        self.chk_show_ids.setChecked(False)
        self.chk_show_ids.setToolTip(
            "Draw each cell's ID number at its centroid.\n"
            "Keyboard shortcut: I")
        self.chk_show_ids.toggled.connect(self._on_toggle_ids)
        tools.addWidget(self.chk_show_ids)
        tools.addStretch()
        layout.addLayout(tools)

        # Canvas
        self.canvas = MaskCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # Below the canvas: a tabbed control panel that consolidates
        # the GT-labeling row (used when creating training data) and
        # the new pipeline-review row (Save Edits / Save & Next /
        # Filter Cells / Delete by Cell ID — used by ic295_review.py).
        # Tabs are MUCH narrower than the combined ~1700px row was,
        # so the window fits on a 1440-wide laptop screen.
        self.review_tabs = QTabWidget()
        self.review_tabs.setDocumentMode(True)  # flatter look on macOS

        # ── Tab 1: Pipeline review ── (the ic295_review workflow)
        review_page = QWidget()
        rev = QHBoxLayout(review_page)
        rev.setContentsMargins(8, 4, 8, 4)
        # The button objects are created below in the GT section block
        # and then re-parented here; we just allocate placeholders so
        # the tab order is right.

        # ── Tab 2: GT labeling ── (training-data workflow)
        gt_page = QWidget()
        gt_outer = QVBoxLayout(gt_page)
        gt_outer.setContentsMargins(8, 4, 8, 4)
        gt = QHBoxLayout()
        gt.addWidget(QLabel("GT frame:"))
        self.btn_gt_prev = QPushButton("⟪ Prev")
        self.btn_gt_prev.setToolTip(
            "Jump to previous target frame (Shift+Left).")
        self.btn_gt_prev.clicked.connect(self.prev_target_frame)
        gt.addWidget(self.btn_gt_prev)
        self.btn_gt_next = QPushButton("Next ⟫")
        self.btn_gt_next.setToolTip(
            "Jump to next target frame (Shift+Right).")
        self.btn_gt_next.clicked.connect(self.next_target_frame)
        gt.addWidget(self.btn_gt_next)
        gt.addWidget(QLabel("  Every N frames:"))
        self.gt_jump_spin = QSpinBox()
        self.gt_jump_spin.setRange(1, 1000)
        self.gt_jump_spin.setValue(self.gt_jump_size)
        self.gt_jump_spin.setToolTip(
            "Frame stride for target sampling. Ignored if a\n"
            "GT_FRAMES.txt file is found in the recording's folder.")
        self.gt_jump_spin.valueChanged.connect(self._on_jump_size)
        gt.addWidget(self.gt_jump_spin)
        btn_load_gt = QPushButton("Load GT_FRAMES.txt")
        btn_load_gt.setToolTip(
            "Load an explicit target-frame list from a file.\n"
            "Auto-detected when you open a recording in a folder\n"
            "with GT_FRAMES.txt next to the .ome.tif.")
        btn_load_gt.clicked.connect(self._on_load_gt_frames)
        gt.addWidget(btn_load_gt)
        btn_gt_save = QPushButton("Save GT for this frame")
        btn_gt_save.setToolTip(
            "Save current frame's mask as the labelled GT for this\n"
            "frame into the recording's gt_masks/ folder.")
        btn_gt_save.clicked.connect(self._on_save_gt_frame)
        gt.addWidget(btn_gt_save)
        btn_gt_save_all = QPushButton("Save all GT")
        btn_gt_save_all.setToolTip(
            "Save every frame that has any labels into gt_masks/.\n"
            "Skips empty frames. (Ctrl+Shift+G)")
        btn_gt_save_all.clicked.connect(self._on_save_all_gt)
        gt.addWidget(btn_gt_save_all)
        # ── Pipeline-output save / cleanup — placed in the
        # "Pipeline Review" tab (rev layout), not the GT row, so they
        # don't bloat the GT-labeling controls. ──
        # Frame nav (mirrors Left/Right arrow shortcuts but visible
        # in the panel for users who haven't memorised the keys).
        self.btn_rev_prev = QPushButton("◀ Prev")
        self.btn_rev_prev.setToolTip(
            "Go to previous frame. Same as Left arrow key.\n"
            "Honours the unsaved-edits warning on dirty frames.")
        self.btn_rev_prev.clicked.connect(self.prev_frame)
        rev.addWidget(self.btn_rev_prev)
        self.btn_rev_next = QPushButton("Next ▶")
        self.btn_rev_next.setToolTip(
            "Go to next frame. Same as Right arrow key.\n"
            "Honours the unsaved-edits warning on dirty frames.")
        self.btn_rev_next.clicked.connect(self.next_frame)
        rev.addWidget(self.btn_rev_next)
        rev.addSpacing(12)
        self.btn_save_edits = QPushButton("💾 Save Edits")
        self.btn_save_edits.setToolTip(
            "Overwrite the masks file this editor was launched against "
            "(pipeline_results/masks.npz). Preserves the full schema "
            "(labels + masks foreground + any sidecar arrays).\n"
            "Shortcut: Ctrl+S")
        self.btn_save_edits.clicked.connect(self._on_save_in_place_explicit)
        rev.addWidget(self.btn_save_edits)
        self.btn_save_and_next = QPushButton("💾 Save & Next →")
        self.btn_save_and_next.setToolTip(
            "Save edits in place, then advance to the next frame.\n"
            "Use for fluid frame-by-frame review.\n"
            "Shortcut: Ctrl+Shift+S")
        self.btn_save_and_next.clicked.connect(self._on_save_and_next)
        rev.addWidget(self.btn_save_and_next)
        # Disable both unless a canonical mask path was registered
        # (otherwise there's no in-place target). load_video() will
        # enable them after a successful launch with mask_path set.
        self.btn_save_edits.setEnabled(False)
        self.btn_save_and_next.setEnabled(False)
        rev.addSpacing(20)
        btn_filter = QPushButton("🔍 Filter Cells…")
        btn_filter.setToolTip(
            "Bulk-remove artifacts by criteria — per-frame "
            "(min/max area, edge distance) or per-track (min "
            "lifetime, max mean velocity for static phantoms). "
            "Preview before applying. (Ctrl+Shift+F)")
        btn_filter.clicked.connect(self._on_filter_cells)
        rev.addWidget(btn_filter)
        btn_del_id = QPushButton("🗑 Delete by Cell ID…")
        btn_del_id.setToolTip(
            "Pick a cell ID and remove it from every frame it "
            "appears in. Equivalent to Shift-clicking a cell while "
            "the 'delete' tool is active.")
        btn_del_id.clicked.connect(self._on_delete_by_id)
        rev.addWidget(btn_del_id)
        btn_trim = QPushButton("✂ Trim Edges…")
        btn_trim.setToolTip(
            "Bulk-remove mask pixels within a user-defined band "
            "along one or more image edges. Targets the vignette / "
            "black-margin artefact that produces thin slivers of "
            "mask along an edge across many frames. Preview before "
            "applying.")
        btn_trim.clicked.connect(self._on_trim_edges)
        rev.addWidget(btn_trim)
        rev.addStretch()
        self.chk_warn_unsaved = QCheckBox("Warn on unsaved")
        self.chk_warn_unsaved.setChecked(True)
        self.chk_warn_unsaved.setToolTip(
            "If on, prompt before navigating away from a target frame\n"
            "with unsaved label edits. Turn off if you prefer to\n"
            "manage saves manually (Ctrl+G).")
        self.chk_warn_unsaved.toggled.connect(self._on_warn_toggle)
        gt.addWidget(self.chk_warn_unsaved)
        # Inter-target playback: scrub from current frame to the next
        # target at ~5 fps so the user can verify motion before labeling.
        self.btn_play_to_next = QPushButton("▶ Play to next target")
        self.btn_play_to_next.setCheckable(True)
        self.btn_play_to_next.setToolTip(
            "Scrub forward at ~5 fps until the next target frame.\n"
            "Useful for verifying cell motion + identity before "
            "labelling.")
        self.btn_play_to_next.toggled.connect(self._on_play_toggle)
        gt.addWidget(self.btn_play_to_next)
        self.gt_progress_label = QLabel("Labeled: – / –")
        self.gt_progress_label.setStyleSheet("color: #888; padding-left: 10px;")
        gt.addWidget(self.gt_progress_label)
        gt.addStretch()
        gt_outer.addLayout(gt)
        gt_outer.addStretch()
        # Assemble the tabbed control panel
        self.review_tabs.addTab(review_page, "🔍 Pipeline Review")
        self.review_tabs.addTab(gt_page, "GT Labeling")
        # Default tab is "GT Labeling"; load_video() flips to
        # "Pipeline Review" when launched with a canonical mask_path
        # (ic295_review.py workflow).
        self.review_tabs.setCurrentIndex(1)
        layout.addWidget(self.review_tabs)

        # Channel toggle (shown only when multichannel data is loaded).
        ch = QHBoxLayout()
        self._channel_label = QLabel("Channel:")
        ch.addWidget(self._channel_label)
        self.radio_dic = QRadioButton("DIC")
        self.radio_dic.setChecked(True)
        self.radio_dic.toggled.connect(self._on_channel_toggle)
        self.radio_fluo = QRadioButton("Fluo")
        ch.addWidget(self.radio_dic)
        ch.addWidget(self.radio_fluo)
        self._channel_group = QButtonGroup(self)
        self._channel_group.addButton(self.radio_dic)
        self._channel_group.addButton(self.radio_fluo)
        ch.addStretch()
        # Hidden until a fluo channel is loaded
        for w in (self._channel_label, self.radio_dic, self.radio_fluo):
            w.setVisible(False)
        layout.addLayout(ch)

        # Frame slider — custom subclass so we can paint GT-target ticks.
        from gui.gt_slider import GtTickSlider
        sl = QHBoxLayout()
        sl.addWidget(QLabel("Frame:"))
        self.frame_slider = GtTickSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame)
        sl.addWidget(self.frame_slider)
        self.frame_label = QLabel("0 / 0")
        sl.addWidget(self.frame_label)
        layout.addLayout(sl)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — load a video to begin")

        # Menubar — SAM2 and View entries. On macOS this appears in
        # the global menu bar at top of screen; on Linux/Windows it's
        # attached to the window.
        mb = self.menuBar()
        sam2_menu = mb.addMenu("SAM2")
        act_settings = sam2_menu.addAction("Options…")
        act_settings.triggered.connect(self._on_sam2_settings)
        view_menu = mb.addMenu("View")
        act_overlay = view_menu.addAction("Overlay Options…")
        act_overlay.triggered.connect(self._on_overlay_settings)

        # Keyboard shortcuts
        QShortcut(QKeySequence("Left"), self, activated=self.prev_frame)
        QShortcut(QKeySequence("Right"), self, activated=self.next_frame)
        QShortcut(QKeySequence("Shift+Left"), self,
                  activated=self.prev_target_frame)
        QShortcut(QKeySequence("Shift+Right"), self,
                  activated=self.next_target_frame)
        QShortcut(QKeySequence("Ctrl+G"), self,
                  activated=self._on_save_gt_frame)
        QShortcut(QKeySequence("Ctrl+Shift+G"), self,
                  activated=self._on_save_all_gt)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.redo)
        _save_sc = QShortcut(QKeySequence("Ctrl+S"), self,
                              activated=self._on_save_default)
        # Use ApplicationShortcut context so Ctrl+S fires even when the
        # canvas (QGraphicsView child) has keyboard focus — otherwise
        # the default WidgetShortcut routing eats it during active
        # mouse-driven editing.
        _save_sc.setContext(Qt.ApplicationShortcut)
        # Ctrl+Shift+S = save in place + advance to the next frame
        # (fluid frame-by-frame review).
        _save_next_sc = QShortcut(QKeySequence("Ctrl+Shift+S"), self,
                                    activated=self._on_save_and_next)
        _save_next_sc.setContext(Qt.ApplicationShortcut)
        QShortcut(QKeySequence("B"), self, activated=lambda: self._select_tool("brush"))
        QShortcut(QKeySequence("E"), self, activated=lambda: self._select_tool("eraser"))
        QShortcut(QKeySequence("P"), self, activated=lambda: self._select_tool("polygon"))
        QShortcut(QKeySequence("F"), self, activated=lambda: self._select_tool("fill"))
        QShortcut(QKeySequence("R"), self, activated=lambda: self._select_tool("relabel"))
        QShortcut(QKeySequence("X"), self, activated=lambda: self._select_tool("delete"))
        # `Delete` key as an immediate alias — switch to delete tool
        # and stay there until user picks another.
        QShortcut(QKeySequence("Delete"), self, activated=lambda: self._select_tool("delete"))
        # Filter Cells dialog
        _filter_sc = QShortcut(QKeySequence("Ctrl+Shift+F"), self,
                                 activated=self._on_filter_cells)
        _filter_sc.setContext(Qt.ApplicationShortcut)
        QShortcut(QKeySequence("I"), self,
                  activated=lambda: self.chk_show_ids.toggle())
        # 1-9 select cells 1-9; 0 = cell 10. Shift+1..Shift+9 select
        # cells 11-19; Shift+0 = cell 20.
        for k in range(1, 10):
            QShortcut(QKeySequence(str(k)), self,
                      activated=lambda v=k: self._set_active_cell(v))
        QShortcut(QKeySequence("0"), self,
                  activated=lambda: self._set_active_cell(10))
        for k in range(1, 10):
            QShortcut(QKeySequence(f"Shift+{k}"), self,
                      activated=lambda v=10 + k:
                          self._set_active_cell(v))
        QShortcut(QKeySequence("Shift+0"), self,
                  activated=lambda: self._set_active_cell(20))

    # ------------------------------------------------------------------
    # Drag-and-drop dispatcher. Accepts:
    #   - Video / multichannel TIFF (.ome.tif, .tif, .mp4, .avi, .mov)
    #     → load_video()
    #   - Mask .npz (anything ending in _masks.npz or named masks.npz)
    #     → _load_masks_from()
    #   - Mask folder (folder containing *.png masks or masks.npz)
    #     → _load_masks_from(folder)
    #   - GT_FRAMES.txt → _load_gt_frames_from()
    #   - Folder containing a single .ome.tif → load that recording
    # ------------------------------------------------------------------

    VIDEO_EXTS = (".mp4", ".avi", ".mov", ".tif", ".tiff")
    MASK_EXTS = (".npz",)

    def _dd_accepts(self, mime):
        """Return True if `mime` contains at least one droppable URL."""
        if not mime.hasUrls():
            return False
        for url in mime.urls():
            p = url.toLocalFile()
            if not p:
                continue
            if self._dd_classify(p) is not None:
                return True
        return False

    def _dd_classify(self, path):
        """Categorise a path. Returns one of {'video','mask','gt',
        'folder', None}."""
        if not path:
            return None
        if os.path.isdir(path):
            return "folder"
        low = path.lower()
        if low.endswith(self.VIDEO_EXTS):
            return "video"
        if low.endswith(self.MASK_EXTS):
            return "mask"
        if os.path.basename(path) == "GT_FRAMES.txt":
            return "gt"
        return None

    def _dd_handle(self, mime):
        """Dispatch a drop event. Called by both the main window's
        dropEvent and the canvas's forwarded drop."""
        if not mime.hasUrls():
            return
        # If multiple files dropped, process the first that we
        # recognise. (Mixed drops are rare; users typically drag one.)
        for url in mime.urls():
            path = url.toLocalFile()
            kind = self._dd_classify(path)
            if kind == "video":
                self.load_video(path)
                return
            if kind == "mask":
                if self.masks is None:
                    self.status.showMessage(
                        "Load a recording first, then drop the mask file")
                    return
                self._load_masks_from(path)
                self._redraw()
                self.status.showMessage(
                    f"Loaded masks from {os.path.basename(path)}")
                return
            if kind == "gt":
                self._load_gt_frames_from(path)
                return
            if kind == "folder":
                self._dd_handle_folder(path)
                return

    def _dd_handle_folder(self, folder):
        """Drop of a folder: prefer a single .ome.tif inside; else look
        for a mask stack."""
        tifs = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(self.VIDEO_EXTS)
            and not f.startswith("."))
        # If exactly one recording in the folder, load it.
        if len(tifs) == 1:
            self.load_video(tifs[0])
            return
        # If there's a masks.npz, load it (assuming a recording is
        # already open).
        candidates = [os.path.join(folder, "masks.npz")]
        for f in os.listdir(folder):
            if f.endswith("_masks.npz") or f.endswith(".npz"):
                candidates.append(os.path.join(folder, f))
        for cand in candidates:
            if os.path.exists(cand):
                if self.masks is None:
                    self.status.showMessage(
                        "Load a recording before dropping a mask file")
                    return
                self._load_masks_from(cand)
                self._redraw()
                self.status.showMessage(
                    f"Loaded masks from {os.path.basename(cand)}")
                return
        # Folder of PNG masks?
        if any(f.lower().endswith(".png") and "mask" in f.lower()
               for f in os.listdir(folder)):
            if self.masks is None:
                self.status.showMessage(
                    "Load a recording before dropping a mask folder")
                return
            self._load_masks_from(folder)
            self._redraw()
            self.status.showMessage(
                f"Loaded masks from {os.path.basename(folder)}/")
            return
        # Otherwise tell the user we couldn't make sense of it
        self.status.showMessage(
            f"Folder dropped but no recording or masks found in "
            f"{os.path.basename(folder)}/")

    def dragEnterEvent(self, event):
        if self._dd_accepts(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._dd_accepts(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        self._dd_handle(event.mimeData())
        event.acceptProposedAction()

    # Backwards-compat alias — some external callers may invoke this.
    def _on_drop_file(self, path):
        kind = self._dd_classify(path)
        if kind == "video":
            self.load_video(path)
        else:
            self.status.showMessage(f"Don't know how to load {path}")

    def load_video(self, video_path, mask_path=None):
        try:
            from core.io import detect_channels, load_recording
            self.dic_frames = None
            self.fluo_frames = None
            self.active_channel = "dic"
            n_ch = (detect_channels(video_path)
                    if video_path.lower().endswith((".tif", ".tiff"))
                    else 1)
            if n_ch > 1:
                # Default DIC=ch1, Fluo=ch0 (matches IC295 layout).
                # We don't pop a channel chooser dialog here to keep the
                # editor minimal — the assumption is the user is editing
                # GT for the IC295 layout.
                rec = load_recording(
                    video_path, dic_channel=1, fluo_channel=0)
                self.frames = rec["frames"]
                self.fluo_frames = rec.get("cy5_frames")
                self.dic_frames = self.frames
                meta = rec
            else:
                self.frames = load_video(video_path)
                self.dic_frames = self.frames
                meta = load_metadata(video_path)
            self.name = meta.get("name", os.path.basename(video_path))
            # Cache scale + time metadata for the overlay renderer.
            try:
                self.um_per_px = float(meta.get("um_per_px") or 0) or None
            except (TypeError, ValueError):
                self.um_per_px = None
            try:
                self.dt_min = (
                    float(meta.get("time_interval_min") or 0) or None)
            except (TypeError, ValueError):
                self.dt_min = None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")
            return

        self.video_path = video_path
        n, h, w = self.frames.shape
        self.masks = np.zeros((n, h, w), dtype=np.int32)
        if mask_path and os.path.exists(mask_path):
            self._load_masks_from(mask_path)
            # Record this as the canonical save target. Subsequent
            # drag-drop mask loads (e.g. dragging `masks_original.npz`
            # in to compare) WON'T overwrite this — Save-In-Place
            # always targets the file the editor was launched against.
            if mask_path.lower().endswith(".npz"):
                self.canonical_mask_path = mask_path
                # Track which keys the original file had — on save we
                # round-trip ALL of them (so e.g. an IC295 masks.npz
                # with {labels, masks, fusion_source_stack} comes back
                # with labels updated to the edited stack, masks
                # regenerated from the new label foreground, and
                # fusion_source_stack preserved unchanged).
                try:
                    _peek = np.load(mask_path)
                    self._canonical_original_keys = tuple(_peek.files)
                except Exception:
                    self._canonical_original_keys = ("labels",)
                # The primary key (used by analyze + GUI loaders)
                self._canonical_mask_key = (
                    "labels" if "labels" in self._canonical_original_keys
                    else ("masks" if "masks" in self._canonical_original_keys
                          else self._canonical_original_keys[0]))
                self.setWindowTitle(
                    f"Mask Editor — {os.path.basename(mask_path)} "
                    f"(editing in place)")
                # Enable the in-place save buttons now that there's a
                # canonical target. Defensive getattr — the buttons
                # may not exist yet during the very first build path,
                # but load_video is invoked after _build_ui from the
                # MaskEditor constructor so they should always be there.
                for attr in ("btn_save_edits", "btn_save_and_next"):
                    if getattr(self, attr, None) is not None:
                        getattr(self, attr).setEnabled(True)
                # Default to the "Pipeline Review" tab since we're in
                # the ic295_review.py workflow.
                if hasattr(self, "review_tabs"):
                    self.review_tabs.setCurrentIndex(0)

        self.undo_stacks.clear()
        self.redo_stacks.clear()
        self._dirty_frames.clear()
        self.current_frame = 0
        self.frame_slider.setEnabled(True)
        self.frame_slider.setRange(0, n - 1)
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"0 / {n - 1}")
        # Show/hide channel toggle based on whether fluo data is present
        has_fluo = self.fluo_frames is not None
        for widget in (self._channel_label, self.radio_dic,
                       self.radio_fluo):
            widget.setVisible(has_fluo)
        if has_fluo:
            self.radio_dic.blockSignals(True)
            self.radio_dic.setChecked(True)
            self.radio_dic.blockSignals(False)
        cy5_note = " [+ Fluo channel]" if has_fluo else ""
        self.status.showMessage(
            f"Loaded {self.name}: {n} frames, {w}×{h}{cy5_note}"
        )
        # Auto-detect GT labelling setup in the recording's folder
        try:
            self._auto_detect_gt_setup(video_path)
        except Exception as e:
            self.status.showMessage(f"GT auto-detect skipped: {e}")
        self._redraw()

    def _load_masks_from(self, path):
        """Load a mask stack from .npz or a folder of PNGs."""
        if path.endswith(".npz"):
            data = np.load(path)
            # Prefer the int32 multi-cell stack (`labels`) over the
            # boolean foreground union (`masks`). The IC295 pipeline
            # writes BOTH keys — picking `masks` collapses every cell
            # into a single connected blob and gives every cell ID=1,
            # which is what the legacy code did. The focused GUI
            # loader uses this same priority since 2026-05-29.
            if "labels" in data.files:
                key = "labels"
            elif "masks" in data.files:
                key = "masks"
            else:
                key = list(data.files)[0]
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

    def _on_save_default(self):
        """Pick the right save action: in-place when launched against
        a specific `masks.npz`, GT-folder dialog otherwise."""
        if getattr(self, "canonical_mask_path", None):
            self._on_save_in_place()
        else:
            self._on_save()

    def _on_save_in_place_explicit(self):
        """Save Edits button handler — always tries the in-place save
        path; warns if no canonical target was registered (e.g. the
        editor was opened standalone, not via ic295_review.py)."""
        if not getattr(self, "canonical_mask_path", None):
            QMessageBox.information(
                self, "No in-place target",
                "This editor wasn't launched with a specific masks file "
                "to overwrite.\n\n"
                "Use 'Save Masks' (top toolbar) to export to a folder "
                "instead — or relaunch via:\n"
                "  python main_editor.py <video> <masks.npz>")
            return
        self._on_save_in_place()

    def _on_save_and_next(self):
        """Save edits in place, then advance to the next frame.
        Fluid frame-by-frame review (Ctrl+Shift+S)."""
        if not getattr(self, "canonical_mask_path", None):
            QMessageBox.information(
                self, "No in-place target",
                "Open the editor with a masks file to enable Save & Next.")
            return
        self._on_save_in_place()
        # next_frame() does nothing if we're already at the last frame
        if hasattr(self, "next_frame"):
            self.next_frame()

    def _on_save_in_place(self):
        """Overwrite the masks file the editor was launched against,
        round-tripping every key the original file had so the file's
        schema is preserved. For IC295 pipeline outputs that means:
        `labels` (the int32 multi-cell stack — updated from the editor's
        in-memory edits), `masks` (the boolean foreground union —
        regenerated from the new labels), and `fusion_source_stack`
        (preserved untouched from the original file)."""
        path = getattr(self, "canonical_mask_path", None)
        if not path or self.masks is None:
            return
        keys = getattr(self, "_canonical_original_keys", None) or ("labels",)
        try:
            save = {}
            # Read the original to preserve any non-mask sidecars
            # (e.g. fusion_source_stack) without rebuilding them.
            try:
                _orig = np.load(path)
            except Exception:
                _orig = None
            for k in keys:
                if k == "labels":
                    save[k] = self.masks.astype(np.int32)
                elif k == "masks":
                    # Regenerate the foreground union from the
                    # (possibly edited) label stack.
                    save[k] = (self.masks > 0).astype(bool)
                elif _orig is not None and k in _orig.files:
                    save[k] = _orig[k]
            # If we don't have any key (shouldn't happen), fall back to
            # writing just `labels`.
            if not save:
                save["labels"] = self.masks.astype(np.int32)
            # NB: `np.savez_compressed` auto-appends `.npz` to a
            # filename that doesn't end in `.npz`, so we MUST give the
            # tmp file a `.npz` suffix — otherwise numpy writes to
            # `<path>.tmp.npz` and the subsequent rename of `<path>.tmp`
            # fails (it never existed). Use `.tmp.npz` so the suffix
            # is already correct.
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **save)
            os.replace(tmp, path)
        except Exception as e:
            # Best-effort cleanup of the tmp file before reporting
            try:
                if os.path.exists(path + ".tmp.npz"):
                    os.remove(path + ".tmp.npz")
            except OSError:
                pass
            QMessageBox.critical(self, "Save failed",
                                  f"Couldn't overwrite {path}:\n{e}")
            return
        n_cells = int(self.masks.max())
        self._dirty_frames.clear()
        # Loud, persistent status — bold green for 10 s so the user
        # actually sees the confirmation during active editing.
        self.status.showMessage(
            f"✓ Saved {n_cells} cells → {os.path.basename(path)}  "
            f"({os.path.getsize(path) // 1024} KB)",
            10000)
        try:
            self.status.setStyleSheet(
                "QStatusBar { color: white; "
                "background-color: #2a8a3e; font-weight: bold; }")
            # Reset styling after the message expires so subsequent
            # neutral messages render normally.
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(
                10000,
                lambda: self.status.setStyleSheet(""))
        except Exception:
            pass
        # Console print so reviewer session logs (review_session.log)
        # carry the audit trail too.
        print(f"[mask_editor] saved {n_cells} cells → {path}",
              flush=True)

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

    def _on_eraser_size(self, val):
        self.eraser_size = val

    def _fit_view(self):
        self.canvas.resetTransform()
        self.canvas.setSceneRect(self.canvas.pixmap_item.boundingRect())
        self.canvas.fitInView(
            self.canvas.pixmap_item.boundingRect(),
            Qt.KeepAspectRatio)

    def _on_opacity(self, val):
        self.mask_opacity = val / 100.0
        self._redraw()

    def _on_warn_toggle(self, checked):
        self._warn_on_unsaved = checked

    def _on_toggle_ids(self, checked):
        self.show_ids = checked
        self._redraw()

    def _on_frame(self, idx):
        prev = self.current_frame
        # Confirm-before-navigate when leaving a dirty target frame.
        # Only triggered for TARGET frames so casual scrubbing through
        # non-target frames stays fast. Gated by the user toggle —
        # off by default? No, on by default; user can disable it.
        if (self._warn_on_unsaved
                and prev != idx and prev in self._dirty_frames
                and prev in self._current_target_frames()
                and not getattr(self, "_suppress_dirty_check", False)):
            action = self._prompt_unsaved(prev)
            if action == "cancel":
                # Roll back the slider without re-triggering this guard
                self._suppress_dirty_check = True
                self.frame_slider.setValue(prev)
                self._suppress_dirty_check = False
                return
            if action == "save":
                self._save_gt_for(prev)
            elif action == "discard":
                self._dirty_frames.discard(prev)
        self.current_frame = idx
        self.frame_label.setText(f"{idx} / {self.frame_slider.maximum()}")
        self._refresh_gt_progress()
        self._redraw()

    def _prompt_unsaved(self, frame_idx):
        """Show a confirm dialog. Returns 'save' / 'discard' / 'cancel'."""
        from PyQt5.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved GT changes")
        box.setText(
            f"Frame {frame_idx} has unsaved label edits.\n"
            f"Save as GT before leaving?")
        save_btn = box.addButton("Save and continue", QMessageBox.AcceptRole)
        disc_btn = box.addButton("Discard", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(save_btn)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is save_btn:
            return "save"
        if clicked is disc_btn:
            return "discard"
        return "cancel"

    def _on_play_toggle(self, checked):
        """Start / stop inter-target playback. Steps the slider forward
        at ~5 fps until we hit the next GT target, then stops."""
        from PyQt5.QtCore import QTimer
        if not checked:
            # Stopping
            if self._play_timer is not None:
                self._play_timer.stop()
                self._play_timer = None
            self.btn_play_to_next.setText("▶ Play to next target")
            return
        targets = self._current_target_frames()
        cur = self.frame_slider.value()
        next_targets = [f for f in targets if f > cur]
        if not next_targets:
            self.btn_play_to_next.setChecked(False)
            self.status.showMessage("No target ahead of current frame")
            return
        self._play_target = min(next_targets)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(200)   # 5 fps
        self._play_timer.timeout.connect(self._play_tick)
        self._play_timer.start()
        self.btn_play_to_next.setText("⏸ Pause")
        self.status.showMessage(
            f"Playing forward to F{self._play_target} …")

    def _play_tick(self):
        """One step of inter-target playback."""
        if self.frames is None:
            self.btn_play_to_next.setChecked(False)
            return
        nxt = self.frame_slider.value() + 1
        if nxt > self.frame_slider.maximum():
            self.btn_play_to_next.setChecked(False)
            return
        # Set the next frame WITHOUT triggering the dirty-check guard
        # (playback through transient frames shouldn't pop the dialog).
        self._suppress_dirty_check = True
        self.frame_slider.setValue(nxt)
        self._suppress_dirty_check = False
        if (self._play_target is not None
                and nxt >= self._play_target):
            self.btn_play_to_next.setChecked(False)
            self.status.showMessage(
                f"Arrived at target F{self._play_target}")

    def _save_gt_for(self, frame_idx):
        """Save the mask for a specific frame index to gt_masks/."""
        if self.gt_masks_dir is None:
            self._on_save_gt_frame()
            return
        os.makedirs(self.gt_masks_dir, exist_ok=True)
        from skimage import io as skio
        out_path = os.path.join(
            self.gt_masks_dir, f"mask_F{frame_idx}.png")
        skio.imsave(out_path,
                    self.masks[frame_idx].astype(np.uint16),
                    check_contrast=False)
        self._dirty_frames.discard(frame_idx)
        self._refresh_gt_progress()

    def _on_channel_toggle(self, dic_checked):
        """Swap which channel the canvas displays. Masks are shared."""
        if self.dic_frames is None or self.fluo_frames is None:
            return
        self.active_channel = "dic" if dic_checked else "fluo"
        self.frames = (self.dic_frames if dic_checked
                       else self.fluo_frames)
        self._redraw()

    def prev_frame(self):
        # Use current_frame as the source so the navigation stays
        # consistent even if external callers set self.current_frame
        # directly (the slider's valueChanged signal usually keeps
        # them in sync, but we shouldn't rely on it).
        target = max(0, int(self.current_frame) - 1)
        if target != self.frame_slider.value():
            self.frame_slider.setValue(target)

    def next_frame(self):
        target = min(int(self.current_frame) + 1,
                     self.frame_slider.maximum())
        if target != self.frame_slider.value():
            self.frame_slider.setValue(target)

    # ------------------------------------------------------------------
    # GT-frame navigation
    # ------------------------------------------------------------------

    def _current_target_frames(self):
        """Return the active list of GT target frame indices."""
        if self.gt_frames is not None:
            return self.gt_frames
        if self.frames is None:
            return []
        return list(range(0, len(self.frames), self.gt_jump_size))

    def prev_target_frame(self):
        """Jump to the largest GT frame < current frame."""
        targets = self._current_target_frames()
        if not targets:
            return
        cur = self.frame_slider.value()
        candidates = [f for f in targets if f < cur]
        if candidates:
            self.frame_slider.setValue(max(candidates))
        else:
            # Wrap to the LAST target (so cycling through is easy)
            self.frame_slider.setValue(targets[-1])

    def next_target_frame(self):
        """Jump to the smallest GT frame > current frame."""
        targets = self._current_target_frames()
        if not targets:
            return
        cur = self.frame_slider.value()
        candidates = [f for f in targets if f > cur]
        if candidates:
            self.frame_slider.setValue(min(candidates))
        else:
            self.frame_slider.setValue(targets[0])

    def _on_jump_size(self, n):
        self.gt_jump_size = max(1, int(n))
        # If we had loaded explicit frames from file, switching the
        # stride should clear the file-driven list so the spinbox wins.
        self.gt_frames = None
        self._refresh_gt_progress()

    def _on_load_gt_frames(self):
        """Manually load a GT_FRAMES.txt file."""
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load GT_FRAMES.txt", "",
            "Text files (*.txt);;All files (*)")
        if path:
            self._load_gt_frames_from(path)

    def _load_gt_frames_from(self, path):
        """Parse a GT_FRAMES.txt file (one integer per line, '#' lines
        are comments) and replace the active target list."""
        try:
            frames = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        frames.append(int(line))
                    except ValueError:
                        pass
            if frames:
                self.gt_frames = sorted(set(frames))
                self.status.showMessage(
                    f"Loaded {len(self.gt_frames)} GT frames from "
                    f"{os.path.basename(path)}")
                self._refresh_gt_progress()
        except Exception as e:
            self.status.showMessage(f"GT load failed: {e}")

    def _auto_detect_gt_setup(self, video_path):
        """When a video is loaded, look for GT_FRAMES.txt + gt_masks/
        in the same folder and wire them up automatically."""
        folder = os.path.dirname(os.path.abspath(video_path))
        gt_file = os.path.join(folder, "GT_FRAMES.txt")
        if os.path.exists(gt_file):
            self._load_gt_frames_from(gt_file)
        masks_dir = os.path.join(folder, "gt_masks")
        if os.path.isdir(masks_dir):
            self.gt_masks_dir = masks_dir
        else:
            self.gt_masks_dir = None
        self._refresh_gt_progress()

    def _refresh_gt_progress(self):
        """Update the 'Labeled: 3/10' label by scanning gt_masks/, and
        push the target + labeled state to the slider for tick marks."""
        targets = self._current_target_frames()
        labeled_set = set()
        if self.gt_masks_dir and os.path.isdir(self.gt_masks_dir):
            existing = set(os.listdir(self.gt_masks_dir))
            for fi in targets:
                if (f"mask_F{fi}.png" in existing
                        or f"mask_F{fi:03d}.png" in existing):
                    labeled_set.add(fi)
        # Slider tick marks
        if hasattr(self.frame_slider, "set_target_frames"):
            self.frame_slider.set_target_frames(targets)
            self.frame_slider.set_labeled_frames(labeled_set)
        if not targets:
            self.gt_progress_label.setText("Labeled: – / –")
            return
        # Current-frame status indicator
        cur = self.frame_slider.value() if self.frames is not None else None
        cur_status = ""
        if cur in targets:
            cur_status = " ✓" if cur in labeled_set else " ●"
        self.gt_progress_label.setText(
            f"Labeled: {len(labeled_set)} / {len(targets)}{cur_status}")

    def _on_save_gt_frame(self):
        """Save the current frame's mask into gt_masks/ as
        mask_F<idx>.png. Creates the folder if missing."""
        if self.frames is None or self.masks is None:
            return
        fi = self.frame_slider.value()
        if not self.gt_masks_dir:
            # Default to a sibling folder next to the loaded video.
            from PyQt5.QtWidgets import QFileDialog
            base = os.path.dirname(os.path.abspath(
                getattr(self, "video_path", "")))
            default = os.path.join(base, "gt_masks") if base else ""
            chosen = QFileDialog.getExistingDirectory(
                self, "Choose gt_masks folder", default)
            if not chosen:
                return
            self.gt_masks_dir = chosen
        os.makedirs(self.gt_masks_dir, exist_ok=True)
        from skimage import io as skio
        out_path = os.path.join(
            self.gt_masks_dir, f"mask_F{fi}.png")
        skio.imsave(out_path,
                    self.masks[fi].astype(np.uint16),
                    check_contrast=False)
        self._dirty_frames.discard(fi)
        self.status.showMessage(
            f"Saved GT for F{fi} → {os.path.basename(out_path)}")
        self._refresh_gt_progress()

    def _on_save_all_gt(self):
        """Save every frame that has any label content into gt_masks/.

        Empty frames (no cells painted) are skipped. The save folder is
        prompted once if it isn't already known."""
        if self.frames is None or self.masks is None:
            return
        if not self.gt_masks_dir:
            from PyQt5.QtWidgets import QFileDialog
            base = os.path.dirname(os.path.abspath(
                getattr(self, "video_path", "")))
            default = os.path.join(base, "gt_masks") if base else ""
            chosen = QFileDialog.getExistingDirectory(
                self, "Choose gt_masks folder", default)
            if not chosen:
                return
            self.gt_masks_dir = chosen
        os.makedirs(self.gt_masks_dir, exist_ok=True)
        from skimage import io as skio
        n_saved = 0
        for fi in range(len(self.masks)):
            if not self.masks[fi].any():
                continue
            out_path = os.path.join(
                self.gt_masks_dir, f"mask_F{fi}.png")
            skio.imsave(out_path,
                        self.masks[fi].astype(np.uint16),
                        check_contrast=False)
            self._dirty_frames.discard(fi)
            n_saved += 1
        self.status.showMessage(
            f"Saved GT for {n_saved} frames → {self.gt_masks_dir}")
        self._refresh_gt_progress()

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
            # Mark this frame as having unsaved edits
            self._dirty_frames.add(idx)
        self._stroke_snapshot = None
        # Stroke finished — bring contours (and cell-ID text) back.
        self._redraw()

    def paint_at(self, x, y, tool):
        if self.masks is None:
            return
        idx = self.current_frame
        m = self.masks[idx]
        h, w = m.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        size = self.eraser_size if tool == "eraser" else self.brush_size
        r = max(1, size // 2)
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
        # Mid-stroke: skip per-cell contour traces (and ID text) to keep
        # the brush responsive on frames with many labels. end_stroke
        # triggers a full redraw with contours on mouse release.
        self._redraw(draft=True)

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

    def _on_sam2_settings(self):
        """Menu handler: open the SAM2 Options dialog and update
        self.sam2_settings in place if the user accepts."""
        from gui.mask_editor_sam2_settings import SAM2SettingsDialog
        if SAM2SettingsDialog.show(self, self.sam2_settings):
            self.status.showMessage(
                "SAM2 settings updated — next click uses new values",
                5000)

    def _on_overlay_settings(self):
        """Menu handler: open the Overlay Options dialog (timestamp +
        scale bar) and refresh the canvas if the user accepts."""
        from gui.overlays import OverlaySettingsDialog
        if OverlaySettingsDialog.show(self, self.overlay_settings):
            self._redraw()
            self.status.showMessage("Overlay settings updated", 4000)

    def run_sam2_box_at(self, x0, y0, x1, y1):
        """SAM2 box-drag tool handler. Detect cell within the box and
        add to current frame with self.active_cell as the ID. The
        backend rejects tiny / huge boxes (accidental drags) and
        masks that leak beyond box_area × 1.3.

        Wrapped in try/except so any future Python error inside SAM2
        or the post-process surfaces as a status-bar message instead
        of aborting Qt (PyQt's default behaviour on unhandled
        exceptions inside event-handler call chains is qFatal+abort).
        """
        try:
            self._run_sam2_box_impl(x0, y0, x1, y1)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status.showMessage(
                f"SAM2 box error: {e!r} — see stderr for traceback",
                12000)

    def _run_sam2_box_impl(self, x0, y0, x1, y1):
        if self.masks is None or self.frames is None:
            self.status.showMessage("Load a recording first")
            return
        from gui.mask_editor_sam2_point import (
            predict_at_box, apply_to_labels, id_exists_in_frame)
        target_id = int(self.active_cell)
        fi = self.current_frame
        if id_exists_in_frame(self.masks, fi, target_id):
            self.status.showMessage(
                f"Cell ID {target_id} already present on frame {fi} "
                f"— pick a different ID or delete the existing first",
                8000)
            return
        # Note: we deliberately do NOT call QApplication.processEvents()
        # here — that's a known PyQt footgun inside a mouse-event call
        # chain (reentrancy → reliable SIGABRT). The "running…" status
        # message won't repaint until SAM2 returns, which is fine: the
        # inference is fast enough (100-300 ms on the warm path) that
        # the brief freeze is not jarring.
        self.status.showMessage(
            f"SAM2 box [{x0},{y0}–{x1},{y1}] for ID {target_id}…")
        s = self.sam2_settings
        fluo = (self.fluo_frames[fi]
                if (s.get("use_fluo", True)
                    and getattr(self, "fluo_frames", None) is not None)
                else None)
        result = predict_at_box(
            self.frames[fi], x0, y0, x1, y1,
            fluo_image=fluo,
            fluo_min_max=s.get("fluo_min_max", 30),
            pad_px=s["pad_px"],
            min_box_area_px=s["min_box_area_px"],
            min_area_px=s["min_area_px"],
            max_area_cap_px=s["max_area_cap_px"],
            smooth_radius=s["smooth_radius"],
            keep_largest=s["keep_largest"],
            fill_holes=s["fill_holes"],
            constrain_to_box=s["constrain_to_box"],
            box_expand_frac=s["box_expand_pct"] / 100.0,
            use_tta=s["use_tta"])
        if not result.ok:
            self.status.showMessage(f"SAM2 box: {result.message}", 8000)
            return
        snapshot = apply_to_labels(self.masks, fi, result, target_id)
        self.undo_stacks.setdefault(fi, []).append(snapshot)
        if len(self.undo_stacks[fi]) > self.max_undo:
            self.undo_stacks[fi].pop(0)
        self._dirty_frames.add(fi)
        self._redraw()
        self.status.showMessage(
            f"SAM2 box: added cell {target_id} on frame {fi}  "
            f"({result.message})", 8000)

    def run_sam2_at(self, x, y):
        """SAM2 point-click tool handler. Detect cell at (x, y) and
        add to current frame with self.active_cell as the ID.

        Wrapped in try/except — see _run_sam2_box_impl for rationale.
        """
        try:
            self._run_sam2_point_impl(x, y)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status.showMessage(
                f"SAM2 error: {e!r} — see stderr for traceback",
                12000)

    def _run_sam2_point_impl(self, x, y):
        if self.masks is None or self.frames is None:
            self.status.showMessage("Load a recording first")
            return
        from gui.mask_editor_sam2_point import (
            predict_at_point, apply_to_labels, id_exists_in_frame)
        target_id = int(self.active_cell)
        fi = self.current_frame
        if id_exists_in_frame(self.masks, fi, target_id):
            self.status.showMessage(
                f"Cell ID {target_id} already present on frame {fi} "
                f"— pick a different ID or delete the existing first",
                8000)
            return
        # No QApplication.processEvents() — see note in
        # _run_sam2_box_impl (reentrancy footgun inside mouse-event
        # call chain).
        self.status.showMessage(
            f"SAM2 running at ({x}, {y}) for ID {target_id}…")
        s = self.sam2_settings
        fluo = (self.fluo_frames[fi]
                if (s.get("use_fluo", True)
                    and getattr(self, "fluo_frames", None) is not None)
                else None)
        result = predict_at_point(
            self.frames[fi], x, y,
            fluo_image=fluo,
            fluo_min_max=s.get("fluo_min_max", 30),
            crop_size=s["crop_size"],
            min_area_px=s["min_area_px"],
            max_area_px=s["max_area_cap_px"],
            smooth_radius=s["smooth_radius"],
            keep_largest=s["keep_largest"],
            fill_holes=s["fill_holes"],
            use_tta=s["use_tta"])
        if not result.ok:
            self.status.showMessage(f"SAM2: {result.message}", 8000)
            return
        snapshot = apply_to_labels(self.masks, fi, result, target_id)
        self.undo_stacks.setdefault(fi, []).append(snapshot)
        if len(self.undo_stacks[fi]) > self.max_undo:
            self.undo_stacks[fi].pop(0)
        self._dirty_frames.add(fi)
        self._redraw()
        self.status.showMessage(
            f"SAM2: added cell {target_id} on frame {fi}  "
            f"({result.message})", 8000)

    def delete_cell_track(self, cell_id):
        """Remove the given cell ID from every frame it appears in.
        Pushes per-frame undo entries so each affected frame can be
        reversed with Ctrl+Z."""
        if self.masks is None or cell_id <= 0:
            return
        affected = []
        for i in range(self.masks.shape[0]):
            if (self.masks[i] == cell_id).any():
                affected.append(i)
                self.undo_stacks.setdefault(i, []).append(
                    self.masks[i].copy())
                if len(self.undo_stacks[i]) > self.max_undo:
                    self.undo_stacks[i].pop(0)
                self.redo_stacks.pop(i, None)
                self._dirty_frames.add(i)
        if not affected:
            self.status.showMessage(
                f"Cell {cell_id} not present in any frame")
            return
        self.masks[self.masks == cell_id] = 0
        self.status.showMessage(
            f"Deleted cell {cell_id} from {len(affected)} frame(s)",
            5000)
        print(f"[mask_editor] delete_cell_track: cell {cell_id} "
              f"from {len(affected)} frames", flush=True)
        self._redraw()

    def _on_delete_by_id(self):
        """Prompt for a cell ID and delete it from every frame."""
        if self.masks is None:
            return
        max_id = int(self.masks.max())
        if max_id == 0:
            QMessageBox.information(self, "No cells",
                                      "There are no cells in this stack.")
            return
        cid, ok = QInputDialog.getInt(
            self, "Delete cell ID",
            f"Cell ID to remove from ALL frames (1 - {max_id}):",
            value=1, min=1, max=max_id, step=1)
        if not ok:
            return
        n_frames = int((self.masks == cid).any(axis=(1, 2)).sum())
        if n_frames == 0:
            QMessageBox.information(
                self, "Not present",
                f"Cell {cid} does not appear in any frame.")
            return
        ans = QMessageBox.question(
            self, "Confirm",
            f"Remove cell {cid} from all {n_frames} frame(s) it appears in?",
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        self.delete_cell_track(cid)

    def _on_filter_cells(self):
        """Open the bulk Filter Cells dialog."""
        if self.masks is None:
            QMessageBox.information(self, "No masks",
                                      "Load a recording with masks first.")
            return
        FilterCellsDialog(self).exec_()

    def _on_trim_edges(self):
        """Open the Trim Edges dialog."""
        if self.masks is None:
            QMessageBox.information(self, "No masks",
                                      "Load a recording with masks first.")
            return
        from gui.mask_editor_trim_edges import TrimEdgesDialog
        TrimEdgesDialog(self).exec_()

    def delete_cell(self, x, y):
        """Delete the entire cell whose connected component is under
        the cursor — clears it from the CURRENT frame in one click.
        For wholesale removal of a track across all frames, see
        `delete_cell_track` (Shift+Click)."""
        if self.masks is None:
            return
        idx = self.current_frame
        m = self.masks[idx]
        h, w = m.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        label = int(m[y, x])
        if label == 0:
            self.status.showMessage("No cell under cursor to delete")
            return
        # Find the connected component holding this pixel
        from scipy import ndimage
        labeled, _ = ndimage.label(m == label)
        comp_id = labeled[y, x]
        if comp_id == 0:
            return
        component = labeled == comp_id
        # Push undo for THIS frame only (single-frame action)
        self.undo_stacks.setdefault(idx, []).append(m.copy())
        if len(self.undo_stacks[idx]) > self.max_undo:
            self.undo_stacks[idx].pop(0)
        self.redo_stacks.pop(idx, None)
        m[component] = 0
        self._dirty_frames.add(idx)
        n_px = int(component.sum())
        self.status.showMessage(
            f"Deleted cell {label} from frame {idx} ({n_px:,} px)", 4000)
        self._redraw()

    def relabel_cell(self, x, y):
        """Change the label of the connected component under the cursor."""
        if self.masks is None:
            return
        idx = self.current_frame
        m = self.masks[idx]
        h, w = m.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        old_label = m[y, x]
        if old_label == 0:
            self.status.showMessage("No cell under cursor to relabel")
            return
        if old_label == self.active_cell:
            self.status.showMessage(
                f"Cell already has label {self.active_cell}")
            return
        from scipy import ndimage
        binary = m == old_label
        labeled, n_components = ndimage.label(binary)
        component_id = labeled[y, x]
        if component_id == 0:
            return
        component = labeled == component_id
        self.undo_stacks.setdefault(idx, []).append(m.copy())
        if len(self.undo_stacks[idx]) > self.max_undo:
            self.undo_stacks[idx].pop(0)
        self.redo_stacks.setdefault(idx, []).clear()
        m[component] = self.active_cell
        n_px = int(component.sum())
        self.status.showMessage(
            f"Relabelled cell {old_label} → {self.active_cell} "
            f"({n_px} px, 1 of {n_components} component"
            f"{'s' if n_components > 1 else ''})")
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
        """Set the active cell ID to the lowest unused ID across the
        whole recording (not just the current frame). Reuses IDs left
        free by deleted tracks. Caps at the spinbox max (30)."""
        if self.masks is None:
            return
        used = set(int(v) for v in np.unique(self.masks))
        used.discard(0)
        for cid in range(1, 31):
            if cid not in used:
                self.cell_spin.setValue(cid)
                self.status.showMessage(
                    f"New cell ID: {cid}  ({len(used)} IDs in use)",
                    5000)
                return
        # All 30 IDs used — keep current spinbox value, warn user
        self.status.showMessage(
            "All cell IDs 1–30 are in use across this recording; "
            "delete a track first or pick an ID to replace.",
            8000)

    def _update_cell_color_label(self):
        r, g, b = cell_color(self.active_cell)
        self.cell_color_label.setStyleSheet(
            f"color: rgb({r},{g},{b}); font-size: 18px; font-weight: bold;")
        self.cell_color_label.setText(f" ● Cell {self.active_cell}")

    def _redraw(self, polygon_preview=None, draft=False):
        if self.frames is None:
            return
        idx = self.current_frame
        img = self.frames[idx]
        h, w = img.shape

        if self.masks is not None and self.masks[idx].any():
            rgb = render_label_overlay(
                img, self.masks[idx], opacity=self.mask_opacity,
                active_cell=self.active_cell,
                polygon_preview=polygon_preview,
                show_ids=self.show_ids,
                draw_contours=not draft)
        else:
            rgb = np.stack([img, img, img], axis=-1).copy()
            if polygon_preview and len(polygon_preview) >= 2:
                pts = np.array(polygon_preview, dtype=np.int32)
                cv2.polylines(rgb, [pts], False, (255, 255, 255), 1)

        # Paint timestamp + scale bar overlays directly onto rgb.
        # No-op when both overlay toggles are off in self.overlay_settings.
        try:
            from gui.overlays import draw_overlays
            draw_overlays(rgb, idx, self.um_per_px, self.dt_min,
                          self.overlay_settings)
        except Exception:
            # Never let an overlay failure break the editor render.
            pass

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
