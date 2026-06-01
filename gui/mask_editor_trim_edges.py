"""Trim-edges dialog for the mask editor.

Bulk-zeroes mask pixels within a user-defined band along one or
more image edges. Targets the common artefact pattern where a dark
vignette / black margin at an image edge gets picked up by detection
and shows as a thin sliver of mask along that edge across many
frames.

Workflow:
  Pipeline Review tab → "Trim Edges…" button → dialog
    pick edges (any combination of L/R/T/B)
    pick width (px)
    pick scope (current frame / all frames)
    pick cells (all / single ID)
    Preview → count of px / cells / frames affected
    Apply → mutate masks in place, push per-frame undo
"""
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QCheckBox, QRadioButton, QPushButton, QLabel,
    QButtonGroup,
)


def _build_edge_mask(H, W, edges, width):
    """Boolean (H, W) mask True at every pixel in the edge band."""
    em = np.zeros((H, W), dtype=bool)
    w = max(0, int(width))
    if w == 0:
        return em
    if edges.get("left"):
        em[:, :w] = True
    if edges.get("right"):
        em[:, max(0, W - w):] = True
    if edges.get("top"):
        em[:w, :] = True
    if edges.get("bottom"):
        em[max(0, H - w):, :] = True
    return em


def _compute_targets(masks, edges, width, frames, cell_id):
    """For each frame in `frames`, return (frame_idx, bool_mask) of
    pixels that WOULD be zeroed. Used by both preview and apply."""
    N, H, W = masks.shape
    em = _build_edge_mask(H, W, edges, width)
    if not em.any():
        return []
    out = []
    for i in frames:
        f = masks[i]
        if cell_id is None:
            target = (f > 0) & em
        else:
            target = (f == int(cell_id)) & em
        if target.any():
            out.append((i, target))
    return out


class TrimEdgesDialog(QDialog):
    """Modal dialog: select edges + width + scope + cells, preview,
    apply. Mutates editor.masks in place, pushes per-frame undo, and
    marks affected frames dirty."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.setWindowTitle("Trim Edge Mask Pixels")
        self.setMinimumWidth(420)
        root = QVBoxLayout(self)

        # ── Edges ──
        gb_e = QGroupBox("Edges to trim (check any combination)")
        ge = QFormLayout(gb_e)
        self.cb_left = QCheckBox("Left")
        self.cb_left.setChecked(True)
        self.cb_right = QCheckBox("Right")
        self.cb_top = QCheckBox("Top")
        self.cb_bottom = QCheckBox("Bottom")
        ge.addRow(self.cb_left, self.cb_top)
        ge.addRow(self.cb_right, self.cb_bottom)
        self.sp_width = QSpinBox()
        self.sp_width.setRange(1, 1024)
        self.sp_width.setValue(30)
        self.sp_width.setSuffix(" px")
        self.sp_width.setToolTip(
            "Width of the edge band (applied to every checked edge).")
        ge.addRow("Width:", self.sp_width)
        root.addWidget(gb_e)

        # ── Scope ──
        gb_s = QGroupBox("Frame scope")
        gs = QVBoxLayout(gb_s)
        self.rb_all_frames = QRadioButton("All frames")
        self.rb_all_frames.setChecked(True)
        self.rb_cur_frame = QRadioButton("Current frame only")
        scope_group = QButtonGroup(self)
        scope_group.addButton(self.rb_all_frames)
        scope_group.addButton(self.rb_cur_frame)
        gs.addWidget(self.rb_all_frames)
        gs.addWidget(self.rb_cur_frame)
        root.addWidget(gb_s)

        # ── Cells ──
        gb_c = QGroupBox("Affected cells")
        gc = QHBoxLayout(gb_c)
        self.rb_all_cells = QRadioButton("All cells")
        self.rb_all_cells.setChecked(True)
        self.rb_one_cell = QRadioButton("Cell ID")
        self.sp_cell_id = QSpinBox()
        self.sp_cell_id.setRange(1, 30)
        self.sp_cell_id.setValue(int(editor.active_cell))
        cell_group = QButtonGroup(self)
        cell_group.addButton(self.rb_all_cells)
        cell_group.addButton(self.rb_one_cell)
        gc.addWidget(self.rb_all_cells)
        gc.addWidget(self.rb_one_cell)
        gc.addWidget(self.sp_cell_id)
        gc.addStretch()
        root.addWidget(gb_c)

        # ── Preview readout ──
        self.lbl_preview = QLabel(
            "(click Preview to count affected pixels)")
        self.lbl_preview.setStyleSheet("color: #444;")
        root.addWidget(self.lbl_preview)

        # ── Buttons ──
        btn_row = QHBoxLayout()
        btn_preview = QPushButton("Preview")
        btn_apply = QPushButton("Apply")
        btn_apply.setDefault(True)
        btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(btn_preview)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_apply)
        root.addLayout(btn_row)

        btn_preview.clicked.connect(self._on_preview)
        btn_cancel.clicked.connect(self.reject)
        btn_apply.clicked.connect(self._on_apply)

    def _read(self):
        edges = {
            "left":   self.cb_left.isChecked(),
            "right":  self.cb_right.isChecked(),
            "top":    self.cb_top.isChecked(),
            "bottom": self.cb_bottom.isChecked(),
        }
        width = int(self.sp_width.value())
        if self.rb_cur_frame.isChecked():
            frames = [int(self.editor.current_frame)]
        else:
            frames = list(range(int(self.editor.masks.shape[0])))
        cell_id = (int(self.sp_cell_id.value())
                   if self.rb_one_cell.isChecked() else None)
        return edges, width, frames, cell_id

    def _on_preview(self):
        if self.editor.masks is None:
            self.lbl_preview.setText("(no recording loaded)")
            return
        edges, width, frames, cell_id = self._read()
        if not any(edges.values()):
            self.lbl_preview.setText(
                "Pick at least one edge to trim.")
            return
        targets = _compute_targets(
            self.editor.masks, edges, width, frames, cell_id)
        if not targets:
            self.lbl_preview.setText("No mask pixels in the edge band.")
            return
        total_px = sum(int(t.sum()) for _, t in targets)
        cells = set()
        for i, t in targets:
            cells.update(int(c) for c in
                         np.unique(self.editor.masks[i][t]) if c > 0)
        self.lbl_preview.setText(
            f"Will remove {total_px:,} px across {len(cells)} cell(s) "
            f"on {len(targets)} frame(s).")

    def _on_apply(self):
        if self.editor.masks is None:
            self.reject(); return
        edges, width, frames, cell_id = self._read()
        if not any(edges.values()):
            self.lbl_preview.setText(
                "Pick at least one edge to trim.")
            return
        targets = _compute_targets(
            self.editor.masks, edges, width, frames, cell_id)
        if not targets:
            self.lbl_preview.setText("No mask pixels in the edge band.")
            return
        # Per-frame undo snapshot, then mutate.
        total_px = 0
        for i, target_mask in targets:
            self.editor.undo_stacks.setdefault(i, []).append(
                self.editor.masks[i].copy())
            if (len(self.editor.undo_stacks[i])
                    > self.editor.max_undo):
                self.editor.undo_stacks[i].pop(0)
            self.editor.redo_stacks.pop(i, None)
            self.editor.masks[i][target_mask] = 0
            self.editor._dirty_frames.add(i)
            total_px += int(target_mask.sum())
        self.editor._redraw()
        which = ",".join(k for k, v in edges.items() if v)
        msg = (f"Trimmed {total_px:,} px from {which} edge(s) "
               f"(width {width} px) across {len(targets)} frame(s)")
        self.editor.status.showMessage(msg, 8000)
        print(f"[mask_editor] {msg}", flush=True)
        self.accept()
