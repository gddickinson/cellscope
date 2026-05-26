"""Image viewer with brightness/contrast, pan/zoom, frame slider, mask overlay."""
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton,
    QCheckBox, QRadioButton, QButtonGroup, QFrame,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtGui import QPainter, QColor, QPen


class FrameNavigatorBar(QWidget):
    """Color-coded bar showing detection status per frame.

    Green = cell detected, red = no cell, orange = gap-filled,
    white outline = current frame.
    """

    def __init__(self):
        super().__init__()
        self.n_frames = 0
        self.frame_status = None    # array: 0=none, 1=detected, 2=gap-filled
        self.current_frame = 0
        self.setFixedHeight(14)
        self.setToolTip("Green=detected  Orange=gap-filled  Red=missed")

    def set_status(self, masks, missed_frames=None):
        if masks is None:
            self.frame_status = None
            self.update()
            return
        n = len(masks)
        self.n_frames = n
        self.frame_status = np.zeros(n, dtype=int)
        missed = set(missed_frames or [])
        for i in range(n):
            has_cell = masks[i].any() if masks[i].ndim == 2 else False
            if has_cell:
                self.frame_status[i] = 2 if i in missed else 1
            else:
                self.frame_status[i] = 0
        self.update()

    def set_current(self, idx):
        self.current_frame = idx
        self.update()

    def clear(self):
        self.frame_status = None
        self.update()

    def paintEvent(self, event):
        if self.frame_status is None or self.n_frames == 0:
            return
        p = QPainter(self)
        w = self.width()
        h = self.height()
        n = self.n_frames
        colors = {0: QColor(220, 60, 60), 1: QColor(80, 200, 80),
                  2: QColor(255, 165, 0)}
        bar_w = max(1, w / n)
        for i in range(n):
            x = int(i * w / n)
            x1 = int((i + 1) * w / n)
            p.fillRect(x, 0, x1 - x, h, colors.get(self.frame_status[i],
                                                      QColor(100, 100, 100)))
        # Current frame marker
        cx = int(self.current_frame * w / n)
        cw = max(2, int(bar_w))
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawRect(cx, 0, cw, h - 1)
        p.end()


class ImageViewer(QWidget):
    """Display image frames with mask overlay, B/C controls, pan/zoom.

    When a multichannel recording is loaded (DIC + fluorescence such
    as Cy5/SiR-actin), the viewer stores BOTH stacks and lets the user
    toggle between them with a small radio control. The active stack
    is bound to `self.frames` so the rest of the rendering code stays
    channel-agnostic.
    """

    frame_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.frames = None        # active stack (DIC by default)
        self.dic_frames = None    # DIC channel
        self.fluo_frames = None   # fluorescence channel (e.g. Cy5)
        self.active_channel = "dic"   # "dic" | "fluo"
        # Per-channel brightness/contrast — fluorescence usually needs
        # different defaults, so we remember separate settings per channel.
        self._bc_per_channel = {
            "dic":  {"brightness": 0.0, "contrast": 1.0},
            "fluo": {"brightness": 0.0, "contrast": 1.0},
        }
        self.masks = None
        self.source_stack = None  # (N, H, W) uint8 fusion-source codes
        self.color_by_source = False
        # Pre-Cy5-filter labels (cells dropped by persistence_guard).
        # Same shape as `masks`; rendered semi-transparent magenta when
        # `show_dropped` is on. Lets users assess what the filter cut.
        self.dropped_labels = None
        self.show_dropped = False
        self.current_frame = 0
        self.brightness = 0.0
        self.contrast = 1.0
        self.mask_opacity = 0.4
        self.show_mask = True
        self.show_contour = True
        self.show_ids = False
        self.show_tracks = False
        self._track_centroids = None  # (n_cells, n_frames, 2) cached
        self._xlim = None
        self._ylim = None
        self._dragging = False
        self._drag_start = None
        self._roi_selector = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.fig.patch.set_facecolor("#2b2b2b")
        self.ax = self.fig.add_axes([0.01, 0.01, 0.98, 0.98])
        self.ax.set_facecolor("#2b2b2b")
        self.ax.axis("off")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        layout.addWidget(self.canvas, stretch=1)

        self.nav_bar = FrameNavigatorBar()
        layout.addWidget(self.nav_bar)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame)
        slider_row.addWidget(self.frame_slider, stretch=1)
        self.frame_label = QLabel("0 / 0")
        slider_row.addWidget(self.frame_label)
        layout.addLayout(slider_row)

        # === Row 1: brightness/contrast/opacity + zoom buttons ===
        bc_row = QHBoxLayout()
        bc_row.setSpacing(8)

        bc_row.addWidget(QLabel("Bright:"))
        self.bright_slider = QSlider(Qt.Horizontal)
        self.bright_slider.setRange(-100, 100)
        self.bright_slider.setValue(0)
        self.bright_slider.setMinimumWidth(150)
        self.bright_slider.valueChanged.connect(self._on_bc_changed)
        bc_row.addWidget(self.bright_slider, stretch=2)

        bc_row.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setMinimumWidth(150)
        self.contrast_slider.valueChanged.connect(self._on_bc_changed)
        bc_row.addWidget(self.contrast_slider, stretch=2)

        btn_auto = QPushButton("Auto B/C")
        btn_auto.setToolTip("Auto brightness/contrast (p1/p99 stretch)")
        btn_auto.clicked.connect(self._auto_bc)
        bc_row.addWidget(btn_auto)

        btn_reset_bc = QPushButton("Reset B/C")
        btn_reset_bc.setToolTip("Reset brightness/contrast to defaults")
        btn_reset_bc.clicked.connect(self._reset_bc)
        bc_row.addWidget(btn_reset_bc)

        bc_row.addSpacing(12)
        bc_row.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(40)
        self.opacity_slider.setMinimumWidth(110)
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        bc_row.addWidget(self.opacity_slider, stretch=1)

        bc_row.addSpacing(12)
        btn_zm = QPushButton(" – ")
        btn_zm.setToolTip("Zoom out")
        btn_zm.clicked.connect(self._zoom_out)
        bc_row.addWidget(btn_zm)
        btn_zp = QPushButton(" + ")
        btn_zp.setToolTip("Zoom in")
        btn_zp.clicked.connect(self._zoom_in)
        bc_row.addWidget(btn_zp)
        btn_fit = QPushButton("Fit View")
        btn_fit.setToolTip("Reset zoom and center image")
        btn_fit.clicked.connect(self._zoom_fit)
        bc_row.addWidget(btn_fit)

        layout.addLayout(bc_row)

        # === Row 2: view toggles ===
        view_row = QHBoxLayout()
        view_row.setSpacing(10)

        self.chk_mask = QCheckBox("Mask")
        self.chk_mask.setChecked(True)
        self.chk_mask.toggled.connect(self._on_toggle_mask)
        view_row.addWidget(self.chk_mask)

        self.chk_contour = QCheckBox("Contour")
        self.chk_contour.setChecked(True)
        self.chk_contour.toggled.connect(self._on_toggle_contour)
        view_row.addWidget(self.chk_contour)

        self.chk_ids = QCheckBox("Cell IDs")
        self.chk_ids.setChecked(False)
        self.chk_ids.setToolTip(
            "Draw the cell / track ID at each cell's centroid.")
        self.chk_ids.toggled.connect(self._on_toggle_ids)
        view_row.addWidget(self.chk_ids)

        self.chk_tracks = QCheckBox("Tracks")
        self.chk_tracks.setChecked(False)
        self.chk_tracks.setToolTip(
            "Overlay the trajectory of each cell from frame 0 to the\n"
            "current frame, coloured by cell ID. Computed from the\n"
            "label stack on the fly.")
        self.chk_tracks.toggled.connect(self._on_toggle_tracks)
        view_row.addWidget(self.chk_tracks)

        self.chk_source = QCheckBox("Source ⓘ")
        self.chk_source.setToolTip(
            "Colour cells by which channel detected them:\n"
            "  red    = DIC-only (cellpose_dic found it, Cy5 didn't)\n"
            "  green  = both channels (DIC + Cy5 agreed)\n"
            "  yellow = Cy5-only (cpsam(Cy5) added; DIC missed it)\n\n"
            "Available after running detection with Cy5 fusion ON.")
        self.chk_source.setVisible(False)
        self.chk_source.toggled.connect(self._on_toggle_source)
        view_row.addWidget(self.chk_source)

        self.chk_dropped = QCheckBox("Dropped ⓘ")
        self.chk_dropped.setToolTip(
            "Overlay cells that the Cy5 persistence_guard filter\n"
            "dropped (rendered semi-transparent magenta with a\n"
            "dashed contour). Lets you assess which detections the\n"
            "filter cut so you can adjust filter parameters for\n"
            "future runs.\n\nAvailable after running detection with\n"
            "Cy5 filter ON, or after loading masks_unfiltered.npz.")
        self.chk_dropped.setVisible(False)
        self.chk_dropped.toggled.connect(self._on_toggle_dropped)
        view_row.addWidget(self.chk_dropped)

        # --- Channel toggle (shown only when fluorescence is loaded) ---
        self._channel_divider = QFrame()
        self._channel_divider.setFrameShape(QFrame.VLine)
        self._channel_divider.setFrameShadow(QFrame.Sunken)
        self._channel_divider.setVisible(False)
        view_row.addSpacing(8)
        view_row.addWidget(self._channel_divider)

        self._channel_label = QLabel("Channel:")
        self._channel_label.setVisible(False)
        view_row.addWidget(self._channel_label)

        self.radio_dic = QRadioButton("DIC")
        self.radio_dic.setChecked(True)
        self.radio_dic.setVisible(False)
        self.radio_dic.setToolTip(
            "Show the DIC (differential interference contrast) channel.")
        view_row.addWidget(self.radio_dic)

        self.radio_fluo = QRadioButton("Fluo")
        self.radio_fluo.setVisible(False)
        self.radio_fluo.setToolTip(
            "Show the fluorescence channel (e.g. Cy5 / SiR-actin).\n"
            "Each channel keeps its own brightness/contrast settings.")
        view_row.addWidget(self.radio_fluo)

        self._channel_group = QButtonGroup(self)
        self._channel_group.addButton(self.radio_dic)
        self._channel_group.addButton(self.radio_fluo)
        self.radio_dic.toggled.connect(self._on_channel_toggle)

        view_row.addStretch()
        layout.addLayout(view_row)

    def _full_extent(self):
        if self.frames is None:
            return (0, 100), (100, 0)
        H, W = self.frames[0].shape[:2]
        return (0, W), (H, 0)

    def set_data(self, frames, masks=None, fluo_frames=None):
        """Load image data into the viewer.

        Args:
            frames: (N, H, W) uint8 — the primary DIC channel (always
                shown by default).
            masks: optional (N, H, W) bool/int — cell-mask overlay.
            fluo_frames: optional (N, H, W) uint8 — fluorescence
                channel (e.g. Cy5 / SiR-actin). When given, the
                Channel toggle (DIC / Fluo) appears in the control
                row. When None, the toggle stays hidden.
        """
        self.dic_frames = frames
        self.fluo_frames = fluo_frames
        # Reset both channels' BC to defaults on new data
        self._bc_per_channel = {
            "dic":  {"brightness": 0.0, "contrast": 1.0},
            "fluo": {"brightness": 0.0, "contrast": 1.0},
        }
        # Default channel is DIC
        self.active_channel = "dic"
        self.frames = frames
        self.masks = masks
        self.current_frame = 0
        self._xlim, self._ylim = self._full_extent()
        n = len(frames) if frames is not None else 0
        self.frame_slider.setEnabled(n > 0)
        self.frame_slider.setRange(0, max(0, n - 1))
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"0 / {max(0, n - 1)}")

        # Show/hide the channel toggle based on whether fluo data is present
        has_fluo = fluo_frames is not None
        for w in (self._channel_divider, self._channel_label,
                  self.radio_dic, self.radio_fluo):
            w.setVisible(has_fluo)
        if has_fluo:
            # Reset to DIC without triggering the toggle handler
            self.radio_dic.blockSignals(True)
            self.radio_dic.setChecked(True)
            self.radio_dic.blockSignals(False)

        # Reset the B/C sliders to neutral (we just reset the stored
        # values too). Block signals so the redraw doesn't fire twice.
        for sl, val in [(self.bright_slider, 0), (self.contrast_slider, 100)]:
            sl.blockSignals(True)
            sl.setValue(val)
            sl.blockSignals(False)
        self.brightness = 0.0
        self.contrast = 1.0

        self._redraw()

    def _on_channel_toggle(self, dic_checked):
        """Swap displayed channel. dic_checked is True when DIC is
        selected; False when Fluo is selected (mutually exclusive)."""
        if self.dic_frames is None or self.fluo_frames is None:
            return
        # Remember the current channel's B/C before swapping
        self._bc_per_channel[self.active_channel] = {
            "brightness": self.brightness,
            "contrast": self.contrast,
        }
        # Swap
        self.active_channel = "dic" if dic_checked else "fluo"
        self.frames = (self.dic_frames if dic_checked
                       else self.fluo_frames)
        # Restore the new channel's B/C
        bc = self._bc_per_channel[self.active_channel]
        self.brightness = bc["brightness"]
        self.contrast = bc["contrast"]
        # Update sliders silently
        self.bright_slider.blockSignals(True)
        self.contrast_slider.blockSignals(True)
        self.bright_slider.setValue(int(self.brightness))
        self.contrast_slider.setValue(int(self.contrast * 100))
        self.bright_slider.blockSignals(False)
        self.contrast_slider.blockSignals(False)
        self._redraw()

    def update_masks(self, masks):
        self.masks = masks
        # Track centroids depend on the labels stack — invalidate so
        # the toggle recomputes them on next use.
        self._track_centroids = None
        self._redraw()

    def _compute_track_centroids(self):
        """Build (n_cells, n_frames, 2) array of (y, x) centroids from
        the labels stack. NaN where the cell is absent in that frame.
        Used by the Tracks overlay."""
        if (self.masks is None or self.masks.dtype == bool
                or self.masks.max() == 0):
            self._track_centroids = None
            return
        n_frames = len(self.masks)
        n_cells = int(self.masks.max())
        centroids = np.full((n_cells, n_frames, 2), np.nan,
                            dtype=np.float32)
        for fi in range(n_frames):
            lab_frame = self.masks[fi]
            if lab_frame.max() == 0:
                continue
            for lab in range(1, n_cells + 1):
                m = lab_frame == lab
                if not m.any():
                    continue
                ys, xs = np.where(m)
                centroids[lab - 1, fi] = (float(ys.mean()),
                                          float(xs.mean()))
        self._track_centroids = centroids

    def _apply_bc(self, frame):
        f = frame.astype(np.float32)
        f = f * self.contrast + self.brightness
        return np.clip(f, 0, 255).astype(np.uint8)

    def _render_frame(self, idx):
        """Build the display image (RGB array) for frame idx."""
        import cv2
        img = self._apply_bc(self.frames[idx])
        has_mask = (self.masks is not None and self.masks[idx].any())
        has_dropped = (self.show_dropped
                        and self.dropped_labels is not None
                        and idx < len(self.dropped_labels)
                        and self.dropped_labels[idx].any())
        if not has_mask and not has_dropped:
            return None
        is_multi = (has_mask and self.masks.dtype != bool
                     and self.masks[idx].max() > 1)
        rgb = np.stack([img, img, img], axis=-1).astype(np.float32)

        # If source colouring is on AND a source stack is loaded, use
        # red/yellow/green per-cell instead of the cell-ID palette.
        use_source = (self.color_by_source
                      and self.source_stack is not None)

        # Dropped-cell overlay — only the cells the Cy5 filter
        # rejected. Drawn BEFORE kept masks so the kept colour wins
        # when the two overlap (they shouldn't, but be defensive).
        if has_dropped:
            dropped_frame = self.dropped_labels[idx]
            # Suppress dropped pixels under a kept cell (avoids
            # double-coloured regions if the filter just shrank a
            # track without removing it).
            if has_mask:
                kept_mask = self.masks[idx] > 0
                drop_mask_any = (dropped_frame > 0) & ~kept_mask
            else:
                drop_mask_any = dropped_frame > 0
            if self.show_mask and drop_mask_any.any():
                # Magenta fill, 60% of normal opacity (dimmer than kept)
                a = self.mask_opacity * 0.6
                rgb[drop_mask_any] = (
                    rgb[drop_mask_any] * (1 - a)
                    + np.array([255, 0, 255]) * a)

        if self.show_mask and has_mask:
            if is_multi:
                from gui.mask_editor_multicell import cell_color
                for lab in range(1, int(self.masks[idx].max()) + 1):
                    m = self.masks[idx] == lab
                    if not m.any():
                        continue
                    if use_source:
                        # cv2 BGR → matplotlib RGB swap for the fill
                        bgr = self._cell_source_color(lab, idx)
                        c = np.array([bgr[2], bgr[1], bgr[0]],
                                     dtype=np.float32)
                    else:
                        c = np.array(cell_color(lab), dtype=np.float32)
                    alpha = self.mask_opacity
                    rgb[m] = rgb[m] * (1 - alpha) + c * alpha
            else:
                m = self.masks[idx] > 0
                alpha = self.mask_opacity
                rgb[m] = rgb[m] * (1 - alpha) + np.array([0, 255, 0]) * alpha

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        if self.show_contour and has_mask:
            if is_multi:
                from gui.mask_editor_multicell import cell_color
                for lab in range(1, int(self.masks[idx].max()) + 1):
                    m = self.masks[idx] == lab
                    if not m.any():
                        continue
                    contours, _ = cv2.findContours(
                        m.astype(np.uint8), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE)
                    if use_source:
                        color = self._cell_source_color(lab, idx)
                    else:
                        color = cell_color(lab)
                    cv2.drawContours(rgb, contours, -1, color, 1)
            else:
                m = (self.masks[idx] > 0).astype(np.uint8)
                contours, _ = cv2.findContours(
                    m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)

        # Dropped-cell contour — magenta, 2 px so it reads as
        # distinct from the kept-cell contour (1 px green).
        if self.show_contour and has_dropped:
            dropped_frame = self.dropped_labels[idx]
            for lab in range(1, int(dropped_frame.max()) + 1):
                m = dropped_frame == lab
                if not m.any():
                    continue
                contours, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, contours, -1, (255, 0, 255), 2)

        return rgb

    def _redraw(self):
        if self.frames is None:
            return
        self.ax.clear()
        self.ax.axis("off")
        idx = self.current_frame
        rgb = self._render_frame(idx)
        if rgb is not None:
            self.ax.imshow(rgb)
        else:
            img = self._apply_bc(self.frames[idx])
            self.ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        # Track trajectories — drawn before IDs so labels stay on top
        if (self.show_tracks and self._track_centroids is not None):
            self._draw_track_trails(idx)
        # Cell ID labels overlaid on matplotlib axes (after imshow)
        if (self.show_ids and self.masks is not None
                and self.masks.dtype != bool
                and self.masks[idx].max() > 0):
            self._draw_cell_ids(idx)
        if self._xlim and self._ylim:
            self.ax.set_xlim(self._xlim)
            self.ax.set_ylim(self._ylim)
        if self._roi_selector is not None:
            self._roi_selector.draw_on_axes(self.ax)
        self.canvas.draw_idle()

    def _draw_track_trails(self, current_idx):
        """Plot each cell's centroid trajectory from frame 0 up to
        the current frame, coloured by cell ID and using the same
        palette as the cell masks. A small filled circle marks the
        cell's current position."""
        if self._track_centroids is None:
            return
        from gui.mask_editor_multicell import cell_color
        n_cells = self._track_centroids.shape[0]
        for cell_idx in range(n_cells):
            lab = cell_idx + 1
            trail = self._track_centroids[cell_idx, :current_idx + 1]
            valid = ~np.isnan(trail[:, 0])
            if valid.sum() < 1:
                continue
            ys = trail[valid, 0]
            xs = trail[valid, 1]
            # Convert cv2 BGR colour palette to matplotlib RGB
            bgr = cell_color(lab)
            rgb_color = (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)
            if valid.sum() >= 2:
                self.ax.plot(xs, ys, "-", color=rgb_color, lw=1.8,
                             alpha=0.9, solid_capstyle="round")
            # Mark current position
            self.ax.plot(xs[-1], ys[-1], "o", color=rgb_color,
                         markersize=4, markeredgecolor="white",
                         markeredgewidth=0.5)

    def _draw_cell_ids(self, idx):
        """Annotate each cell with its label ID at the mask centroid."""
        lab_frame = self.masks[idx]
        for lab in range(1, int(lab_frame.max()) + 1):
            m = lab_frame == lab
            if not m.any():
                continue
            ys, xs = np.where(m)
            cy, cx = float(ys.mean()), float(xs.mean())
            self.ax.text(
                cx, cy, str(int(lab)),
                color="white", fontsize=9, fontweight="bold",
                ha="center", va="center",
                bbox=dict(facecolor="black", alpha=0.55,
                          edgecolor="none", pad=1.5))

    # --- Frame navigation ---
    def _on_frame(self, idx):
        self.current_frame = idx
        self.frame_label.setText(f"{idx} / {self.frame_slider.maximum()}")
        self.nav_bar.set_current(idx)
        self._redraw()
        self.frame_changed.emit(idx)

    # --- Brightness / Contrast ---
    def _on_bc_changed(self, _=None):
        self.brightness = self.bright_slider.value()
        self.contrast = self.contrast_slider.value() / 100.0
        self._redraw()

    def _auto_bc(self):
        """Set B/C sliders to stretch current frame's p1-p99 to 0-255."""
        if self.frames is None:
            return
        f = self.frames[self.current_frame].astype(np.float32)
        p1, p99 = np.percentile(f, [1, 99])
        if p99 <= p1:
            p99 = p1 + 1
        c = 255.0 / (p99 - p1)
        b = -p1 * c
        self.contrast_slider.setValue(int(round(c * 100)))
        self.bright_slider.setValue(int(round(b)))

    def _reset_bc(self):
        self.bright_slider.setValue(0)
        self.contrast_slider.setValue(100)

    def _on_opacity(self, val):
        self.mask_opacity = val / 100.0
        self._redraw()

    def _on_toggle_mask(self, checked):
        self.show_mask = checked
        self._redraw()

    def _on_toggle_contour(self, checked):
        self.show_contour = checked
        self._redraw()

    def _on_toggle_ids(self, checked):
        self.show_ids = checked
        self._redraw()

    def _on_toggle_tracks(self, checked):
        self.show_tracks = checked
        # Compute centroids on first use; cached until masks change.
        if checked and self._track_centroids is None:
            self._compute_track_centroids()
        self._redraw()

    def _on_toggle_source(self, checked):
        self.color_by_source = checked
        self._redraw()

    def _on_toggle_dropped(self, checked):
        self.show_dropped = checked
        self._redraw()

    def set_dropped_labels(self, labels):
        """Attach pre-Cy5-filter labels (N, H, W) int32 — same shape
        as the kept-cell stack. Cells present here but NOT in the
        kept stack are what the filter dropped. Enables the
        "Dropped" checkbox. Pass None to clear."""
        self.dropped_labels = labels
        has_drop = labels is not None
        self.chk_dropped.setVisible(has_drop)
        if not has_drop and self.show_dropped:
            self.show_dropped = False
            self.chk_dropped.setChecked(False)
        self._redraw()

    def set_source_stack(self, source_stack):
        """Attach the per-pixel fusion source map (N, H, W) uint8.

        Codes: 0=bg, 1=dic_only, 2=cy5_only, 3=both. Enables the
        "Source" checkbox in the viewer control row. Pass None to
        clear (hides the checkbox)."""
        self.source_stack = source_stack
        has_src = source_stack is not None
        self.chk_source.setVisible(has_src)
        if not has_src and self.color_by_source:
            self.color_by_source = False
            self.chk_source.setChecked(False)
        self._redraw()

    def _cell_source_color(self, lab, frame_idx):
        """Return (B, G, R) cv2 colour for the cell `lab` in frame
        `frame_idx`, based on its dominant pixel source. Returns the
        regular per-cell palette colour if no source map is loaded."""
        if (self.source_stack is None
                or frame_idx >= len(self.source_stack)):
            from gui.mask_editor_multicell import cell_color
            return cell_color(lab)
        mask = self.masks[frame_idx] == lab
        if not mask.any():
            from gui.mask_editor_multicell import cell_color
            return cell_color(lab)
        src_under = self.source_stack[frame_idx][mask]
        n_dic = int((src_under == 1).sum())
        n_cy5 = int((src_under == 2).sum())
        n_both = int((src_under == 3).sum())
        # OpenCV uses BGR, matplotlib RGB — keep this in BGR:
        # red:    (0, 0, 255)
        # yellow: (0, 255, 255)
        # lime:   (0, 255, 0)
        if n_both >= max(n_dic, n_cy5):
            return (0, 255, 0)        # lime / both
        if n_cy5 > n_dic:
            return (0, 255, 255)      # yellow / cy5-only
        return (0, 0, 255)            # red / dic-only

    # --- Zoom ---
    def _zoom_at(self, factor, cx=None, cy=None):
        """Zoom by factor, centered on (cx, cy) in data coords."""
        xl, xr = self._xlim
        yb, yt = self._ylim
        if cx is None:
            cx = (xl + xr) / 2
        if cy is None:
            cy = (yb + yt) / 2
        new_hw = (xr - xl) / factor
        new_hh = (yb - yt) / factor
        # Keep center at (cx, cy)
        self._xlim = (cx - new_hw / 2, cx + new_hw / 2)
        self._ylim = (cy + abs(new_hh) / 2, cy - abs(new_hh) / 2)
        self._clamp_limits()
        self._redraw()

    def _clamp_limits(self):
        """Prevent panning beyond the image boundaries."""
        if self.frames is None:
            return
        H, W = self.frames[0].shape[:2]
        xl, xr = self._xlim
        yb, yt = self._ylim
        vw = xr - xl
        vh = yb - yt
        if vw >= W:
            xl, xr = 0, W
        else:
            if xl < 0:
                xl, xr = 0, vw
            if xr > W:
                xl, xr = W - vw, W
        if abs(vh) >= H:
            yb, yt = H, 0
        else:
            if yt < 0:
                yt, yb = 0, abs(vh)
            if yb > H:
                yb, yt = H, H - abs(vh)
        self._xlim = (xl, xr)
        self._ylim = (yb, yt)

    def _zoom_in(self):
        self._zoom_at(1.4)

    def _zoom_out(self):
        self._zoom_at(1 / 1.4)
        fe = self._full_extent()
        xl, xr = self._xlim
        if (xr - xl) >= (fe[0][1] - fe[0][0]):
            self._zoom_fit()

    def _zoom_fit(self):
        self._xlim, self._ylim = self._full_extent()
        self._redraw()

    def _on_scroll(self, event):
        if self.frames is None:
            return
        factor = 1.2 if event.button == "up" else 1 / 1.2
        cx = event.xdata if event.xdata is not None else None
        cy = event.ydata if event.ydata is not None else None
        self._zoom_at(factor, cx, cy)

    # --- Pan / Drag ---
    def _on_press(self, event):
        if event.button in (2, 3) or (event.button == 1 and event.key == "control"):
            self._dragging = True
            self._drag_start = (event.x, event.y)
            self.canvas.setCursor(Qt.ClosedHandCursor)

    def _on_release(self, event):
        if self._dragging:
            self._dragging = False
            self.canvas.setCursor(Qt.ArrowCursor)

    def _on_motion(self, event):
        if not self._dragging or self._drag_start is None:
            return
        if event.x is None or event.y is None:
            return
        dx_px = event.x - self._drag_start[0]
        dy_px = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        xl, xr = self._xlim
        yb, yt = self._ylim
        # Convert pixel drag to data units
        ax_bbox = self.ax.get_window_extent()
        if ax_bbox.width == 0 or ax_bbox.height == 0:
            return
        dx_data = -dx_px * (xr - xl) / ax_bbox.width
        dy_data = dy_px * (yb - yt) / ax_bbox.height
        self._xlim = (xl + dx_data, xr + dx_data)
        self._ylim = (yb + dy_data, yt + dy_data)
        self._clamp_limits()
        self._redraw()
