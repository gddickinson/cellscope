"""Image viewer with brightness/contrast, pan/zoom, frame slider, mask overlay."""
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton,
    QCheckBox,
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
    """Display DIC frames with mask overlay, B/C controls, pan/zoom."""

    frame_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.frames = None
        self.masks = None
        self.current_frame = 0
        self.brightness = 0.0
        self.contrast = 1.0
        self.mask_opacity = 0.4
        self.show_mask = True
        self.show_contour = True
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

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Bright:"))
        self.bright_slider = QSlider(Qt.Horizontal)
        self.bright_slider.setRange(-100, 100)
        self.bright_slider.setValue(0)
        self.bright_slider.setMaximumWidth(110)
        self.bright_slider.valueChanged.connect(self._on_bc_changed)
        ctrl_row.addWidget(self.bright_slider)

        ctrl_row.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setMaximumWidth(110)
        self.contrast_slider.valueChanged.connect(self._on_bc_changed)
        ctrl_row.addWidget(self.contrast_slider)

        btn_auto = QPushButton("Auto B/C")
        btn_auto.setToolTip("Auto brightness/contrast (p1/p99 stretch)")
        btn_auto.clicked.connect(self._auto_bc)
        ctrl_row.addWidget(btn_auto)

        btn_reset_bc = QPushButton("Reset B/C")
        btn_reset_bc.setToolTip("Reset brightness/contrast to defaults")
        btn_reset_bc.clicked.connect(self._reset_bc)
        ctrl_row.addWidget(btn_reset_bc)

        ctrl_row.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(40)
        self.opacity_slider.setMaximumWidth(80)
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        ctrl_row.addWidget(self.opacity_slider)

        btn_zm = QPushButton(" - ")
        btn_zm.setToolTip("Zoom out")
        btn_zm.clicked.connect(self._zoom_out)
        ctrl_row.addWidget(btn_zm)
        btn_zp = QPushButton(" + ")
        btn_zp.setToolTip("Zoom in")
        btn_zp.clicked.connect(self._zoom_in)
        ctrl_row.addWidget(btn_zp)
        btn_fit = QPushButton("Fit View")
        btn_fit.setToolTip("Reset zoom and center image")
        btn_fit.clicked.connect(self._zoom_fit)
        ctrl_row.addWidget(btn_fit)

        self.chk_mask = QCheckBox("Mask")
        self.chk_mask.setChecked(True)
        self.chk_mask.toggled.connect(self._on_toggle_mask)
        ctrl_row.addWidget(self.chk_mask)
        self.chk_contour = QCheckBox("Contour")
        self.chk_contour.setChecked(True)
        self.chk_contour.toggled.connect(self._on_toggle_contour)
        ctrl_row.addWidget(self.chk_contour)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

    def _full_extent(self):
        if self.frames is None:
            return (0, 100), (100, 0)
        H, W = self.frames[0].shape[:2]
        return (0, W), (H, 0)

    def set_data(self, frames, masks=None):
        self.frames = frames
        self.masks = masks
        self.current_frame = 0
        self._xlim, self._ylim = self._full_extent()
        n = len(frames) if frames is not None else 0
        self.frame_slider.setEnabled(n > 0)
        self.frame_slider.setRange(0, max(0, n - 1))
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"0 / {max(0, n - 1)}")
        self._redraw()

    def update_masks(self, masks):
        self.masks = masks
        self._redraw()

    def _apply_bc(self, frame):
        f = frame.astype(np.float32)
        f = f * self.contrast + self.brightness
        return np.clip(f, 0, 255).astype(np.uint8)

    def _render_frame(self, idx):
        """Build the display image (RGB array) for frame idx."""
        import cv2
        img = self._apply_bc(self.frames[idx])
        has_mask = (self.masks is not None and self.masks[idx].any())
        if not has_mask:
            return None
        is_multi = self.masks.dtype != bool and self.masks[idx].max() > 1
        rgb = np.stack([img, img, img], axis=-1).astype(np.float32)

        if self.show_mask:
            if is_multi:
                from gui.mask_editor_multicell import cell_color
                for lab in range(1, int(self.masks[idx].max()) + 1):
                    m = self.masks[idx] == lab
                    if not m.any():
                        continue
                    c = np.array(cell_color(lab), dtype=np.float32)
                    alpha = self.mask_opacity
                    rgb[m] = rgb[m] * (1 - alpha) + c * alpha
            else:
                m = self.masks[idx] > 0
                alpha = self.mask_opacity
                rgb[m] = rgb[m] * (1 - alpha) + np.array([0, 255, 0]) * alpha

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        if self.show_contour:
            if is_multi:
                from gui.mask_editor_multicell import cell_color
                for lab in range(1, int(self.masks[idx].max()) + 1):
                    m = self.masks[idx] == lab
                    if not m.any():
                        continue
                    contours, _ = cv2.findContours(
                        m.astype(np.uint8), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(rgb, contours, -1,
                                    cell_color(lab), 1)
            else:
                m = (self.masks[idx] > 0).astype(np.uint8)
                contours, _ = cv2.findContours(
                    m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)

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
        if self._xlim and self._ylim:
            self.ax.set_xlim(self._xlim)
            self.ax.set_ylim(self._ylim)
        if self._roi_selector is not None:
            self._roi_selector.draw_on_axes(self.ax)
        self.canvas.draw_idle()

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
