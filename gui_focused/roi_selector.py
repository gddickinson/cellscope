"""ROI selector for the focused GUI image viewer.

Supports rectangle, ellipse, and polygon ROI shapes. The ROI is
stored as a binary mask (H, W) applied to every frame before
detection — pixels outside the ROI are zeroed.
"""
import numpy as np
import cv2
from matplotlib.widgets import RectangleSelector, EllipseSelector
from matplotlib.patches import Rectangle, Ellipse, Polygon
from matplotlib.lines import Line2D


class ROISelector:
    """Manages ROI selection on an ImageViewer's matplotlib axes."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.roi_mask = None
        self._shape_type = None
        self._shape_kwargs = None
        self._selector = None
        self._poly_points = []
        self._poly_line = None
        self._poly_cid_press = None
        self.active = False
        self.on_roi_drawn = None    # callback when ROI is created

    def has_roi(self):
        return self.roi_mask is not None

    def start(self, shape="rectangle"):
        """Begin interactive ROI selection."""
        self._clear_selectors()
        ax = self.viewer.ax
        if shape == "rectangle":
            self._selector = RectangleSelector(
                ax, self._on_rect_select, useblit=True,
                button=[1], minspanx=5, minspany=5,
                props=dict(edgecolor="#ffcc00", facecolor="none",
                           linewidth=2, linestyle="--"),
                interactive=True)
        elif shape == "ellipse":
            self._selector = EllipseSelector(
                ax, self._on_ellipse_select, useblit=True,
                button=[1], minspanx=5, minspany=5,
                props=dict(edgecolor="#ffcc00", facecolor="none",
                           linewidth=2, linestyle="--"),
                interactive=True)
        elif shape == "polygon":
            self._poly_points = []
            self._poly_cid_press = self.viewer.canvas.mpl_connect(
                "button_press_event", self._on_poly_click)
        self.viewer.canvas.draw_idle()

    def clear(self):
        """Remove ROI entirely."""
        self.roi_mask = None
        self._shape_type = None
        self._shape_kwargs = None
        self.active = False
        self._clear_selectors()
        self.viewer._redraw()

    def _clear_selectors(self):
        if self._selector:
            self._selector.set_active(False)
            self._selector = None
        if self._poly_cid_press:
            self.viewer.canvas.mpl_disconnect(self._poly_cid_press)
            self._poly_cid_press = None
        self._poly_points = []
        if self._poly_line:
            try:
                self._poly_line.remove()
            except ValueError:
                pass
            self._poly_line = None

    def _make_mask(self, shape_type, **kwargs):
        if self.viewer.frames is None:
            return None
        H, W = self.viewer.frames[0].shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        if shape_type == "rectangle":
            x0, y0, x1, y1 = kwargs["bounds"]
            r0, r1 = int(max(0, y0)), int(min(H, y1))
            c0, c1 = int(max(0, x0)), int(min(W, x1))
            mask[r0:r1, c0:c1] = 1
        elif shape_type == "ellipse":
            cx, cy = kwargs["center"]
            rx, ry = kwargs["radii"]
            cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)),
                        0, 0, 360, 1, -1)
        elif shape_type == "polygon":
            pts = np.array(kwargs["points"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)

    def _finish_roi(self, shape_type, **kwargs):
        """Store ROI shape, create mask, activate, and redraw."""
        self.roi_mask = self._make_mask(shape_type, **kwargs)
        self._shape_type = shape_type
        self._shape_kwargs = kwargs
        self.active = True
        self._clear_selectors()
        self.viewer._redraw()
        if self.on_roi_drawn:
            self.on_roi_drawn()

    def _on_rect_select(self, eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return
        bounds = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self._finish_roi("rectangle", bounds=bounds)

    def _on_ellipse_select(self, eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
        if rx < 3 or ry < 3:
            return
        self._finish_roi("ellipse", center=(cx, cy), radii=(rx, ry))

    def _on_poly_click(self, event):
        if event.inaxes != self.viewer.ax:
            return
        if event.button == 1:
            self._poly_points.append((event.xdata, event.ydata))
            self._draw_poly_preview()
        elif event.button == 3 and len(self._poly_points) >= 3:
            pts = [(int(x), int(y)) for x, y in self._poly_points]
            self._finish_roi("polygon", points=pts)

    def _draw_poly_preview(self):
        if self._poly_line:
            try:
                self._poly_line.remove()
            except ValueError:
                pass
            self._poly_line = None
        if len(self._poly_points) >= 2:
            xs = [p[0] for p in self._poly_points]
            ys = [p[1] for p in self._poly_points]
            self._poly_line = Line2D(xs, ys, color="#ffcc00",
                                     linewidth=2, linestyle="--",
                                     marker="o", markersize=4)
            self.viewer.ax.add_line(self._poly_line)
        self.viewer.canvas.draw_idle()

    def draw_on_axes(self, ax):
        """Re-draw the ROI overlay on the given axes. Called by the
        viewer after each _redraw() so the ROI persists across frames."""
        if not self.active or self._shape_type is None:
            return
        kw = self._shape_kwargs
        style = dict(edgecolor="#ffcc00", facecolor="none",
                     linewidth=2, linestyle="--")
        if self._shape_type == "rectangle":
            b = kw["bounds"]
            patch = Rectangle(
                (b[0], b[1]), b[2] - b[0], b[3] - b[1], **style)
        elif self._shape_type == "ellipse":
            cx, cy = kw["center"]
            rx, ry = kw["radii"]
            patch = Ellipse((cx, cy), rx * 2, ry * 2, **style)
        elif self._shape_type == "polygon":
            patch = Polygon(kw["points"], closed=True, **style)
        else:
            return
        ax.add_patch(patch)

    def apply_to_frames(self, frames):
        """Zero out pixels outside the ROI mask. Returns modified copy."""
        if self.roi_mask is None:
            return frames
        out = frames.copy()
        out[:, ~self.roi_mask] = 0
        return out
