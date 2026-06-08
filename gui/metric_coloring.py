"""Colour cells by an analysis result (a per-cell heat-map overlay).

Shared by the focused GUI viewer (`gui_focused/image_viewer.py`) and the
mask editor (`gui/mask_editor.py`). Both normally colour each cell by its
ID (the categorical palette in `gui.mask_editor_multicell.cell_color`).
This module lets them instead colour each cell by a measured value:

  - a per-TRACK scalar  (mean speed, persistence, % time rounded, …) →
    the cell keeps one colour across the whole recording, from a
    continuous matplotlib colormap; or
  - a per-FRAME value   (cell state, area, speed at this frame) →
    the cell's colour can change frame to frame.

The metric values come from `core.mask_metrics.compute_label_metrics`
(computed straight from the label stack, so colouring needs no Analyze
run). Pick a metric with `get_metric(name)`, build a `MetricColorizer`
from the metrics dict + spec, and ask it for `color_for_cell(cid, frame)`
or `lut_for_frame(labels_frame, frame)`.

`MetricLegend` is a thin QWidget that draws the colour key (a gradient
bar with min/max for continuous metrics, labelled swatches for the
categorical cell-state metric).
"""
from __future__ import annotations

import numpy as np

from core.mask_metrics import STATE_CODE

# Neutral grey for cells with no measurable value (absent in frame, NaN).
GRAY = (128, 128, 128)

# Cell-state palette (RGB). Binary model: green = spread (adherent),
# red = rounded (balled-up), grey = unknown/too-small.
STATE_COLORS = {
    0: (130, 130, 130),   # unknown
    1: (60, 180, 75),     # spread  — green
    2: (230, 50, 50),     # rounded — red
}
_CODE_TO_LABEL = {0: "unknown", 1: "spread", 2: "rounded"}

# Default colour-by option (matches the GUIs' native ID palette).
ID_METRIC = "Cell ID"


class MetricSpec:
    """Describes one colour-by option."""

    __slots__ = ("name", "kind", "key", "cmap", "vmin", "vmax", "unit")

    def __init__(self, name, kind, key=None, cmap="viridis",
                 vmin=None, vmax=None, unit=""):
        self.name = name        # dropdown label
        self.kind = kind        # "id" | "state" | "track" | "frame"
        self.key = key          # metric key in per_track / per_frame
        self.cmap = cmap        # matplotlib colormap name (continuous)
        self.vmin = vmin        # fixed lower bound (None → auto)
        self.vmax = vmax        # fixed upper bound (None → auto)
        self.unit = unit


# Ordered list of colour-by options. "Cell ID" + "Cell state" first,
# then per-track scalars, then per-frame values.
METRICS = [
    MetricSpec(ID_METRIC, "id"),
    MetricSpec("Cell state (rounded / spread)", "state"),
    MetricSpec("Mean speed", "track", "mean_speed", "viridis",
               unit="µm/min"),
    MetricSpec("Persistence", "track", "persistence", "plasma",
               vmin=0.0, vmax=1.0),
    MetricSpec("Net displacement", "track", "net_displacement",
               "viridis", unit="µm"),
    MetricSpec("Total distance", "track", "total_distance", "viridis",
               unit="µm"),
    MetricSpec("Mean area", "track", "mean_area", "cividis", unit="µm²"),
    MetricSpec("Mean circularity", "track", "mean_circularity",
               "coolwarm", vmin=0.0, vmax=1.0),
    MetricSpec("Mean solidity", "track", "mean_solidity", "coolwarm",
               vmin=0.0, vmax=1.0),
    MetricSpec("% time rounded", "track", "frac_rounded", "magma",
               vmin=0.0, vmax=1.0),
    MetricSpec("Frames tracked", "track", "frames_tracked", "viridis"),
    MetricSpec("Area — per frame", "frame", "area", "cividis", unit="µm²"),
    MetricSpec("Circularity — per frame", "frame", "circularity",
               "coolwarm", vmin=0.0, vmax=1.0),
    MetricSpec("Speed — per frame", "frame", "speed", "viridis",
               unit="µm/min"),
]
_BY_NAME = {m.name: m for m in METRICS}


def metric_names():
    """Ordered list of colour-by option names for a dropdown."""
    return [m.name for m in METRICS]


def get_metric(name):
    """MetricSpec for `name` (defaults to the Cell-ID spec)."""
    return _BY_NAME.get(name, METRICS[0])


def _get_cmap(name):
    try:
        from matplotlib import colormaps
        return colormaps[name]
    except Exception:  # older matplotlib
        from matplotlib import cm
        return cm.get_cmap(name)


def value_to_rgb(value, vmin, vmax, cmap):
    """Map a scalar through a colormap → (r, g, b) ints in 0-255.

    NaN / None → GRAY. A degenerate range maps everything to the
    colormap midpoint.
    """
    if value is None or not np.isfinite(value):
        return GRAY
    if vmax is None or vmin is None or vmax <= vmin:
        t = 0.5
    else:
        t = (float(value) - vmin) / (vmax - vmin)
        t = min(1.0, max(0.0, t))
    r, g, b, _ = cmap(t)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


class MetricColorizer:
    """Maps (cell_id, frame) → RGB for one selected metric."""

    def __init__(self, metrics, spec):
        self.metrics = metrics or {}
        self.spec = spec
        self.is_categorical = spec.kind in ("state", "id")
        self._cmap = (None if self.is_categorical
                      else _get_cmap(spec.cmap))
        self.vmin, self.vmax = self._compute_range()

    def _compute_range(self):
        spec = self.spec
        if self.is_categorical:
            return (None, None)
        vals = []
        if spec.kind == "track":
            for d in self.metrics.get("per_track", {}).values():
                v = d.get(spec.key)
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
        elif spec.kind == "frame":
            for d in self.metrics.get("per_frame", {}).values():
                arr = np.asarray(d.get(spec.key), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    vals.append(arr)
            vals = (np.concatenate(vals) if vals else [])
        vmin = spec.vmin
        vmax = spec.vmax
        arr = np.asarray(vals, dtype=float)
        if arr.size:
            # Robust 2-98 percentile range when not pinned by the spec.
            if vmin is None:
                vmin = float(np.percentile(arr, 2))
            if vmax is None:
                vmax = float(np.percentile(arr, 98))
            if vmax <= vmin:  # all-equal data — widen slightly
                vmax = vmin + 1e-6
        else:
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax
        return (vmin, vmax)

    def value_for_cell(self, cid, frame_idx):
        """Raw metric value for one cell (None if unavailable)."""
        spec = self.spec
        if spec.kind == "track":
            d = self.metrics.get("per_track", {}).get(int(cid))
            return None if d is None else d.get(spec.key)
        if spec.kind == "frame":
            d = self.metrics.get("per_frame", {}).get(int(cid))
            if d is None:
                return None
            arr = d.get(spec.key)
            if arr is None or frame_idx >= len(arr):
                return None
            return float(arr[frame_idx])
        return None

    def color_for_cell(self, cid, frame_idx):
        """RGB (0-255) for cell `cid` at frame `frame_idx`."""
        spec = self.spec
        cid = int(cid)
        if spec.kind == "id":
            from gui.mask_editor_multicell import cell_color
            return cell_color(cid)
        if spec.kind == "state":
            codes = self.metrics.get("states", {}).get(cid)
            if codes is None or frame_idx >= len(codes):
                return GRAY
            return STATE_COLORS.get(int(codes[frame_idx]), GRAY)
        val = self.value_for_cell(cid, frame_idx)
        return value_to_rgb(val, self.vmin, self.vmax, self._cmap)

    def lut_for_frame(self, labels_frame, frame_idx):
        """{cell_id: (r, g, b)} for every cell present in `labels_frame`."""
        ids = np.unique(labels_frame)
        return {int(c): self.color_for_cell(int(c), frame_idx)
                for c in ids.tolist() if c > 0}

    # --- Legend helpers ---------------------------------------------
    def legend_kind(self):
        return "state" if self.spec.kind == "state" else "gradient"

    def state_entries(self):
        """[(label, (r,g,b)), …] in the order spread→rounded→unknown.
        Driven by STATE_COLORS so it tracks the state model automatically."""
        order = [1, 2, 0]  # spread, rounded, unknown
        return [(_CODE_TO_LABEL[c], STATE_COLORS[c])
                for c in order if c in STATE_COLORS]

    def gradient_label(self):
        u = f" ({self.spec.unit})" if self.spec.unit else ""
        return f"{self.spec.name}{u}"


# ----------------------------------------------------------------------
# Legend widget (used by both GUIs)
# ----------------------------------------------------------------------
try:
    from PyQt5.QtWidgets import QWidget
    from PyQt5.QtGui import QPainter, QColor, QFont
    from PyQt5.QtCore import Qt
    _HAVE_QT = True
except Exception:  # headless import of the colour logic without Qt
    _HAVE_QT = False
    QWidget = object


class MetricLegend(QWidget):
    """Compact colour key: a gradient bar (continuous metrics) or
    labelled swatches (cell state). Hidden until a metric is set."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._colorizer = None
        self.setFixedHeight(26)
        self.setVisible(False)

    def set_colorizer(self, colorizer):
        """Attach a MetricColorizer (or None to hide the legend)."""
        self._colorizer = colorizer
        active = colorizer is not None and colorizer.spec.kind != "id"
        self.setVisible(active)
        self.update()

    def paintEvent(self, event):  # noqa: N802 (Qt signature)
        c = self._colorizer
        if c is None or c.spec.kind == "id":
            return
        p = QPainter(self)
        f = QFont()
        f.setPointSize(8)
        p.setFont(f)
        w, h = self.width(), self.height()
        if c.legend_kind() == "state":
            self._paint_state(p, c, w, h)
        else:
            self._paint_gradient(p, c, w, h)
        p.end()

    def _paint_state(self, p, c, w, h):
        x = 4
        for label, (r, g, b) in c.state_entries():
            p.fillRect(x, 5, 14, h - 12, QColor(r, g, b))
            p.setPen(QColor(220, 220, 220))
            p.drawText(x + 18, h - 8, label)
            x += 18 + p.fontMetrics().horizontalAdvance(label) + 14

    def _paint_gradient(self, p, c, w, h):
        # Layout:  [label]  vmin [====gradient====] vmax
        fm = p.fontMetrics()
        label = c.gradient_label()
        vmin_s, vmax_s = _fmt(c.vmin), _fmt(c.vmax)
        p.setPen(QColor(220, 220, 220))
        p.drawText(4, h - 8, label)
        x = 4 + fm.horizontalAdvance(label) + 12
        p.drawText(x, h - 8, vmin_s)
        bar_x0 = x + fm.horizontalAdvance(vmin_s) + 6
        bar_x1 = w - (fm.horizontalAdvance(vmax_s) + 10)
        bar_w = max(10, bar_x1 - bar_x0)
        cmap = c._cmap
        for i in range(bar_w):
            t = i / max(1, bar_w - 1)
            r, g, b, _ = cmap(t)
            p.fillRect(bar_x0 + i, 6, 1, h - 14,
                       QColor(int(r * 255), int(g * 255), int(b * 255)))
        p.drawText(bar_x1 + 4, h - 8, vmax_s)


def _fmt(v):
    if v is None or not np.isfinite(v):
        return "?"
    av = abs(v)
    if av >= 100:
        return f"{v:.0f}"
    if av >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"
