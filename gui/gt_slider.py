"""QSlider subclass that paints coloured tick marks at GT target frames.

Usage:
    s = GtTickSlider(Qt.Horizontal)
    s.setRange(0, n_frames - 1)
    s.set_target_frames([0, 10, 20, ...])
    s.set_labeled_frames({0, 10})    # green ticks
    # unlabeled targets get yellow ticks

The tick marks are drawn ABOVE the slider track so they don't interfere
with the user's interaction. Repaints whenever target/labeled state
changes.
"""
from PyQt5.QtWidgets import QSlider, QStyle, QStyleOptionSlider
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QColor, QPen


class GtTickSlider(QSlider):
    """Horizontal slider with custom-painted GT-target tick marks."""

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self._target_frames = []   # list[int]
        self._labeled_frames = set()  # subset of target_frames
        # Reserve vertical space above the groove for the ticks.
        self.setMinimumHeight(30)

    def set_target_frames(self, frames):
        """Set the GT-target frame indices to display."""
        self._target_frames = list(frames or [])
        self.update()

    def set_labeled_frames(self, frames):
        """Mark which target frames have a saved mask (green tick)."""
        self._labeled_frames = set(frames or [])
        self.update()

    def paintEvent(self, event):
        # Let Qt paint the underlying slider first
        super().paintEvent(event)
        if not self._target_frames:
            return

        # Compute the slider track's horizontal extent. Use QStyle's
        # subcontrol rects so we land on the actual groove pixels.
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        handle_rect = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)

        # The handle width matters because Qt offsets the track by half
        # the handle so the handle centre maps to [min, max].
        x0 = groove_rect.x() + handle_rect.width() // 2
        x1 = groove_rect.x() + groove_rect.width() - handle_rect.width() // 2
        track_width = max(1, x1 - x0)

        n_min, n_max = self.minimum(), self.maximum()
        span = max(1, n_max - n_min)

        # Ticks go ABOVE the groove. Slim vertical bars.
        tick_top = groove_rect.y() - 8
        tick_bottom = groove_rect.y() - 1
        tick_height = max(4, tick_bottom - tick_top)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        for fi in self._target_frames:
            if not (n_min <= fi <= n_max):
                continue
            x = x0 + int(round(track_width * (fi - n_min) / span))
            is_labeled = fi in self._labeled_frames
            color = QColor("#3eb049") if is_labeled else QColor("#e0a700")
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawLine(x, tick_top, x, tick_bottom)

        painter.end()
