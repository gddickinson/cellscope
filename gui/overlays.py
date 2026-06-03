"""Timestamp + scale-bar overlay rendering for both GUIs.

A single module shared by gui/mask_editor.py and gui_focused/image_viewer.py
so the overlay behaviour is consistent. Draws directly onto the RGB
numpy array before display, using cv2 — no Qt dependency for the
draw functions themselves (only the settings dialog needs Qt).

Settings are a plain dict that callers mutate in place via the
OverlaySettingsDialog. draw_overlays() reads from that dict and is
a no-op when both "show_timestamp" and "show_scale_bar" are False —
zero render-path cost when overlays are off.

Convention: positions are one of "top-left", "top-right",
"bottom-left", "bottom-right". Colours are (R, G, B) tuples.
"""
import cv2
import numpy as np


POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")
TIMESTAMP_FORMATS = ("MM:SS", "HH:MM:SS", "min", "h")


def default_overlay_settings():
    """Defaults — overlays OFF until the user opens the dialog.

    Returned as a fresh dict so callers can mutate freely.
    """
    return {
        # Timestamp
        "show_timestamp": False,
        "timestamp_position": "top-right",
        "timestamp_format": "MM:SS",
        "timestamp_color": [255, 255, 255],   # RGB (lists so JSON-safe)
        "timestamp_font_scale": 1.2,
        "timestamp_thickness": 2,
        # Scale bar
        "show_scale_bar": False,
        "scale_bar_position": "bottom-right",
        "scale_bar_length_um": 100,
        "scale_bar_thickness_px": 6,
        "scale_bar_color": [255, 255, 255],
        "scale_bar_show_label": True,
        "scale_bar_label_font_scale": 0.9,
    }


def _fmt_time(frame_idx, dt_min, fmt):
    """Format the elapsed time for frame_idx given dt_min per frame."""
    t = float(frame_idx) * float(dt_min)   # total minutes
    if fmt == "MM:SS":
        m = int(t)
        s = int(round((t - m) * 60))
        return f"{m:02d}:{s:02d}"
    if fmt == "HH:MM:SS":
        h = int(t // 60)
        rem = t - h * 60
        m = int(rem)
        s = int(round((rem - m) * 60))
        return f"{h:02d}:{m:02d}:{s:02d}"
    if fmt == "h":
        return f"{t / 60.0:.2f} h"
    return f"{t:.0f} min"


def _anchor(W, H, w, h, pos, pad):
    """Top-left corner (x, y) for a w×h box at the given anchor."""
    if pos == "top-left":
        return pad, pad
    if pos == "top-right":
        return W - w - pad, pad
    if pos == "bottom-left":
        return pad, H - h - pad
    return W - w - pad, H - h - pad   # bottom-right default


def _bgr(rgb_tuple):
    """Convert RGB (R,G,B) → BGR for cv2 drawing on an RGB array.

    cv2 paints in BGR; our rgb array is RGB. To get the user's
    intended RGB colour to appear correctly we pass (R, G, B) to cv2
    on an RGB-ordered array — the channels happen to line up.
    Renamed for clarity; treat any cv2.putText/line/rectangle call
    as accepting our R/G/B value here.
    """
    return (int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2]))


def draw_overlays(rgb, frame_idx, um_per_px, dt_min, settings):
    """Paint timestamp + scale bar onto rgb (modified in place).

    rgb: (H, W, 3) uint8 — modified in place
    frame_idx: int — current frame index
    um_per_px: float — pixel size in microns. Pass None or 0 to
        disable the scale bar (degraded gracefully — only timestamp
        will render).
    dt_min: float — time-step in minutes between frames. Pass None
        or 0 to disable the timestamp.
    settings: dict (see default_overlay_settings)
    """
    if rgb is None or rgb.ndim != 3:
        return
    H, W = rgb.shape[:2]
    pad = max(10, int(0.015 * min(H, W)))

    show_ts = bool(settings.get("show_timestamp")) and dt_min
    show_sb = bool(settings.get("show_scale_bar")) and um_per_px

    if show_ts:
        fmt = settings.get("timestamp_format", "MM:SS")
        text = _fmt_time(frame_idx, dt_min, fmt)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = float(settings.get("timestamp_font_scale", 1.0))
        thick = int(settings.get("timestamp_thickness", 2))
        color = _bgr(settings.get(
            "timestamp_color", [255, 255, 255]))
        (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
        x, y = _anchor(W, H, tw, th + bl, settings.get(
            "timestamp_position", "top-right"), pad)
        # putText origin is at the baseline; shift down so the box
        # corner matches the anchor.
        cv2.putText(rgb, text, (x, y + th), font, scale, color,
                    thick, cv2.LINE_AA)

    if show_sb:
        length_um = float(settings.get("scale_bar_length_um", 100))
        bar_px = max(1, int(round(length_um / float(um_per_px))))
        thick = int(settings.get("scale_bar_thickness_px", 6))
        color = _bgr(settings.get(
            "scale_bar_color", [255, 255, 255]))
        show_label = bool(settings.get("scale_bar_show_label", True))
        label = f"{int(length_um)} um"
        font = cv2.FONT_HERSHEY_SIMPLEX
        lscale = float(settings.get("scale_bar_label_font_scale", 0.9))
        lthick = max(1, int(thick / 2))
        if show_label:
            (lw, lh), lbl_baseline = cv2.getTextSize(
                label, font, lscale, lthick)
            box_w = max(bar_px, lw)
            box_h = thick + 4 + lh + lbl_baseline
        else:
            box_w = bar_px
            box_h = thick
        bx, by = _anchor(W, H, box_w, box_h, settings.get(
            "scale_bar_position", "bottom-right"), pad)
        # The bar sits at the bottom of the box; label above.
        bar_x = bx + (box_w - bar_px) // 2
        bar_y = by + box_h - thick
        cv2.rectangle(rgb, (bar_x, bar_y),
                      (bar_x + bar_px, bar_y + thick), color, -1)
        if show_label:
            (lw, lh), _ = cv2.getTextSize(label, font, lscale, lthick)
            lx = bx + (box_w - lw) // 2
            ly = bar_y - 4
            cv2.putText(rgb, label, (lx, ly), font, lscale, color,
                        lthick, cv2.LINE_AA)


# ─────── settings dialog ───────
class OverlaySettingsDialog:
    """Modal dialog editing an overlay settings dict in place."""

    @staticmethod
    def show(parent, settings):
        from PyQt5.QtWidgets import (
            QDialog, QFormLayout, QGroupBox, QVBoxLayout, QHBoxLayout,
            QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
            QComboBox, QColorDialog, QLabel,
        )
        from PyQt5.QtGui import QColor

        dlg = QDialog(parent)
        dlg.setWindowTitle("Overlay options — timestamp + scale bar")
        dlg.setMinimumWidth(420)
        root = QVBoxLayout(dlg)

        # ── Timestamp ──
        gb_ts = QGroupBox("Timestamp")
        f_ts = QFormLayout(gb_ts)
        cb_show_ts = QCheckBox("Show timestamp")
        cb_show_ts.setChecked(bool(settings["show_timestamp"]))
        f_ts.addRow(cb_show_ts)
        cmb_ts_pos = QComboBox()
        cmb_ts_pos.addItems(POSITIONS)
        cmb_ts_pos.setCurrentText(settings["timestamp_position"])
        f_ts.addRow("Position:", cmb_ts_pos)
        cmb_ts_fmt = QComboBox()
        cmb_ts_fmt.addItems(TIMESTAMP_FORMATS)
        cmb_ts_fmt.setCurrentText(settings["timestamp_format"])
        cmb_ts_fmt.setToolTip(
            "MM:SS — minutes:seconds, HH:MM:SS — hours:minutes:seconds,\n"
            "min — total minutes, h — fractional hours")
        f_ts.addRow("Format:", cmb_ts_fmt)
        sp_ts_font = QDoubleSpinBox()
        sp_ts_font.setRange(0.2, 5.0)
        sp_ts_font.setSingleStep(0.1)
        sp_ts_font.setValue(float(settings["timestamp_font_scale"]))
        f_ts.addRow("Font scale:", sp_ts_font)
        sp_ts_thick = QSpinBox()
        sp_ts_thick.setRange(1, 10)
        sp_ts_thick.setValue(int(settings["timestamp_thickness"]))
        f_ts.addRow("Font thickness:", sp_ts_thick)
        btn_ts_color = QPushButton()
        ts_col = list(settings["timestamp_color"])

        def _set_btn(btn, rgb):
            btn.setStyleSheet(
                f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); "
                f"color: {'#000' if sum(rgb) > 380 else '#fff'};")
            btn.setText(f"R={rgb[0]} G={rgb[1]} B={rgb[2]}")
        _set_btn(btn_ts_color, ts_col)

        def _pick_ts_color():
            c = QColorDialog.getColor(
                QColor(ts_col[0], ts_col[1], ts_col[2]), dlg,
                "Timestamp colour")
            if c.isValid():
                ts_col[:] = [c.red(), c.green(), c.blue()]
                _set_btn(btn_ts_color, ts_col)
        btn_ts_color.clicked.connect(_pick_ts_color)
        f_ts.addRow("Colour:", btn_ts_color)
        root.addWidget(gb_ts)

        # ── Scale bar ──
        gb_sb = QGroupBox("Scale bar")
        f_sb = QFormLayout(gb_sb)
        cb_show_sb = QCheckBox("Show scale bar")
        cb_show_sb.setChecked(bool(settings["show_scale_bar"]))
        f_sb.addRow(cb_show_sb)
        cmb_sb_pos = QComboBox()
        cmb_sb_pos.addItems(POSITIONS)
        cmb_sb_pos.setCurrentText(settings["scale_bar_position"])
        f_sb.addRow("Position:", cmb_sb_pos)
        sp_sb_len = QSpinBox()
        sp_sb_len.setRange(1, 10000)
        sp_sb_len.setSuffix(" µm")
        sp_sb_len.setValue(int(settings["scale_bar_length_um"]))
        f_sb.addRow("Length:", sp_sb_len)
        sp_sb_thick = QSpinBox()
        sp_sb_thick.setRange(1, 50)
        sp_sb_thick.setSuffix(" px")
        sp_sb_thick.setValue(int(settings["scale_bar_thickness_px"]))
        f_sb.addRow("Thickness:", sp_sb_thick)
        cb_sb_label = QCheckBox("Show length label")
        cb_sb_label.setChecked(bool(settings["scale_bar_show_label"]))
        f_sb.addRow(cb_sb_label)
        sp_sb_label_font = QDoubleSpinBox()
        sp_sb_label_font.setRange(0.2, 5.0)
        sp_sb_label_font.setSingleStep(0.1)
        sp_sb_label_font.setValue(
            float(settings["scale_bar_label_font_scale"]))
        f_sb.addRow("Label font scale:", sp_sb_label_font)
        btn_sb_color = QPushButton()
        sb_col = list(settings["scale_bar_color"])
        _set_btn(btn_sb_color, sb_col)

        def _pick_sb_color():
            c = QColorDialog.getColor(
                QColor(sb_col[0], sb_col[1], sb_col[2]), dlg,
                "Scale-bar colour")
            if c.isValid():
                sb_col[:] = [c.red(), c.green(), c.blue()]
                _set_btn(btn_sb_color, sb_col)
        btn_sb_color.clicked.connect(_pick_sb_color)
        f_sb.addRow("Colour:", btn_sb_color)
        root.addWidget(gb_sb)

        # Buttons
        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset to defaults")
        btn_cancel = QPushButton("Cancel")
        btn_ok = QPushButton("OK")
        btn_ok.setDefault(True)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        root.addLayout(btn_row)

        def _do_reset():
            d = default_overlay_settings()
            cb_show_ts.setChecked(d["show_timestamp"])
            cmb_ts_pos.setCurrentText(d["timestamp_position"])
            cmb_ts_fmt.setCurrentText(d["timestamp_format"])
            sp_ts_font.setValue(d["timestamp_font_scale"])
            sp_ts_thick.setValue(d["timestamp_thickness"])
            ts_col[:] = list(d["timestamp_color"])
            _set_btn(btn_ts_color, ts_col)
            cb_show_sb.setChecked(d["show_scale_bar"])
            cmb_sb_pos.setCurrentText(d["scale_bar_position"])
            sp_sb_len.setValue(d["scale_bar_length_um"])
            sp_sb_thick.setValue(d["scale_bar_thickness_px"])
            cb_sb_label.setChecked(d["scale_bar_show_label"])
            sp_sb_label_font.setValue(d["scale_bar_label_font_scale"])
            sb_col[:] = list(d["scale_bar_color"])
            _set_btn(btn_sb_color, sb_col)
        btn_reset.clicked.connect(_do_reset)
        btn_cancel.clicked.connect(dlg.reject)
        btn_ok.clicked.connect(dlg.accept)

        if dlg.exec_() == QDialog.Accepted:
            settings["show_timestamp"]           = cb_show_ts.isChecked()
            settings["timestamp_position"]       = cmb_ts_pos.currentText()
            settings["timestamp_format"]         = cmb_ts_fmt.currentText()
            settings["timestamp_font_scale"]     = sp_ts_font.value()
            settings["timestamp_thickness"]      = sp_ts_thick.value()
            settings["timestamp_color"]          = list(ts_col)
            settings["show_scale_bar"]           = cb_show_sb.isChecked()
            settings["scale_bar_position"]       = cmb_sb_pos.currentText()
            settings["scale_bar_length_um"]      = sp_sb_len.value()
            settings["scale_bar_thickness_px"]   = sp_sb_thick.value()
            settings["scale_bar_show_label"]     = cb_sb_label.isChecked()
            settings["scale_bar_label_font_scale"] = sp_sb_label_font.value()
            settings["scale_bar_color"]          = list(sb_col)
            return True
        return False
