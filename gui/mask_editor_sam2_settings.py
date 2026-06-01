"""Settings dialog for the SAM2 mask-editor tools.

Holds the SAM2SettingsDialog + default_settings(). Split out from
gui/mask_editor_sam2_point.py so the inference module stays under
the project's 500-line cap.

The dialog mutates a settings dict in place; both run_sam2_at and
run_sam2_box_at on the editor read from that dict at click time, so
changes take effect on the very next click.
"""


def default_settings():
    """Defaults for every SAM2 knob the editor exposes via the
    Settings dialog. Matches the function-level defaults in
    predict_at_point / predict_at_box. Returned as a fresh dict so
    callers can mutate freely."""
    return {
        # mask cleanup
        "smooth_radius": 2,
        "keep_largest": True,
        "fill_holes": True,
        # area guards (shared between point and box)
        "min_area_px": 200,
        "max_area_cap_px": 50000,
        # point tool
        "crop_size": 512,
        # box tool
        "pad_px": 64,
        "min_box_area_px": 4096,
        "constrain_to_box": True,
        "box_expand_pct": 10,
        # quality / speed
        "use_tta": False,
        # input channels
        "use_fluo": True,
        "fluo_min_max": 30,
    }


class SAM2SettingsDialog:
    """Modal dialog that edits a settings dict in place."""

    @staticmethod
    def show(parent, settings):
        from PyQt5.QtWidgets import (
            QDialog, QFormLayout, QGroupBox, QVBoxLayout, QHBoxLayout,
            QSpinBox, QCheckBox, QPushButton,
        )
        dlg = QDialog(parent)
        dlg.setWindowTitle("SAM2 Options")
        dlg.setMinimumWidth(380)
        root = QVBoxLayout(dlg)

        # ── Mask cleanup ──
        gb_clean = QGroupBox("Mask cleanup (applied to every SAM2 output)")
        f_clean = QFormLayout(gb_clean)
        sp_smooth = QSpinBox()
        sp_smooth.setRange(0, 10)
        sp_smooth.setValue(int(settings["smooth_radius"]))
        sp_smooth.setToolTip(
            "Morphological-closing disk radius (px). Smooths the "
            "jagged single-pixel boundary; 0 disables.")
        f_clean.addRow("Smoothing radius (px):", sp_smooth)
        cb_largest = QCheckBox(
            "Keep only the largest connected component")
        cb_largest.setChecked(bool(settings["keep_largest"]))
        cb_largest.setToolTip(
            "Drop detached noise blobs from the prediction.")
        f_clean.addRow(cb_largest)
        cb_holes = QCheckBox("Fill internal holes")
        cb_holes.setChecked(bool(settings["fill_holes"]))
        cb_holes.setToolTip(
            "Fill background pixels fully enclosed by the mask.")
        f_clean.addRow(cb_holes)
        root.addWidget(gb_clean)

        # ── Area guards ──
        gb_area = QGroupBox("Area guards (reject obviously-wrong masks)")
        f_area = QFormLayout(gb_area)
        sp_min = QSpinBox()
        sp_min.setRange(0, 100000)
        sp_min.setValue(int(settings["min_area_px"]))
        sp_min.setToolTip("Reject masks smaller than this many px.")
        f_area.addRow("Min area (px):", sp_min)
        sp_maxcap = QSpinBox()
        sp_maxcap.setRange(1000, 1_000_000)
        sp_maxcap.setSingleStep(5000)
        sp_maxcap.setValue(int(settings["max_area_cap_px"]))
        sp_maxcap.setToolTip(
            "Hard upper bound. Box tool adapts to box_area×1.3 with "
            "this as the floor.")
        f_area.addRow("Max area cap (px):", sp_maxcap)
        root.addWidget(gb_area)

        # ── Point tool ──
        gb_point = QGroupBox("Point-click tool")
        f_point = QFormLayout(gb_point)
        sp_crop = QSpinBox()
        sp_crop.setRange(128, 2048)
        sp_crop.setSingleStep(64)
        sp_crop.setValue(int(settings["crop_size"]))
        sp_crop.setToolTip(
            "Side of the square crop around each click. Larger = "
            "more context but slower (encoder cost is O(N²)).")
        f_point.addRow("Crop size (px):", sp_crop)
        root.addWidget(gb_point)

        # ── Box tool ──
        gb_box = QGroupBox("Box-drag tool")
        f_box = QFormLayout(gb_box)
        sp_pad = QSpinBox()
        sp_pad.setRange(0, 256)
        sp_pad.setValue(int(settings["pad_px"]))
        sp_pad.setToolTip(
            "Crop expands this many px outside the drawn box on each "
            "side to give SAM2 some context.")
        f_box.addRow("Box padding (px):", sp_pad)
        sp_minbox = QSpinBox()
        sp_minbox.setRange(100, 100000)
        sp_minbox.setSingleStep(500)
        sp_minbox.setValue(int(settings["min_box_area_px"]))
        sp_minbox.setToolTip(
            "Reject drags whose box area is smaller than this. "
            "Catches accidental clicks-without-drag.")
        f_box.addRow("Min box area (px):", sp_minbox)
        cb_constrain = QCheckBox(
            "Constrain mask to drawn box (+ margin)")
        cb_constrain.setChecked(bool(settings["constrain_to_box"]))
        cb_constrain.setToolTip(
            "SAM2 box prompts are a SOFT constraint — masks can leak "
            "outside, especially when the box is around a subobject "
            "(e.g. nucleus only). When this is on, the predicted "
            "mask gets clipped to the box plus a percentage margin "
            "so it never extends far beyond what you drew.")
        f_box.addRow(cb_constrain)
        sp_expand = QSpinBox()
        sp_expand.setRange(0, 100)
        sp_expand.setSingleStep(5)
        sp_expand.setSuffix(" %")
        sp_expand.setValue(int(settings["box_expand_pct"]))
        sp_expand.setToolTip(
            "How much to expand the box before clipping. 0 = mask "
            "must stay exactly inside the drawn rectangle; 10 % "
            "(default) = small slack for imperfect drawing.")
        f_box.addRow("Box expand margin:", sp_expand)
        root.addWidget(gb_box)

        # ── Quality / speed ──
        gb_q = QGroupBox("Quality (slower)")
        f_q = QFormLayout(gb_q)
        cb_tta = QCheckBox(
            "Use test-time augmentation (4 rotations, ~4× slower)")
        cb_tta.setChecked(bool(settings["use_tta"]))
        cb_tta.setToolTip(
            "Run SAM2 at 0°/90°/180°/270° rotations of the crop and "
            "majority-vote at 0.5. Catches features SAM2 misses at "
            "the default orientation; particularly useful for cells "
            "with low-contrast or asymmetric boundaries. Cost: ~250 ms "
            "per click instead of ~60 ms (warm); first call still pays "
            "the model-load tax (~3 s).")
        f_q.addRow(cb_tta)
        root.addWidget(gb_q)

        # ── Input channels ──
        gb_inp = QGroupBox("Input channels")
        f_inp = QFormLayout(gb_inp)
        cb_fluo = QCheckBox(
            "Use fluorescence channel when present")
        cb_fluo.setChecked(bool(settings["use_fluo"]))
        cb_fluo.setToolTip(
            "When the recording has a fluorescence channel (e.g. Cy5), "
            "pack it into SAM2's red channel alongside DIC in "
            "green+blue. SAM2's encoder sees both signals — "
            "fluorescence as a localisation cue (nucleus stains "
            "are bright), DIC for boundary detection. No extra "
            "inference cost — same one encoder pass with a richer "
            "input.")
        f_inp.addRow(cb_fluo)
        sp_fluo_min = QSpinBox()
        sp_fluo_min.setRange(0, 255)
        sp_fluo_min.setValue(int(settings["fluo_min_max"]))
        sp_fluo_min.setToolTip(
            "Per-crop guard against weak fluorescence harming the "
            "detection. If the crop's brightest fluo pixel is below "
            "this value (0-255), we silently fall back to DIC-only "
            "for that cell — strong-Cy5 cells use both channels, "
            "weak-Cy5 cells use just DIC. The status bar shows "
            '"+Cy5" or "Cy5 off (weak)" after each click so you '
            "can see which path was taken.")
        f_inp.addRow("Min fluo signal (0-255):", sp_fluo_min)
        root.addWidget(gb_inp)

        # ── Buttons ──
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
            d = default_settings()
            sp_smooth.setValue(d["smooth_radius"])
            cb_largest.setChecked(d["keep_largest"])
            cb_holes.setChecked(d["fill_holes"])
            sp_min.setValue(d["min_area_px"])
            sp_maxcap.setValue(d["max_area_cap_px"])
            sp_crop.setValue(d["crop_size"])
            sp_pad.setValue(d["pad_px"])
            sp_minbox.setValue(d["min_box_area_px"])
            cb_constrain.setChecked(d["constrain_to_box"])
            sp_expand.setValue(d["box_expand_pct"])
            cb_tta.setChecked(d["use_tta"])
            cb_fluo.setChecked(d["use_fluo"])
            sp_fluo_min.setValue(d["fluo_min_max"])

        btn_reset.clicked.connect(_do_reset)
        btn_cancel.clicked.connect(dlg.reject)
        btn_ok.clicked.connect(dlg.accept)

        if dlg.exec_() == QDialog.Accepted:
            settings["smooth_radius"]    = sp_smooth.value()
            settings["keep_largest"]     = cb_largest.isChecked()
            settings["fill_holes"]       = cb_holes.isChecked()
            settings["min_area_px"]      = sp_min.value()
            settings["max_area_cap_px"]  = sp_maxcap.value()
            settings["crop_size"]        = sp_crop.value()
            settings["pad_px"]           = sp_pad.value()
            settings["min_box_area_px"]  = sp_minbox.value()
            settings["constrain_to_box"] = cb_constrain.isChecked()
            settings["box_expand_pct"]   = sp_expand.value()
            settings["use_tta"]          = cb_tta.isChecked()
            settings["use_fluo"]         = cb_fluo.isChecked()
            settings["fluo_min_max"]     = sp_fluo_min.value()
            return True
        return False
