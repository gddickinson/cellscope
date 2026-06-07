"""Named detection speed/quality presets for the focused GUI.

A preset is a bundle of overrides applied on top of `DEFAULTS` to the
focused GUI's detection widgets — a one-click way to trade speed for
quality. "Default (Balanced)" is the canonical pipeline (no overrides);
the others step away from it toward speed or quality. After applying a
preset the user can still tweak any individual option.

Per the project's single-source-of-truth rule, the *baseline* for every
knob comes from `core.pipeline_defaults.DEFAULTS` — presets only record
the deltas. Switching to any preset resets the FULL set of tunable knobs
(`PRESET_PARAMS`), so the result is deterministic regardless of prior
edits.

The speed/quality axis is dominated by: downsample factor, whether the
(expensive) gap-fill cascade runs and how many of its phases (SAM2,
CP3 fallback), test-time augmentation, DeepSea refinement, mirror-pad,
and cpsam precision (fp32 is ~1.26× faster than bf16 on Apple MPS).
"""
from core.pipeline_defaults import DEFAULTS as _PD

# Ordered fastest → highest quality (drives the GUI dropdown order).
PRESET_ORDER = ["Fast", "Medium", "Default (Balanced)", "Highest Quality"]

# The full set of knobs a preset controls. Any param NOT named in a
# preset's dict below falls back to the DEFAULTS-backed baseline, so
# every preset fully determines all of these.
PRESET_PARAMS = [
    "downsample", "use_gap_fill", "use_sam2_video_gap_fill",
    "gap_fill_crop", "gap_fill_augment", "use_deepsea", "use_tta",
    "use_mirror_pad", "use_bfloat16", "use_fallback",
]


def _baseline():
    """The 'Default (Balanced)' values — straight from DEFAULTS (+ the
    GUI's 'auto' downsample default)."""
    return {
        "downsample": "auto",
        "use_gap_fill": _PD.use_gap_fill,
        "use_sam2_video_gap_fill": _PD.use_sam2_video_gap_fill,
        "gap_fill_crop": _PD.gap_fill_crop,
        "gap_fill_augment": _PD.gap_fill_augment,
        "use_deepsea": _PD.use_deepsea,
        "use_tta": _PD.use_tta,
        "use_mirror_pad": str(_PD.use_mirror_pad).lower(),
        "use_bfloat16": _PD.use_bfloat16,
        "use_fallback": _PD.use_fallback,
    }


# Deltas vs the baseline. "Default (Balanced)" is empty (= baseline).
DETECTION_PRESETS = {
    # Fastest possible — accept poor quality. Skip the whole gap-fill
    # cascade (the dominant cost), aggressive downsample, fp32, no TTA /
    # DeepSea / mirror-pad / CP3 fallback.
    "Fast": {
        "downsample": "3",
        "use_gap_fill": False,
        "use_sam2_video_gap_fill": False,
        "use_deepsea": False,
        "use_tta": False,
        "use_mirror_pad": "off",
        "use_bfloat16": False,
        "use_fallback": False,
    },
    # Middle ground — keep cheap gap-fill (crop + no-augment) but drop
    # the expensive Phase-3 SAM2 + Phase-2 CP3 fallback; fp32 speedup;
    # DeepSea on. Noticeably faster than Default, modest quality cost.
    "Medium": {
        "use_sam2_video_gap_fill": False,
        "use_bfloat16": False,
        "use_fallback": False,
    },
    # Canonical validated pipeline — no overrides.
    "Default (Balanced)": {},
    # Spare no compute — full resolution, all gap-fill phases, full-frame
    # always-augment gap-fill, TTA, DeepSea, mirror-pad on, CP3 fallback.
    "Highest Quality": {
        "downsample": "off",
        "use_gap_fill": True,
        "use_sam2_video_gap_fill": True,
        "gap_fill_crop": False,
        "gap_fill_augment": True,
        "use_deepsea": True,
        "use_tta": True,
        "use_mirror_pad": "on",
        "use_bfloat16": True,
        "use_fallback": True,
    },
}


def preset_values(name):
    """Resolve a preset name to a full {param: value} dict over
    PRESET_PARAMS (baseline filled in for any unspecified knob)."""
    vals = _baseline()
    vals.update(DETECTION_PRESETS.get(name, {}))
    return vals


def _set_combo_data(combo, data):
    for i in range(combo.count()):
        if combo.itemData(i) == data:
            combo.setCurrentIndex(i)
            return
    # tolerate value-as-text fallback
    idx = combo.findText(str(data))
    if idx >= 0:
        combo.setCurrentIndex(idx)


def _set_combo_text(combo, text):
    idx = combo.findText(str(text))
    if idx >= 0:
        combo.setCurrentIndex(idx)


def apply_preset_to_panel(panel, name):
    """Set the focused GUI ParamsPanel's detection widgets to `name`.

    Drives the existing widgets only — no new param plumbing. Returns the
    resolved values dict (handy for tests / logging)."""
    v = preset_values(name)
    _set_combo_data(panel.downsample, v["downsample"])
    _set_combo_text(panel.use_mirror_pad, v["use_mirror_pad"])
    panel.use_gap_fill.setChecked(v["use_gap_fill"])
    panel.use_sam2_video_gap_fill.setChecked(v["use_sam2_video_gap_fill"])
    panel.gap_fill_crop.setChecked(v["gap_fill_crop"])
    panel.gap_fill_augment.setChecked(v["gap_fill_augment"])
    panel.use_deepsea.setChecked(v["use_deepsea"])
    panel.use_tta.setChecked(v["use_tta"])
    panel.use_bfloat16.setChecked(v["use_bfloat16"])
    panel.use_fallback.setChecked(v["use_fallback"])
    return v
