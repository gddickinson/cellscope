"""Pure recording-level decision rules of the detection pipeline (no models, no I/O).

Split out of `core/pipeline_defaults.py` (2026-11-12) to keep that file under the
500-line budget; `pipeline_defaults` re-exports every name here, so import from
there as before. Contents: the auto-select threshold + probe constants and
`decide_backbone`, the mirror-padding policy (`should_use_mirror_pad`) and the
auto-downsample policy (`resolve_downsample`).
"""
from __future__ import annotations

# Cell-density threshold for the auto-switch between cpsam_dic
# (single-cell-biased) and raw cpsam (handles touching cells).
# 1.5 picks raw cpsam when the 75th-percentile cell count across
# sampled frames is ≥2 (rounded). Single-cell sequences stay on
# cpsam_dic.
#
# The probe is 11 frames + p75 (not median) so a few sparse frames
# at the start/end of a recording (e.g. cells migrating into the
# field of view) can't drag the median down and route a dense field
# to single-cell mode. Pos68_DMSO surfaced this: a 5-frame median-1.0
# probe routed a 9–14-cells/frame recording to cpsam_dic, causing
# 11.2 mean FN/frame in evaluation. With p75 across 11 samples the
# few sparse frames are tolerated without losing the multi-cell
# signal in the rest of the recording.
MULTICELL_THRESHOLD = 1.5
N_SAMPLE_FRAMES_FOR_AUTO = 11
PROBE_AGGREGATOR_PERCENTILE = 75

# Mirror-padding policy: pad the image by MIRROR_PAD_PX before each
# cpsam call when the detection-resolution min dim is at least
# MIRROR_PAD_MIN_DIM_PX. Below that threshold the reflection border
# becomes a too-large fraction of the frame and cpsam mis-detects
# the padding region as cell-like, causing -0.10+ F1 regressions
# (validated on small legacy ignasi recordings, 2026-05-21).
MIRROR_PAD_PX = 50
MIRROR_PAD_MIN_DIM_PX = 1000


def should_use_mirror_pad(use_mirror_pad, det_frame_shape):
    """Resolve the use_mirror_pad config against an image shape.

    Args:
      use_mirror_pad: "auto" | "on" | "off" | True | False | None
      det_frame_shape: (H, W) at the resolution detection runs at

    Returns:
      bool — whether to pad
    """
    if use_mirror_pad in (None, "off", "Off", "OFF", False):
        return False
    if use_mirror_pad in ("on", "On", "ON", True):
        return True
    # "auto" or anything else
    if det_frame_shape is None:
        return False
    H, W = det_frame_shape[-2:]
    return min(H, W) >= MIRROR_PAD_MIN_DIM_PX


# ------------------------------------------------------------------
# Auto-downsample policy
# ------------------------------------------------------------------
# Cutoffs calibrated from GT runs:
#   Pos7_WT (2048²):     ds=2 F1 0.75 → 0.82 (CLEAR WIN)
#   IC293 Pos3 (1028²):  ds=2 F1 0.90 → 0.87 (slight cost, OK)
#   IC293 Pos0 (438×759): ds=2 F1 0.92 → 0.82 (HURTS — cells become
#                                            too small for cpsam)
#   ignasi_control (438×759, 15 frames): ds=2 F1 0.80 → 0.80 (no change)
# Conclusion: max-dim 800-1000 is the empirical breakpoint. Below
# that, downsampling pushes cells under cpsam's ~30 px diameter
# prior and the model misses them.
DOWNSAMPLE_SMALL_PX = 1100   # below this: don't downsample. Raised
                              # from 900 so 1024² recordings keep
                              # full resolution by default — at 2×,
                              # cpsam fuses touching cells whose ~30-px
                              # original diameter shrinks below 20 px
                              # (test2 DMSO_busy cells 3+4 case).
DOWNSAMPLE_LARGE_PX = 1500   # above this: ds=2 is a clear win


def resolve_downsample(spec, frame_shape,
                        small_px=DOWNSAMPLE_SMALL_PX,
                        large_px=DOWNSAMPLE_LARGE_PX):
    """Resolve a downsample spec to an integer factor.

    Args:
      spec: 'auto', 'off', or an integer-like (1, 2, '2', ...).
      frame_shape: (H, W) of a single frame, used when spec='auto'.
      small_px / large_px: cutoff thresholds on max(H, W).

    Returns:
      (factor, reason_string)
    """
    if spec in (None, "off", "Off", "OFF"):
        return 1, "downsample disabled"
    if isinstance(spec, str) and spec.lower() == "auto":
        h, w = frame_shape[:2]
        max_dim = max(h, w)
        if max_dim < small_px:
            return 1, (f"auto: max dim {max_dim} < {small_px}; cells "
                       f"would be too small at ds≥2")
        if max_dim < large_px:
            return 2, (f"auto: max dim {max_dim} in "
                       f"[{small_px}, {large_px}); ds=2 keeps cells "
                       f"resolvable while giving a ~3× speedup")
        return 2, (f"auto: max dim {max_dim} ≥ {large_px}; ds=2 "
                   f"gives ~5× speedup with minimal accuracy loss")
    # Explicit integer
    try:
        factor = int(spec)
    except (TypeError, ValueError):
        raise ValueError(f"downsample spec not understood: {spec!r}")
    if factor < 1:
        raise ValueError(f"downsample factor must be ≥ 1, got {factor}")
    return factor, f"explicit factor {factor}"


def decide_backbone(counts):
    """The auto-select decision from per-frame probe cell counts (pure).

    Uses the 75th-percentile sample count rather than the median so a
    few sparse frames (cells entering/leaving the FoV, early-time
    recordings before cells migrate in) can't drag the decision down
    in an otherwise dense field. The median is kept in the info dict
    for diagnostic comparability. Returns the info dict; its "choice"
    is "cpsam" (multi-cell) or "cpsam_dic" (single-cell).
    """
    import numpy as np
    counts_arr = np.asarray(counts, dtype=float)
    decision_count = float(np.percentile(
        counts_arr, PROBE_AGGREGATOR_PERCENTILE))
    info = {
        "cell_counts_at_sample": list(counts),
        "median_count": float(np.median(counts_arr)),
        "p75_count": decision_count,
        "decision_count": decision_count,
        "aggregator": f"p{PROBE_AGGREGATOR_PERCENTILE}",
        "threshold": MULTICELL_THRESHOLD,
    }
    if decision_count >= MULTICELL_THRESHOLD:
        info["choice"] = "cpsam"
        info["reason"] = (
            f"p{PROBE_AGGREGATOR_PERCENTILE} {decision_count:.1f} "
            f"cells/frame ≥ {MULTICELL_THRESHOLD}: multi-cell scene, "
            f"switching to raw cpsam (cpsam_dic merges touching cells)")
    else:
        info["choice"] = "cpsam_dic"
        info["reason"] = (
            f"p{PROBE_AGGREGATOR_PERCENTILE} {decision_count:.1f} "
            f"cells/frame < {MULTICELL_THRESHOLD}: single-cell scene, "
            f"using cpsam_dic (tighter boundaries on isolated cells)")
    return info
