"""Single source of truth for the focused-GUI + multichannel pipeline
defaults.

CellScope has TWO config layers:

  1. `config.py` (project root)
     The legacy config that the older `gui/` GUI + `core/auto_params`,
     `core/detection`, `core/refinement`, `core/edge_dynamics`,
     `core/boundary_rf`, etc. import. It defines edge-snap params,
     Fourier descriptors, CRF iterations, optical-flow Farneback
     params, RF filter banks, plotting colours, and a stale
     `MIN_CELL_AREA_PX = 300`. Leave it alone unless you're touching
     the legacy refinement pipeline.

  2. `core/pipeline_defaults.py`  (this file — NEW, 2026-05-14)
     The defaults consumed by the FOCUSED GUI (`gui_focused/`), the
     multichannel pipeline (`hybrid_dic.py`, `hybrid_cpsam_multi.py`,
     `dic_cy5_fusion.py`), and any programmatic / batch script.
     Values like `min_area_px = 500`, `use_cy5_fusion`,
     `cy5_filter_mode = "multi_metric"`, etc. live here.

When the two overlap (e.g. cell-area threshold), `pipeline_defaults.py`
wins for the active pipeline. The legacy `config.py` value remains
because legacy callers haven't been audited.

If a default changes:
  1. Edit it here.
  2. Run the GUI smoke test (`scripts/test_focused_gui.py`).
  3. Existing pickled / saved projects keep their saved values; only
     fresh runs pick up the new default.

Why this exists: in May 2026 a programmatic re-run of the DMSO_busy
demo produced different detections than the GUI run on the same
recording. Root cause was a hard-coded `model_path="data/models/
cellpose_dic"` + `min_area_px=200` in the script, contradicting the
GUI defaults (cpsam_dic auto-selected + min_area_px=500). One source
of truth prevents this kind of drift.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional

# ------------------------------------------------------------------
# Where to find the GUI-installed models. Resolution order is the same
# as `gui_focused/params_panel.py::_populate_dic_models`: cpsam_dic
# first (best, ViT fine-tune), then v3 > v2 > v1.
# ------------------------------------------------------------------
DIC_MODEL_PREFERENCE = (
    "cpsam_dic",
    "cellpose_dic_v3",
    "cellpose_dic_v2",
    "cellpose_dic",
)


def resolve_dic_model_path(project_root: Optional[str] = None) -> str:
    """Return the best available DIC model path.

    Mirrors the "Auto (best available)" choice from the GUI dropdown.
    If `project_root` is None, paths are returned relative to the
    current working directory (matching `core.detection`'s convention).
    """
    base = (os.path.join(project_root, "data/models")
            if project_root else "data/models")
    for name in DIC_MODEL_PREFERENCE:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    # Fall back to the oldest name even if it doesn't exist — caller
    # gets a clear FileNotFoundError instead of None.
    return os.path.join(base, DIC_MODEL_PREFERENCE[-1])


# Cell-density threshold for the auto-switch between cpsam_dic
# (single-cell-biased) and raw cpsam (handles touching cells).
# 1.5 picks raw cpsam when MEDIAN cell count across sampled frames is
# ≥2 (rounded). Single-cell sequences stay on cpsam_dic.
MULTICELL_THRESHOLD = 1.5
N_SAMPLE_FRAMES_FOR_AUTO = 5


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
DOWNSAMPLE_SMALL_PX = 900    # below this: don't downsample
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


_PROBE_SCRIPT = '''
import sys, os, json, warnings, logging, numpy as np
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)

from cellpose import models
data = np.load("{input_path}", allow_pickle=True)
frames = data["frames"]
indices = data["indices"]
min_area_px = int({min_area_px})
m = models.CellposeModel(gpu=True, pretrained_model="{model_path}")
counts = []
for fi in indices:
    labels = m.eval(frames[int(fi)], augment=False)[0]
    n = sum(1 for c in range(1, int(labels.max()) + 1)
            if (labels == c).sum() >= min_area_px)
    counts.append(int(n))
print("PROBE_OK", json.dumps(counts))
'''


def _run_probe_subprocess(frames, indices, model_path, min_area_px,
                           env_name="cellpose4", project_root=None):
    """Run the cell-count probe in `env_name` via subprocess so we can
    use a CP4 model (cpsam_dic) from a CP3-only context."""
    import subprocess
    import tempfile
    import numpy as np

    if project_root is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))

    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "input.npz")
        sub_frames = np.array(
            [frames[int(i)] for i in indices])
        np.savez_compressed(
            inp, frames=sub_frames,
            indices=np.arange(len(sub_frames)))
        script = _PROBE_SCRIPT.format(
            input_path=inp, model_path=model_path,
            min_area_px=min_area_px)
        proc = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", script],
            capture_output=True, text=True, timeout=600,
            cwd=project_root)
        for line in proc.stdout.splitlines():
            if line.startswith("PROBE_OK "):
                import json as _json
                return _json.loads(line[len("PROBE_OK "):])
        raise RuntimeError(
            f"probe subprocess failed:\nSTDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr[-500:]}")


def select_pipeline_for_recording(
        frames, min_area_px=None, sample_n=N_SAMPLE_FRAMES_FOR_AUTO,
        verbose=False):
    """Pick the right detection pipeline + model for a recording.

    Probes a few frames with cpsam_dic, counts cells per frame, and
    returns either ('cpsam_dic', model_path) or ('cpsam', None) based
    on whether the recording looks multi-cell.

    Why this exists: cpsam_dic was fine-tuned on single-cell VAMPIRE
    crops and tends to MERGE touching cells in multi-cell scenes
    (proved on IC293 Pos3 — F1 0.65 → 0.90 by switching to raw cpsam).
    Raw cpsam is the better default for crowded recordings; cpsam_dic
    gives slightly tighter boundaries on isolated single cells. This
    function lets the runner pick automatically.

    The probe runs in the cellpose4 env via subprocess so cpsam_dic
    (a CP4 ViT fine-tune) loads correctly even when the calling
    process is in the cellpose (CP3) env.

    Args:
      frames: (N, H, W) uint8 stack
      min_area_px: drop detections smaller than this when counting
      sample_n: number of frames to sample evenly across the recording
      verbose: print decision rationale

    Returns:
      tuple ('cpsam_dic' | 'cpsam', model_path_or_None, info_dict)
    """
    if min_area_px is None:
        min_area_px = DEFAULTS.min_area_px

    import numpy as np
    n_frames = len(frames)
    if n_frames == 0:
        return ("cpsam_dic", resolve_dic_model_path(),
                {"reason": "no frames"})

    indices = list(np.linspace(0, n_frames - 1, min(sample_n, n_frames),
                                dtype=int))
    cpsam_dic_path = resolve_dic_model_path()
    if verbose:
        print(f"[auto-select] probing {len(indices)} frames "
              f"with cpsam_dic ({cpsam_dic_path}, via cellpose4 "
              f"subprocess) …")
    counts = _run_probe_subprocess(
        frames, indices, cpsam_dic_path, min_area_px)

    median_count = float(np.median(counts))
    info = {
        "sampled_frames": [int(i) for i in indices],
        "cell_counts_at_sample": counts,
        "median_count": median_count,
        "threshold": MULTICELL_THRESHOLD,
    }

    if median_count >= MULTICELL_THRESHOLD:
        info["choice"] = "cpsam"
        info["reason"] = (
            f"median {median_count:.1f} cells/frame ≥ "
            f"{MULTICELL_THRESHOLD}: multi-cell scene, switching to "
            f"raw cpsam (cpsam_dic merges touching cells)")
        if verbose:
            print(f"[auto-select] {info['reason']}")
        return "cpsam", None, info

    info["choice"] = "cpsam_dic"
    info["reason"] = (
        f"median {median_count:.1f} cells/frame < "
        f"{MULTICELL_THRESHOLD}: single-cell scene, using cpsam_dic "
        f"(tighter boundaries on isolated cells)")
    if verbose:
        print(f"[auto-select] {info['reason']}")
    return "cpsam_dic", cpsam_dic_path, info


# ------------------------------------------------------------------
# Detection / refinement / tracking defaults.
# ------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineDefaults:
    # --- Detection ---
    # Both pixel and physical-unit defaults are provided. Code that
    # knows the recording's um_per_px should call `pixel_thresholds()`
    # below to get the per-recording px values, falling back to these
    # raw px defaults when scale is unknown. See
    # `core/pipeline_defaults.py::pixel_thresholds` for the conversion
    # helpers.
    min_area_px: int = 200
    # Physical-unit equivalent. At IC295's 0.6523 µm/px:
    #   200 px × 0.6523² µm²/px² ≈ 85 µm² — small but real (balled
    #   keratinocytes can project to ~80-150 µm² in DIC).
    min_area_um2: float = 85.0

    # Expected cell diameter (typical spread keratinocyte). Cellpose
    # uses this to internally scale features so the model's training-
    # diameter prior matches the input. None = let cellpose auto-
    # estimate from image content (works at IC295 scale).
    expected_cell_diameter_um: float = 30.0

    expected_cells: int = 0   # 0 = unknown, scan_cell_count() may set
    search_radius_px: int = 150
    # Physical-unit equivalent for tracker max-hop and gap-fill search.
    # Keratinocytes move <0.5-2 µm/min on DIC time-lapse; our
    # 10-min/frame interval gives <20 µm/frame max, so 100 µm /min
    # search radius covers ~5 frames of motion.
    max_hop_um_per_min: float = 15.0
    search_radius_um: float = 100.0
    min_track_length: int = 3
    # Max consecutive frames a track can be undetected before the
    # Hungarian tracker declares it dead. Was hardcoded to 10 in
    # core/multi_cell.py until May 2026 — bumped to 15 because
    # cells in our recordings can be undetectable for ~10 frames
    # while transitioning shape (mitosis, retraction, de-attachment)
    # at our 10-min/frame interval. Track-identity preservation
    # across these gaps prevents ID swaps like cell9→cell8.
    max_gap_frames: int = 15

    # --- Quality / refinement toggles ---
    use_preprocess: bool = True       # temporal bg + spatial HP (CP3 only)
    use_deepsea: bool = True
    use_retry: bool = True
    use_fallback: bool = True         # CP3 fallback subprocess
    use_gap_fill: bool = True
    use_sam2_video_gap_fill: bool = True
    # SAM2 video propagation between Phase 2 (CP3 fallback) and Phase 4
    # (simple mask translation). Catches cells that biologically retract
    # / dim too much for cpsam to detect at the gap frame, but where
    # SAM2's memory attention can follow them through the gap. Adds
    # ~1 sec per gap frame; turn off if cpsam(augment) is enough for
    # your data.
    use_tta: bool = False             # cpsam test-time augmentation

    # --- Tiling (only relevant for cpsam multi mode) ---
    use_tiling: bool = False
    tile_grid: int = 2

    # --- ROI ---
    use_roi: bool = False

    # --- Cy5 multichannel ---
    # When a Cy5 channel is loaded, fusion + recovery are auto-enabled
    # by set_cy5_available() in the GUI. We mirror that here.
    use_cy5_fusion: bool = False
    use_cy5_recovery: bool = False
    cy5_filter_mode: str = "multi_metric"   # canonical string id
    cy5_filter_threshold: float = 0.15
    cy5_fusion_jaccard_thresh: float = 0.30
    cy5_fusion_max_overlap_frac: float = 0.50
    cy5_fusion_augment_cpsam: bool = False

    # --- Analytics compute flags ---
    compute_tracking: bool = True
    compute_morphology: bool = True
    compute_edge: bool = True
    compute_boundary: bool = True
    compute_area_stability: bool = True
    compute_membrane: bool = False
    compute_vampire: bool = False
    vampire_n_clusters: int = 5
    compute_state_classification: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    def pixel_thresholds(self, um_per_px: Optional[float] = None,
                          time_interval_min: Optional[float] = None,
                          ) -> dict:
        """Convert physical-unit defaults to per-recording pixel values.

        When `um_per_px` is None we just return the raw `_px` defaults
        unchanged (preserves behaviour for code that doesn't know the
        recording's scale yet).

        Returns a dict with the keys callers actually pass to the
        pipeline: ``min_area_px``, ``max_hop_px``, ``search_radius_px``,
        ``cell_diameter_px``.
        """
        if not um_per_px or um_per_px <= 0:
            return {
                "min_area_px": self.min_area_px,
                "max_hop_px": int(round(
                    self.max_hop_um_per_min
                    * (time_interval_min or 1.0)
                    / 1.0)) if time_interval_min else 150,
                "search_radius_px": self.search_radius_px,
                "cell_diameter_px": None,
            }
        ums_per_px2 = um_per_px ** 2
        min_area_px = max(1, int(round(self.min_area_um2 / ums_per_px2)))
        cell_diameter_px = int(round(
            self.expected_cell_diameter_um / um_per_px))
        # Max hop = (max speed µm/min) * (frame interval min) / um_per_px
        if time_interval_min and time_interval_min > 0:
            max_hop_px = max(10, int(round(
                self.max_hop_um_per_min * time_interval_min
                / um_per_px)))
        else:
            max_hop_px = self.search_radius_px
        search_radius_px = max(10, int(round(
            self.search_radius_um / um_per_px)))
        return {
            "min_area_px": min_area_px,
            "max_hop_px": max_hop_px,
            "search_radius_px": search_radius_px,
            "cell_diameter_px": cell_diameter_px,
        }

    def with_cy5_available(self, has_cy5: bool) -> "PipelineDefaults":
        """Return a copy with Cy5-specific defaults flipped on when
        a fluorescence channel is present. Matches the GUI's
        `set_cy5_available()` behaviour exactly."""
        if not has_cy5:
            return self
        from dataclasses import replace
        return replace(
            self,
            use_cy5_fusion=True,
            use_cy5_recovery=True,
            cy5_filter_mode="multi_metric",
        )


# Frozen singleton for convenience.
DEFAULTS = PipelineDefaults()


def get_defaults(has_cy5: bool = False) -> PipelineDefaults:
    """Return the canonical defaults, with Cy5 paths auto-enabled
    when a fluorescence channel is loaded."""
    return DEFAULTS.with_cy5_available(has_cy5)


# ------------------------------------------------------------------
# Cy5 filter mode IDs (used by core.cy5_filter.apply_cy5_filter and
# params_panel's dropdown — keep in lockstep).
# ------------------------------------------------------------------
CY5_FILTER_MODES = (
    "off",
    "conservative",
    "conservative_strict",
    "adaptive",
    "adaptive_loose",
    "multi_metric",
    "composite_score",
    "consensus",
    "temporal_stability",
    "threshold",
)


# ------------------------------------------------------------------
# Diff helper: compare an actual params dict against the defaults so
# RUN_METADATA.md can record only the deviations.
# ------------------------------------------------------------------
def diff_from_defaults(params: dict,
                        has_cy5: bool = False) -> dict:
    """Return only the keys in `params` whose values differ from the
    canonical defaults. Used by write_run_metadata to produce a
    compact, readable summary."""
    base = get_defaults(has_cy5).as_dict()
    out = {}
    for k, v in params.items():
        if k not in base:
            out[k] = v
        elif base[k] != v:
            out[k] = {"used": v, "default": base[k]}
    return out
