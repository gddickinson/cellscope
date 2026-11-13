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
     Values like `min_area_px`, `use_cy5_fusion`, `cy5_filter_mode`,
     the tracker weights, etc. live here.

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
from dataclasses import dataclass, asdict
from typing import Optional

# Re-exported so `core.pipeline_defaults` stays the single import point.
from core.pipeline_policy import (  # noqa: F401
    DOWNSAMPLE_LARGE_PX, DOWNSAMPLE_SMALL_PX, MIRROR_PAD_MIN_DIM_PX, MIRROR_PAD_PX,
    MULTICELL_THRESHOLD, N_SAMPLE_FRAMES_FOR_AUTO, PROBE_AGGREGATOR_PERCENTILE,
    decide_backbone, resolve_downsample, should_use_mirror_pad)
from core.auto_select import (  # noqa: F401
    DIC_MODEL_PREFERENCE, _PROBE_SCRIPT, _run_probe_subprocess, resolve_dic_model_path,
    select_pipeline_for_recording)


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
    #   endothelial cells can project to ~80-150 µm² in DIC).
    min_area_um2: float = 85.0

    # Expected cell diameter (typical spread endothelial cell). Cellpose
    # uses this to internally scale features so the model's training-
    # diameter prior matches the input. None = let cellpose auto-
    # estimate from image content (works at IC295 scale).
    expected_cell_diameter_um: float = 30.0

    expected_cells: int = 0   # 0 = unknown, scan_cell_count() may set
    search_radius_px: int = 150
    # Physical-unit equivalent for tracker max-hop and gap-fill search.
    # Endothelial cells move <0.5-2 µm/min on DIC time-lapse; our
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
    # Hungarian cost-matrix weights (core.multi_cell.track_all_cells):
    #   cost = w_dist * distance + w_iou * (1 - IoU) * max_hop
    #          + w_area * |d area| / max(area) * max_hop
    # These are the weights the tracker has used since the IoU + area
    # terms were wired in (commit 60ea19c, Pos7-WT GT ID consistency
    # 0.88 -> 0.97); they were literals in the function signature until
    # 2026-11-12 and moved here so every caller reads one source.
    track_w_dist: float = 1.0
    track_w_iou: float = 1.0
    track_w_area: float = 0.5

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
    # Gap-fill Phase 1 acceleration: segment only an adaptive crop
    # around the interpolated centroid instead of the whole frame.
    # Phase 1 only ever accepts a cell within `search_radius` of that
    # centroid, so a crop of half-size (search_radius + cell-radius +
    # margin) provably contains every acceptable candidate — same
    # recall, ~(frame/crop)² less compute. Falls back to the full frame
    # when the adaptive crop isn't smaller. Validated vs reviewed-mask
    # GT (scripts/bench_gap_fill.py).
    gap_fill_crop: bool = True
    # Gap-fill Phase 1 augment: run cpsam with augment=True (4-rotation
    # TTA = 4 forward passes) only as a FALLBACK when the fast
    # augment=False pass misses. augment is the dominant per-gap cost at
    # the downsampled production resolution — it's a call count, not
    # pixels, so it cuts time even when the crop can't engage. GT
    # validated (scripts/bench_gap_fill.py, --downsample 2): crop +
    # no-augment matches crop + augment (0 good fills dropped, IoU
    # Δ −0.011) at ~2.1×; the augment fallback keeps recall ≥ the
    # always-augment path. Set True to always augment (old behaviour).
    gap_fill_augment: bool = False

    # cpsam weight precision. cellpose 4.x defaults to bfloat16 (16-bit).
    # On NVIDIA/CUDA bf16 is the fast path; on Apple Silicon (MPS) it is
    # the SLOW path — fp32 (use_bfloat16=False) is ~1.26× faster there
    # (validated across 5 IC295 recordings, density 2→19 cells/frame,
    # 2026-06-06). Default kept at True (bf16) = cellpose's default /
    # current behaviour — flip to False to opt into the MPS fp32 speedup.
    # Per-frame foreground IoU vs reviewed masks was a wash (mean Δ
    # +0.0004; dense recordings ~identical) but NOT certified loss-free
    # at the tracked-output level, so this stays opt-in until a
    # full-pipeline F1 validation gates a default change.
    use_bfloat16: bool = True

    # --- Mirror padding (auto-enabled on large detection frames) ---
    # When the detection-resolution image is large enough that the
    # reflection border is a small fraction of the frame
    # (min(H, W) ≥ MIRROR_PAD_MIN_DIM_PX), pad the image by
    # MIRROR_PAD_PX before each cpsam call so cells right at the FoV
    # edge sit in the interior of a "complete" reflection — cpsam
    # segments them as full cells instead of partial profiles.
    # Validated on 9 GT recordings (2026-05-21): +0.018 mean F1
    # across 6 IC295 recordings (2048² → ds=2 → 1024² detection),
    # with a small -0.016 regression on Pos20_KO that's outweighed
    # by gains of +0.006 to +0.034 on the other 5. SKIPPED on small
    # legacy ignasi recordings (438² or 1028×828) where the
    # reflection border is too large a fraction of the frame.
    # "auto" → enabled iff min_det_dim ≥ MIRROR_PAD_MIN_DIM_PX.
    use_mirror_pad: str = "auto"      # "auto" | "on" | "off"

    # --- cpsam-on-Cy5 union (Method D, 2026-05-22) ---
    # Run cpsam on the Cy5 channel too (when present), union the
    # detected cells with the DIC-detected cells via NMS. The
    # *bbox-detection* validation showed +0.11-0.26 F1 on dense
    # recordings (Pos68_DMSO, Pos31_GOF), but the full-pipeline
    # re-run (2026-05-23) showed those gains EVAPORATED:
    #   Pos68_DMSO: 0.56 → 0.56  (no change after Cy5 filter)
    #   Pos31_GOF:  0.52 → 0.52  (no change)
    # The downstream multi-metric Cy5 filter rejects cpsam(Cy5)-
    # detected cells because their Cy5 signal characteristics don't
    # match what the filter expects from threshold-based Cy5 fusion.
    # Default OFF until the filter is tuned to admit cpsam(Cy5) cells
    # (queued investigation, 2026-05-23). Cost when enabled: ~1.5×
    # runtime.
    use_cpsam_cy5_union: bool = False

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
    cy5_filter_mode: str = "persistence_guard"   # canonical string id
    cy5_filter_threshold: float = 0.15
    # persistence_guard v2 sub-parameters (see core/cy5_filter.py
    # persistence_guard_filter docstring for full rationale):
    cy5_pg_min_lifetime: int = 35
    cy5_pg_static_velocity_px: float = 3.0
    cy5_pg_static_shape_iou: float = 0.85
    # Cy5 fusion sub-parameters:
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
    # On by default: the endothelial cell motility analysis is confounded
    # by balled-up (rounded mitotic / dying) cells, so every GUI now
    # stratifies motility by cell state out of the box — matching the
    # IC295 batch (scripts/ic295_analyze_one.py runs state unless
    # --no-state). Turn off per-run if you only need bulk metrics.
    compute_state_classification: bool = True

    # --- Single-cell curation (opt-in; for curated single-cell crops) ---
    # When a recording is hand-cropped to ONE target cell under known
    # priors (isolated, flattened, non-dividing, present every frame,
    # roughly centred), run core.single_cell_curation AFTER detection to
    # assemble that one cell's track: texture-based artifact + optical-
    # shadow rejection, label-agnostic ID-stitching, SAM2 gap recovery,
    # and tracking through rounded states. Flags (never drops) exceptions
    # (division / two persistent cells / unrecoverable frames). OFF by
    # default → ordinary multi-cell detection is completely unchanged.
    # (Built + validated on the IC293 EC-migration crops, 2026-06-17.)
    single_cell_curation: bool = False
    sc_present_every_frame: bool = True
    sc_no_dividing: bool = True
    sc_isolated: bool = True
    sc_roughly_centered: bool = True
    sc_expected_cell_area_um2: float = 0.0   # 0 = infer per recording

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
            cy5_filter_mode="persistence_guard",
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
