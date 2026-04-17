"""Dataclasses holding every user-adjustable pipeline parameter.

These are the single source of truth for "what does the user want this
run to do". Both GUIs construct a `RunParams` from their option panels
and pass it through workers into the core pipeline. Serializing to JSON
enables presets and reproducible run logs.

Defaults mirror config.py. User-facing labels live in the panels, not here.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
import json

from config import (
    CELLPOSE_MODEL_NAME, CELLPOSE_FLOW_THRESHOLD, CELLPOSE_CELLPROB_THRESHOLD,
    EDGE_SNAP_DEFAULT_SEARCH, EDGE_SNAP_DEFAULT_MAX_STEP,
    EDGE_SNAP_DEFAULT_SMOOTH_SIGMA,
    FOURIER_N_DESCRIPTORS, TEMPORAL_SMOOTH_SIGMA,
    MIN_CELL_AREA_PX,
)


# --- Detection ---

@dataclass
class DetectionParams:
    """All cellpose detection settings."""
    # Mode: "default" (cellpose+flow), "cascade", "threshold_retry"
    mode: str = "default"

    # Cellpose thresholds (primary pass)
    flow_threshold: float = CELLPOSE_FLOW_THRESHOLD
    cellprob_threshold: float = CELLPOSE_CELLPROB_THRESHOLD
    # Cellpose `diameter` hint in pixels. None = auto-estimate.
    # Critical for cells that don't match the model's training scale
    # (e.g., diameter=50 lifted Ignasi IC293 IoU from 0.55 → 0.84).
    diameter: float = 0.0  # 0 means None (auto)
    # Phase 15c — cellpose built-in test-time augmentation.
    # +1 detected frame / +317 px-per-frame on pos0_wt, 0.8× runtime.
    # Stacks with mode="tiled" for larger wins on OOD recordings.
    detect_augment: bool = False

    # Model selection
    model_name: str = CELLPOSE_MODEL_NAME
    model_path: Optional[str] = None  # None = use MODEL_DIR/model_name

    # Minimum mask area (px) to count as a real detection
    min_area_px: int = MIN_CELL_AREA_PX

    # Optical-flow fusion (default mode)
    use_smart_fusion: bool = False
    max_flow_weight: float = 1.0
    flow_method: str = "farneback"  # "farneback" | "farneback_clean" | "framediff"

    # threshold_retry mode
    retry_cellprob_thresholds: List[float] = field(
        default_factory=lambda: [-2.0, -4.0]
    )
    interpolate_gaps: bool = True  # fill remaining empty frames temporally

    # cascade mode
    cascade_gt_model_path: Optional[str] = None  # None = auto-detect

    # Expected number of cells per frame (0 = unknown/auto, 1 = single-cell).
    # When > 0, used for post-detection filtering: keep only the top-N
    # largest cells per frame, treating the rest as debris.
    # 0 = no filtering, detect as many cells as the model finds.
    expected_cells: int = 0

    # --- Preprocessing (applied to frames BEFORE cellpose) ---
    # None/empty = disabled. See `core.preprocess.preprocess_sequence`.
    preprocess_temporal_method: Optional[str] = None  # 'median'/'mean'/'min'/'max'/None
    preprocess_spatial_highpass_sigma: float = 0.0     # 0 = disabled
    preprocess_debris_diameter_um: float = 0.0         # 0 = disabled
    preprocess_pixel_size_um: float = 0.0              # 0 = use recording's value


# --- Refinement ---

@dataclass
class RefinementParams:
    """All boundary-refinement settings."""
    # Mode: "iterative" | "fixed" | "snap_only" | "rf_only" | "none"
    # | "alt_method" — use a classical alt-segmentation method (see
    #   `alt_segmentation.METHODS`) instead of the RF stack.
    mode: str = "fixed"

    # --- Cropping ---
    # crop_mode: "none" | "global" | "per_cell"
    #   global   — one bbox (union of all masks + padding) for the whole stack
    #   per_cell — each frame gets its own padded bbox (better for cells that
    #              move a lot, and for per-frame classical methods)
    crop_mode: str = "global"
    # Kept for backwards compat — True = "global", False = "none"
    use_crop: bool = True
    crop_padding_px: int = 50
    per_cell_padding_px: int = 30

    # --- Alt-segmentation method (only used when mode="alt_method") ---
    # Key from alt_segmentation.METHODS. Examples:
    #   "gmm_3"  (best non-RF, overall IoU 0.639 in benchmarks)
    #   "morph_gradient_r5"
    #   "local_std_sigma4"  (best on cKO)
    #   "otsu", "adaptive", "multi_otsu_3", "canny_fill",
    #   "watershed", "kmeans_3"
    alt_method: str = "gmm_3"

    # --- Hybrid RF method (only used when mode="hybrid") ---
    # Key from hybrid_rf.METHODS:
    #   "localstd_rf" — local-std seed → RF boundary refinement
    #   "gmm_rf"      — GMM seed → RF boundary refinement
    #   "ensemble"    — avg(RF, local_std, GMM) → threshold
    hybrid_method: str = "gmm_rf"
    hybrid_rf_threshold: float = 0.6  # threshold for RF-refined band
    hybrid_ensemble_threshold: float = 0.5  # for ensemble mode
    hybrid_boundary_dilate: int = 10  # seed-mask dilation for band
    hybrid_boundary_erode: int = 5    # seed-mask erosion for band

    # --- SAM refinement (only used when mode="sam") ---
    # Model type: 'vit_b' (ships, ~360 MB) | 'vit_l' | 'vit_h'
    #   or SAM2 variants: 'hiera_t' | 'hiera_s' | 'hiera_b+' | 'hiera_l'
    sam_model_type: str = "vit_b"
    # "v1" (original SAM) or "v2" (SAM2 with hiera backbone).
    # When sam_version="v2", sam_model_type must be one of the hiera_*.
    sam_version: str = "v1"
    # Whether to pass the seed mask as a low-res logit mask prompt
    # (in addition to centroid point + bbox). Usually helps.
    sam_use_mask_prompt: bool = True

    # Pre-refinement stage. If True, runs MedSAM on the detection masks
    # BEFORE the configured refinement mode. The resulting tightened
    # masks are passed as input to the chosen refiner (Chan-Vese, RF,
    # snap, etc.). Lets users stack MedSAM's bbox-precise boundary
    # with a finishing refiner. On Ignasi: cellpose+MedSAM alone hits
    # 0.860 IoU (15/15 > 0.8); +Chan-Vese can push specific frames.
    pre_refine_medsam: bool = False

    # Post-MedSAM: combine with DeepSea pretrained segmentation via
    # pixel-OR + fill + largest component. Recovers boundary pixels
    # MedSAM under-segments (Ignasi: 0.860 → 0.885 mean IoU,
    # 9/15 → 14/15 frames > 0.85). Requires DeepSea ckpt at
    # data/models/deepsea/segmentation.pth.
    union_with_deepsea: bool = False

    # --- RF filter bank (reserved for future on-the-fly RF retraining) ---
    # "default" = use the saved rf_boundary_model.pkl
    # Other values will trigger per-run RF retraining once that feature
    # lands. Values match test_segmentation_methods FILTER_CONFIGS:
    # "fine_scale", "minimal_texture", "edge_focused", "texture_rich",
    # "gabor_bank", "kitchen_sink", "coarse_scale".
    rf_filter_bank: str = "default"

    # Per-step toggles (applies to fixed and iterative modes)
    use_rf: bool = True
    use_snap: bool = True
    use_fourier: bool = True
    use_crf: bool = True
    use_temporal: bool = True

    # RF isoline
    rf_threshold: float = 0.75
    rf_use_isoline: bool = True

    # Edge snap
    snap_search_radius: int = EDGE_SNAP_DEFAULT_SEARCH
    snap_max_step: int = EDGE_SNAP_DEFAULT_MAX_STEP
    snap_smooth_sigma: float = EDGE_SNAP_DEFAULT_SMOOTH_SIGMA
    snap_use_membrane_edge: bool = False
    snap_membrane_alpha: float = 0.7

    # Fourier contour smoothing
    fourier_n_descriptors: int = FOURIER_N_DESCRIPTORS

    # Temporal boundary smoothing
    temporal_sigma: float = TEMPORAL_SMOOTH_SIGMA

    # Cascade-aware: skip refinement on frames detected by GT model
    skip_cascade_gt_frames: bool = True


# --- Analysis ---

@dataclass
class AnalysisParams:
    """Which downstream analytics to compute + scale overrides."""
    # Per-analytic switches
    compute_tracking: bool = True
    compute_morphology: bool = True
    compute_edge_dynamics: bool = True
    compute_boundary_confidence: bool = True
    compute_area_stability: bool = True
    compute_membrane_quality: bool = False  # opt-in (slower)

    # Scale/timing overrides (None = use recording JSON sidecar)
    override_um_per_px: Optional[float] = None
    override_time_interval_min: Optional[float] = None


# --- Top-level ---

@dataclass
class RunParams:
    """Complete parameter set for one pipeline run."""
    detection: DetectionParams = field(default_factory=DetectionParams)
    refinement: RefinementParams = field(default_factory=RefinementParams)
    analysis: AnalysisParams = field(default_factory=AnalysisParams)

    # Preset provenance (for the run log)
    preset_name: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "RunParams":
        d = dict(data)
        det = DetectionParams(**d.pop("detection", {}))
        ref = RefinementParams(**d.pop("refinement", {}))
        ana = AnalysisParams(**d.pop("analysis", {}))
        return cls(detection=det, refinement=ref, analysis=ana,
                   preset_name=d.get("preset_name", "default"))

    @classmethod
    def from_json(cls, s: str) -> "RunParams":
        return cls.from_dict(json.loads(s))
