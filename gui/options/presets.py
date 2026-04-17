"""Named parameter presets.

Built-in presets capture recommended configurations from the project's
experimental history (SESSION_LOG.md). Users can also save/load their
own presets as JSON files in data/presets/.
"""
import os
import json
from typing import Dict, List

from config import PROJECT_DIR
from gui.options.params import (
    RunParams, DetectionParams, RefinementParams, AnalysisParams,
)

USER_PRESET_DIR = os.path.join(PROJECT_DIR, "data", "presets")


# --- Built-in presets ---

def _recommended_pipeline() -> RunParams:
    """Phase 13 benchmark winner — best overall pipeline.

    Stage 1: cascade detection (GT model → cellpose_dic → interpolate)
             — 0.740 IoU, 100% detection rate on our recordings.
    Stage 2: hybrid_gmm_rf refinement using the **default_combined**
             RF bank (trained on our manual GT + Jesse VAMPIRE crops).
             This is the current best — 0.356 mean IoU on held-out
             crops (+14% vs the Phase 12 version; +39% on GoF).

    This is the default for any NEW analysis unless you have a specific
    reason to choose otherwise. See `results/evaluation/combined_bank_
    comparison.csv` for the benchmark.
    """
    return RunParams(
        detection=DetectionParams(
            mode="cascade",
            flow_threshold=0.0,
            cellprob_threshold=0.0,
        ),
        refinement=RefinementParams(
            mode="hybrid",
            hybrid_method="gmm_rf",
            hybrid_rf_threshold=0.6,
            hybrid_boundary_dilate=10,
            hybrid_boundary_erode=5,
            crop_mode="per_cell",
            per_cell_padding_px=30,
            rf_filter_bank="default_combined",
            skip_cascade_gt_frames=True,
        ),
        analysis=AnalysisParams(),
        preset_name="★ Recommended pipeline (Phase 13 winner)",
    )


def _rf_only_combined() -> RunParams:
    """Fastest strong pure-RF refinement on the combined dataset.

    Phase 13c overall winner at **0.367 mean IoU** on the held-out
    OOD test — slightly edges out default_combined (0.362). Use this
    when the recording matches the training distribution (our DIC
    recordings, Jesse SparseSlow recordings); weaker on GoF so use
    the Recommended hybrid preset for heterogeneous data.
    """
    return RunParams(
        detection=DetectionParams(
            mode="cascade",
            flow_threshold=0.0,
            cellprob_threshold=0.0,
        ),
        refinement=RefinementParams(
            mode="fixed",
            rf_filter_bank="best_combo_combined",
            rf_threshold=0.5,
            rf_use_isoline=True,
            crop_mode="global",
            skip_cascade_gt_frames=True,
        ),
        analysis=AnalysisParams(),
        preset_name="RF only (best_combo_combined, fast)",
    )


def _best_for_our_recordings() -> RunParams:
    """Legacy optimal pipeline for our full-frame DIC keratinocyte videos.

    Matches SESSION_LOG Test 20/21: cellpose_dic at ft=0.0 ct=0.0 →
    RF isoline refinement. Crop enabled for speed.

    Kept as an alternative for users who prefer the full-stack RF
    refinement. For the Phase 12 winner see "Recommended pipeline".
    """
    return RunParams(
        detection=DetectionParams(
            mode="default",
            flow_threshold=0.0,
            cellprob_threshold=0.0,
        ),
        refinement=RefinementParams(
            mode="fixed",
            use_crop=True,
            use_rf=True, use_snap=True, use_fourier=True,
            use_crf=True, use_temporal=True,
        ),
        analysis=AnalysisParams(),
        preset_name="Best for our recordings (legacy RF stack)",
    )


def _best_for_jesse() -> RunParams:
    """Threshold-retry detection for pre-cropped Jesse data.

    Recovers missed Jesse frames (74% → 96% detection at ft=0.0)
    without regressing our data, and fills any remaining gaps.
    """
    return RunParams(
        detection=DetectionParams(
            mode="threshold_retry",
            flow_threshold=0.0,
            cellprob_threshold=0.0,
            retry_cellprob_thresholds=[-2.0, -4.0],
            interpolate_gaps=True,
        ),
        refinement=RefinementParams(
            mode="fixed",
            use_crop=True,
        ),
        analysis=AnalysisParams(),
        preset_name="Best for Jesse",
    )


def _fastest() -> RunParams:
    """Minimum-latency pipeline: cellpose → snap only → analyze."""
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="snap_only",
            use_crop=True,
        ),
        analysis=AnalysisParams(compute_membrane_quality=False),
        preset_name="Fastest",
    )


def _cascade_no_refine() -> RunParams:
    """Cascade detection (GT → original → interpolate) with no refinement.

    Produces the best RAW boundaries (SESSION_LOG: ctrl IoU 0.872).
    Good for visual inspection of what cellpose+GT-retrain can do alone.
    """
    return RunParams(
        detection=DetectionParams(mode="cascade"),
        refinement=RefinementParams(mode="none", use_crop=False),
        analysis=AnalysisParams(),
        preset_name="Cascade (no refine)",
    )


def _full_pipeline_iterative() -> RunParams:
    """Threshold-retry detect + iterative refinement (3 presets tried)."""
    return RunParams(
        detection=DetectionParams(
            mode="threshold_retry",
            flow_threshold=0.0,
            cellprob_threshold=0.0,
        ),
        refinement=RefinementParams(
            mode="iterative",
            use_crop=True,
        ),
        analysis=AnalysisParams(compute_membrane_quality=True),
        preset_name="Full pipeline (iterative)",
    )


def _minimal_detection_only() -> RunParams:
    """Just cellpose detection with no refinement, for debugging."""
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(mode="none"),
        analysis=AnalysisParams(
            compute_morphology=False,
            compute_edge_dynamics=False,
            compute_membrane_quality=False,
        ),
        preset_name="Minimal (detection only)",
    )


def _alt_method_gmm_per_cell() -> RunParams:
    """Classical GMM segmentation on per-cell crops (strongest non-RF)."""
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="alt_method",
            alt_method="gmm_3",
            crop_mode="per_cell",
            per_cell_padding_px=30,
            use_crop=True,
        ),
        analysis=AnalysisParams(),
        preset_name="Alt: GMM per-cell",
    )


def _tiled_tta_detection() -> RunParams:
    """Tiled (3x3_o64) + TTA (cellpose augment=True) — strongest
    detection on OOD recordings.

    Phase 15a sweep winner (tiled 3x3_o64) + Phase 15c TTA (+1
    detected frame, +317 px on pos0_wt) stacked.
    """
    return RunParams(
        detection=DetectionParams(mode="tiled", detect_augment=True),
        refinement=RefinementParams(mode="none"),
        analysis=AnalysisParams(),
        preset_name="★ Tiled + TTA (strongest OOD detection)",
    )


def _tiled_detection() -> RunParams:
    """Tiled (2×2 with overlap) cellpose detection — Phase 14i winner
    for large/OOD recordings where internal cellpose tiling misses cells.

    On Jesse 1024² recordings: doubles detection rate (5/10 → 10/10 on
    pos0_wt) and adds 2× pixels vs full-frame. Runs 2.5× faster
    (smaller tiles = faster per-tile inference).
    """
    return RunParams(
        detection=DetectionParams(mode="tiled"),
        refinement=RefinementParams(mode="none"),
        analysis=AnalysisParams(),
        preset_name="★ Tiled detection (large/OOD recordings)",
    )


def _ignasi_tuned_plus_medsam() -> RunParams:
    """Ignasi-tuned detection + MedSAM refinement.

    On Ignasi's 15 GT frames: cellpose_combined_robust @ diameter=50,
    cellprob=-2.0 alone gets 0.838 mean IoU (12/15 frames > 0.8).
    Adding MedSAM refinement lifts to **0.860 mean and 15/15 frames
    above 0.8 IoU** — eliminates all weak-detection outliers by
    pulling the foundation-model refiner across the boundary.

    Use for cropped single-cell DIC recordings where every frame
    needs reliable boundary quality.
    """
    return RunParams(
        detection=DetectionParams(
            mode="default",
            model_name="cellpose_combined_robust",
            cellprob_threshold=-2.0,
            diameter=50.0,
        ),
        refinement=RefinementParams(
            mode="none",
            pre_refine_medsam=True,  # detection → MedSAM → (no further)
            crop_mode="none",
        ),
        analysis=AnalysisParams(),
        preset_name="★ Ignasi-tuned + MedSAM (15/15 > 0.8 IoU)",
    )


def _ignasi_hybrid_cpsam_multi() -> RunParams:
    """Multi-cell hybrid cpsam — for recordings with 2-3 cells.

    Pipeline:
      1. cpsam at defaults → keep ALL instance labels
      2. Debris filter (area < 500 px)
      3. Missed frames → cellpose fallback (subprocess in cellpose env)
      4. Per-cell DeepSea refinement (preserves multi-cell identity)
      5. Hungarian tracker (cross-frame identity)

    Tested on Ignasi Pos2-WT (2 cells) and Pos3-WT (2→3, division).

    REQUIRES cellpose4 env active + cellpose env available.
    """
    return RunParams(
        detection=DetectionParams(mode="hybrid_cpsam_multi"),
        refinement=RefinementParams(
            mode="none", crop_mode="none",
            pre_refine_medsam=False,
            union_with_deepsea=False,
        ),
        analysis=AnalysisParams(),
        preset_name="★ Ignasi multi-cell cpsam (2-3 cells)",
    )


def _ignasi_hybrid_cpsam() -> RunParams:
    """Hybrid cpsam + cellpose fallback — best on Ignasi.

    Pipeline:
      1. cpsam at defaults (ViT, cellpose4 env)
      2. Missed frames (area < 500 px) → cellpose_combined_robust
         + MedSAM + DeepSea (via subprocess in cellpose env)
      3. DeepSea union on all frames

    On Ignasi 97-frame recording: 0 missed frames in GT subset,
    2 missed in full recording (frames 78, 80) rescued by fallback.
    Combines cpsam's 0.932 IoU (65 GT) with 100% frame coverage.

    REQUIRES:
      - cellpose4 env active (cellpose >= 4, cpsam)
      - cellpose env available (cellpose 3.x, for CP3 model fallback)
    """
    return RunParams(
        detection=DetectionParams(mode="hybrid_cpsam"),
        refinement=RefinementParams(
            mode="none", crop_mode="none",
            pre_refine_medsam=False,
            union_with_deepsea=False,
        ),
        analysis=AnalysisParams(),
        preset_name="★ Ignasi hybrid cpsam (best, auto-fallback)",
    )


def _ignasi_cpsam() -> RunParams:
    """Cellpose-SAM (cpsam, ViT default in cellpose 4.1.1+).

    On Ignasi 65 GT: mean IoU **0.863** as union of all detections
    (debris penalises union); equivalent to 0.915 if largest-CC is
    applied (cpsam alone, no refinement). Use the cpsam+DeepSea
    union preset instead — it eliminates debris automatically and
    pushes all 65 frames > 0.85.

    REQUIRES the `cellpose4` env (cellpose >= 4):
      conda run -n cellpose4 python main.py
    """
    return RunParams(
        detection=DetectionParams(mode="cpsam"),
        refinement=RefinementParams(mode="none", crop_mode="none"),
        analysis=AnalysisParams(),
        preset_name="★ Ignasi cpsam alone (cellpose 4.1.1 ViT)",
    )


def _ignasi_cpsam_deepsea() -> RunParams:
    """Cellpose-SAM + DeepSea union — best on Ignasi.

    On Ignasi 65 GT: mean IoU **0.932**, **65/65 > 0.85**,
    55/65 > 0.9, min 0.867. Beats cpsam-alone (+0.069) and the
    previous cellpose+MedSAM+DeepSea (+0.025) by clean margins.

    DeepSea union also removes debris — it fills holes then keeps
    largest CC, so cpsam's spurious tiny components get dropped
    automatically.

    REQUIRES the `cellpose4` env (cellpose >= 4):
      conda run -n cellpose4 python main.py

    DO NOT add MedSAM — it tightens cpsam's already-good boundaries
    and DROPS mean IoU by 0.148 (C1 in the combinations benchmark).
    """
    return RunParams(
        detection=DetectionParams(mode="cpsam"),
        refinement=RefinementParams(
            mode="none",
            pre_refine_medsam=False,
            union_with_deepsea=True,
            crop_mode="none",
        ),
        analysis=AnalysisParams(),
        preset_name="★ Ignasi cpsam + DeepSea union (65/65 > 0.85, 0.932)",
    )


def _ignasi_medsam_deepsea_union() -> RunParams:
    """Cellpose + MedSAM + DeepSea union — best Ignasi result.

    Empirical (15 GT frames):
      cellpose+MedSAM:        0.860 mean IoU, 9/15 > 0.85
      + DeepSea union (this): 0.885 mean IoU, 14/15 > 0.85

    Mechanism: MedSAM under-segments by ~5% on phase-contrast
    cytoplasm; DeepSea (trained on phase-contrast) catches the missed
    boundary pixels. Pixel-OR + fill + largest connected component.
    """
    return RunParams(
        detection=DetectionParams(
            mode="default",
            model_name="cellpose_combined_robust",
            cellprob_threshold=-2.0,
            diameter=50.0,
        ),
        refinement=RefinementParams(
            mode="none",
            pre_refine_medsam=True,
            union_with_deepsea=True,
            crop_mode="none",
        ),
        analysis=AnalysisParams(),
        preset_name="★ Ignasi MedSAM + DeepSea union (14/15 > 0.85 IoU)",
    )


def _ignasi_tuned() -> RunParams:
    """Ignasi IC293 tuned: cellpose_combined_robust with diameter=50
    + cellprob_threshold=-2.0. Diameter sweep on 15 manual GT frames
    found this combo lifts mean IoU from 0.55 → 0.84 with 15/15
    detection (12/15 frames at IoU > 0.8).

    Use when cells appear larger or smaller than the model's default
    trained scale (typical sign: low detection rate or detected
    masks much smaller than the visible cell).
    """
    return RunParams(
        detection=DetectionParams(
            mode="default",
            model_name="cellpose_combined_robust",
            cellprob_threshold=-2.0,
            diameter=50.0,
        ),
        refinement=RefinementParams(mode="none"),
        analysis=AnalysisParams(),
        preset_name="★ Ignasi-tuned (combined_robust diameter=50 ct=-2)",
    )


def _robust_v1_plus_chan_vese() -> RunParams:
    """cellpose_combined_robust (v1) + Chan-Vese per-cell refinement.

    Pipeline that won on Ignasi's IC293 cropped single-cell recording
    (cellpose_combined_robust beat cellpose_robust_v2 on real OOD
    crops; Chan-Vese tightens the boundary). Use for cropped single-
    cell DIC recordings from other labs / different microscopes.

    Per-frame analytics (area, speed, edge dynamics) are produced
    by the Analysis tab once Refine + Analyze runs.
    """
    return RunParams(
        detection=DetectionParams(
            mode="default",
            model_name="cellpose_combined_robust",
        ),
        refinement=RefinementParams(
            mode="alt_method",
            alt_method="chan_vese",
            crop_mode="per_cell",
            per_cell_padding_px=30,
        ),
        analysis=AnalysisParams(),
        preset_name="★ Cropped single-cell (robust v1 + Chan-Vese)",
    )


def _robust_cellpose_detection() -> RunParams:
    """Cellpose robust_v2 — retrained with extended augmentation
    (Phase 13d.4, includes elastic deformation + motion blur +
    defocus blur on top of the v1 augmentation set).

    Wins on 9 of 11 perturbation types vs v1 (cellpose_combined_robust):
    +0.064 gamma_bright, +0.048 gamma_dark, +0.045 noise_s8, +0.034
    contrast_low, +0.025 combined, +0.023 contrast_high, +0.019
    noise_s16, +0.009 bgscramble, +0.036 clean. Both fail on brightness
    ±30 (outside trained distribution).

    On clean 30-frame test: 0.733 IoU (v1 was 0.700; cascade is 0.766).
    """
    return RunParams(
        detection=DetectionParams(
            mode="default",
            model_name="cellpose_robust_v2",
        ),
        refinement=RefinementParams(mode="none"),
        analysis=AnalysisParams(),
        preset_name="★ Robust detection (noise-tolerant, cellpose_robust_v2)",
    )


def _alt_method_chan_vese_per_cell() -> RunParams:
    """Morphological Chan-Vese refinement — best non-RF for cKO (Phase 14f)."""
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="alt_method",
            alt_method="chan_vese",
            crop_mode="per_cell",
            per_cell_padding_px=30,
            use_crop=True,
        ),
        analysis=AnalysisParams(),
        preset_name="Alt: Chan-Vese per-cell (cKO winner)",
    )


def _alt_method_watershed_per_cell() -> RunParams:
    """Watershed seeded from cellpose markers, per-cell crop."""
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="alt_method",
            alt_method="watershed",
            crop_mode="per_cell",
            per_cell_padding_px=30,
        ),
        analysis=AnalysisParams(),
        preset_name="Alt: Watershed per-cell",
    )


def _hybrid_gmm_rf_per_cell() -> RunParams:
    """GMM seed + RF boundary refinement, per-cell crops.

    Strong result in test_hybrid_rf.py — GMM gives a stable interior
    mask, RF handles the tricky boundary band. Uses the currently-
    selected RF filter bank.
    """
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="hybrid",
            hybrid_method="gmm_rf",
            hybrid_rf_threshold=0.6,
            hybrid_boundary_dilate=10,
            hybrid_boundary_erode=5,
            crop_mode="per_cell",
            per_cell_padding_px=30,
            rf_filter_bank="default",
        ),
        analysis=AnalysisParams(),
        preset_name="Hybrid: GMM → RF (per-cell)",
    )


def _hybrid_localstd_rf_per_cell() -> RunParams:
    """Local-std seed + RF boundary refinement, per-cell crops."""
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="hybrid",
            hybrid_method="localstd_rf",
            hybrid_rf_threshold=0.6,
            crop_mode="per_cell",
            per_cell_padding_px=30,
            rf_filter_bank="default",
        ),
        analysis=AnalysisParams(),
        preset_name="Hybrid: LocalStd → RF (per-cell)",
    )


def _hybrid_ensemble_per_cell() -> RunParams:
    """Average of RF + local_std + GMM probabilities, per-cell crops."""
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="hybrid",
            hybrid_method="ensemble",
            hybrid_ensemble_threshold=0.5,
            crop_mode="per_cell",
            per_cell_padding_px=30,
            rf_filter_bank="default",
        ),
        analysis=AnalysisParams(),
        preset_name="Hybrid: Ensemble (per-cell)",
    )


def _rf_bank_texture_rich() -> RunParams:
    """Full stack refinement using the texture_rich RF filter bank."""
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="fixed",
            rf_filter_bank="texture_rich",
            crop_mode="global",
            use_crop=True,
        ),
        analysis=AnalysisParams(),
        preset_name="RF bank: texture_rich",
    )


def _rf_bank_fine_scale_gmm_thresh08() -> RunParams:
    """fine_scale+GMM bank at RF threshold 0.8 (user-validated config).

    From test_hybrid_rf.py — RF features include a GMM cell-probability
    channel; threshold=0.8 produced strong boundaries.
    """
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="fixed",
            rf_filter_bank="fine_scale_gmm",
            rf_threshold=0.8,
            rf_use_isoline=True,
            crop_mode="global",
            use_crop=True,
        ),
        analysis=AnalysisParams(),
        preset_name="RF bank: fine_scale+gmm (thr=0.8)",
    )


def _medsam_refinement() -> RunParams:
    """MedSAM (SAM fine-tuned on biomedical images) refinement.

    Best foundation-model refiner tested: +0.145 IoU over SAM vit_b
    on our 20-frame test (mean 0.605 vs 0.460). Especially strong on
    ctrl (0.749). Uses HF transformers API.
    """
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="medsam",
            crop_mode="per_cell",
            per_cell_padding_px=30,
        ),
        analysis=AnalysisParams(),
        preset_name="MedSAM refinement (HF biomedical)",
    )


def _sam_refinement() -> RunParams:
    """SAM-based refinement (per-cell crop; no full-stack RF).

    Foundation-model alternative to the RF pipeline. Historically
    produces smoother boundaries than RF but may miss fine features.
    """
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="sam",
            sam_model_type="vit_b",
            sam_use_mask_prompt=True,
            crop_mode="per_cell",
            per_cell_padding_px=30,
        ),
        analysis=AnalysisParams(),
        preset_name="SAM refinement (vit_b)",
    )


def _jesse_full_retrain_threshold_retry() -> RunParams:
    """Jesse-focused: splits-trained model + threshold_retry.

    Uses `cellpose_full_retrain` (trained on 2,425 images from
    data/splits/train) as the primary detector — it produces tighter
    boundaries than cellpose_dic on frames it finds. Threshold-retry
    rescues the frames it misses on Jesse-like pre-cropped data by
    re-running with lower cellprob_threshold, then temporal
    interpolation fills any remaining gaps.

    Refinement: full stack with the default RF bank at global crop.
    """
    return RunParams(
        detection=DetectionParams(
            mode="threshold_retry",
            model_name="cellpose_full_retrain",
            flow_threshold=0.0,
            cellprob_threshold=0.0,
            retry_cellprob_thresholds=[-2.0, -4.0],
            interpolate_gaps=True,
        ),
        refinement=RefinementParams(
            mode="fixed",
            use_crop=True,
            crop_mode="global",
        ),
        analysis=AnalysisParams(),
        preset_name="Jesse: full_retrain + threshold_retry",
    )


def _cleaned_detection_fixed() -> RunParams:
    """Preprocessing + default detection + full-stack refinement.

    Temporal median background subtraction and spatial high-pass
    suppress static content and illumination drift before cellpose.
    Useful on recordings with debris, vignetting, or drift.
    """
    return RunParams(
        detection=DetectionParams(
            mode="default",
            preprocess_temporal_method="median",
            preprocess_spatial_highpass_sigma=40.0,
        ),
        refinement=RefinementParams(
            mode="fixed",
            use_crop=True,
            crop_mode="global",
        ),
        analysis=AnalysisParams(),
        preset_name="Cleaned (median bg + HP) + full stack",
    )


def _rf_bank_best_combo() -> RunParams:
    """Cherry-picked bank: fine gradients + coarse local_std + meijering
    + morph_gradient + GMM prob + entropy + intensity.

    Overall IoU 0.792 (ctrl 0.896, cKO 0.687) — not the overall winner
    but balanced, with strong cKO performance for a compact 16-feature
    model.
    """
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="fixed",
            rf_filter_bank="best_combo",
            crop_mode="global",
            use_crop=True,
        ),
        analysis=AnalysisParams(),
        preset_name="RF bank: best_combo",
    )


def _rf_bank_fine_scale_morph_thresh09() -> RunParams:
    """fine_scale+morph bank at RF threshold 0.9.

    From test_hybrid_rf.py — RF features include multi-scale
    morphological gradient; threshold=0.9 gave tight boundaries.
    """
    return RunParams(
        detection=DetectionParams(mode="default"),
        refinement=RefinementParams(
            mode="fixed",
            rf_filter_bank="fine_scale_morph",
            rf_threshold=0.9,
            rf_use_isoline=True,
            crop_mode="global",
            use_crop=True,
        ),
        analysis=AnalysisParams(),
        preset_name="RF bank: fine_scale+morph (thr=0.9)",
    )


BUILTIN_PRESETS: Dict[str, callable] = {
    "★ Recommended pipeline (Phase 13 winner)": _recommended_pipeline,
    "★ Robust detection (noise-tolerant, cellpose_robust_v2)": _robust_cellpose_detection,
    "★ Cropped single-cell (robust v1 + Chan-Vese)": _robust_v1_plus_chan_vese,
    "★ Ignasi-tuned (combined_robust diameter=50 ct=-2)": _ignasi_tuned,
    "★ Ignasi-tuned + MedSAM (15/15 > 0.8 IoU)": _ignasi_tuned_plus_medsam,
    "★ Ignasi multi-cell cpsam (2-3 cells)": _ignasi_hybrid_cpsam_multi,
    "★ Ignasi hybrid cpsam (best, auto-fallback)": _ignasi_hybrid_cpsam,
    "★ Ignasi cpsam + DeepSea union (65/65 > 0.85, 0.932)": _ignasi_cpsam_deepsea,
    "★ Ignasi cpsam alone (cellpose 4.1.1 ViT)": _ignasi_cpsam,
    "★ Ignasi MedSAM + DeepSea union (14/15 > 0.85 IoU)": _ignasi_medsam_deepsea_union,
    "★ Tiled detection (large/OOD recordings)": _tiled_detection,
    "★ Tiled + TTA (strongest OOD detection)": _tiled_tta_detection,
    "RF only (best_combo_combined, fast)": _rf_only_combined,
    "Best for our recordings (legacy RF stack)":
        _best_for_our_recordings,
    "Best for Jesse": _best_for_jesse,
    "Full pipeline (iterative)": _full_pipeline_iterative,
    "Cascade (no refine)": _cascade_no_refine,
    "Fastest": _fastest,
    "Alt: GMM per-cell": _alt_method_gmm_per_cell,
    "Alt: Chan-Vese per-cell (cKO winner)": _alt_method_chan_vese_per_cell,
    "Alt: Watershed per-cell": _alt_method_watershed_per_cell,
    "Hybrid: GMM → RF (per-cell)": _hybrid_gmm_rf_per_cell,
    "Hybrid: LocalStd → RF (per-cell)": _hybrid_localstd_rf_per_cell,
    "Hybrid: Ensemble (per-cell)": _hybrid_ensemble_per_cell,
    "SAM refinement (vit_b)": _sam_refinement,
    "MedSAM refinement (HF biomedical)": _medsam_refinement,
    "Cleaned (median bg + HP) + full stack": _cleaned_detection_fixed,
    "Jesse: full_retrain + threshold_retry":
        _jesse_full_retrain_threshold_retry,
    "RF bank: texture_rich": _rf_bank_texture_rich,
    "RF bank: best_combo": _rf_bank_best_combo,
    "RF bank: fine_scale+gmm (thr=0.8)": _rf_bank_fine_scale_gmm_thresh08,
    "RF bank: fine_scale+morph (thr=0.9)": _rf_bank_fine_scale_morph_thresh09,
    "Minimal (detection only)": _minimal_detection_only,
}


# --- User preset I/O ---

def list_user_presets() -> List[str]:
    """Return the names of user-saved presets in data/presets/."""
    if not os.path.isdir(USER_PRESET_DIR):
        return []
    out = []
    for f in sorted(os.listdir(USER_PRESET_DIR)):
        if f.endswith(".json"):
            out.append(os.path.splitext(f)[0])
    return out


def load_user_preset(name: str) -> RunParams:
    """Load a user preset by name (filename without .json)."""
    path = os.path.join(USER_PRESET_DIR, f"{name}.json")
    with open(path) as f:
        return RunParams.from_json(f.read())


def save_user_preset(name: str, params: RunParams) -> str:
    """Save params under data/presets/{name}.json. Returns the path."""
    os.makedirs(USER_PRESET_DIR, exist_ok=True)
    path = os.path.join(USER_PRESET_DIR, f"{name}.json")
    saved = RunParams.from_dict(params.to_dict())
    saved.preset_name = name
    with open(path, "w") as f:
        f.write(saved.to_json())
    return path


def delete_user_preset(name: str) -> bool:
    """Delete a user preset. Returns True if deleted."""
    path = os.path.join(USER_PRESET_DIR, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def all_preset_names() -> List[str]:
    """Built-ins first, then user presets."""
    return list(BUILTIN_PRESETS.keys()) + list_user_presets()


def load_preset(name: str) -> RunParams:
    """Resolve a preset name to RunParams — checks built-ins first."""
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name]()
    return load_user_preset(name)
