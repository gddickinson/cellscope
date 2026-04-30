"""Imaging modality detection and per-modality pipeline routing.

Supports two modalities:
  - Phase-contrast: high contrast cells on bright background.
    Best pipeline: cpsam + DeepSea union.
  - DIC (Differential Interference Contrast): cells appear as 3D
    relief with textured background. Best pipeline: cellpose_dic
    with preprocessing + DeepSea union.

Auto-detection uses image statistics to distinguish the two:
  - DIC has higher local texture variance (relief shading)
  - Phase-contrast has higher global contrast (dark cells on bright bg)
  - DIC background is more uniform in intensity histogram
"""
import numpy as np
import cv2
import logging

log = logging.getLogger(__name__)

MODALITIES = ("auto", "dic", "phase_contrast")


def detect_modality(frames, n_sample=5):
    """Auto-detect imaging modality from image statistics.

    Uses two discriminating features calibrated on known recordings:
    1. Local texture variance (DIC ~190-215, phase-contrast ~100-120)
    2. Intensity range (phase-contrast uses wider dynamic range)

    Calibrated on:
      DIC: Jesse's VAMPIRE crops (Pos10-WT=215, Pos58-KO=192, Pos64=205)
      Phase-contrast: Ignasi's crops (Pos0-WT=117, Pos3-WT=105)

    Returns: 'dic' or 'phase_contrast'
    """
    n = len(frames)
    indices = np.linspace(0, n - 1, min(n_sample, n), dtype=int)
    samples = frames[indices]

    texture_scores = []
    range_scores = []

    for img in samples:
        texture_scores.append(_local_texture_variance(img))
        range_scores.append(_intensity_range_ratio(img))

    mean_texture = np.mean(texture_scores)
    mean_range = np.mean(range_scores)

    # DIC: high texture (>150), narrow effective range (<0.5)
    # Phase-contrast: lower texture (<150), wider effective range (>0.5)
    # Threshold at texture=150 separates known DIC from phase-contrast
    is_dic = mean_texture > 150

    modality = "dic" if is_dic else "phase_contrast"
    log.info("Modality detection: texture=%.1f range=%.2f → %s",
             mean_texture, mean_range, modality)
    return modality


def _local_texture_variance(img):
    """Mean local variance in 16x16 blocks. High for DIC texture."""
    h, w = img.shape
    block = 16
    variances = []
    for r in range(0, h - block, block):
        for c in range(0, w - block, block):
            patch = img[r:r+block, c:c+block].astype(np.float32)
            variances.append(np.var(patch))
    return float(np.mean(variances)) if variances else 0.0


def _intensity_range_ratio(img):
    """Effective intensity range as fraction of [0, 255].
    Phase-contrast uses wider range; DIC shade-corrected is narrower."""
    p2 = np.percentile(img, 2)
    p98 = np.percentile(img, 98)
    return float((p98 - p2) / 255.0)


def get_pipeline_config(modality):
    """Return recommended pipeline settings for a modality.

    Returns dict with keys that map to detect params.
    """
    if modality == "dic":
        import os
        # cpsam_dic (CP4 ViT fine-tune) wins on every benchmark we've
        # measured: +0.06 IoU on our-GT, +0.42 IoU on VAMPIRE OOD.
        # Loads via cellpose4 env (subprocess delegation in
        # core/hybrid_dic.py). Fall back to CP3 fine-tunes if absent.
        cpsam_dic = "data/models/cpsam_dic"
        v3 = "data/models/cellpose_dic_v3"
        v2 = "data/models/cellpose_dic_v2"
        v1 = "data/models/cellpose_dic"
        if os.path.exists(cpsam_dic):
            model_path, detector = cpsam_dic, "cpsam_dic"
        elif os.path.exists(v3):
            model_path, detector = v3, "cellpose_dic_v3"
        elif os.path.exists(v2):
            model_path, detector = v2, "cellpose_dic_v2"
        else:
            model_path, detector = v1, "cellpose_dic"
        # cpsam_dic was trained on raw VAMPIRE crops — preprocessing
        # tends to hurt. CP3 fine-tunes still want the median/HP step.
        is_cpsam = detector.startswith("cpsam_dic")
        return {
            "preprocess": not is_cpsam,
            "preprocess_temporal_method": "median",
            "preprocess_highpass_sigma": 40.0,
            "detector": detector,
            "model_path": model_path,
            "flow_threshold": 0.0,
            "cellprob_threshold": 0.0,
            "use_deepsea": True,
            "use_fallback": True,
            "retry_thresholds": () if is_cpsam else (-2.0,),
            "description": (
                f"DIC: {detector} (cellpose4 subprocess) + DeepSea"
                if is_cpsam
                else f"DIC: {detector} + preprocessing + DeepSea"),
        }
    else:
        return {
            "preprocess": False,
            "detector": "cpsam",
            "use_deepsea": True,
            "use_fallback": True,
            "description": "Phase-contrast: cpsam + DeepSea union",
        }
