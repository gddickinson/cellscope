"""Automatic parameter selection for edge refinement.

The user does not adjust knobs. For each recording, this module tries
several edge-snap configurations and picks the best by:

    score = membrane_score_after × stability_factor

where `membrane_score` is a texture-aware metric that distinguishes
true cell boundaries from internal features (organelles, nucleus).
It combines intensity contrast (DIC bright/dark across the membrane)
with texture contrast (cell interior is more textured than background).

`stability_factor` decays smoothly from 1.0 to 0 as the temporal IoU
drop relative to baseline grows from `AUTO_IOU_DROP_OK` to
`AUTO_IOU_DROP_REJECT`. A safety floor (`AUTO_MIN_IOU`) catches
catastrophic topology collapse.

The previous version used `boundary_confidence` (mean image gradient
along the contour). That metric is fooled by interior features and
rewards edge snap for moving the boundary into the cytoplasm. The
membrane score is a much more honest signal.
"""
import numpy as np

from core.refinement import refine_pipeline
from core.evaluation import mean_consecutive_iou
from core.membrane_quality import membrane_score_timeseries
from config import (
    AUTO_REFINE_CONFIGS, AUTO_MIN_IOU,
    AUTO_IOU_DROP_OK, AUTO_IOU_DROP_REJECT,
)


def _score_config(frames, base_masks, refined_masks,
                  base_score_mean, base_iou):
    """Score = membrane score × smooth stability factor.

    The stability factor is 1.0 if IoU drop ≤ AUTO_IOU_DROP_OK,
    decays linearly to 0 at AUTO_IOU_DROP_REJECT, and is 0 beyond.
    """
    new_scores = membrane_score_timeseries(frames, refined_masks)
    new_iou = mean_consecutive_iou(refined_masks)
    new_score_mean = float(np.nanmean(new_scores))

    if new_iou < AUTO_MIN_IOU:
        return 0.0, new_score_mean, new_iou

    iou_drop = max(0.0, base_iou - new_iou)

    if iou_drop <= AUTO_IOU_DROP_OK:
        stability = 1.0
    elif iou_drop >= AUTO_IOU_DROP_REJECT:
        stability = 0.0
    else:
        span = AUTO_IOU_DROP_REJECT - AUTO_IOU_DROP_OK
        stability = 1.0 - (iou_drop - AUTO_IOU_DROP_OK) / span

    score = new_score_mean * stability
    return score, new_score_mean, new_iou


def auto_select_refinement(frames, base_masks, progress_fn=None):
    """Try AUTO_REFINE_CONFIGS, return the best one applied.

    Args:
        frames: (N, H, W) uint8
        base_masks: (N, H, W) bool — detection output
        progress_fn: optional callable(message_str)

    Returns:
        best_masks: (N, H, W) bool — refined masks for the chosen config
        chosen_cfg: dict — the chosen configuration
        scores: list of dicts — diagnostics per tried config
    """
    base_scores = membrane_score_timeseries(frames, base_masks)
    base_score_mean = float(np.nanmean(base_scores))
    base_iou = mean_consecutive_iou(base_masks)

    if progress_fn:
        progress_fn(
            f"Baseline membrane score={base_score_mean:.1f}, "
            f"iou={base_iou:.3f}"
        )

    # Baseline is always a candidate (in case no refinement helps)
    best = ({"name": "baseline", "search": 0, "max_step": 0,
             "smooth": 0, "fourier": False}, base_masks)
    best_score = base_score_mean
    scores = [{
        "name": "baseline", "search": 0, "max_step": 0, "smooth": 0,
        "fourier": False, "score": base_score_mean,
        "conf_after": base_score_mean, "iou_after": base_iou,
    }]

    for cfg in AUTO_REFINE_CONFIGS:
        if progress_fn:
            progress_fn(f"Trying '{cfg['name']}'...")
        refined = refine_pipeline(
            frames, base_masks,
            search_radius=cfg["search"], max_step=cfg["max_step"],
            smooth_sigma=cfg["smooth"],
            use_fourier=cfg.get("fourier", True),
        )
        score, conf, iou = _score_config(
            frames, base_masks, refined, base_score_mean, base_iou
        )
        scores.append({
            "name": cfg["name"], "search": cfg["search"],
            "max_step": cfg["max_step"], "smooth": cfg["smooth"],
            "fourier": cfg.get("fourier", True),
            "score": float(score), "conf_after": conf,
            "iou_after": iou,
        })
        if progress_fn:
            f_tag = "+f" if cfg.get("fourier", True) else "  "
            progress_fn(
                f"  {cfg['name']:14s} {f_tag}: membr={conf:5.1f} "
                f"iou={iou:.3f} score={score:.3f}"
            )
        if score > best_score:
            best_score = score
            best = (cfg, refined)

    # Apply parsimony rule: among configurations within 15% of the best
    # score, pick the LEAST aggressive (smallest search radius). This
    # avoids over-snapping when more aggressive configs offer only
    # marginal improvement, which is when over-shrinking into interior
    # features tends to happen. The threshold is intentionally generous
    # because the membrane score is noisy and small differences are
    # not meaningful.
    score_threshold = best_score * 0.85
    near_best = [
        (s, idx) for idx, s in enumerate(scores)
        if s["score"] >= score_threshold
    ]
    if near_best and len(near_best) > 1:
        # Sort by aggressiveness (search radius), then by score descending
        near_best.sort(key=lambda x: (x[0]["search"], -x[0]["score"]))
        gentlest = near_best[0][0]
        if (gentlest["search"] != best[0].get("search", 0) or
                gentlest.get("fourier", False) != best[0].get("fourier", False)):
            # Re-run the chosen config to get its mask
            if gentlest["search"] == 0:
                # baseline
                refined = base_masks
            else:
                refined = refine_pipeline(
                    frames, base_masks,
                    search_radius=gentlest["search"],
                    max_step=gentlest["max_step"],
                    smooth_sigma=gentlest["smooth"],
                    use_fourier=gentlest.get("fourier", False),
                )
            best = (gentlest, refined)
            if progress_fn:
                progress_fn(
                    f"  Parsimony: prefer gentler '{gentlest['name']}' "
                    f"(score {gentlest['score']:.2f}) over best "
                    f"(score {best_score:.2f})"
                )

    chosen_cfg, refined_masks = best

    # Apply temporal boundary smoothing as a final step. This improves
    # IoU substantially (control: 0.866 → ~0.898) at virtually no cost
    # to per-frame metrics. Always-on because it's the biggest free win.
    from core.contour import temporal_smooth_polar_boundaries
    from core.tracking import extract_centroids
    from config import TEMPORAL_SMOOTH_SIGMA

    if progress_fn:
        progress_fn("Applying temporal boundary smoothing...")
    centroids = extract_centroids(refined_masks)
    refined_masks = temporal_smooth_polar_boundaries(
        refined_masks, centroids, temporal_sigma=TEMPORAL_SMOOTH_SIGMA
    )
    if progress_fn:
        progress_fn(
            f"Chosen: '{chosen_cfg['name']}' "
            f"(search={chosen_cfg['search']}, "
            f"max_step={chosen_cfg['max_step']})"
        )
    return refined_masks, chosen_cfg, scores
