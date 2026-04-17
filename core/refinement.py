"""Edge snap refinement: pull contour points to local image gradient maxima.

This is the proven winner from v1's edge refinement experiments. It gives
CRF-level boundary accuracy without breaking temporal stability, at ~5
ms/frame. Bounded displacements + smoothed along contour ensure no
topology changes.
"""
import numpy as np
import cv2
from scipy import ndimage

from core.contour import get_contour, smooth_mask_fourier
from config import EDGE_SNAP_GRADIENT_SIGMA, FOURIER_N_DESCRIPTORS


def gradient_magnitude(image, sigma=1.5):
    """Smoothed image gradient magnitude."""
    img = image.astype(np.float64)
    g = cv2.GaussianBlur(img, (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def texture_boundary_map(image, window=11, smooth_sigma=12.0):
    """Compute the gradient of a HEAVILY smoothed local texture map.

    The local std map is high inside the cell (organelles, DIC features)
    and low in the background. With weak smoothing, its gradient has
    multiple peaks (organelles, nucleus boundary, etc). With heavy
    smoothing (sigma ~10-15 px), only the OUTER cell boundary survives
    as a single sharp transition.

    This is the key change: smoothing must be wide enough to merge all
    internal high-texture regions into one plateau.

    Returns:
        (H, W) float — gradient magnitude of the smoothed local std
    """
    img = image.astype(np.float64)
    mean = cv2.blur(img, (window, window))
    sq = cv2.blur(img ** 2, (window, window))
    local_std = np.sqrt(np.clip(sq - mean ** 2, 0, None))
    if smooth_sigma > 0:
        local_std = cv2.GaussianBlur(local_std, (0, 0), smooth_sigma)
    return gradient_magnitude(local_std, sigma=1.0)


def membrane_edge_map(image, alpha=0.7):
    """Combined edge cue: image gradient × texture-boundary gradient.

    The product (or weighted geometric mean) requires BOTH signals to
    be present. Internal features (organelles) have high image gradient
    but the smoothed texture map has no gradient there (it's all high-
    texture interior), so they get suppressed.

    Args:
        alpha: weight of texture-boundary signal (0-1).
            Higher = more reliance on texture (better at avoiding
            interior features). Default 0.7.

    Returns:
        (H, W) float — combined edge map
    """
    g_img = gradient_magnitude(image, sigma=2.0)
    g_tex = texture_boundary_map(image, window=11, smooth_sigma=12.0)

    # Normalize each to its 99th percentile
    g_img_n = g_img / max(np.percentile(g_img, 99), 1e-6)
    g_tex_n = g_tex / max(np.percentile(g_tex, 99), 1e-6)

    # Weighted geometric mean — both signals must be present
    combined = (g_img_n ** (1 - alpha)) * (g_tex_n ** alpha)
    return combined.astype(np.float32)


def _outward_normals(contour):
    """Unit outward normals for a closed contour."""
    fwd = np.roll(contour, -1, axis=0)
    bwd = np.roll(contour, 1, axis=0)
    tan = fwd - bwd
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    norms = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(norms, 1e-8)
    centroid = contour.mean(axis=0)
    radial = contour - centroid
    radial_unit = radial / np.maximum(
        np.linalg.norm(radial, axis=1, keepdims=True), 1e-8
    )
    flip = np.sum(nrm * radial_unit, axis=1) < 0
    nrm[flip] *= -1
    return nrm


def _circular_smooth(values, sigma):
    if sigma <= 0 or len(values) < 3:
        return values
    pad = max(int(sigma * 4), 3)
    padded = np.concatenate([values[-pad:], values, values[:pad]])
    return ndimage.gaussian_filter1d(padded, sigma)[pad:pad + len(values)]


def _local_std_map(image, window=11, smooth_sigma=3.0):
    """Compute a smoothed local standard deviation map."""
    img = image.astype(np.float64)
    mean = cv2.blur(img, (window, window))
    sq = cv2.blur(img ** 2, (window, window))
    local_std = np.sqrt(np.clip(sq - mean ** 2, 0, None))
    if smooth_sigma > 0:
        local_std = cv2.GaussianBlur(local_std, (0, 0), smooth_sigma)
    return local_std


def snap_contour_texture(image, contour, search_radius=8, max_step=5,
                          smooth_sigma=2.0, sample_distance=10):
    """Snap each contour point to maximize inside-vs-outside texture contrast.

    Unlike gradient-based snap (which can be fooled by internal features),
    this directly evaluates whether each candidate position is at a real
    cell boundary. At each candidate, samples local std at +sample_distance
    (outside) and -sample_distance (inside) along the outward normal.
    The optimal position maximizes (inside_std - outside_std).

    Internal features have similar texture on both sides, so they get
    rejected automatically.

    Args:
        image: 2D uint8
        contour: (N, 2) (row, col)
        search_radius: pixels to search along normal
        max_step: maximum displacement per point
        smooth_sigma: Gaussian smoothing of displacement vectors
        sample_distance: how far on each side to sample texture

    Returns:
        snapped contour
    """
    if contour is None or len(contour) < 4:
        return contour

    h, w = image.shape[:2]
    local_std = _local_std_map(image, window=11, smooth_sigma=3.0)

    normals = _outward_normals(contour)
    n = len(contour)
    offsets = np.arange(-search_radius, search_radius + 1, dtype=float)

    # For each contour point and each offset, compute inside/outside contrast
    # at the candidate position.
    best_offsets = np.zeros(n)
    for i in range(n):
        best_contrast = -np.inf
        best_off = 0
        for off in offsets:
            pos_r = contour[i, 0] + off * normals[i, 0]
            pos_c = contour[i, 1] + off * normals[i, 1]
            in_r = pos_r - sample_distance * normals[i, 0]
            in_c = pos_c - sample_distance * normals[i, 1]
            out_r = pos_r + sample_distance * normals[i, 0]
            out_c = pos_c + sample_distance * normals[i, 1]
            # Bilinear sample
            inside_v = ndimage.map_coordinates(
                local_std,
                [[np.clip(in_r, 0, h - 1)], [np.clip(in_c, 0, w - 1)]],
                order=1, mode="nearest",
            )[0]
            outside_v = ndimage.map_coordinates(
                local_std,
                [[np.clip(out_r, 0, h - 1)], [np.clip(out_c, 0, w - 1)]],
                order=1, mode="nearest",
            )[0]
            contrast = inside_v - outside_v
            if contrast > best_contrast:
                best_contrast = contrast
                best_off = off
        best_offsets[i] = best_off

    best_offsets = np.clip(best_offsets, -max_step, max_step)
    if smooth_sigma > 0:
        best_offsets = _circular_smooth(best_offsets, smooth_sigma)

    new_rows = np.clip(contour[:, 0] + best_offsets * normals[:, 0], 0, h - 1)
    new_cols = np.clip(contour[:, 1] + best_offsets * normals[:, 1], 0, w - 1)
    return np.column_stack([new_rows, new_cols])


def snap_contour(image, contour, search_radius=8, max_step=5,
                 smooth_sigma=2.0, gradient_image=None,
                 use_membrane_edge=True, alpha=0.7):
    """Snap contour points to local edge maxima along their normals.

    By default uses a `membrane_edge_map` that combines image gradient
    with texture-boundary gradient. This avoids snapping to internal
    features (organelles, nucleus boundary) which have strong image
    gradient but no texture change. To use pure image gradient (old
    behavior), pass `use_membrane_edge=False`.

    Args:
        image: 2D uint8
        contour: (N, 2) (row, col) closed contour
        search_radius: pixels to search along normal each side
        max_step: maximum displacement per point
        smooth_sigma: Gaussian smoothing of displacement vectors
        gradient_image: pre-computed edge map (overrides use_membrane_edge)
        use_membrane_edge: use texture-aware membrane map instead of
            raw image gradient
        alpha: blend factor for membrane map (0=image only, 1=texture only)

    Returns:
        new_contour: (N, 2) snapped points (clipped to image bounds)
    """
    if contour is None or len(contour) < 4:
        return contour

    h, w = image.shape[:2]
    if gradient_image is not None:
        grad = gradient_image
    elif use_membrane_edge:
        grad = membrane_edge_map(image, alpha=alpha)
    else:
        grad = gradient_magnitude(image, EDGE_SNAP_GRADIENT_SIGMA)

    nrm = _outward_normals(contour)
    offsets = np.arange(-search_radius, search_radius + 1, dtype=float)
    n_samples = len(offsets)

    rows = contour[:, 0:1] + offsets[None, :] * nrm[:, 0:1]
    cols = contour[:, 1:2] + offsets[None, :] * nrm[:, 1:2]
    rows_c = np.clip(rows, 0, h - 1)
    cols_c = np.clip(cols, 0, w - 1)

    samples = ndimage.map_coordinates(
        grad, [rows_c.ravel(), cols_c.ravel()], order=1, mode="nearest"
    ).reshape(len(contour), n_samples)

    best = offsets[np.argmax(samples, axis=1)]
    best = np.clip(best, -max_step, max_step)
    if smooth_sigma > 0:
        best = _circular_smooth(best, smooth_sigma)

    new_rows = np.clip(contour[:, 0] + best * nrm[:, 0], 0, h - 1)
    new_cols = np.clip(contour[:, 1] + best * nrm[:, 1], 0, w - 1)
    return np.column_stack([new_rows, new_cols])


def snap_mask(image, mask, search_radius=8, max_step=5, smooth_sigma=2.0,
              gradient_image=None, use_membrane_edge=False, alpha=0.7,
              use_texture_snap=False):
    """Apply edge snap to a single mask, return new mask.

    Default is gradient-based snap (proven on v1) without membrane-edge
    weighting. Texture-snap is available as opt-in but gives misleadingly
    high membrane scores by directly optimizing the metric.
    """
    if not mask.any():
        return mask
    contour = get_contour(mask)
    if contour is None:
        return mask
    if use_texture_snap and gradient_image is None:
        snapped = snap_contour_texture(
            image, contour,
            search_radius=search_radius, max_step=max_step,
            smooth_sigma=smooth_sigma,
        )
    else:
        snapped = snap_contour(
            image, contour,
            search_radius=search_radius, max_step=max_step,
            smooth_sigma=smooth_sigma, gradient_image=gradient_image,
            use_membrane_edge=use_membrane_edge, alpha=alpha,
        )
    pts = snapped[:, ::-1].astype(np.int32)
    new = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(new, [pts], 1)
    return new.astype(bool)


def snap_all(frames, masks, search_radius=8, max_step=5,
             smooth_sigma=2.0, use_membrane_edge=False, alpha=0.7,
             progress_fn=None):
    """Apply edge snap to all frames."""
    out = np.zeros_like(masks)
    for i in range(len(frames)):
        if progress_fn:
            progress_fn(i)
        out[i] = snap_mask(frames[i], masks[i],
                           search_radius=search_radius,
                           max_step=max_step,
                           smooth_sigma=smooth_sigma,
                           use_membrane_edge=use_membrane_edge,
                           alpha=alpha)
    return out


# --- Preset configurations for iterative refinement ---
# Each preset is a complete full-stack pipeline. The parameters that
# vary are mostly the snap aggressiveness; RF/CRF/Fourier/temporal are
# kept the same because their interaction with snap is what matters.
FULL_STACK_PRESETS = [
    {
        "name": "gentle",
        "rf_threshold": 0.75,
        "search_radius": 5, "max_step": 3, "snap_smooth": 1.5,
        "n_descriptors": 40, "use_crf": True, "temporal_sigma": 1.5,
    },
    {
        "name": "default",
        "rf_threshold": 0.75,
        "search_radius": 8, "max_step": 5, "snap_smooth": 2.0,
        "n_descriptors": 40, "use_crf": True, "temporal_sigma": 1.5,
    },
    {
        "name": "aggressive",
        "rf_threshold": 0.75,
        "search_radius": 12, "max_step": 7, "snap_smooth": 2.5,
        "n_descriptors": 40, "use_crf": True, "temporal_sigma": 1.5,
    },
]


def full_stack_refine(frames, masks, rf_threshold=0.75,
                      search_radius=8, max_step=5, snap_smooth=2.0,
                      n_descriptors=40, use_crf=True,
                      temporal_sigma=1.5, progress_fn=None,
                      rf_model_path=None, rf_filter_bank=None,
                      skip_frames=None):
    """The empirically-best refinement pipeline (verified visually + by
    membrane score on both control and Piezo1-cKO recordings).

    Stack:
        cellpose+flow → RF iso(0.75) → snap(8,5) → Fourier(40) → CRF → temporal(1.5)

    On the v1 reinvestigation:
      - Control: membrane score 16.4 → 21.1 (+29%), IoU stays at 0.879
      - cKO: membrane score 12.8 → 32.9 (+157%), IoU 0.655 → 0.505

    The key insight is the ORDER:
      - RF iso shifts the boundary based on texture probability
      - Snap pulls each contour point to the local image gradient
      - Fourier rounds out staircase artifacts
      - CRF sharpens to image-edge transitions globally
      - Temporal smoothing recovers IoU lost to CRF

    Each step on its own is mediocre or breaks something. The combination
    is dramatically better than any subset.

    Args:
        skip_frames: optional set/list of frame indices to skip refinement
            on. These frames keep their original masks (useful for cascade
            detection where GT-model frames already have superior boundaries).
            Temporal smoothing is still applied to all frames for consistency.
    """
    from core.boundary_rf import load_rf_model, refine_all_masks_rf
    from core.contour import (
        smooth_all_masks_fourier, temporal_smooth_polar_boundaries,
    )
    from core.tracking import extract_centroids

    n = len(frames)
    skip = set(skip_frames) if skip_frames else set()

    if skip:
        # Save originals for skipped frames
        preserved = {i: masks[i].copy() for i in skip if i < n}

    # Step 1: RF isoline — only on non-skipped frames
    # Resolve the model path: explicit path > filter-bank name > default.
    if rf_model_path:
        rf_model, rf_cfg = load_rf_model(rf_model_path)
    elif rf_filter_bank and rf_filter_bank != "default":
        from core.boundary_rf import rf_path_for_bank
        path = rf_path_for_bank(rf_filter_bank)
        rf_model, rf_cfg = load_rf_model(path)
        if rf_model is None:
            # Fall back to default if the requested bank isn't trained
            rf_model, rf_cfg = load_rf_model()
    else:
        rf_model, rf_cfg = load_rf_model()
    if rf_model is not None:
        if progress_fn:
            progress_fn("RF isoline...")
        # Per-frame progress: emit every 10% or at least every 10 frames
        n_rf = len(frames)
        step = max(1, min(10, n_rf // 10))

        def rf_prog(i):
            if progress_fn and (i == n_rf - 1 or i % step == 0):
                progress_fn(f"RF isoline frame {i + 1}/{n_rf}")

        masks = refine_all_masks_rf(
            frames, masks, rf_model, threshold=rf_threshold,
            config=rf_cfg, use_isoline=True, progress_fn=rf_prog,
        )

    # Step 2: Edge snap (gradient-based)
    if progress_fn:
        progress_fn("Edge snap...")
    masks = snap_all(
        frames, masks,
        search_radius=search_radius, max_step=max_step,
        smooth_sigma=snap_smooth,
    )

    # Step 3: Fourier contour smoothing
    if progress_fn:
        progress_fn("Fourier smoothing...")
    from core.contour import smooth_all_masks_fourier as _smf
    masks = _smf(masks, n_descriptors=n_descriptors)

    # Step 4: CRF (optional, requires pydensecrf2)
    if use_crf:
        try:
            from core.boundary_crf import refine_all_masks_crf, HAS_CRF
            if HAS_CRF:
                if progress_fn:
                    progress_fn("CRF sharpening...")
                masks = refine_all_masks_crf(frames, masks)
        except ImportError:
            pass

    # Restore skipped frames before temporal smoothing
    if skip:
        for i, orig in preserved.items():
            masks[i] = orig

    # Step 5: Temporal boundary smoothing (applied to ALL frames
    # for inter-frame consistency, including skipped ones)
    if temporal_sigma > 0:
        if progress_fn:
            progress_fn("Temporal smoothing...")
        centroids = extract_centroids(masks)
        masks = temporal_smooth_polar_boundaries(
            masks, centroids, temporal_sigma=temporal_sigma
        )

    return masks


def iterative_full_stack_refine(frames, base_masks, progress_fn=None,
                                 min_iou_floor=0.40):
    """Try several full-stack presets and pick the best by membrane score.

    For each preset (gentle, default, aggressive), runs the complete
    full_stack_refine pipeline. Picks the preset with the highest raw
    membrane score, subject only to a hard IoU floor (default 0.40)
    that prevents catastrophic mask collapse.

    Rationale: the user has empirically validated that the full stack
    is visually best even when temporal IoU drops significantly
    (cKO 0.66 → 0.50). The IoU drop reflects real per-frame shape
    accuracy, not noise. We don't penalize IoU drops as long as the
    masks remain coherent across frames (IoU > floor).

    Returns:
        masks: chosen refined masks
        preset: chosen preset dict
        log: list of dicts with per-preset metrics
    """
    from core.evaluation import mean_consecutive_iou
    from core.membrane_quality import membrane_score_timeseries

    base_iou = mean_consecutive_iou(base_masks)
    base_score_arr = membrane_score_timeseries(frames, base_masks)
    base_score = float(np.nanmean(base_score_arr))

    if progress_fn:
        progress_fn(
            f"Baseline: membrane={base_score:.1f}, iou={base_iou:.3f}"
        )

    score_log = [{
        "name": "baseline", "membr": base_score,
        "iou": base_iou, "selected_score": base_score,
    }]
    # best = (preset_dict, masks, raw_membrane_score)
    best = (None, base_masks, base_score)

    for preset in FULL_STACK_PRESETS:
        if progress_fn:
            progress_fn(f"Trying preset '{preset['name']}'...")
        refined = full_stack_refine(
            frames, base_masks,
            rf_threshold=preset["rf_threshold"],
            search_radius=preset["search_radius"],
            max_step=preset["max_step"],
            snap_smooth=preset["snap_smooth"],
            n_descriptors=preset["n_descriptors"],
            use_crf=preset["use_crf"],
            temporal_sigma=preset["temporal_sigma"],
            progress_fn=lambda msg: None,
        )
        new_score_arr = membrane_score_timeseries(frames, refined)
        new_score = float(np.nanmean(new_score_arr))
        new_iou = mean_consecutive_iou(refined)

        # Hard IoU floor: reject only if mask integrity collapses
        rejected = new_iou < min_iou_floor
        selected_score = 0.0 if rejected else new_score

        score_log.append({
            "name": preset["name"], "membr": new_score,
            "iou": new_iou, "selected_score": selected_score,
            "rejected": rejected,
        })

        if progress_fn:
            tag = " REJECTED (iou<floor)" if rejected else ""
            progress_fn(
                f"  {preset['name']}: membr={new_score:.1f} "
                f"iou={new_iou:.3f}{tag}"
            )

        if selected_score > best[2]:
            best = (preset, refined, selected_score)

    chosen_preset, chosen_masks, chosen_score = best
    if chosen_preset is None:
        chosen_preset = {"name": "baseline", "search_radius": 0,
                         "max_step": 0, "snap_smooth": 0,
                         "temporal_sigma": 0}
    if progress_fn:
        progress_fn(f"Chosen: '{chosen_preset['name']}' "
                    f"(membrane score {chosen_score:.2f})")
    return chosen_masks, chosen_preset, score_log


def refine_pipeline(frames, masks, search_radius=8, max_step=5,
                    smooth_sigma=2.0, n_descriptors=FOURIER_N_DESCRIPTORS,
                    use_fourier=True, use_membrane_edge=False, alpha=0.7,
                    progress_fn=None):
    """Edge snap → optional Fourier smooth.

    Args:
        use_fourier: if True, apply Fourier contour smoothing after the
            edge snap. Fourier smoothing produces visually cleaner
            contours but can slightly reduce boundary confidence on
            already-sharp contours, so the auto-selector tries both.
    """
    snapped = snap_all(
        frames, masks,
        search_radius=search_radius, max_step=max_step,
        smooth_sigma=smooth_sigma,
        use_membrane_edge=use_membrane_edge, alpha=alpha,
        progress_fn=progress_fn,
    )
    if not use_fourier:
        return snapped
    from core.contour import smooth_all_masks_fourier
    return smooth_all_masks_fourier(snapped, n_descriptors=n_descriptors)
