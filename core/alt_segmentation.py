"""Alternative (non-RF) segmentation methods for boundary refinement.

These methods take a cellpose mask + image crop and compute a refined
binary mask using classical image processing. Ported from the
test_segmentation_methods.py benchmark script in the original
CellScope project.

Each function has the signature:
    method(image, cellpose_mask, **kwargs) -> boolean mask

and is expected to be called on a cropped region around the cell
(either a global bbox or per-cell bbox).

From DETECTION_METHODS_REPORT.md benchmarks (full-frame, not crops):
    GMM (3 components)     : ctrl 0.690, cKO 0.588, overall 0.639
    Morph gradient (r=5)   : ctrl 0.646, cKO 0.620, overall 0.633
    Local std (σ=4)        : ctrl 0.558, cKO 0.673, overall 0.615
    Watershed              : ctrl 0.653, cKO 0.373, overall 0.513

RF isoline remains the single-best refinement (0.810 overall), but
these alternatives run with no training data and can be useful for
quick comparisons or when RF training data is unavailable.
"""
import numpy as np
from scipy import ndimage
from skimage import filters, morphology, segmentation, feature
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from core.evaluation import compute_iou


# --- Shared utilities ---

def _cleanup_mask(mask, min_area=100):
    """Fill holes and keep the largest connected component."""
    mask = ndimage.binary_fill_holes(mask)
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return (labeled == largest) & (sizes[largest - 1] >= min_area)


def _pick_best_against(cp_mask, candidates):
    """Given a cellpose mask and several candidate masks, return the one
    with highest IoU against cp_mask. Used to resolve DIC polarity
    ambiguity (cell can be darker or brighter than background).
    """
    best = None
    best_iou = -1
    for c in candidates:
        c = _cleanup_mask(c)
        iou = compute_iou(c, cp_mask)
        if iou > best_iou:
            best = c
            best_iou = iou
    return best if best is not None else cp_mask


# --- Individual methods ---

def seg_otsu(image, cp_mask, **_):
    """Global Otsu thresholding; pick the polarity that best matches cp."""
    t = filters.threshold_otsu(image)
    return _pick_best_against(cp_mask, [image > t, image < t])


def seg_adaptive(image, cp_mask, block_size=51, **_):
    """Adaptive (local) thresholding."""
    if block_size % 2 == 0:
        block_size += 1
    t = filters.threshold_local(image, block_size, method="gaussian")
    return _pick_best_against(cp_mask, [image > t, image < t])


def seg_multi_otsu(image, cp_mask, classes=3, **_):
    """Multi-Otsu; take the class region most overlapping with cp_mask."""
    try:
        thresholds = filters.threshold_multiotsu(image, classes=classes)
    except ValueError:
        return seg_otsu(image, cp_mask)
    regions = np.digitize(image, bins=thresholds)
    return _pick_best_against(
        cp_mask, [regions == c for c in range(classes)],
    )


def seg_local_std(image, cp_mask, sigma=4, **_):
    """Segment based on local texture (std dev) — cell interior has
    higher texture variance than background."""
    k = int(4 * sigma + 1) | 1
    img = image.astype(float)
    mean = ndimage.uniform_filter(img, k)
    mean_sq = ndimage.uniform_filter(img ** 2, k)
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    t = filters.threshold_otsu(std)
    return _cleanup_mask(std > t)


def seg_canny_fill(image, cp_mask, sigma=2, **_):
    """Canny edge detection → dilate → fill enclosed regions."""
    edges = feature.canny(image.astype(float), sigma=sigma)
    edges = morphology.binary_dilation(edges, morphology.disk(1))
    filled = ndimage.binary_fill_holes(edges)
    return _cleanup_mask(filled)


def seg_watershed(image, cp_mask, core_erode=5, bg_dilate=10, **_):
    """Watershed from markers derived from cellpose mask.
    Cell core (eroded cp) vs background (outside dilated cp).
    """
    gradient = filters.sobel(image.astype(float))
    markers = np.zeros(image.shape, dtype=int)
    eroded = ndimage.binary_erosion(cp_mask, iterations=core_erode)
    markers[eroded] = 1
    far_bg = ~ndimage.binary_dilation(cp_mask, iterations=bg_dilate)
    markers[far_bg] = 2
    ws = segmentation.watershed(gradient, markers)
    return _cleanup_mask(ws == 1)


def seg_kmeans(image, cp_mask, n_clusters=3, **_):
    """K-means pixel clustering; pick cluster best matching cp_mask.
    Features: intensity + normalized coordinates (weak spatial prior).
    """
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    X = np.column_stack([
        image.ravel().astype(float),
        (yy.ravel() / max(h, 1) * 50),
        (xx.ravel() / max(w, 1) * 50),
    ])
    km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
    labels = km.fit_predict(X).reshape(h, w)
    return _pick_best_against(
        cp_mask, [labels == c for c in range(n_clusters)],
    )


def seg_gmm(image, cp_mask, n_components=3, **_):
    """Gaussian Mixture Model pixel classification on intensity + local std.

    This was the best non-RF method in benchmarks (overall IoU 0.639).
    """
    img = image.astype(float)
    std = ndimage.uniform_filter(img ** 2, 9) - \
        ndimage.uniform_filter(img, 9) ** 2
    std = np.sqrt(np.maximum(std, 0))
    X = np.column_stack([img.ravel(), std.ravel()])
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(X).reshape(image.shape)
    return _pick_best_against(
        cp_mask, [labels == c for c in range(n_components)],
    )


def _boundary_confidence(image, mask, band_px=3):
    """Proxy confidence from boundary sharpness.

    Returns mean gradient magnitude along the cell boundary band,
    normalized to [0, 1] (1 = sharp, 0 = diffuse). Used to gate
    Chan-Vese: high-confidence detections should NOT be refined
    further (CV over-tightens strong boundaries — Phase 14f).
    """
    if not mask.any():
        return 0.0
    img_f = image.astype(np.float32)
    gy, gx = np.gradient(img_f)
    grad_mag = np.sqrt(gy**2 + gx**2)
    inner = ndimage.binary_erosion(mask, iterations=band_px)
    outer = ndimage.binary_dilation(mask, iterations=band_px)
    band = outer & ~inner
    if not band.any():
        return 0.0
    mean_grad = float(grad_mag[band].mean())
    # Normalize: DIC images typically have gradients in [0, 30].
    return float(np.clip(mean_grad / 30.0, 0.0, 1.0))


def seg_chan_vese_confidence_gated(image, cp_mask, threshold=0.35,
                                    iterations=20, **kwargs):
    """Chan-Vese refinement GATED by boundary confidence.

    If the seed mask's boundary already has high gradient sharpness
    (confidence >= threshold), pass through unchanged — CV would
    over-tighten. If confidence < threshold, apply Chan-Vese.

    This targets Phase 14f's finding that Chan-Vese helps weak
    detections (+41% on cellpose_dic) but hurts strong ones
    (-29% on cascade).
    """
    conf = _boundary_confidence(image, cp_mask)
    if conf >= threshold:
        return cp_mask.astype(bool)
    return seg_chan_vese(image, cp_mask, iterations=iterations, **kwargs)


def seg_chan_vese(image, cp_mask, iterations=35, smoothing=1, **_):
    """Morphological Chan-Vese active contour (region-based).

    Starts from `cp_mask` as the level-set init and evolves it to
    match intensity statistics inside vs outside the contour. Handles
    topology changes (splits/merges) automatically — useful for
    filopodia-rich cells where RF can get stranded components.

    Reference: Marquez-Neila et al. 2014 (scikit-image's
    `morphological_chan_vese`).
    """
    img_f = image.astype(np.float32)
    if img_f.max() > 1.0:
        img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-9)
    if not np.any(cp_mask):
        return cp_mask.astype(bool)
    evolved = segmentation.morphological_chan_vese(
        img_f, num_iter=iterations, init_level_set=cp_mask.astype(np.int8),
        smoothing=smoothing,
    )
    return _cleanup_mask(evolved > 0)


def seg_morph_gac(image, cp_mask, iterations=40, smoothing=1,
                  balloon=-1, threshold=0.5, **_):
    """Morphological geodesic active contour (edge-based).

    Uses the image gradient as the stopping function; the contour
    flows along strong edges. `balloon=-1` contracts the init mask
    toward the nearest gradient maximum — good for tightening around
    a cell boundary when cellpose oversegments.
    """
    from skimage.segmentation import morphological_geodesic_active_contour
    img_f = image.astype(np.float32)
    if img_f.max() > 1.0:
        img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-9)
    if not np.any(cp_mask):
        return cp_mask.astype(bool)
    from skimage.segmentation import inverse_gaussian_gradient
    gimg = inverse_gaussian_gradient(img_f)
    evolved = morphological_geodesic_active_contour(
        gimg, num_iter=iterations,
        init_level_set=cp_mask.astype(np.int8),
        smoothing=smoothing, balloon=balloon, threshold=threshold,
    )
    return _cleanup_mask(evolved > 0)


def seg_morph_gradient(image, cp_mask, radius=5, **_):
    """Morphological gradient (dilation − erosion) → Otsu → fill."""
    selem = morphology.disk(radius)
    img = image.astype(np.uint16 if image.dtype != np.uint8 else np.uint8)
    # morphology.dilation/erosion require integer types; cast if needed
    if img.dtype not in (np.uint8, np.uint16):
        img = (image - image.min()).astype(np.uint8)
    grad = (morphology.dilation(img, selem).astype(int)
            - morphology.erosion(img, selem).astype(int))
    t = filters.threshold_otsu(grad)
    filled = ndimage.binary_fill_holes(grad > t)
    return _cleanup_mask(filled)


# --- Registry for GUI wiring ---

METHODS = {
    "otsu": (seg_otsu, "Otsu (global threshold)"),
    "adaptive": (seg_adaptive, "Adaptive threshold (local)"),
    "multi_otsu_3": (lambda img, cp, **k: seg_multi_otsu(img, cp, 3),
                     "Multi-Otsu (3 classes)"),
    "multi_otsu_4": (lambda img, cp, **k: seg_multi_otsu(img, cp, 4),
                     "Multi-Otsu (4 classes)"),
    "local_std_sigma2": (lambda img, cp, **k: seg_local_std(img, cp, 2),
                         "Local std (σ=2)"),
    "local_std_sigma4": (lambda img, cp, **k: seg_local_std(img, cp, 4),
                         "Local std (σ=4)"),
    "local_std_sigma8": (lambda img, cp, **k: seg_local_std(img, cp, 8),
                         "Local std (σ=8)"),
    "canny_fill": (seg_canny_fill, "Canny edge → fill"),
    "watershed": (seg_watershed, "Watershed (from cellpose markers)"),
    "kmeans_3": (lambda img, cp, **k: seg_kmeans(img, cp, 3),
                 "K-means (3 clusters)"),
    "kmeans_4": (lambda img, cp, **k: seg_kmeans(img, cp, 4),
                 "K-means (4 clusters)"),
    "gmm_3": (lambda img, cp, **k: seg_gmm(img, cp, 3),
              "GMM (3 components, best non-RF)"),
    "gmm_4": (lambda img, cp, **k: seg_gmm(img, cp, 4),
              "GMM (4 components)"),
    "morph_gradient_r3": (lambda img, cp, **k: seg_morph_gradient(img, cp, 3),
                          "Morphological gradient (r=3)"),
    "morph_gradient_r5": (lambda img, cp, **k: seg_morph_gradient(img, cp, 5),
                          "Morphological gradient (r=5)"),
    "chan_vese": (seg_chan_vese,
                  "Morphological Chan-Vese (region-based AC)"),
    "chan_vese_gated": (seg_chan_vese_confidence_gated,
                         "Chan-Vese gated by boundary sharpness"),
    "morph_gac": (seg_morph_gac,
                  "Morphological geodesic AC (edge-based)"),
}


def apply_method_to_stack(frames, masks, method_key):
    """Apply an alt segmentation method frame-by-frame.

    Args:
        frames: (N, H, W) uint8
        masks:  (N, H, W) bool — cellpose seed masks (used for polarity
            disambiguation and as fallback when the method fails)
        method_key: one of METHODS.keys()

    Returns:
        (N, H, W) bool — refined masks
    """
    if method_key not in METHODS:
        raise KeyError(f"Unknown alt segmentation method: {method_key}")
    fn, _ = METHODS[method_key]
    out = np.zeros_like(masks)
    for i in range(len(frames)):
        if not masks[i].any():
            continue
        try:
            out[i] = fn(frames[i], masks[i])
        except Exception:
            # On failure, fall back to cellpose
            out[i] = masks[i]
    return out
