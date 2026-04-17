"""Hybrid refinement: combine classical seeds with the RF boundary model.

Three strategies, ported from `test_hybrid_rf.py` in the original
CellScope project:

  - hybrid_localstd_rf : local-std coarse mask  → RF refines a narrow
                         boundary band around it
  - hybrid_gmm_rf      : GMM coarse mask        → RF refines boundary
  - hybrid_ensemble    : average(RF, local_std, GMM) → threshold

All methods use the currently-selected RF model (`rf_filter_bank`), so
the user's bank choice still applies.

Per-frame design — works seamlessly with `crop_mode='per_cell'`.
"""
import numpy as np
from scipy import ndimage
from skimage import filters
from sklearn.mixture import GaussianMixture

from core.boundary_rf import (
    load_rf_model, predict_cell_probability, rf_path_for_bank,
)
from core.alt_segmentation import _cleanup_mask


# --- Shared feature computations ---

def _compute_local_std(img, sigma=4):
    k = int(4 * sigma + 1) | 1
    f = img.astype(float)
    mean = ndimage.uniform_filter(f, k)
    mean_sq = ndimage.uniform_filter(f ** 2, k)
    return np.sqrt(np.maximum(mean_sq - mean ** 2, 0))


def _compute_gmm_cell_prob(img, n_components=3):
    """Per-pixel probability of belonging to the high-texture (cell)
    GMM component. Picks the component with highest mean local_std
    as the cell cluster.
    """
    h, w = img.shape
    std_map = _compute_local_std(img, 4)
    X = np.column_stack([img.ravel().astype(float), std_map.ravel()])
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    probs = gmm.predict_proba(X)
    cell_comp = int(np.argmax(gmm.means_[:, 1]))
    return probs[:, cell_comp].reshape(h, w).astype(np.float32)


def _load_rf(rf_filter_bank=None):
    """Load the RF model corresponding to the selected bank (or default)."""
    if rf_filter_bank and rf_filter_bank != "default":
        rf, cfg = load_rf_model(rf_path_for_bank(rf_filter_bank))
        if rf is not None:
            return rf, cfg
    return load_rf_model()


# --- Hybrid methods ---

def hybrid_localstd_rf(img, cp_mask, rf_filter_bank=None,
                       std_sigma=4, rf_thresh=0.6,
                       dilate=10, erode=5):
    """Local-std coarse mask → RF refines the boundary band.

    Local-std gives a stable interior estimate; RF is used only near
    the boundary where its per-pixel discrimination matters.
    """
    rf, cfg = _load_rf(rf_filter_bank)
    if rf is None:
        return cp_mask

    std_map = _compute_local_std(img.astype(float), std_sigma)
    t = filters.threshold_otsu(std_map)
    coarse = _cleanup_mask(std_map > t)

    prob = predict_cell_probability(img, rf, cfg)

    dilated = ndimage.binary_dilation(coarse, iterations=dilate)
    eroded = ndimage.binary_erosion(coarse, iterations=erode)
    band = dilated & ~eroded

    combined = coarse.copy()
    combined[band] = prob[band] >= rf_thresh
    return _cleanup_mask(combined)


def hybrid_gmm_rf(img, cp_mask, rf_filter_bank=None,
                  rf_thresh=0.6, dilate=10, erode=5,
                  n_components=3):
    """GMM coarse mask → RF refines the boundary band."""
    rf, cfg = _load_rf(rf_filter_bank)
    if rf is None:
        return cp_mask

    gmm_p = _compute_gmm_cell_prob(img, n_components=n_components)
    coarse = _cleanup_mask(gmm_p > 0.5)

    prob = predict_cell_probability(img, rf, cfg)

    dilated = ndimage.binary_dilation(coarse, iterations=dilate)
    eroded = ndimage.binary_erosion(coarse, iterations=erode)
    band = dilated & ~eroded

    combined = coarse.copy()
    combined[band] = prob[band] >= rf_thresh
    return _cleanup_mask(combined)


def hybrid_ensemble(img, cp_mask, rf_filter_bank=None,
                    thresh=0.5, std_sigma=4):
    """Average(RF prob, normalized local_std, GMM prob) → threshold.

    All three probability maps are rescaled to [0, 1]; their mean is
    thresholded. Acts as a robust consensus estimator.
    """
    rf, cfg = _load_rf(rf_filter_bank)
    if rf is None:
        return cp_mask

    rf_prob = predict_cell_probability(img, rf, cfg)
    std_map = _compute_local_std(img.astype(float), std_sigma)
    std_prob = std_map / max(std_map.max(), 1e-6)
    gmm_prob = _compute_gmm_cell_prob(img)

    avg = (rf_prob + std_prob + gmm_prob) / 3.0
    return _cleanup_mask(avg >= thresh)


# --- Registry for GUI wiring ---

METHODS = {
    "localstd_rf": (hybrid_localstd_rf,
                    "LocalStd seed → RF boundary refinement"),
    "gmm_rf": (hybrid_gmm_rf,
               "GMM seed → RF boundary refinement"),
    "ensemble": (hybrid_ensemble,
                 "Ensemble: avg(RF, local_std, GMM)"),
}


def apply_method_to_stack(frames, masks, method_key,
                          rf_filter_bank=None, **kwargs):
    """Apply a hybrid method frame-by-frame."""
    if method_key not in METHODS:
        raise KeyError(f"Unknown hybrid method: {method_key}")
    fn, _ = METHODS[method_key]
    out = np.zeros_like(masks)
    for i in range(len(frames)):
        if not masks[i].any():
            continue
        try:
            out[i] = fn(frames[i], masks[i],
                        rf_filter_bank=rf_filter_bank, **kwargs)
        except Exception:
            out[i] = masks[i]
    return out
