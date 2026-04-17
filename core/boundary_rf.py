"""Random forest pixel classifier for boundary refinement.

Inference-only port from v1 (`boundary_rf.py`). The trained model and
filter config are bundled in `data/`. Used as the FIRST refinement step
in the full stack: cellpose+flow → RF iso → snap → Fourier → CRF → temporal.

The RF was found (in v1's reinvestigation) to give a small membrane-score
benefit ON ITS OWN — but more importantly, it shifts the mask in a way
that helps the downstream snap and CRF steps land on better positions.
The full stack outperforms any subset of these methods.
"""
import os
import json
import pickle
import warnings
import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, measure
from skimage.filters import meijering, sato, frangi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

warnings.filterwarnings("ignore", "InconsistentVersionWarning")
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

from config import PROJECT_DIR

MODEL_PATH = os.path.join(PROJECT_DIR, "data", "models", "rf_boundary_model.pkl")
CONFIG_PATH = os.path.join(PROJECT_DIR, "data", "models", "filter_config.json")
MODEL_DIR = os.path.join(PROJECT_DIR, "data", "models")


def rf_path_for_bank(bank_name: str) -> str:
    """Return the on-disk path for an RF filter-bank model.

    'default' → rf_boundary_model.pkl (legacy winning model)
    other     → rf_bank_<name>.pkl (trained by scripts/train_rf_filter_banks.py)
    """
    if bank_name in (None, "", "default"):
        return MODEL_PATH
    return os.path.join(MODEL_DIR, f"rf_bank_{bank_name}.pkl")


def list_available_rf_banks() -> list:
    """Return the list of trained RF filter-bank model names.

    Always includes "default" (the legacy winning model) plus any
    rf_bank_<name>.pkl files found in data/models/.
    """
    out = ["default"]
    if os.path.isdir(MODEL_DIR):
        for fn in sorted(os.listdir(MODEL_DIR)):
            if fn.startswith("rf_bank_") and fn.endswith(".pkl"):
                out.append(fn[len("rf_bank_"):-len(".pkl")])
    return out


def load_filter_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def compute_one_filter(image, filt_def):
    """Compute a single multi-scale filter from its config definition."""
    img = image.astype(np.float64)
    name = filt_def["name"]
    sigmas = filt_def.get("sigmas", [1, 2, 4, 8])
    results = []

    if name == "gaussian":
        for s in sigmas:
            results.append((cv2.GaussianBlur(img, (0, 0), s), f"gauss_s{s}"))
    elif name == "gradient":
        for s in sigmas:
            g = cv2.GaussianBlur(img, (0, 0), s)
            gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
            results.append((np.sqrt(gx ** 2 + gy ** 2), f"grad_s{s}"))
    elif name == "scharr":
        for s in sigmas:
            g = cv2.GaussianBlur(img, (0, 0), s)
            sx = cv2.Scharr(g, cv2.CV_64F, 1, 0)
            sy = cv2.Scharr(g, cv2.CV_64F, 0, 1)
            results.append((np.sqrt(sx ** 2 + sy ** 2), f"scharr_s{s}"))
    elif name == "log":
        for s in sigmas:
            g = cv2.GaussianBlur(img, (0, 0), s)
            results.append((np.abs(cv2.Laplacian(g, cv2.CV_64F)), f"log_s{s}"))
    elif name == "local_std":
        for s in sigmas:
            g = cv2.GaussianBlur(img, (0, 0), s)
            k = int(s * 4 + 1) | 1
            mean = cv2.blur(g, (k, k))
            mean_sq = cv2.blur(g ** 2, (k, k))
            results.append((np.sqrt(np.clip(mean_sq - mean ** 2, 0, None)), f"std_s{s}"))
    elif name == "meijering":
        results.append((meijering(img, sigmas=sigmas, black_ridges=False), "meijering"))
    elif name == "sato":
        results.append((sato(img, sigmas=sigmas, black_ridges=False), "sato"))
    elif name == "frangi":
        results.append((frangi(img, sigmas=sigmas, black_ridges=False), "frangi"))
    elif name == "hessian":
        s = filt_def.get("sigma", 3)
        H = hessian_matrix(img, sigma=s, use_gaussian_derivatives=False)
        evals = hessian_matrix_eigvals(H)
        results.append((np.abs(evals[0]), f"hessian_s{s}"))
    elif name == "hessian_multi":
        for s in sigmas:
            H = hessian_matrix(img, sigma=s, use_gaussian_derivatives=False)
            evals = hessian_matrix_eigvals(H)
            results.append((np.maximum(np.abs(evals[0]), np.abs(evals[1])), f"hess_s{s}"))
    elif name == "dog":
        for s1 in sigmas:
            s2 = s1 * 1.6
            g1 = cv2.GaussianBlur(img, (0, 0), s1)
            g2 = cv2.GaussianBlur(img, (0, 0), s2)
            results.append((np.abs(g1 - g2), f"dog_s{s1}"))
    elif name == "abs_dev_median":
        smooth = filt_def.get("smooth_sigma", 5)
        median_bg = np.median(img)
        abs_dev = cv2.GaussianBlur(np.abs(img - median_bg), (0, 0), smooth)
        results.append((abs_dev, "abs_dev_median"))
    elif name == "entropy":
        from skimage.filters.rank import entropy as rank_entropy
        from skimage.morphology import disk
        radii = filt_def.get("radii", [3, 5])
        mx = img.max() if img.max() > 0 else 1
        img8 = (img / mx * 255).astype(np.uint8)
        for r in radii:
            results.append((rank_entropy(img8, disk(r)).astype(np.float64),
                             f"entropy_r{r}"))
    elif name == "intensity":
        results.append((img.copy(), "intensity"))
    elif name == "laplacian":
        for s in sigmas:
            g = cv2.GaussianBlur(img, (0, 0), s)
            results.append((cv2.Laplacian(g, cv2.CV_64F), f"lap_s{s}"))
    elif name == "gabor":
        from skimage.filters import gabor
        freqs = filt_def.get("freqs", [0.1, 0.2, 0.3])
        thetas = filt_def.get("thetas", [0.0, 0.7853981633974483,
                                           1.5707963267948966])
        for freq in freqs:
            for theta in thetas:
                real, _ = gabor(img, frequency=freq, theta=theta)
                results.append(
                    (real, f"gabor_f{freq:.2f}_t{theta:.2f}")
                )
    elif name == "lbp":
        from skimage.feature import local_binary_pattern
        radii = filt_def.get("radii", [1, 2, 3])
        for r in radii:
            n_pts = 8 * r
            lbp = local_binary_pattern(img, n_pts, r, method="uniform")
            results.append((lbp.astype(np.float64), f"lbp_r{r}"))
    elif name == "morph_gradient":
        from skimage import morphology as _morph
        radii = filt_def.get("radii", [2, 3, 5])
        # morphology requires integer types — rescale if needed
        if image.dtype == np.uint8:
            src = image
        else:
            mn, mx = float(img.min()), float(img.max())
            if mx > mn:
                src = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                src = np.zeros(img.shape, dtype=np.uint8)
        for r in radii:
            selem = _morph.disk(r)
            mg = (_morph.dilation(src, selem).astype(np.float64)
                  - _morph.erosion(src, selem).astype(np.float64))
            results.append((mg, f"morph_grad_r{r}"))
    elif name == "gmm_prob":
        # Per-pixel probability of belonging to the high-texture GMM
        # component (cell interior). See core.hybrid_rf._compute_gmm_cell_prob.
        from core.hybrid_rf import _compute_gmm_cell_prob
        n_comp = filt_def.get("n_components", 3)
        results.append(
            (_compute_gmm_cell_prob(image, n_components=n_comp)
             .astype(np.float64), f"gmm_prob_k{n_comp}")
        )

    return results


def compute_filter_bank(image, config=None):
    """Compute the full multi-channel filter bank for an image."""
    if config is None:
        config = load_filter_config()
    features = []
    for filt_def in config["filters"]:
        for arr, _name in compute_one_filter(image, filt_def):
            features.append(arr)
    return np.stack(features, axis=-1).astype(np.float32)


def load_rf_model(path=MODEL_PATH):
    """Load the trained RF model and its filter config."""
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "model" in data:
        return data["model"], data.get("config", load_filter_config())
    return data, load_filter_config()


def predict_cell_probability(image, rf_model, config=None):
    bank = compute_filter_bank(image, config)
    h, w, n = bank.shape
    probs = rf_model.predict_proba(bank.reshape(-1, n))[:, 1]
    return probs.reshape(h, w)


def refine_mask_with_rf(image, cellpose_mask, rf_model, threshold=0.75,
                        config=None, min_area=300, use_isoline=True):
    """Refine a single mask using RF probability."""
    if not cellpose_mask.any():
        return cellpose_mask

    prob = predict_cell_probability(image, rf_model, config)

    if use_isoline:
        return _refine_with_isoline(prob, cellpose_mask, threshold, min_area)

    rf_mask = ndimage.binary_fill_holes(prob >= threshold)
    rf_mask = morphology.remove_small_objects(rf_mask, min_size=min_area)
    labeled = measure.label(rf_mask.astype(np.uint8))
    if labeled.max() == 0:
        return cellpose_mask
    best_label, best_overlap = 0, 0
    for region in measure.regionprops(labeled):
        overlap = np.logical_and(labeled == region.label, cellpose_mask).sum()
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = region.label
    return (labeled == best_label) if best_label else cellpose_mask


def _refine_with_isoline(prob, cellpose_mask, threshold, min_area):
    """Extract boundary as the threshold isoline of the probability map."""
    contours = measure.find_contours(prob, threshold)
    if not contours:
        return cellpose_mask
    h, w = prob.shape
    best_overlap = 0
    best_mask = None
    for contour in contours:
        pts = contour[:, ::-1].astype(np.int32)
        test_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(test_mask, [pts], 1)
        if test_mask.sum() < min_area:
            continue
        overlap = np.logical_and(test_mask > 0, cellpose_mask).sum()
        if overlap > best_overlap:
            best_overlap = overlap
            best_mask = test_mask
    if best_mask is None:
        return cellpose_mask
    return best_mask.astype(bool)


def refine_all_masks_rf(frames, cellpose_masks, rf_model, threshold=0.75,
                        config=None, progress_fn=None, use_isoline=True):
    n = len(frames)
    refined = np.zeros_like(cellpose_masks)
    for i in range(n):
        if progress_fn:
            progress_fn(i)
        refined[i] = refine_mask_with_rf(
            frames[i], cellpose_masks[i], rf_model, threshold,
            config=config, use_isoline=use_isoline,
        )
    return refined
