"""VAMPIRE shape mode analysis integration.

Wraps the vampire-analysis package to provide shape mode
decomposition, clustering, and heterogeneity metrics for
CellScope's per-frame cell masks.

Reference:
    Lam et al., "A robust unsupervised machine-learning method to
    quantify the morphological heterogeneity of cells and nuclei",
    Nature Protocols 2021.
"""
import numpy as np
import logging

log = logging.getLogger(__name__)

N_CONTOUR_POINTS = 50


def extract_contour_from_mask(mask):
    """Extract a single cell contour from a binary mask.

    Returns (2, N_CONTOUR_POINTS) array or None if no cell found.
    """
    from vampire.extraction import get_contour_from_object
    from vampire.processing import register_contour, sample_contour

    if not mask.any():
        return None
    try:
        contour = get_contour_from_object(mask.astype(np.uint8))
        if contour is None or contour.shape[1] < 10:
            return None
        sampled = sample_contour(contour, N_CONTOUR_POINTS)
        registered = register_contour(sampled)
        return registered
    except Exception as e:
        log.debug("Contour extraction failed: %s", e)
        return None


def extract_contours_from_stack(masks):
    """Extract registered contours from a mask stack.

    Args:
        masks: (N, H, W) bool array — one cell per frame

    Returns:
        contours: (M, 2*N_CONTOUR_POINTS) array — valid frames only
        valid_indices: list of frame indices with valid contours
    """
    contours = []
    valid_idx = []
    for i in range(len(masks)):
        c = extract_contour_from_mask(masks[i])
        if c is not None:
            contours.append(c.flatten())
            valid_idx.append(i)
    if not contours:
        return np.array([]), []
    return np.array(contours), valid_idx


def compute_shape_modes(contours, n_clusters=5, n_pc=20):
    """Run VAMPIRE PCA + K-means on extracted contours.

    Args:
        contours: (M, 2*N_CONTOUR_POINTS) array from
                  extract_contours_from_stack
        n_clusters: number of shape mode clusters
        n_pc: number of principal components for clustering

    Returns:
        dict with:
            principal_directions: eigenvectors
            principal_components: PC scores per contour
            cluster_ids: (M,) int array — shape mode per contour
            centroids: (n_clusters, n_pc) cluster centers
            mean_contour: mean shape
            explained_variance: variance ratio per PC
    """
    from vampire.analysis import pca_contours, cluster_contours

    if len(contours) < n_clusters:
        log.warning("Too few contours (%d) for %d clusters",
                     len(contours), n_clusters)
        n_clusters = max(2, len(contours) // 2)

    n_pc = min(n_pc, len(contours) - 1, contours.shape[1])

    mean_contour = contours.mean(axis=0)
    centered = contours - mean_contour

    principal_directions, pc = pca_contours(centered)

    contours_df, centroids = cluster_contours(
        pc, contours, num_clusters=n_clusters,
        num_pc=n_pc, random_state=42)

    cluster_ids = contours_df["cluster_id"].values.astype(int)

    total_var = np.var(centered, axis=0).sum()
    explained = np.array([
        np.var(pc[:, i]) / total_var for i in range(min(n_pc, pc.shape[1]))
    ])

    return {
        "principal_directions": principal_directions,
        "principal_components": pc,
        "cluster_ids": cluster_ids,
        "centroids": centroids,
        "mean_contour": mean_contour,
        "explained_variance": explained,
        "n_clusters": n_clusters,
        "n_contours": len(contours),
    }


def compute_heterogeneity(cluster_ids, n_clusters):
    """Compute Shannon entropy of shape mode distribution.

    Returns:
        dict with entropy, mode_counts, mode_fractions
    """
    counts = np.bincount(cluster_ids, minlength=n_clusters)
    total = counts.sum()
    if total == 0:
        return {"entropy": 0, "mode_counts": counts.tolist(),
                "mode_fractions": [0] * n_clusters}
    fracs = counts / total
    entropy = -np.sum(fracs[fracs > 0] * np.log2(fracs[fracs > 0]))
    return {
        "entropy": float(entropy),
        "max_entropy": float(np.log2(n_clusters)),
        "normalized_entropy": float(entropy / max(np.log2(n_clusters), 1)),
        "mode_counts": counts.tolist(),
        "mode_fractions": fracs.tolist(),
    }


def run_vampire_analysis(masks, n_clusters=5, n_pc=20):
    """Full VAMPIRE pipeline on a mask stack.

    Args:
        masks: (N, H, W) bool — per-frame cell masks
        n_clusters: number of shape modes
        n_pc: PCA components for clustering

    Returns:
        dict with all VAMPIRE results + heterogeneity, or None if
        too few valid contours.
    """
    contours, valid_idx = extract_contours_from_stack(masks)
    if len(contours) < 5:
        log.warning("Too few valid contours (%d) for VAMPIRE",
                     len(contours))
        return None

    modes = compute_shape_modes(contours, n_clusters, n_pc)
    hetero = compute_heterogeneity(modes["cluster_ids"],
                                    modes["n_clusters"])

    # Build per-frame mode assignment (NaN for frames without contour)
    n_frames = len(masks)
    frame_modes = np.full(n_frames, np.nan)
    for i, fi in enumerate(valid_idx):
        frame_modes[fi] = modes["cluster_ids"][i]

    return {
        "contours": contours,
        "valid_indices": valid_idx,
        "modes": modes,
        "heterogeneity": hetero,
        "frame_modes": frame_modes,
        "n_clusters": modes["n_clusters"],
        "n_contours": modes["n_contours"],
    }
