"""Quality metrics for segmentation masks."""
import numpy as np
import cv2

from core.contour import get_contour


def compute_iou(mask_a, mask_b):
    """Intersection over Union."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def boundary_confidence(image, mask, sigma=2.0):
    """Mean image gradient magnitude along the mask contour.

    Higher = boundary sits on a real edge.
    """
    contour = get_contour(mask)
    if contour is None:
        return float("nan")
    img = image.astype(np.float64)
    g = cv2.GaussianBlur(img, (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    rows = np.clip(contour[:, 0].astype(int), 0, image.shape[0] - 1)
    cols = np.clip(contour[:, 1].astype(int), 0, image.shape[1] - 1)
    return float(np.mean(grad[rows, cols]))


def boundary_confidence_timeseries(frames, masks, sigma=2.0):
    """Boundary confidence per frame."""
    n = len(frames)
    out = np.full(n, np.nan)
    for i in range(n):
        if masks[i].any():
            out[i] = boundary_confidence(frames[i], masks[i], sigma)
    return out


def mean_consecutive_iou(masks):
    """Mean IoU between consecutive frames."""
    ious = []
    for i in range(len(masks) - 1):
        if masks[i].any() and masks[i + 1].any():
            ious.append(compute_iou(masks[i], masks[i + 1]))
    return float(np.mean(ious)) if ious else float("nan")


def area_stability(masks, um_per_px=1.0):
    """Per-recording stability summary."""
    n = len(masks)
    areas = np.array([m.sum() * um_per_px ** 2 for m in masks])
    valid = areas[areas > 0]
    if len(valid) < 2:
        return {
            "areas_um2": areas, "mean_area_um2": float("nan"),
            "area_cv": float("nan"), "max_min_ratio": float("nan"),
            "n_large_jumps": 0, "mean_consec_iou": float("nan"),
        }
    cv = float(np.std(valid) / np.mean(valid))
    ratio = float(valid.max() / valid.min())
    changes = np.abs(np.diff(areas[areas > 0])) / (areas[areas > 0][:-1] + 1)
    n_jumps = int((changes > 0.3).sum())
    return {
        "areas_um2": areas,
        "mean_area_um2": float(np.mean(valid)),
        "area_cv": cv,
        "max_min_ratio": ratio,
        "n_large_jumps": n_jumps,
        "mean_consec_iou": mean_consecutive_iou(masks),
    }
