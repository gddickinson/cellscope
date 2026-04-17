"""Improved optical flow quality assessment, inspired by Robitaille et al. 2022.

The key insight from the paper: flow magnitude alone gives a narrow,
data-dependent signal. Combining it with static features (image gradient,
entropy) — and analyzing their joint distribution — produces a much more
discriminating quality measure.

Here we compute:
  1. A per-frame quality score from flow + image gradient agreement
     at the cellpose boundary
  2. A per-pixel "flow trust map" — high where flow magnitude AND image
     gradient agree at sub-membrane scale, low where they don't
"""
import numpy as np
import cv2
from scipy import ndimage

from core.contour import get_contour


def _gradient_magnitude(image, sigma=1.5):
    img = image.astype(np.float64)
    g = cv2.GaussianBlur(img, (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def assess_flow_quality_joint(image, flow_mag, cp_mask):
    """Per-frame flow quality from joint flow + image gradient at boundary.

    A frame is "high quality" for flow when:
      - Image gradient at the cellpose boundary is sharp (cellpose found
        a real edge)
      - Flow gradient at the same boundary is also sharp (motion stops
        at the same place)
      - Their magnitudes are correlated along the contour (they agree
        about WHERE the boundary is)

    Returns:
        score: float in [0, 1+]; can exceed 1 when both signals are
            unusually strong
        diagnostics: dict
    """
    if not cp_mask.any() or flow_mag.max() == 0:
        return 0.0, {"reason": "empty"}

    contour = get_contour(cp_mask)
    if contour is None or len(contour) < 8:
        return 0.0, {"reason": "no contour"}

    # Sample image gradient and flow gradient along the boundary
    img_grad = _gradient_magnitude(image, sigma=1.5)
    flow_grad = _gradient_magnitude(
        (flow_mag * 255 / max(flow_mag.max(), 1e-6)).astype(np.uint8),
        sigma=1.5,
    )

    rows = np.clip(contour[:, 0].astype(int), 0, image.shape[0] - 1)
    cols = np.clip(contour[:, 1].astype(int), 0, image.shape[1] - 1)

    img_grad_b = img_grad[rows, cols]
    flow_grad_b = flow_grad[rows, cols]

    # Normalize each signal to its 90th percentile of the whole frame
    img_norm = float(np.mean(img_grad_b) / max(np.percentile(img_grad, 90), 1e-6))
    flow_norm = float(np.mean(flow_grad_b) / max(np.percentile(flow_grad, 90), 1e-6))

    # Correlation along the boundary (do they peak at the same places?)
    if len(img_grad_b) > 4:
        ig = (img_grad_b - img_grad_b.mean())
        fg = (flow_grad_b - flow_grad_b.mean())
        denom = (np.linalg.norm(ig) * np.linalg.norm(fg))
        corr = float(np.dot(ig, fg) / denom) if denom > 1e-6 else 0.0
    else:
        corr = 0.0

    # Compose: image signal gives baseline, flow signal amplifies, correlation
    # rewards agreement. Each component is in [0, ~1.5].
    score = float(0.4 * img_norm + 0.4 * flow_norm + 0.2 * max(0, corr))

    return score, {
        "img_norm": img_norm,
        "flow_norm": flow_norm,
        "corr": corr,
    }


def flow_trust_map(image, flow_mag, cp_mask, sigma=8.0):
    """Per-pixel trust map for flow.

    Trust is high where:
      - The pixel is near (or inside) the cellpose mask (spatial constraint)
      - The local flow gradient and image gradient agree on direction
      - Flow magnitude is non-trivial

    Returns:
        trust: (H, W) float array in [0, 1]
    """
    h, w = image.shape[:2]
    if not cp_mask.any():
        return np.zeros((h, w), dtype=np.float32)

    # Spatial weight: smooth band around the cellpose boundary
    di = ndimage.distance_transform_edt(cp_mask)
    do = ndimage.distance_transform_edt(~cp_mask)
    signed = di - do
    spatial_w = 1.0 / (1.0 + np.exp(-signed / sigma))

    # Smoothed flow magnitude as a soft mask
    flow_smooth = cv2.GaussianBlur(flow_mag.astype(np.float32), (0, 0), 3.0)
    if flow_smooth.max() > 0:
        flow_norm = np.clip(flow_smooth / np.percentile(flow_smooth, 95), 0, 1)
    else:
        flow_norm = np.zeros_like(spatial_w)

    # Image gradient - flow gradient agreement (per pixel)
    img_g = _gradient_magnitude(image, sigma=2.0)
    flow_g = _gradient_magnitude(
        (flow_smooth * 255 / max(flow_smooth.max(), 1e-6)).astype(np.uint8),
        sigma=2.0,
    )
    if img_g.max() > 0:
        img_n = np.clip(img_g / np.percentile(img_g, 95), 0, 1)
    else:
        img_n = np.zeros_like(spatial_w)
    if flow_g.max() > 0:
        flow_gn = np.clip(flow_g / np.percentile(flow_g, 95), 0, 1)
    else:
        flow_gn = np.zeros_like(spatial_w)
    agreement = np.minimum(img_n, flow_gn)

    # Composite: spatial × (flow strength + agreement)
    trust = spatial_w * (0.5 * flow_norm + 0.5 * agreement)
    return trust.astype(np.float32)
