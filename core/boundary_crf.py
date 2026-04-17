"""DenseCRF post-processing for boundary sharpening.

Ported from v1. Uses fully-connected pairwise CRF to enforce that
nearby pixels with similar intensity belong to the same class.
On its own this gives sharp edges but breaks temporal consistency;
when followed by temporal_smooth_polar_boundaries the IoU is recovered
while the edge sharpness is kept.
"""
import numpy as np
import cv2

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    HAS_CRF = True
except ImportError:
    HAS_CRF = False


def refine_mask_crf(image, mask_or_prob, crf_iters=5,
                    sxy_gauss=3, compat_gauss=3,
                    sxy_bilateral=40, srgb_bilateral=5,
                    compat_bilateral=10):
    """Refine a single mask using DenseCRF.

    Args:
        image: 2D uint8 (DIC frame)
        mask_or_prob: 2D bool mask OR float probability in [0, 1]
        crf_iters: number of CRF iterations
        sxy_*, srgb_*, compat_*: CRF kernel parameters

    Returns:
        refined_mask: 2D bool array
    """
    if not HAS_CRF:
        raise ImportError("pydensecrf2 not available")

    h, w = image.shape[:2]

    if mask_or_prob.dtype == bool or mask_or_prob.max() > 1:
        prob_fg = np.where(mask_or_prob > 0, 0.9, 0.1).astype(np.float32)
    else:
        prob_fg = np.clip(mask_or_prob.astype(np.float32), 0.01, 0.99)
    prob_bg = 1.0 - prob_fg
    probs = np.stack([prob_bg, prob_fg], axis=0)

    unary = unary_from_softmax(probs)
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(
        sxy=sxy_gauss, compat=compat_gauss,
        kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    if image.ndim == 2:
        img_3ch = np.stack([image, image, image], axis=-1)
    else:
        img_3ch = image
    d.addPairwiseBilateral(
        sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=img_3ch,
        compat=compat_bilateral,
        kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    Q = d.inference(crf_iters)
    labels = np.argmax(Q, axis=0).reshape(h, w)
    return labels.astype(bool)


def refine_all_masks_crf(frames, masks, progress_fn=None, **kwargs):
    """Apply CRF refinement to all frames."""
    if not HAS_CRF:
        raise ImportError("pydensecrf2 not available")
    n = len(frames)
    refined = np.zeros_like(masks)
    for i in range(n):
        if progress_fn:
            progress_fn(i)
        refined[i] = refine_mask_crf(frames[i], masks[i], **kwargs)
    return refined
