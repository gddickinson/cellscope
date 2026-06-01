"""SAM2 point-and-click cell detection for the mask editor.

A new editor tool: pick "sam2" in the tool palette, then left-click
anywhere on the canvas. The clicked point is passed to SAM2 (Hiera
Tiny) on a small crop centred at the click. The highest-confidence
mask is added to the current frame with the active cell ID from the
cell-spinner.

Guards before apply (see predict_at_point):
  - the clicked pixel must lie inside the predicted mask
  - the predicted mask must be at least min_area_px pixels
The editor additionally refuses if the active ID already labels a
cell in this frame — use a different ID, or delete the existing
cell first.

On success the prior label state is pushed to the undo stack and
the frame is marked dirty; Cmd+Z restores.

Why a small crop: SAM2's encoder cost is roughly O(N²). A 512×512
crop encodes ~16× faster than a full 2048×2048 frame, giving every
click ~100-200 ms of latency on M1 Max (MPS). No first-click warm-up
surprise (which a full-frame + cache approach would impose).
"""
import os
import numpy as np
import torch

# Lazy singleton — loading SAM2 takes ~3-5 s; do it on first use only.
_PREDICTOR = None
_DEFAULT_CKPT = "data/models/sam2/sam2.1_hiera_tiny.pt"
_DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ensure_predictor(checkpoint=_DEFAULT_CKPT, config=_DEFAULT_CFG):
    """Lazy-load SAM2 image predictor on first use."""
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not os.path.isabs(checkpoint):
        proj = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        ckpt = os.path.join(proj, checkpoint)
    else:
        ckpt = checkpoint
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {ckpt}")
    sam2_model = build_sam2(config, ckpt, device=_device())
    _PREDICTOR = SAM2ImagePredictor(sam2_model)
    return _PREDICTOR


def _grayscale_to_rgb(arr):
    """SAM2 expects 3-channel RGB; tile a single-channel DIC image."""
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return np.concatenate([arr, arr, arr], axis=-1)
    return arr


def _compute_crop(x, y, H, W, size):
    half = size // 2
    y0 = max(0, y - half)
    y1 = min(H, y + half)
    x0 = max(0, x - half)
    x1 = min(W, x + half)
    return x0, y0, x1, y1


class SAM2PointResult:
    """Outcome of a single point-prediction attempt."""

    def __init__(self, ok, mask=None, x0=0, y0=0, score=0.0,
                 area=0, message=""):
        self.ok = ok          # True iff SAM2 + guards succeeded
        self.mask = mask      # bool (h_crop, w_crop) when ok
        self.x0 = x0          # crop origin (x) in full-frame coords
        self.y0 = y0          # crop origin (y) in full-frame coords
        self.score = score    # SAM2 confidence
        self.area = area
        self.message = message


def predict_at_point(frame_image, click_x, click_y,
                     crop_size=512, min_area_px=200,
                     max_area_px=50000):
    """Run SAM2 at (click_x, click_y) on a cropped region of frame_image.

    Returns a SAM2PointResult. On failure result.ok is False and
    result.message describes why; the caller should surface that to
    the user and not touch the masks.

    max_area_px guards against SAM2 segmenting empty/low-contrast
    background as one giant blob. Default 50000 px is ~10× a typical
    keratinocyte cell area (~5000 px at 0.6523 µm/px) — generous
    enough not to clip legitimately large cells, tight enough to
    reject the 200k-300k-px background blobs SAM2 produces when the
    click happens to land on featureless space.
    """
    H, W = frame_image.shape[:2]
    if not (0 <= click_x < W and 0 <= click_y < H):
        return SAM2PointResult(False, message="click out of bounds")

    x0, y0, x1, y1 = _compute_crop(click_x, click_y, H, W, crop_size)
    crop = frame_image[y0:y1, x0:x1]
    crop_rgb = _grayscale_to_rgb(crop)
    local_x = click_x - x0
    local_y = click_y - y0

    try:
        predictor = _ensure_predictor()
    except Exception as e:
        return SAM2PointResult(
            False, message=f"SAM2 load failed: {e!r}")

    try:
        with torch.inference_mode():
            predictor.set_image(crop_rgb)
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[local_x, local_y]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
    except Exception as e:
        return SAM2PointResult(
            False, message=f"SAM2 inference failed: {e!r}")

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(bool)
    score = float(scores[best_idx])

    if not best_mask[local_y, local_x]:
        return SAM2PointResult(
            False,
            message=("click not inside detected mask — try clicking "
                     "nearer the cell centre"))

    area = int(best_mask.sum())
    if area < min_area_px:
        return SAM2PointResult(
            False,
            message=(f"detected mask too small ({area} px) — likely "
                     f"background"))
    if area > max_area_px:
        return SAM2PointResult(
            False,
            message=(f"detected mask too large ({area} px) — likely "
                     f"background blob; click closer to the cell"))

    return SAM2PointResult(
        True, mask=best_mask, x0=x0, y0=y0,
        score=score, area=area,
        message=f"area={area} px, score={score:.2f}")


def id_exists_in_frame(labels_3d, frame_idx, target_id):
    """True if any pixel of frame_idx is labelled target_id."""
    return bool(np.any(labels_3d[frame_idx] == target_id))


def apply_to_labels(labels_3d, frame_idx, result, target_id):
    """Paste result.mask into labels_3d[frame_idx] with target_id.

    Returns the pre-modification snapshot of labels_3d[frame_idx]
    for the caller to push onto the undo stack.

    Pixels outside the predicted mask — even within the crop bounds
    — are untouched. Existing labels at pixels INSIDE the new mask
    are overwritten (the user clicked there on purpose).
    """
    snapshot = labels_3d[frame_idx].copy()
    h_crop, w_crop = result.mask.shape
    y0 = result.y0
    x0 = result.x0
    region = labels_3d[frame_idx, y0:y0 + h_crop, x0:x0 + w_crop]
    region[result.mask] = target_id
    labels_3d[frame_idx, y0:y0 + h_crop, x0:x0 + w_crop] = region
    return snapshot
