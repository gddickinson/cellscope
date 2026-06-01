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


def _compose_rgb(dic, fluo=None, fluo_min_max=30):
    """Build a 3-channel uint8 RGB image for SAM2 from microscopy
    channels.

    No fluo: tile DIC to all 3 channels (same as _grayscale_to_rgb).
    With fluo + meaningful signal: pack as R=fluo, G=DIC, B=DIC.
      SAM2's encoder then sees the fluorescence signal as a
      localisation cue (nucleus stains are bright on the R channel)
      while still having DIC for boundary information in G+B. No
      extra inference cost — same one encoder pass with a richer
      input.
    With fluo but weak signal (crop's max < fluo_min_max):
      auto fall back to DIC-only. Otherwise SAM2 would see a
      near-black R channel which can be misinterpreted as a
      shadow / boundary feature, subtly pulling the mask. Default
      threshold 30/255 ≈ 12 % of full range — anything dimmer is
      mostly noise.

    Convention R=fluo follows microscopy display norms (Cy5 → red).
    Channels must already have matching (H, W) shapes.
    """
    if fluo is None or dic.shape != fluo.shape:
        return _grayscale_to_rgb(dic), False
    if int(fluo.max()) < int(fluo_min_max):
        return _grayscale_to_rgb(dic), False
    return np.stack([fluo, dic, dic], axis=-1), True


def _compute_crop(x, y, H, W, size):
    half = size // 2
    y0 = max(0, y - half)
    y1 = min(H, y + half)
    x0 = max(0, x - half)
    x1 = min(W, x + half)
    return x0, y0, x1, y1


from gui.mask_editor_sam2_ensemble import _predict_with_tta  # noqa: E402


def _disk_struct(radius):
    """Circular structuring element of given pixel radius."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def _postprocess_mask(raw_mask, smooth_radius=2,
                      keep_largest=True, fill_holes=True):
    """Clean up a raw SAM2 mask. Applied to every predicted mask
    before the guards run (so the area check sees the final, cleaned
    mask, not the raw noisy one).

    Steps, in order:
      1. keep_largest: drop detached connected components, keep only
         the biggest. Removes the small "noise blobs" SAM2 sometimes
         outputs alongside the main cell on textured DIC backgrounds.
      2. fill_holes: fill background pixels fully surrounded by the
         mask. Handles single-pixel gaps inside the cell body that
         would otherwise show as little holes.
      3. smooth_radius: morphological closing with a disk of given
         radius. Smooths the jagged single-pixel-staircase boundary
         SAM2 produces on low-contrast edges. Default radius 2 is
         conservative — closes gaps up to ~4 px wide without changing
         overall cell shape.

    Returns a clean bool mask of the same shape as raw_mask.
    """
    from scipy import ndimage
    m = raw_mask.astype(bool)
    if not m.any():
        return m
    if keep_largest:
        labeled, n = ndimage.label(m)
        if n > 1:
            sizes = ndimage.sum(m, labeled, range(1, n + 1))
            pick = int(np.argmax(sizes)) + 1
            m = labeled == pick
    if fill_holes:
        m = ndimage.binary_fill_holes(m)
    if smooth_radius > 0:
        struct = _disk_struct(smooth_radius)
        m = ndimage.binary_closing(m, structure=struct)
    return m.astype(bool)


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
                     fluo_image=None, fluo_min_max=30,
                     crop_size=512, min_area_px=200,
                     max_area_px=50000,
                     smooth_radius=2, keep_largest=True,
                     fill_holes=True,
                     use_tta=False):
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
    fluo_crop = (fluo_image[y0:y1, x0:x1]
                 if fluo_image is not None else None)
    crop_rgb, used_fluo = _compose_rgb(
        crop, fluo_crop, fluo_min_max=fluo_min_max)
    local_x = click_x - x0
    local_y = click_y - y0

    try:
        predictor = _ensure_predictor()
    except Exception as e:
        return SAM2PointResult(
            False, message=f"SAM2 load failed: {e!r}")

    try:
        with torch.inference_mode():
            raw_mask, score = _predict_with_tta(
                predictor, crop_rgb,
                point_coords=np.array([[local_x, local_y]]),
                point_labels=np.array([1]),
                box=None, allowed=None, use_tta=use_tta)
    except Exception as e:
        return SAM2PointResult(
            False, message=f"SAM2 inference failed: {e!r}")

    best_mask = _postprocess_mask(
        raw_mask,
        smooth_radius=smooth_radius,
        keep_largest=keep_largest,
        fill_holes=fill_holes)

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

    fluo_note = "" if fluo_image is None else (
        ", +Cy5" if used_fluo else ", Cy5 off (weak)")
    return SAM2PointResult(
        True, mask=best_mask, x0=x0, y0=y0,
        score=score, area=area,
        message=f"area={area} px, score={score:.2f}{fluo_note}")


def predict_at_box(frame_image, x0, y0, x1, y1,
                   fluo_image=None, fluo_min_max=30,
                   pad_px=64,
                   min_box_area_px=4096,
                   max_box_area_frac=0.5,
                   min_area_px=200,
                   max_area_cap_px=50000,
                   smooth_radius=2, keep_largest=True,
                   fill_holes=True,
                   constrain_to_box=True,
                   box_expand_frac=0.10,
                   use_tta=False):
    """Run SAM2 with a BOX prompt (+ implicit centroid click) on a
    padded crop of frame_image. Returns a SAM2PointResult.

    Box prompts give SAM2 a critical extra signal — the spatial
    extent — which dramatically improves segmentation on DIC images
    where cell boundaries are low-contrast (typical undershoot mode
    of the point-only tool). The centroid click disambiguates which
    cell when the box accidentally clips a neighbour.

    Guards:
      - min_box_area_px: rejects accidental near-zero drags (default
        4096 = ~64×64; smaller is probably a fumble, not a drag)
      - max_box_area_frac: rejects boxes spanning more than half the
        frame area (likely accidental over-drag)
      - min_area_px: rejects tiny predicted masks
      - per-call adaptive max area: max(box_area × 1.3, max_area_cap_px)
        — the user defined the upper bound by drawing the box, but
        we keep a generous floor so SAM2 still has room

    For convenience the input box coords are normalised internally
    (swap if dragged right-to-left or bottom-to-top) and clipped to
    the frame.
    """
    H, W = frame_image.shape[:2]
    x0n, x1n = int(min(x0, x1)), int(max(x0, x1))
    y0n, y1n = int(min(y0, y1)), int(max(y0, y1))
    x0n = max(0, min(W, x0n))
    x1n = max(0, min(W, x1n))
    y0n = max(0, min(H, y0n))
    y1n = max(0, min(H, y1n))
    box_w, box_h = x1n - x0n, y1n - y0n
    box_area = box_w * box_h

    if box_area < min_box_area_px:
        return SAM2PointResult(
            False,
            message=(f"box too small ({box_w}×{box_h}); drag a "
                     f"rectangle around the whole cell"))
    if box_area > max_box_area_frac * H * W:
        return SAM2PointResult(
            False,
            message=(f"box too large ({box_w}×{box_h}); draw "
                     f"tighter around a single cell"))

    cx0 = max(0, x0n - pad_px)
    cy0 = max(0, y0n - pad_px)
    cx1 = min(W, x1n + pad_px)
    cy1 = min(H, y1n + pad_px)
    crop = frame_image[cy0:cy1, cx0:cx1]
    fluo_crop = (fluo_image[cy0:cy1, cx0:cx1]
                 if fluo_image is not None else None)
    crop_rgb, used_fluo = _compose_rgb(
        crop, fluo_crop, fluo_min_max=fluo_min_max)

    local_box = np.array(
        [x0n - cx0, y0n - cy0, x1n - cx0, y1n - cy0], dtype=np.float32)
    centroid_x = (x0n + x1n) / 2.0
    centroid_y = (y0n + y1n) / 2.0
    local_cx = centroid_x - cx0
    local_cy = centroid_y - cy0

    try:
        predictor = _ensure_predictor()
    except Exception as e:
        return SAM2PointResult(
            False, message=f"SAM2 load failed: {e!r}")

    # Build the box+margin "allowed" region once in crop coords.
    # _predict_with_tta rotates it per rotation to keep the bias
    # consistent across TTA augmentations. constrain_to_box=False
    # disables both the bias and the hard clip; allowed is then None.
    crop_h, crop_w = crop.shape[:2]
    box_w_local = local_box[2] - local_box[0]
    box_h_local = local_box[3] - local_box[1]
    mx = int(box_w_local * box_expand_frac)
    my = int(box_h_local * box_expand_frac)
    allowed = None
    if constrain_to_box:
        allowed = np.zeros((crop_h, crop_w), dtype=bool)
        ax0 = max(0, int(local_box[0]) - mx)
        ay0 = max(0, int(local_box[1]) - my)
        ax1 = min(crop_w, int(local_box[2]) + mx)
        ay1 = min(crop_h, int(local_box[3]) + my)
        allowed[ay0:ay1, ax0:ax1] = True

    try:
        with torch.inference_mode():
            raw_mask, score = _predict_with_tta(
                predictor, crop_rgb,
                point_coords=np.array([[local_cx, local_cy]]),
                point_labels=np.array([1]),
                box=local_box, allowed=allowed, use_tta=use_tta)
    except Exception as e:
        return SAM2PointResult(
            False, message=f"SAM2 inference failed: {e!r}")

    # Hard clip: drop any predicted pixels outside the allowed
    # region. This is the second line of defence — even if the
    # picked candidate still leaks, we force the result to respect
    # the user's box (plus margin).
    clipped_pixels = 0
    if constrain_to_box and allowed is not None:
        before = int(raw_mask.sum())
        raw_mask = raw_mask & allowed
        clipped_pixels = before - int(raw_mask.sum())

    best_mask = _postprocess_mask(
        raw_mask,
        smooth_radius=smooth_radius,
        keep_largest=keep_largest,
        fill_holes=fill_holes)

    area = int(best_mask.sum())
    if area < min_area_px:
        return SAM2PointResult(
            False,
            message=(f"detected mask too small ({area} px) — try a "
                     f"tighter box around the cell"))
    adaptive_max = max(int(box_area * 1.3), max_area_cap_px)
    if area > adaptive_max:
        return SAM2PointResult(
            False,
            message=(f"detected mask too large ({area} px, box was "
                     f"{box_area}) — SAM2 leaked beyond the box; "
                     f"try a tighter box"))

    clip_note = (f", clipped {clipped_pixels} px to box+"
                 f"{int(box_expand_frac * 100)}%"
                 if clipped_pixels > 0 else "")
    fluo_note = "" if fluo_image is None else (
        ", +Cy5" if used_fluo else ", Cy5 off (weak)")
    return SAM2PointResult(
        True, mask=best_mask, x0=cx0, y0=cy0,
        score=score, area=area,
        message=(f"box: area={area} px, score={score:.2f}"
                 f"{clip_note}{fluo_note}"))


def apply_to_labels(labels_3d, frame_idx, result, target_id):
    """Paste result.mask into labels_3d[frame_idx] with target_id.

    Returns the pre-modification snapshot of labels_3d[frame_idx]
    so the caller can push it onto the undo stack.

    Pixels outside the predicted mask — even within the crop bounds
    — are untouched. Existing labels at pixels INSIDE the new mask
    are overwritten (the user prompted there on purpose).
    """
    snapshot = labels_3d[frame_idx].copy()
    h_crop, w_crop = result.mask.shape
    y0 = result.y0
    x0 = result.x0
    region = labels_3d[frame_idx, y0:y0 + h_crop, x0:x0 + w_crop]
    region[result.mask] = target_id
    labels_3d[frame_idx, y0:y0 + h_crop, x0:x0 + w_crop] = region
    return snapshot


def id_exists_in_frame(labels_3d, frame_idx, target_id):
    """True if any pixel of frame_idx is labelled target_id."""
    return bool(np.any(labels_3d[frame_idx] == target_id))

