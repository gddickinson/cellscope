"""SAM (Segment Anything Model) boundary refinement.

SAM is a general-purpose foundation model trained on ~1B diverse masks.
Given prompts (centroid point + bounding box + optional rough mask),
it returns a precise boundary. Useful as an alternative refinement
path — historically on DIC keratinocytes it produces smooth blob-like
contours and misses filopodia, but bundled as an option for users who
want to compare against the RF pipeline or need generic foundation-
model behaviour on non-keratinocyte DIC data.

Requires `segment_anything` package and a SAM checkpoint under
`data/models/sam/sam_vit_{b,l,h}_*.pth`. The `vit_b` model ships with
this project; `vit_l` / `vit_h` can be downloaded if desired.

Port of the old CellScope/core/sam_refine.py — paths updated to
match the consolidated project layout.
"""
import os
import numpy as np
import torch
from scipy import ndimage

from config import PROJECT_DIR

_SAM_DIR = os.path.join(PROJECT_DIR, "data", "models", "sam")
_MODEL_CACHE = {}

# SAM checkpoint filenames shipped with the segment-anything package
_CHECKPOINT_FILES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def sam_checkpoint_path(model_type="vit_b") -> str:
    """Return the expected on-disk path for a SAM checkpoint."""
    return os.path.join(_SAM_DIR, _CHECKPOINT_FILES[model_type])


def list_available_sam_models() -> list:
    """Return SAM model types for which a checkpoint is present."""
    return [mt for mt in _CHECKPOINT_FILES
            if os.path.exists(sam_checkpoint_path(mt))]


def load_sam(model_type="vit_b", device=None):
    """Load (and cache) a SAM predictor.

    Raises FileNotFoundError if the checkpoint isn't present.
    """
    if model_type in _MODEL_CACHE:
        return _MODEL_CACHE[model_type]

    from segment_anything import sam_model_registry, SamPredictor

    ckpt = sam_checkpoint_path(model_type)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {ckpt}\n"
            "Expected location: data/models/sam/\n"
            "Download vit_b (~360 MB) with:\n"
            "  mkdir -p data/models/sam && \\\n"
            "  curl -L -o data/models/sam/sam_vit_b_01ec64.pth \\\n"
            "  https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_b_01ec64.pth"
        )

    if device is None:
        # Env override for debugging — `SAM_FORCE_CPU=1` forces CPU
        if os.environ.get("SAM_FORCE_CPU") == "1":
            device = "cpu"
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    _MODEL_CACHE[model_type] = predictor
    return predictor


def _mask_to_prompts(mask, n_fg_points=3, n_bg_points=2, padding=10):
    """Build SAM prompts from a seed mask.

    Returns:
        point_coords: (N, 2) array of (x, y)
        point_labels: (N,) array, 1=foreground / 0=background
        box: (x1, y1, x2, y2) bounding box
        mask_input: None (computed separately if use_mask_prompt=True)
    """
    if not mask.any():
        return None, None, None, None

    ys, xs = np.where(mask)
    h, w = mask.shape

    x1 = max(0, xs.min() - padding)
    y1 = max(0, ys.min() - padding)
    x2 = min(w, xs.max() + padding)
    y2 = min(h, ys.max() + padding)
    box = np.array([x1, y1, x2, y2])

    # Foreground: centroid + evenly-spaced interior points
    cy, cx = ndimage.center_of_mass(mask)
    fg_points = [(cx, cy)]
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask, iterations=3)
    if eroded.any():
        eys, exs = np.where(eroded)
        step = max(1, len(eys) // max(n_fg_points - 1, 1))
        for i in range(0, len(eys), step):
            if len(fg_points) >= n_fg_points:
                break
            fg_points.append((exs[i], eys[i]))

    # Background: image corners not inside the mask
    bg_points = []
    for cx_, cy_ in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if not mask[cy_, cx_] and len(bg_points) < n_bg_points:
            bg_points.append((cx_, cy_))

    point_coords = np.array(fg_points + bg_points, dtype=np.float32)
    point_labels = np.array(
        [1] * len(fg_points) + [0] * len(bg_points), dtype=np.int32
    )
    return point_coords, point_labels, box, None


def refine_with_sam(image, seed_mask, predictor=None,
                    use_mask_prompt=True):
    """Refine a single seed mask using SAM."""
    if predictor is None:
        predictor = load_sam()
    if not seed_mask.any():
        return seed_mask

    # SAM wants RGB uint8
    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1)
    else:
        rgb = image
    rgb = rgb.astype(np.uint8)

    predictor.set_image(rgb)
    point_coords, point_labels, box, _ = _mask_to_prompts(seed_mask)

    mask_input = None
    if use_mask_prompt:
        from scipy.ndimage import zoom
        h, w = seed_mask.shape
        low_res = zoom(
            seed_mask.astype(np.float32), (256 / h, 256 / w), order=1,
        )
        mask_input = (low_res * 2 - 1) * 10  # [0,1] → approx [-10, 10]
        mask_input = mask_input[None, ...]

    masks, scores, _ = predictor.predict(
        point_coords=point_coords, point_labels=point_labels,
        box=box[None, :], mask_input=mask_input,
        multimask_output=True,
    )
    # Best mask by predicted IoU score
    best_idx = int(np.argmax(scores))
    refined = masks[best_idx]
    if not refined.any():
        return seed_mask  # safety fallback
    return refined.astype(bool)


def refine_all_with_sam(frames, masks, model_type="vit_b",
                        use_mask_prompt=True, progress_fn=None):
    """Apply SAM refinement to every frame."""
    predictor = load_sam(model_type)
    refined = np.zeros_like(masks)
    for i in range(len(frames)):
        if progress_fn:
            progress_fn(i)
        refined[i] = refine_with_sam(
            frames[i], masks[i], predictor=predictor,
            use_mask_prompt=use_mask_prompt,
        )
    return refined


# ──────────────────────── SAM2 support ────────────────────────

_SAM2_DIR = os.path.join(PROJECT_DIR, "data", "models", "sam2")
_SAM2_CACHE = {}

_SAM2_CHECKPOINTS = {
    "hiera_t":  ("sam2.1_hiera_tiny.pt",   "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "hiera_s":  ("sam2.1_hiera_small.pt",  "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "hiera_b+": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "hiera_l":  ("sam2.1_hiera_large.pt",  "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


def list_available_sam2_models() -> list:
    return [k for k, (fn, _) in _SAM2_CHECKPOINTS.items()
            if os.path.exists(os.path.join(_SAM2_DIR, fn))]


def load_sam2(model_type="hiera_t", device=None):
    """Load (and cache) a SAM2 image predictor."""
    if model_type in _SAM2_CACHE:
        return _SAM2_CACHE[model_type]
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt_name, cfg = _SAM2_CHECKPOINTS[model_type]
    ckpt = os.path.join(_SAM2_DIR, ckpt_name)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {ckpt}. Download from "
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
            f"{ckpt_name}"
        )

    if device is None:
        if os.environ.get("SAM_FORCE_CPU") == "1":
            device = "cpu"
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model = build_sam2(cfg, ckpt, device=device)
    predictor = SAM2ImagePredictor(model)
    _SAM2_CACHE[model_type] = predictor
    return predictor


def refine_with_sam2(image, seed_mask, predictor=None,
                     use_mask_prompt=True):
    """Refine a seed mask using SAM2 (API matches refine_with_sam)."""
    if predictor is None:
        predictor = load_sam2()
    if not seed_mask.any():
        return seed_mask

    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1)
    else:
        rgb = image
    rgb = rgb.astype(np.uint8)

    predictor.set_image(rgb)
    point_coords, point_labels, box, _ = _mask_to_prompts(seed_mask)

    mask_input = None
    if use_mask_prompt:
        from scipy.ndimage import zoom
        h, w = seed_mask.shape
        low_res = zoom(
            seed_mask.astype(np.float32), (256 / h, 256 / w), order=1,
        )
        mask_input = (low_res * 2 - 1) * 10
        mask_input = mask_input[None, ...]

    masks, scores, _ = predictor.predict(
        point_coords=point_coords, point_labels=point_labels,
        box=box[None, :], mask_input=mask_input,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    refined = masks[best_idx]
    if not refined.any():
        return seed_mask
    return refined.astype(bool)


def refine_all_with_sam2(frames, masks, model_type="hiera_t",
                         use_mask_prompt=True, progress_fn=None):
    predictor = load_sam2(model_type)
    refined = np.zeros_like(masks)
    for i in range(len(frames)):
        if progress_fn:
            progress_fn(i)
        refined[i] = refine_with_sam2(
            frames[i], masks[i], predictor=predictor,
            use_mask_prompt=use_mask_prompt,
        )
    return refined
