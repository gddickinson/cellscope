"""MedSAM refinement — HuggingFace SAM format (Phase 14e).

MedSAM (Ma et al. 2024) is SAM fine-tuned on ~1M biomedical images.
The canonical checkpoint at `flaviagiammarino/medsam-vit-base` uses
the HuggingFace `transformers.SamModel` format, which is NOT
compatible with the `segment_anything` library used by `core/
sam_refine.py`. This module wraps that API separately.

Entry point: `refine_with_medsam(image, seed_mask)`.
"""
import os
import numpy as np
import torch

_MODEL_CACHE = {}


def load_medsam(repo_id="flaviagiammarino/medsam-vit-base", device=None):
    if repo_id in _MODEL_CACHE:
        return _MODEL_CACHE[repo_id]
    from transformers import SamModel, SamProcessor

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model = SamModel.from_pretrained(repo_id).to(device).eval()
    processor = SamProcessor.from_pretrained(repo_id)
    _MODEL_CACHE[repo_id] = (model, processor, device)
    return model, processor, device


def refine_with_medsam(image, seed_mask):
    """MedSAM refinement — prompted by bbox of the seed mask."""
    if not seed_mask.any():
        return seed_mask
    model, processor, device = load_medsam()

    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1)
    else:
        rgb = image
    rgb = rgb.astype(np.uint8)

    ys, xs = np.where(seed_mask)
    h, w = seed_mask.shape
    pad = 10
    box = [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
           min(w, int(xs.max()) + pad), min(h, int(ys.max()) + pad)]

    inputs = processor(rgb, input_boxes=[[box]], return_tensors="pt")
    # MPS doesn't support float64 — cast before moving
    coerced = {}
    for k, v in inputs.items():
        if v.dtype == torch.float64:
            v = v.float()
        coerced[k] = v.to(device)
    inputs = coerced
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    masks = processor.image_processor.post_process_masks(
        out.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0].numpy().squeeze()
    if not masks.any():
        return seed_mask
    return masks.astype(bool)


def refine_all_with_medsam(frames, masks, progress_fn=None):
    refined = np.zeros_like(masks)
    for i in range(len(frames)):
        if progress_fn:
            progress_fn(i)
        refined[i] = refine_with_medsam(frames[i], masks[i])
    return refined
