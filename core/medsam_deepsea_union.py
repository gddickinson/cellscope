"""MedSAM + DeepSea union refinement.

Empirical finding (Ignasi IC293, 15 GT frames):
    cellpose+MedSAM alone: 0.860 mean IoU, 9/15 > 0.85
    + DeepSea union:       0.885 mean IoU, 14/15 > 0.85

MedSAM's bbox-precise boundary tends to under-segment by ~5%
(we measured this directly). DeepSea's pretrained phase-contrast
segmentation over-segments the cell into multiple parts but its
union mask covers MedSAM's missed boundary pixels. Pixel-OR of
the two recovers most of the under-segmented area without adding
much false-positive area.

Use after MedSAM refinement; takes the per-frame MedSAM mask +
runs DeepSea on the same image, returns the union (with fill +
largest connected component cleanup).

Requires DeepSea's segmentation.pth at data/models/deepsea/
segmentation.pth (project-relative). If absent, raises FileNotFoundError.
"""
import os
import sys
import numpy as np
import cv2
from scipy import ndimage as ndi


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEEPSEA_DIR = os.path.join(_PROJECT_ROOT, "data", "models", "deepsea")
_DEEPSEA_CKPT = os.path.join(_DEEPSEA_DIR, "segmentation.pth")
_DS_MODEL = None
_DS_TFM = None
_DS_DEVICE = None


def _load_deepsea():
    global _DS_MODEL, _DS_TFM, _DS_DEVICE
    if _DS_MODEL is not None:
        return _DS_MODEL, _DS_TFM, _DS_DEVICE
    if not os.path.exists(_DEEPSEA_CKPT):
        raise FileNotFoundError(
            f"DeepSea checkpoint not found: {_DEEPSEA_CKPT}\n"
            "Run the Setup Wizard or copy segmentation.pth to "
            "data/models/deepsea/")
    if _DEEPSEA_DIR not in sys.path:
        sys.path.insert(0, _DEEPSEA_DIR)
    import torch
    import torchvision.transforms as transforms
    from model import DeepSeaSegmentation
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSeaSegmentation(n_channels=1, n_classes=2, bilinear=True)
    ckpt = torch.load(_DEEPSEA_CKPT, map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([383, 512]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    _DS_MODEL, _DS_TFM, _DS_DEVICE = model, tfm, device
    return model, tfm, device


def _deepsea_predict(image):
    """Run DeepSea on a uint8 image. Returns full-image bool mask
    (union of all detected cells; not largest CC)."""
    import torch
    from skimage.morphology import remove_small_objects
    model, tfm, device = _load_deepsea()
    H, W = image.shape
    t = tfm(image).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        mp, _ = model(t.unsqueeze(0))
    arr = mp.argmax(dim=1).cpu().numpy()[0]
    clean = remove_small_objects(arr > 0, min_size=20, connectivity=1)
    full = cv2.resize(clean.astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST) > 0
    return full


def union_with_deepsea(image, medsam_mask):
    """Return union of medsam_mask and DeepSea segmentation, then
    fill holes and keep the largest connected component."""
    if not medsam_mask.any():
        return medsam_mask
    try:
        ds_mask = _deepsea_predict(image)
    except Exception:
        return medsam_mask  # graceful fallback
    union = medsam_mask | ds_mask
    union = ndi.binary_fill_holes(union)
    lbl, _ = ndi.label(union)
    if lbl.max() > 1:
        sizes = ndi.sum(union, lbl, range(1, lbl.max() + 1))
        union = (lbl == int(np.argmax(sizes)) + 1)
    return union.astype(bool)


def union_with_deepsea_all(frames, masks, progress_fn=None):
    """Apply union_with_deepsea to every frame."""
    out = np.zeros_like(masks)
    for i in range(len(frames)):
        if progress_fn:
            progress_fn(i)
        out[i] = union_with_deepsea(frames[i], masks[i])
    return out
