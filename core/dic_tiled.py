"""Tiled DIC detection using VAMPIRE-sized crops.

cellpose_dic_v2 was trained on VAMPIRE crops (~300-500px). Running it
on full 1024² frames creates a scale mismatch. This module tiles
frames into training-distribution-sized patches, runs detection per
tile, and merges results.

Tile size default: 448px (median of VAMPIRE crop dimensions) with
64px overlap to avoid boundary artifacts.
"""
import logging
import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)

DEFAULT_TILE_SIZE = 448
DEFAULT_OVERLAP = 64


def _make_tile_grid(H, W, tile_size=DEFAULT_TILE_SIZE,
                    overlap=DEFAULT_OVERLAP):
    """Generate (r0, r1, c0, c1) bboxes for tiling."""
    step = tile_size - overlap
    bboxes = []
    for r0 in range(0, H, step):
        r1 = min(r0 + tile_size, H)
        if r1 - r0 < tile_size // 2:
            r0 = max(0, r1 - tile_size)
        for c0 in range(0, W, step):
            c1 = min(c0 + tile_size, W)
            if c1 - c0 < tile_size // 2:
                c0 = max(0, c1 - tile_size)
            bboxes.append((r0, r1, c0, c1))
    # Deduplicate
    return list(set(bboxes))


def detect_dic_tiled(frames, model_path=None, tile_size=DEFAULT_TILE_SIZE,
                     overlap=DEFAULT_OVERLAP, min_area_px=200,
                     flow_threshold=0.0, cellprob_threshold=0.0,
                     progress_fn=None):
    """Tiled cellpose detection for DIC recordings.

    Tiles each frame into patches matching the VAMPIRE training
    crop size, runs cellpose_dic per tile, merges by union, then
    keeps all significant connected components (for multi-cell).

    Returns: (N, H, W) int32 label stack (0=bg, 1..K=cells).
    """
    import os
    from core.detection import detect_cellpose

    if model_path is None:
        v3 = "data/models/cellpose_dic_v3"
        v2 = "data/models/cellpose_dic_v2"
        v1 = "data/models/cellpose_dic"
        if os.path.exists(v3):
            model_path = v3
        elif os.path.exists(v2):
            model_path = v2
        else:
            model_path = v1

    n, H, W = frames.shape
    bboxes = _make_tile_grid(H, W, tile_size, overlap)
    log.info("DIC tiled: %dx%d frames, %d tiles of %dpx (overlap %d)",
             H, W, len(bboxes), tile_size, overlap)

    labels_out = np.zeros(frames.shape, dtype=np.int32)

    for i in range(n):
        if progress_fn and (i % 5 == 0 or i == n - 1):
            progress_fn("Tiled DIC %d/%d" % (i + 1, n),
                        int(100 * i / max(n - 1, 1)))

        # Detect per tile
        frame_mask = np.zeros((H, W), dtype=bool)
        for (r0, r1, c0, c1) in bboxes:
            tile = frames[i, r0:r1, c0:c1][np.newaxis]
            tile_mask = detect_cellpose(
                tile, gpu=True, model_path=model_path,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold)
            frame_mask[r0:r1, c0:c1] |= tile_mask[0]

        # Label connected components, filter by area
        if frame_mask.any():
            lbl, n_cc = ndimage.label(frame_mask)
            new_id = 0
            for cc in range(1, n_cc + 1):
                if (lbl == cc).sum() >= min_area_px:
                    new_id += 1
                    labels_out[i][lbl == cc] = new_id

    total_cells = sum(int(labels_out[i].max()) for i in range(n))
    log.info("DIC tiled: %d total cell detections across %d frames",
             total_cells, n)
    return labels_out


def detect_dic_tiled_single(frames, model_path=None,
                            tile_size=DEFAULT_TILE_SIZE,
                            overlap=DEFAULT_OVERLAP,
                            min_area_px=200,
                            progress_fn=None):
    """Single-cell variant: returns (N, H, W) bool (largest cell only)."""
    labels = detect_dic_tiled(
        frames, model_path=model_path,
        tile_size=tile_size, overlap=overlap,
        min_area_px=min_area_px, progress_fn=progress_fn)
    masks = np.zeros(frames.shape, dtype=bool)
    for i in range(len(frames)):
        if labels[i].max() > 0:
            # Keep largest
            best_id = 0
            best_area = 0
            for cid in range(1, int(labels[i].max()) + 1):
                area = (labels[i] == cid).sum()
                if area > best_area:
                    best_area = area
                    best_id = cid
            masks[i] = labels[i] == best_id
    return masks
