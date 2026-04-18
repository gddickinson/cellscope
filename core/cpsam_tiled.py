"""Tiled cpsam detection for large FOV recordings.

Splits the image into NxN tiles with overlap, runs cpsam per tile,
and merges results. Best for dense multi-cell fields (5+ cells/frame).
NOT recommended for sparse recordings — the ViT needs context.
"""
import numpy as np
import logging

log = logging.getLogger(__name__)


def detect_cpsam_tiled(frames, n_tiles=(2, 2), overlap=64,
                        min_area=50, augment=False, progress_fn=None):
    """Run cpsam on tiles and merge results.

    Args:
        frames: (N, H, W) uint8
        n_tiles: (rows, cols) tile grid
        overlap: pixels of overlap between adjacent tiles
        min_area: minimum cell area in pixels
        augment: use TTA (augment=True per tile)
        progress_fn: callback(msg, pct)

    Returns:
        (N, H, W) int32 label stack (per-frame instance IDs)
    """
    from cellpose import models
    m = models.CellposeModel(gpu=True)

    N, H, W = frames.shape
    ty, tx = n_tiles
    tile_h = H // ty
    tile_w = W // tx
    result = np.zeros((N, H, W), dtype=np.int32)

    for i in range(N):
        if progress_fn and (i % 5 == 0 or i == N - 1):
            progress_fn(f"Tiled cpsam {i+1}/{N} ({ty}x{tx})",
                        int(100 * i / max(N - 1, 1)))

        frame = frames[i]
        all_masks = np.zeros((H, W), dtype=np.int32)
        next_label = 1

        for row in range(ty):
            for col in range(tx):
                r0 = max(0, row * tile_h - overlap)
                r1 = min(H, (row + 1) * tile_h + overlap)
                c0 = max(0, col * tile_w - overlap)
                c1 = min(W, (col + 1) * tile_w + overlap)

                tile = frame[r0:r1, c0:c1]
                tile_masks, _, _ = m.eval(tile, augment=augment)

                # Only place cells within the non-overlap inner region
                inner_r0 = row * tile_h
                inner_r1 = min(H, (row + 1) * tile_h)
                inner_c0 = col * tile_w
                inner_c1 = min(W, (col + 1) * tile_w)

                tr0 = inner_r0 - r0
                tr1 = tr0 + (inner_r1 - inner_r0)
                tc0 = inner_c0 - c0
                tc1 = tc0 + (inner_c1 - inner_c0)

                inner_tile = tile_masks[tr0:tr1, tc0:tc1]
                for lab in np.unique(inner_tile):
                    if lab == 0:
                        continue
                    cell = inner_tile == lab
                    if cell.sum() < min_area:
                        continue
                    target = all_masks[inner_r0:inner_r1,
                                       inner_c0:inner_c1]
                    free = target == 0
                    target[free & cell] = next_label
                    next_label += 1

        result[i] = all_masks

    return result
