"""Per-cell mask cleanup: fill holes + drop disconnected specks.

cpsam masks occasionally have (a) small enclosed holes and (b) tiny
rogue components detached from the cell body. Both corrupt shape
descriptors — a hole or a speck lengthens the perimeter and shrinks the
area/convex ratio, dragging circularity and solidity DOWN — which is
exactly the signal the rounded/spread classifier keys on. Cleaning here
(non-destructively, at shape-measurement time) fixes the metrics without
touching the stored masks.

`clean_cell_mask(mask)` returns a cleaned boolean mask:
  1. keep only connected components ≥ `min_component_frac` of the largest
     (removes rogue specks; a single cell is one blob),
  2. fill enclosed holes.
"""
from __future__ import annotations

import numpy as np


def clean_cell_mask(mask, min_component_frac=0.10):
    """Fill holes + remove small disconnected components for ONE cell.

    Args:
        mask: 2-D bool/int mask of a single cell.
        min_component_frac: drop connected components smaller than this
            fraction of the largest component's area (rogue-speck removal).
    Returns a 2-D bool mask. No-op on empty masks.
    """
    from scipy import ndimage as ndi
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    lbl, n = ndi.label(m)
    if n > 1:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0                      # background
        keep = np.where(sizes >= min_component_frac * sizes.max())[0]
        m = np.isin(lbl, keep)
    m = ndi.binary_fill_holes(m)
    return m
