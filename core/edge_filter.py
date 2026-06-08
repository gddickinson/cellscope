"""Detect cell masks truncated by the image border.

A cell whose mask reaches the edge of the field of view is *cut off* — we
only see part of it. Its outline, and therefore every shape metric
(area, circularity, solidity, …) and the rounded/spread state derived
from them, is unreliable. Such a cell-frame must be excluded from SHAPE
and STATE analysis, while still being counted (the cell exists) and kept
in its track (the centroid still anchors identity).

`mask_touches_edge` is the single primitive. It is deliberately tiny and
stateless so it can be unit-tested and called from any chokepoint that
holds a FULL-FRAME mask (`core.cell_state.shape_metrics_for_mask`, the
labelling samplers, the annotation GUI). It must NOT be called on a mask
that has already been cropped to its bounding box — a bbox crop touches
its own border by construction, so every cell would read as edge.
"""
from __future__ import annotations

import numpy as np


def mask_touches_edge(mask_bool, margin: int = 0) -> bool:
    """True if `mask_bool` reaches within `margin` px of any image border.

    `mask_bool` is a 2-D full-frame boolean mask for ONE cell-frame; its
    shape IS the frame size (this is why the input must be full-frame, not
    a bbox crop). `margin=0` means literal border contact (a pixel on
    row 0, col 0, the last row, or the last col) — the default. A larger
    margin also flags cells whose membrane likely continues just off-frame.

    Empty masks return False (nothing to truncate).
    """
    if mask_bool is None or not np.any(mask_bool):
        return False
    h, w = mask_bool.shape[:2]
    m = int(margin)
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    r0, r1 = np.argmax(rows), h - 1 - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), w - 1 - np.argmax(cols[::-1])
    return bool(r0 <= m or c0 <= m or r1 >= h - 1 - m or c1 >= w - 1 - m)


def bbox_touches_edge(r0, r1, c0, c1, h, w, margin: int = 0) -> bool:
    """Edge test from a precomputed full-frame bounding box.

    For callers that already extracted (r0, r1, c0, c1) from the
    full-frame mask before cropping to the bbox (e.g. the labelling
    feature extractors) — pass the frame size (h, w) explicitly so the
    test is done in full-frame coordinates, never against the crop.
    """
    m = int(margin)
    return bool(r0 <= m or c0 <= m or r1 >= h - 1 - m or c1 >= w - 1 - m)
