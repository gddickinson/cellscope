"""Multi-cell label helpers for the mask editor.

Provides per-cell colored overlay rendering and label utilities
for the mask editor's int32 label mode.
"""
import numpy as np
import cv2

CELL_COLORS = [
    (0, 255, 0),       # 1: green (default / single-cell compat)
    (255, 80, 80),     # 2: red
    (80, 140, 255),    # 3: blue
    (255, 220, 50),    # 4: yellow
    (255, 50, 220),    # 5: magenta
    (50, 255, 255),    # 6: cyan
    (255, 160, 50),    # 7: orange
    (180, 100, 255),   # 8: purple
    (100, 255, 100),   # 9: lime
]


def cell_color(cell_id):
    """Return RGB tuple for a 1-indexed cell ID."""
    if cell_id <= 0:
        return (128, 128, 128)
    return CELL_COLORS[(cell_id - 1) % len(CELL_COLORS)]


def render_label_overlay(frame, labels, opacity=0.4,
                         active_cell=None, polygon_preview=None):
    """Render a frame with colored per-cell label overlay.

    Args:
        frame: (H, W) uint8 grayscale
        labels: (H, W) int32, 0=bg, 1,2,...=cell IDs
        opacity: overlay blend factor
        active_cell: highlight this cell's contour thicker
        polygon_preview: list of (x, y) points to draw as preview

    Returns:
        (H, W, 3) uint8 RGB image
    """
    h, w = frame.shape
    rgb = np.stack([frame, frame, frame], axis=-1).astype(np.float32)

    for cell_id in range(1, int(labels.max()) + 1):
        mask = labels == cell_id
        if not mask.any():
            continue
        color = np.array(cell_color(cell_id), dtype=np.float32)
        rgb[mask] = rgb[mask] * (1 - opacity) + color * opacity

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Draw contours per cell
    for cell_id in range(1, int(labels.max()) + 1):
        mask = labels == cell_id
        if not mask.any():
            continue
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)
        color = cell_color(cell_id)
        thickness = 2 if cell_id == active_cell else 1
        cv2.drawContours(rgb, contours, -1, color, thickness)

    if polygon_preview and len(polygon_preview) >= 2:
        pts = np.array(polygon_preview, dtype=np.int32)
        cv2.polylines(rgb, [pts], False, (255, 255, 255), 1)

    return rgb


def labels_to_bool(labels):
    """Convert int32 labels to bool (any cell = True)."""
    return labels > 0


def bool_to_labels(masks):
    """Convert bool masks to int32 labels (all pixels = cell 1)."""
    return masks.astype(np.int32)


def next_cell_id(labels):
    """Return the next unused cell ID in a label frame."""
    return int(labels.max()) + 1 if labels.max() > 0 else 1
