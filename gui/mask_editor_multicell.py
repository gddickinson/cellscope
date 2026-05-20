"""Multi-cell label helpers for the mask editor.

Provides per-cell colored overlay rendering and label utilities
for the mask editor's int32 label mode.
"""
import numpy as np
import cv2

CELL_COLORS = [
    (0, 255, 0),       # 1:  green (default / single-cell compat)
    (255, 80, 80),     # 2:  red
    (80, 140, 255),    # 3:  blue
    (255, 220, 50),    # 4:  yellow
    (255, 50, 220),    # 5:  magenta
    (50, 255, 255),    # 6:  cyan
    (255, 160, 50),    # 7:  orange
    (180, 100, 255),   # 8:  purple
    (100, 255, 100),   # 9:  lime
    (255, 130, 180),   # 10: pink
    (0, 200, 150),     # 11: teal
    (200, 130, 80),    # 12: brown
    (180, 200, 60),    # 13: olive
    (80, 220, 220),    # 14: turquoise
    (255, 100, 100),   # 15: salmon
    (130, 80, 200),    # 16: indigo
    (240, 240, 110),   # 17: pale yellow
    (170, 250, 90),    # 18: chartreuse
    (240, 130, 200),   # 19: rose
    (90, 180, 240),    # 20: sky blue
    (255, 0, 128),     # 21: fuchsia
    (60, 140, 60),     # 22: forest green
    (30, 60, 180),     # 23: navy
    (255, 200, 0),     # 24: gold
    (255, 127, 80),    # 25: coral
    (160, 210, 255),   # 26: periwinkle
    (255, 180, 90),    # 27: peach
    (140, 40, 40),     # 28: maroon
    (140, 255, 200),   # 29: mint
    (210, 180, 140),   # 30: tan
]


def cell_color(cell_id):
    """Return RGB tuple for a 1-indexed cell ID."""
    if cell_id <= 0:
        return (128, 128, 128)
    return CELL_COLORS[(cell_id - 1) % len(CELL_COLORS)]


def render_label_overlay(frame, labels, opacity=0.4,
                         active_cell=None, polygon_preview=None,
                         show_ids=False, draw_contours=True):
    """Render a frame with colored per-cell label overlay.

    Args:
        draw_contours: when False, skip per-cell contour traces (and
            cell-ID text). Used mid-stroke so the mouse stays
            responsive on frames with many labels; release the mouse
            and the next redraw brings the contours back.
        frame: (H, W) uint8 grayscale
        labels: (H, W) int32, 0=bg, 1,2,...=cell IDs
        opacity: overlay blend factor
        active_cell: highlight this cell's contour thicker
        polygon_preview: list of (x, y) points to draw as preview
        show_ids: when True, draw the cell ID number at each cell's
            centroid in white text on a translucent black background

    Returns:
        (H, W, 3) uint8 RGB image
    """
    h, w = frame.shape
    rgb = np.stack([frame, frame, frame], axis=-1).astype(np.float32)

    # Single-pass color-LUT blend. Replaces an O(N_cells) loop of
    # full-frame `labels == cell_id` boolean comparisons + per-cell
    # blends with a constant-cost lookup + masked blend, so renderer
    # cost stays flat as the user adds more labelled cells.
    max_id = int(labels.max()) if labels.size else 0
    if max_id > 0:
        lut = np.zeros((max_id + 1, 3), dtype=np.float32)
        present = np.unique(labels)
        present = present[present > 0]
        for cid in present:
            lut[cid] = cell_color(int(cid))
        is_cell = labels > 0
        if is_cell.any():
            colors_per_pixel = lut[labels]
            rgb[is_cell] = (rgb[is_cell] * (1 - opacity)
                            + colors_per_pixel[is_cell] * opacity)

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Draw contours only for cell IDs actually present in the frame.
    # Skipping deleted IDs avoids wasted full-frame `==` scans.
    if max_id > 0 and draw_contours:
        for cid in present:
            cell_id = int(cid)
            mask = labels == cell_id
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE)
            color = cell_color(cell_id)
            thickness = 2 if cell_id == active_cell else 1
            cv2.drawContours(rgb, contours, -1, color, thickness)

    if polygon_preview and len(polygon_preview) >= 2:
        pts = np.array(polygon_preview, dtype=np.int32)
        cv2.polylines(rgb, [pts], False, (255, 255, 255), 1)

    if show_ids and max_id > 0 and draw_contours:
        # Pixel size scales with image — keep text readable on both
        # small crops (256²) and full frames (2048²).
        font_scale = max(0.5, min(2.0, min(h, w) / 600.0))
        thickness = max(1, int(round(font_scale * 1.5)))
        for cid in present:
            cell_id = int(cid)
            mask = labels == cell_id
            ys, xs = np.where(mask)
            cy, cx = int(ys.mean()), int(xs.mean())
            label = str(cell_id)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x0 = cx - tw // 2
            y0 = cy + th // 2
            # Black rounded-rect background for legibility
            pad = max(2, int(round(font_scale * 2)))
            cv2.rectangle(
                rgb,
                (x0 - pad, y0 - th - pad),
                (x0 + tw + pad, y0 + pad),
                (0, 0, 0), -1)
            cv2.putText(
                rgb, label, (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

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
