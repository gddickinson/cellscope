"""4-rotation TTA helpers for the SAM2 mask-editor tools.

Split out from gui/mask_editor_sam2_point.py to keep that file
under the project's 500-line cap. Pure inference orchestration —
no Qt, no model loading. The predictor object is passed in by the
caller (predict_at_point / predict_at_box).
"""
import numpy as np


def _rot_coords_ccw(x, y, k, src_w, src_h):
    """Rotate a point (x, y) by k × 90° counter-clockwise.

    src_w / src_h are the SOURCE image dimensions (pre-rotation).
    Returns (x', y') in the rotated image.
    """
    k = k % 4
    if k == 0:
        return float(x), float(y)
    if k == 1:
        return float(y), float(src_w - 1 - x)
    if k == 2:
        return float(src_w - 1 - x), float(src_h - 1 - y)
    return float(src_h - 1 - y), float(x)


def _rot_box_ccw(box, k, src_w, src_h):
    """Rotate axis-aligned box [x0,y0,x1,y1] by k × 90° CCW.

    Result is still axis-aligned (the box's two corners may swap
    positions; we re-normalise so x0<x1, y0<y1).
    """
    x0, y0, x1, y1 = (
        float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    p1 = _rot_coords_ccw(x0, y0, k, src_w, src_h)
    p2 = _rot_coords_ccw(x1, y1, k, src_w, src_h)
    return np.array([
        min(p1[0], p2[0]), min(p1[1], p2[1]),
        max(p1[0], p2[0]), max(p1[1], p2[1]),
    ], dtype=np.float32)


def _predict_with_tta(predictor, crop_rgb, point_coords, point_labels,
                      box, allowed, use_tta):
    """Run SAM2 once or with 4-rotation TTA.

    Returns (best_mask, score) where best_mask is a 2-D bool array
    in original (un-rotated) crop coordinates and score is the SAM2
    confidence (per-rotation average when TTA is on).

    When `allowed` is given (the box+margin region in crop coords),
    candidate selection per rotation is biased toward fits-the-box
    via score × inside_frac. Otherwise (point mode) the highest-
    confidence candidate is taken.
    """
    crop_h, crop_w = crop_rgb.shape[:2]

    def _pick(masks, scores, allowed_in_rot):
        if allowed_in_rot is None:
            return int(np.argmax(scores))
        best_idx = 0
        best_combined = -1.0
        for i in range(len(masks)):
            m_i = masks[i].astype(bool)
            tot = int(m_i.sum())
            if tot == 0:
                continue
            inside_frac = float((m_i & allowed_in_rot).sum()) / tot
            combined = float(scores[i]) * inside_frac
            if combined > best_combined:
                best_combined = combined
                best_idx = i
        return best_idx

    if not use_tta:
        predictor.set_image(crop_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        best_idx = _pick(masks, scores, allowed)
        return masks[best_idx].astype(bool), float(scores[best_idx])

    # 4-rotation TTA. Per rotation: rotate image + prompts + allowed
    # region, predict, pick best candidate, rotate mask back to k=0.
    # Final mask = (sum of 4 binary masks) >= 2 — i.e. majority vote.
    accum = None
    score_sum = 0.0
    for k in range(4):
        rot_img = np.ascontiguousarray(np.rot90(crop_rgb, k))
        rot_pts = (np.array(
            [_rot_coords_ccw(p[0], p[1], k, crop_w, crop_h)
             for p in point_coords], dtype=np.float32)
            if point_coords is not None else None)
        rot_box = (_rot_box_ccw(box, k, crop_w, crop_h)
                   if box is not None else None)
        rot_allowed = (np.ascontiguousarray(np.rot90(allowed, k))
                       if allowed is not None else None)
        predictor.set_image(rot_img)
        masks, scores, _ = predictor.predict(
            point_coords=rot_pts,
            point_labels=point_labels,
            box=rot_box,
            multimask_output=True,
        )
        best_idx = _pick(masks, scores, rot_allowed)
        mask_back = np.ascontiguousarray(
            np.rot90(masks[best_idx].astype(bool), -k))
        accum = (mask_back.astype(np.float32) if accum is None
                 else accum + mask_back.astype(np.float32))
        score_sum += float(scores[best_idx])
    final = accum >= 2.0  # ≥ 2 of 4 rotations agree
    return final, score_sum / 4.0
