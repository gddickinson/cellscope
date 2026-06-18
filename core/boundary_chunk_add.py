"""Chunk-add boundary refinement for single-cell DIC masks.

Validated on IC293 (2026-06-17): recover LARGE solid missing cell lobes
(protrusions cpsam / cpsam_dic under-segmented) WITHOUT disturbing the
existing boundary. The method is self-limiting — already-clean masks
change +0%, and on low-SNR crops (CNR ≈ 0.3) it adds essentially nothing
(+0–1%) because the RF cell-probability never clears the gate there. See
`ic293_analysis/CURATION_DECISIONS.md` for the full exploration that
selected this over isoline/union/threshold-sweep/SAM2-seeded refinement.

Method (per frame, per cell):

  1. candidate = (RF P(cell) > threshold) AND background (not in any mask)
  2. morphological OPENING (disk `open_radius`) -> keep only SOLID regions,
     dropping the thin noisy peripheral band regardless of threshold.
  3. connected components of the candidate; keep a component iff it is
       - large    (>= `min_chunk_px`), AND
       - solid    (survives the opening), AND
       - adjacent (touches the cell's `bridge_px` dilation).
  4. graft the kept chunks onto the mask, fill interior holes.

The OPENING (not the threshold) is what prevents over-inflation: a low
threshold finds more area but the thin band is still removed, so only
genuine lobes survive. P>0.65 (vs 0.85) catches moderate-confidence lobes
(e.g. IC293 Pos11_cell2) while clean cells and low-SNR crops stay ~+0%.

This module is INFERENCE-only — it consumes the trained RF in
`core/boundary_rf.py`. It never edits masks in place; callers decide
what to persist.
"""
from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import (
    binary_fill_holes, binary_dilation, binary_opening, label as cc_label)
from skimage.morphology import disk


@dataclass
class ChunkAddParams:
    """Validated IC293 defaults (px at the recording's native resolution)."""
    threshold: float = 0.65       # RF P(cell) gate
    open_radius: int = 4          # disk radius for the solid-lobe opening
    min_chunk_px: int = 250       # minimum lobe size to graft
    bridge_px: int = 8            # max gap to the cell to count a chunk adjacent
    max_contact_frac: float = 1.0  # LOCALITY gate: reject a chunk that touches
    #   more than this fraction of the cell's boundary. A genuine missing lobe
    #   attaches over a SMALL base arc (contact_frac ≈ 0.02–0.15); a perimeter-
    #   hugging RIND / over-expansion touches a long arc (≈ 0.5–1.0). 1.0 = off
    #   (legacy). IC293-fit value: 0.35 (cleanly separates lobe from rind).
    contact_dilate_px: int = 3    # tolerance when measuring chunk↔cell contact


DEFAULT_CHUNK_ADD = ChunkAddParams()


def chunk_add_mask(prob, mask, occupied=None, params=DEFAULT_CHUNK_ADD):
    """Graft large solid RF lobes onto a single binary cell mask.

    prob     : (H, W) float — RF cell-probability for the frame.
    mask     : (H, W) bool  — current cell footprint.
    occupied : (H, W) bool  — pixels already claimed (other cells / earlier
               grafts) and excluded from the candidate pool so a chunk is
               never assigned to two cells. None -> only `mask` is occupied.

    Returns (new_mask, added_px). `new_mask` always contains `mask`
    (chunk-add only ADDS), minus any `occupied & ~mask` pixels a hole-fill
    might otherwise have swallowed.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask, 0
    if occupied is None:
        occupied = mask
    other = occupied & ~mask                       # pixels owned by other cells
    cand = (prob > params.threshold) & ~occupied
    if not cand.any():
        return mask, 0
    thick = binary_opening(cand, disk(params.open_radius))
    lab, n = cc_label(cand)
    near = binary_dilation(mask, iterations=params.bridge_px)
    # LOCALITY gate setup: the cell's boundary rim + its pixel count, used to
    # measure how much of the cell each candidate chunk wraps (rind vs lobe).
    gate_contact = params.max_contact_frac < 1.0
    if gate_contact:
        from scipy.ndimage import binary_erosion
        rim = mask & ~binary_erosion(mask, iterations=1)
        rim_n = max(int(rim.sum()), 1)
    keep = np.zeros_like(mask)
    for c in range(1, n + 1):
        comp = lab == c
        if not (comp.sum() >= params.min_chunk_px
                and (comp & thick).any() and (comp & near).any()):
            continue
        if gate_contact:
            contact = binary_dilation(comp, iterations=params.contact_dilate_px) & rim
            if contact.sum() / rim_n > params.max_contact_frac:
                continue                            # perimeter rind / over-expansion
        keep |= comp
    if not keep.any():
        return mask, 0
    new = binary_fill_holes(mask | keep)
    new &= ~other                                  # never swallow another cell
    return new, int((new & ~mask).sum())


def refine_label_stack(labels, frames, rf_model, config,
                       params=DEFAULT_CHUNK_ADD, prob_fn=None, probs=None,
                       progress=None):
    """Apply chunk-add to every cell in a (T, H, W) int label stack.

    For each frame the RF probability is computed once; each label's
    footprint is refined against a background-only candidate pool. Larger
    cells claim contested chunks first and claimed pixels are removed from
    the pool, so grafts never overlap and label identity is preserved.

    probs    : optional precomputed per-frame probability, indexed by frame t
               (list/array length T; entries may be None for empty frames).
               Use this to share one RF pass across several param settings —
               do NOT cache by `id(frames[t])` (ephemeral views recycle ids).
    prob_fn  : optional override `img -> (H,W) prob` (used only when `probs`
               is None). progress : optional callable(t, T).

    Returns (new_labels, stats).
    """
    labels = np.asarray(labels)
    if prob_fn is None:
        from core.boundary_rf import predict_cell_probability

        def prob_fn(img):
            return predict_cell_probability(img, rf_model, config)

    out = np.zeros_like(labels)
    T = labels.shape[0]
    frame_added = []
    n_frames_changed = 0
    area_before = area_after = 0
    max_frame_pct, max_frame = 0.0, -1
    for t in range(T):
        if progress is not None:
            progress(t, T)
        lab_t = labels[t]
        ids = [int(v) for v in np.unique(lab_t) if v > 0]
        if not ids:
            continue
        prob = probs[t] if probs is not None else prob_fn(frames[t])
        occupied = lab_t > 0                # never steal pixels from any cell
        ids.sort(key=lambda v: int((lab_t == v).sum()), reverse=True)
        added_t = fb_t = fa_t = 0
        for v in ids:
            M = lab_t == v
            new, added = chunk_add_mask(prob, M, occupied=occupied, params=params)
            out[t][new] = v
            occupied = occupied | new
            added_t += added
            fb_t += int(M.sum())
            fa_t += int(new.sum())
        area_before += fb_t
        area_after += fa_t
        frame_added.append(int(added_t))
        if added_t > 0:
            n_frames_changed += 1
            pct = (fa_t - fb_t) / max(fb_t, 1) * 100.0
            if pct > max_frame_pct:
                max_frame_pct, max_frame = pct, t

    denom = max(area_before, 1)
    stats = {
        "n_frames": int(T),
        "n_frames_present": int(sum(1 for v in frame_added)),
        "n_frames_changed": int(n_frames_changed),
        "area_before_px": int(area_before),
        "area_after_px": int(area_after),
        "area_delta_px": int(area_after - area_before),
        "area_delta_pct": float((area_after - area_before) / denom * 100.0),
        "max_frame_delta_pct": float(max_frame_pct),
        "max_frame": int(max_frame),
        "frame_added_px": frame_added,
        "params": asdict(params),
    }
    return out, stats
