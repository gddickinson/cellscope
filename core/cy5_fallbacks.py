"""Cy5-driven fail-safes for the multi-channel pipeline.

Three tiers of recovery for cells that DIC-led detection misses:

  Tier 2 — recovery via DIC crop
    `find_cy5_missed_regions`: locate Cy5+ regions not covered by
       any DIC mask
    `recover_missed_cells_via_dic_crop`: for each candidate, crop
       DIC + re-run cpsam (TTA cheap on small crop) and place the
       result back

  Tier 3 — gap fill for tracked cells
    Triggered when a track has a gap at frame N AND there is bright
    Cy5 signal at the interpolated centroid → fill with a Cy5-derived
    mask. Three fill strategies:
      * cy5_threshold: bright Cy5 region nearest to centroid
      * cy5_cleanup:   threshold + hole-fill + light morphological
                       smoothing
      * cy5_cpsam:     run cpsam on the Cy5 crop (cleanest edges)

The temporal anchor (cell present in flanking frames) prevents
hallucinating cells from random Cy5 artefacts.
"""
import numpy as np
from scipy.ndimage import binary_dilation


def find_cy5_missed_regions(dic_label_frame, cy5_uint8,
                             k_mad=4.0, min_area_px=50,
                             dic_dilate_px=10):
    """Find Cy5+ regions not covered by any DIC mask.

    A region is "Cy5+" if its pixels are k * MAD above the median.
    A region is "uncovered" if no DIC mask within `dic_dilate_px`
    pixels overlaps it. Returns list of dicts:
      [{'centroid': (cy, cx), 'bbox': (r0, r1, c0, c1),
        'mask': bool array of the local Cy5 region}, ...]

    Use as input to `recover_missed_cells_via_dic_crop`.
    """
    from scipy.ndimage import label as cc_label
    cy5_med = np.median(cy5_uint8)
    cy5_mad = max(1.0, float(np.median(
        np.abs(cy5_uint8.astype(np.float32) - cy5_med))))
    bright = cy5_uint8.astype(np.float32) > cy5_med + k_mad * cy5_mad

    # Subtract regions already covered by DIC masks (dilated for safety)
    dic_present = dic_label_frame > 0
    if dic_present.any():
        dic_coverage = binary_dilation(dic_present,
                                        iterations=dic_dilate_px)
    else:
        dic_coverage = np.zeros_like(dic_present)
    candidate_mask = bright & ~dic_coverage

    cc, n = cc_label(candidate_mask)
    out = []
    for cid in range(1, n + 1):
        region = cc == cid
        if region.sum() < min_area_px:
            continue
        ys, xs = np.where(region)
        out.append({
            "centroid": (float(ys.mean()), float(xs.mean())),
            "bbox": (int(ys.min()), int(ys.max()) + 1,
                     int(xs.min()), int(xs.max()) + 1),
            "mask": region,
            "cy5_max": float(cy5_uint8[region].max()),
            "area_px": int(region.sum()),
        })
    return out


def recover_missed_cells_via_dic_crop(dic_uint8, dic_label_frame,
                                       cy5_uint8, cpsam_model,
                                       pad_px=64, use_tta=False,
                                       min_recovered_area=100,
                                       k_mad=4.0):
    """For each Cy5+ region not covered by DIC masks, crop and re-run
    cpsam on the DIC crop to recover the missed cell.

    Args:
      dic_uint8: (H, W) DIC frame
      dic_label_frame: (H, W) int32 — current cpsam masks
      cy5_uint8: (H, W) Cy5/actin frame
      cpsam_model: a loaded `cellpose.models.CellposeModel`
      pad_px: padding around the candidate bbox before cropping
      use_tta: enable cpsam test-time augmentation on the small crop
        (cheap because crop is small; ~5s vs ~75s for full frame)
      min_recovered_area: drop recovered masks smaller than this
      k_mad: Cy5 brightness threshold = median + k * MAD

    Returns (updated_label_frame, n_recovered, info_list).
    """
    H, W = dic_uint8.shape
    candidates = find_cy5_missed_regions(
        dic_label_frame, cy5_uint8,
        k_mad=k_mad, min_area_px=min_recovered_area // 2)
    if not candidates:
        return dic_label_frame, 0, []

    out = dic_label_frame.copy()
    next_id = int(out.max()) + 1
    n_recovered = 0
    info = []

    for cand in candidates:
        r0, r1, c0, c1 = cand["bbox"]
        # Pad for context
        r0p = max(0, r0 - pad_px)
        r1p = min(H, r1 + pad_px)
        c0p = max(0, c0 - pad_px)
        c1p = min(W, c1 + pad_px)
        if (r1p - r0p) < 64 or (c1p - c0p) < 64:
            continue  # too small for cpsam

        crop_dic = dic_uint8[r0p:r1p, c0p:c1p]
        crop_cand = cand["mask"][r0p:r1p, c0p:c1p]
        try:
            crop_masks, _, _ = cpsam_model.eval(crop_dic,
                                                 augment=use_tta)
        except Exception:
            continue
        crop_masks = crop_masks.astype(np.int32)
        if crop_masks.max() == 0:
            continue

        # Pick the cpsam mask in the crop that overlaps the Cy5
        # candidate region most
        best_lab, best_ovl = 0, 0
        for lab in range(1, int(crop_masks.max()) + 1):
            ovl = int(np.logical_and(crop_masks == lab,
                                      crop_cand).sum())
            if ovl > best_ovl:
                best_ovl, best_lab = ovl, lab
        if best_lab == 0:
            continue

        recovered_local = (crop_masks == best_lab)
        recovered_full = np.zeros_like(out, dtype=bool)
        recovered_full[r0p:r1p, c0p:c1p] = recovered_local
        # Don't overwrite existing cells — only add to background pixels
        recovered_full &= (out == 0)
        if recovered_full.sum() < min_recovered_area:
            continue

        out[recovered_full] = next_id
        info.append({
            "centroid": cand["centroid"],
            "bbox": cand["bbox"],
            "recovered_id": int(next_id),
            "recovered_area_px": int(recovered_full.sum()),
            "cy5_max": cand["cy5_max"],
        })
        next_id += 1
        n_recovered += 1

    return out, n_recovered, info


# ────────────────────────────────────────────────────────────────────
# Tier-3 fail-safe: temporal Cy5 gap fill
#
# When a track has a gap at frame N (cell was tracked on N-1 and
# N+1 but lost on N) AND there is bright Cy5 signal at the
# interpolated centroid, the cell IS there — even if cpsam(DIC)
# couldn't find it on the full frame OR on a crop. Fill with a
# Cy5-derived mask. Requires the temporal anchor so we don't
# hallucinate cells from stray Cy5 artefacts.
# ────────────────────────────────────────────────────────────────────


def _interpolate_centroid(track_stack, frame_idx, max_lookback=5):
    """Linear interpolation between last present and next present
    frames. Returns (cy, cx) or None if no anchors found."""
    n = len(track_stack)
    # Last present before frame_idx
    prev_fi, prev_cent = None, None
    for fi in range(frame_idx - 1, max(-1, frame_idx - max_lookback - 1),
                     -1):
        if fi < 0:
            break
        if track_stack[fi].any():
            ys, xs = np.where(track_stack[fi])
            prev_fi = fi
            prev_cent = (float(ys.mean()), float(xs.mean()))
            break
    next_fi, next_cent = None, None
    for fi in range(frame_idx + 1,
                     min(n, frame_idx + max_lookback + 1)):
        if fi >= n:
            break
        if track_stack[fi].any():
            ys, xs = np.where(track_stack[fi])
            next_fi = fi
            next_cent = (float(ys.mean()), float(xs.mean()))
            break
    if prev_cent and next_cent:
        t = (frame_idx - prev_fi) / (next_fi - prev_fi)
        return (prev_cent[0] + t * (next_cent[0] - prev_cent[0]),
                prev_cent[1] + t * (next_cent[1] - prev_cent[1]))
    return prev_cent or next_cent  # one-sided is acceptable


def _cy5_fill_threshold(cy5_uint8, centroid, search_radius=80,
                         k_mad=3.0, min_area=80):
    """Threshold-only fill: bright Cy5 region nearest to centroid."""
    from scipy.ndimage import label as cc_label
    H, W = cy5_uint8.shape
    cy, cx = int(centroid[0]), int(centroid[1])
    r0, r1 = max(0, cy - search_radius), min(H, cy + search_radius)
    c0, c1 = max(0, cx - search_radius), min(W, cx + search_radius)
    crop = cy5_uint8[r0:r1, c0:c1]
    med = float(np.median(crop))
    mad = max(1.0, float(np.median(np.abs(crop.astype(np.float32) - med))))
    bright = crop > med + k_mad * mad
    cc, n = cc_label(bright)
    if n == 0:
        return None
    # Pick the connected component nearest to the centroid
    best_dist, best_id = float("inf"), 0
    for cid in range(1, n + 1):
        m = cc == cid
        if m.sum() < min_area:
            continue
        ys, xs = np.where(m)
        local_cy, local_cx = ys.mean(), xs.mean()
        global_cy = local_cy + r0
        global_cx = local_cx + c0
        d = (global_cy - cy) ** 2 + (global_cx - cx) ** 2
        if d < best_dist:
            best_dist = d
            best_id = cid
    if best_id == 0:
        return None
    region = cc == best_id
    out = np.zeros((H, W), dtype=bool)
    out[r0:r1, c0:c1] = region
    return out


def _cy5_fill_cleanup(cy5_uint8, centroid, **kwargs):
    """Threshold fill + hole fill + light morphological smoothing."""
    from scipy.ndimage import binary_fill_holes, binary_opening
    base = _cy5_fill_threshold(cy5_uint8, centroid, **kwargs)
    if base is None:
        return None
    base = binary_fill_holes(base)
    base = binary_opening(base, iterations=2)
    return base if base.any() else None


def _cy5_fill_cpsam(cy5_uint8, dic_uint8, centroid, cpsam_model,
                     search_radius=80, pad_px=32, use_tta=True):
    """Run cpsam on a Cy5 crop near the centroid, pick the mask that
    overlaps the centroid most closely. cpsam handles fluorescence
    well, so this gives cleaner edges than thresholding."""
    H, W = cy5_uint8.shape
    cy, cx = int(centroid[0]), int(centroid[1])
    r0 = max(0, cy - search_radius - pad_px)
    r1 = min(H, cy + search_radius + pad_px)
    c0 = max(0, cx - search_radius - pad_px)
    c1 = min(W, cx + search_radius + pad_px)
    if (r1 - r0) < 64 or (c1 - c0) < 64:
        return None
    crop = cy5_uint8[r0:r1, c0:c1]
    try:
        masks, _, _ = cpsam_model.eval(crop, augment=use_tta)
    except Exception:
        return None
    if masks.max() == 0:
        return None
    masks = masks.astype(np.int32)
    # Pick the cpsam mask whose centroid is closest to (cy - r0, cx - c0)
    best_dist, best_lab = float("inf"), 0
    for lab in range(1, int(masks.max()) + 1):
        m = masks == lab
        if not m.any():
            continue
        ys, xs = np.where(m)
        d = (ys.mean() + r0 - cy) ** 2 + (xs.mean() + c0 - cx) ** 2
        if d < best_dist:
            best_dist = d
            best_lab = lab
    if best_lab == 0:
        return None
    out = np.zeros((H, W), dtype=bool)
    out[r0:r1, c0:c1] = (masks == best_lab)
    return out


def cy5_gap_fill_for_track(track_stack, dic_frames, cy5_frames,
                            strategy="cy5_cpsam",
                            cpsam_model=None,
                            search_radius=80, k_mad=3.0,
                            min_area=80, max_lookback=5):
    """Fill internal gaps in one track using Cy5 evidence.

    Default strategy is `cy5_cpsam` — runs cpsam on the Cy5 crop so
    cell morphology is checked, not just intensity. This rejects
    fluorescence artefacts (round dots, edge halos) that wouldn't
    pass cpsam's learned cell-shape prior.

    A gap is a frame where the track has no mask but flanking frames
    do (within `max_lookback`). For each gap:
      1. Interpolate the expected centroid from anchors
      2. Try the chosen `strategy` to derive a mask from Cy5 nearby
      3. If found, write it into the track's stack[gap_idx]

    strategy ∈ {'cy5_threshold', 'cy5_cleanup', 'cy5_cpsam'}.
    Returns count of gaps filled.

    `dic_frames` is required by the cy5_cpsam strategy (it currently
    runs cpsam on the Cy5 crop, but the API allows future strategies
    that combine both channels). Pass cpsam_model only if needed.
    """
    n_frames = len(track_stack)
    filled = 0
    for fi in range(n_frames):
        if track_stack[fi].any():
            continue
        cent = _interpolate_centroid(track_stack, fi,
                                      max_lookback=max_lookback)
        if cent is None:
            continue
        if strategy == "cy5_threshold":
            mask = _cy5_fill_threshold(cy5_frames[fi], cent,
                                        search_radius=search_radius,
                                        k_mad=k_mad,
                                        min_area=min_area)
        elif strategy == "cy5_cleanup":
            mask = _cy5_fill_cleanup(cy5_frames[fi], cent,
                                      search_radius=search_radius,
                                      k_mad=k_mad,
                                      min_area=min_area)
        elif strategy == "cy5_cpsam":
            if cpsam_model is None:
                raise ValueError(
                    "strategy='cy5_cpsam' requires cpsam_model")
            mask = _cy5_fill_cpsam(cy5_frames[fi], dic_frames[fi],
                                    cent, cpsam_model,
                                    search_radius=search_radius)
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        if mask is not None and mask.sum() >= min_area:
            track_stack[fi] = mask
            filled += 1
    return filled
