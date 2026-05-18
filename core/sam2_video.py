"""SAM2 video-mode mask propagation through detection gaps.

When cpsam(augment) and the CP3 fallback both fail to detect a tracked
cell at frame F (typical when the cell is biologically retracting,
mitotic, or dimming), SAM2 video propagation uses the cell's
last-known mask as a memory-attention anchor and propagates it forward
through the gap. SAM2 was trained on natural video so it follows objects
through low-contrast and shape-change frames better than per-frame
detectors.

Two entry points:

  `propagate_through_gap(frames, anchor_frame, anchor_mask,
                          gap_frames, …)`
    The atomic operation: given a known mask at one anchor frame and a
    list of consecutive gap frames, returns a list of propagated masks
    (one per gap frame).

  `fill_track_gaps_with_sam2(tracks, frames, …)`
    Pipeline-level helper: for each track with internal gaps, finds
    the nearest preceding mask, propagates SAM2 forward through the
    gap, accepts any mask above min_area into the track stack.

Cost: ~3 fps on Apple MPS (vit-tiny backbone). For our 40-frame demos
with 5-10 gaps per track, total cost ~5-15 s — comparable to Phase 1
cpsam(augment).

Checkpoint expected at `data/models/sam2/sam2.1_hiera_tiny.pt`.
Hardcoded to the 'tiny' variant for speed; switch to base/large only
if accuracy demands it.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import logging
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "data/models/sam2/sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_predictor(checkpoint=DEFAULT_CHECKPOINT, config=DEFAULT_CONFIG):
    """Lazy-load SAM2 video predictor on best available device."""
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(config, checkpoint, device=_device())


def _dump_frames_to_jpg(frames, out_dir):
    """SAM2 video reads from {00000.jpg, 00001.jpg, ...}. Dumps a
    sequence of uint8 frames to `out_dir`."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.jpg"), f)


def propagate_through_gap(frames, anchor_frame_idx, anchor_mask,
                            gap_frame_indices, predictor=None,
                            min_area=20, max_hop_px=150,
                            max_area_change_factor=3.0):
    """Propagate `anchor_mask` from `anchor_frame_idx` through a list
    of gap frame indices using SAM2 video memory attention.

    SAM2 follows image content, so without constraints it can drift to
    a brighter/larger nearby cell when the original target dims out.
    Two safeguards:
      - Reject propagated masks whose centroid moves > max_hop_px
        from the previous valid mask.
      - Reject propagated masks whose area changes by more than
        `max_area_change_factor` (e.g. 3.0 means area can shrink to
        1/3 or grow to 3× the previous valid area, but not more).
    Once a frame is rejected, propagation through subsequent frames
    is also stopped (cumulative-drift is not recoverable).

    Args:
      frames: (N, H, W) uint8 image stack the gaps live in
      anchor_frame_idx: int, frame index where anchor_mask is correct
      anchor_mask: (H, W) bool, the cell's mask at the anchor
      gap_frame_indices: list[int] frames to propagate INTO. Must all
        be ≥ anchor_frame_idx (forward propagation only).
      predictor: a pre-built SAM2 predictor; lazy-loaded if None
      min_area: discard returned masks smaller than this (px)
      max_hop_px: per-frame centroid drift tolerance
      max_area_change_factor: per-frame max area change ratio

    Returns dict {frame_idx: bool_mask}. Frames where SAM2's output
    failed any safety check are absent from the dict.
    """
    if not gap_frame_indices:
        return {}
    forward_gaps = [g for g in gap_frame_indices if g > anchor_frame_idx]
    if not forward_gaps:
        return {}
    last = max(forward_gaps)

    # Slice frames from anchor to last gap inclusive, dump to JPGs
    sub = frames[anchor_frame_idx:last + 1]
    if len(sub) < 2:
        return {}

    if predictor is None:
        predictor = _build_predictor()

    # Reference centroid + area for drift checks (start from anchor)
    a_ys, a_xs = np.where(anchor_mask)
    last_centroid = (float(a_ys.mean()), float(a_xs.mean()))
    last_area = int(anchor_mask.sum())

    out = {}
    rejected = False
    with tempfile.TemporaryDirectory() as tmp:
        _dump_frames_to_jpg(sub, tmp)
        state = predictor.init_state(video_path=tmp)
        # Anchor mask is at LOCAL frame 0 in the dumped sub-sequence.
        predictor.add_new_mask(state, frame_idx=0, obj_id=1,
                                mask=anchor_mask)
        for local_idx, obj_ids, mask_logits in (
                predictor.propagate_in_video(state)):
            if rejected:
                # Once SAM2 has drifted, don't trust further frames
                # in this segment — cumulative drift is not recoverable.
                continue
            global_idx = anchor_frame_idx + local_idx
            if global_idx not in forward_gaps:
                continue
            for i, oid in enumerate(obj_ids):
                if oid != 1:
                    continue
                m = (mask_logits[i] > 0.0).cpu().numpy().squeeze().astype(bool)
                area = int(m.sum())
                if area < min_area:
                    rejected = True
                    break
                ys, xs = np.where(m)
                cy, cx = float(ys.mean()), float(xs.mean())
                hop = ((cy - last_centroid[0]) ** 2
                       + (cx - last_centroid[1]) ** 2) ** 0.5
                if hop > max_hop_px:
                    log.info(
                        "SAM2 drift rejected at F%d: hop %.0f px "
                        "(>%.0f px from F%d)",
                        global_idx, hop, max_hop_px,
                        global_idx - 1)
                    rejected = True
                    break
                area_ratio = area / max(last_area, 1)
                if (area_ratio > max_area_change_factor
                        or area_ratio < 1.0 / max_area_change_factor):
                    log.info(
                        "SAM2 area change rejected at F%d: "
                        "%d → %d px (×%.1f)",
                        global_idx, last_area, area, area_ratio)
                    rejected = True
                    break
                out[global_idx] = m
                last_centroid = (cy, cx)
                last_area = area
    return out


def fill_track_gaps_with_sam2(tracks, frames, min_area=200,
                                checkpoint=DEFAULT_CHECKPOINT,
                                config=DEFAULT_CONFIG,
                                progress_fn=None):
    """For each track with internal gaps, propagate SAM2 forward
    from the nearest preceding mask. Modifies tracks in-place.

    Returns total count of frames filled. Each filled frame is added
    to `track["sam2_propagated_frames"]` so analytics can flag it.

    Skip strategy: tracks without internal gaps are untouched. Tracks
    where the only gap is BEFORE the first detection (no anchor) are
    skipped — SAM2 propagates forward from a known mask, not into the
    past.
    """
    n_frames = len(frames)
    work_items = []   # (track_idx, anchor_frame, gaps[])
    for tid, track in enumerate(tracks):
        active = np.array([track["stack"][i].any() for i in range(n_frames)])
        if active.sum() < 2:
            continue
        first = int(np.argmax(active))
        last = n_frames - 1 - int(np.argmax(active[::-1]))
        # Walk through the active range, finding gap segments + their
        # preceding anchor.
        i = first
        while i <= last:
            if active[i]:
                anchor = i
                # Skip past consecutive present frames
                while i <= last and active[i]:
                    i += 1
                # Now i is the first gap frame after `anchor`
                gap_run = []
                while i <= last and not active[i]:
                    gap_run.append(i)
                    i += 1
                if gap_run:
                    work_items.append((tid, anchor, gap_run))
            else:
                i += 1
    if not work_items:
        log.info("SAM2 gap-fill: no eligible gap segments")
        return 0

    log.info("SAM2 gap-fill: %d gap segments across %d tracks",
             len(work_items), len(set(w[0] for w in work_items)))

    if progress_fn:
        progress_fn("Loading SAM2 video predictor", 0)
    predictor = _build_predictor(checkpoint=checkpoint, config=config)

    filled = 0
    for n, (tid, anchor, gaps) in enumerate(work_items):
        if progress_fn:
            progress_fn(
                f"SAM2 gap-fill {n + 1}/{len(work_items)} "
                f"(track {tid + 1}, gap len {len(gaps)})",
                int(100 * n / max(len(work_items) - 1, 1)))
        track = tracks[tid]
        anchor_mask = track["stack"][anchor]
        try:
            propagated = propagate_through_gap(
                frames, anchor, anchor_mask, gaps,
                predictor=predictor, min_area=min_area)
        except Exception as e:
            log.warning(
                "SAM2 propagation failed for track %d gap %d-%d: %s",
                tid + 1, gaps[0], gaps[-1], e)
            continue
        track.setdefault("sam2_propagated_frames", set())
        for fi, mask in propagated.items():
            track["stack"][fi] = mask
            track["sam2_propagated_frames"].add(fi)
            filled += 1
    log.info("SAM2 gap-fill filled %d frames across %d gap segments",
             filled, len(work_items))
    if progress_fn:
        progress_fn(f"SAM2 gap-fill done: {filled} frames", 100)
    return filled
