"""Post-hoc division annotator (v2 — biology-aware).

Detects cell-division candidates in an existing tracked label stack.

Algorithm
---------
For each track T at each frame F:

  1. Find the **peak area** in the lookback window before F. (Real
     cell divisions are preceded by mitotic swelling — area roughly
     doubles over a few frames. Comparing F to F-1 alone misses this.)
  2. Test whether area[F] ≤ AREA_RATIO_HALF × peak_area — i.e. a
     halving relative to the peak (not just the previous frame).
  3. Require the parent to have been in balled / transitional state
     in the frames leading up to the split (mitotic rounding).
  4. Search for a new track N appearing within ±DAUGHTER_FRAME_WINDOW
     frames and within MAX_PAIR_DISTANCE_UM µm of the parent.
  5. **Both daughters must grow** — both the parent's post-split
     mask AND the new daughter's track must reach
     SUBSTANTIAL_FRAC × peak_area within POST_GROWTH_WINDOW frames.
     This filters out the 1-frame noise blobs that v1 mistook for
     daughters.
  6. The daughter must persist ≥ MIN_PERSIST_FRAMES consecutive
     frames after appearing.

Score composes proximity × mass-conservation × persistence × pre-
balled prior × pre-swelling magnitude. Every factor multiplies, so
any one being weak pulls the score down.

This module exposes `find_candidates(labels, um_per_px)` returning
(candidates, tracks) where candidates is a sorted list of dicts.
"""
import numpy as np
from skimage import measure

from core.cell_state import (
    shape_metrics_for_mask, classify_state,
    STATE_BALLED, STATE_TRANSITIONAL,
)


# ---------------------------------------------------------------------
# Tunable thresholds (v2 — biology-aware)
# ---------------------------------------------------------------------
AREA_RATIO_HALF = 0.65          # area[F] / peak_area ≤ this → "halved"
LOOKBACK_DELTA = 5              # frames to search back for peak area
                                 # (mitotic prophase ~30-50 min on
                                 # 10-min/frame data = 3-5 frames)
SWELLING_BASELINE_LOOKBACK = 10  # frames before peak for baseline
DAUGHTER_FRAME_WINDOW = 5       # daughter may first-appear up to
                                 # this many frames AFTER the parent's
                                 # halving (or 1 frame BEFORE, to
                                 # tolerate tracker timing noise).
                                 # Daughters that pre-existed >>1 frame
                                 # before the split are NOT daughters
                                 # — they're separate cells.
DAUGHTER_PRE_WINDOW = 1         # tracker timing slack on the early side
POST_GROWTH_WINDOW = 5          # both daughters must reach
                                 # SUBSTANTIAL_FRAC of peak within this
                                 # many frames after split
SUBSTANTIAL_FRAC = 0.30         # min substantial area / peak_area
MIN_PERSIST_FRAMES = 4          # daughter must persist this many
                                 # CONSECUTIVE frames immediately after
                                 # first appearance
MAX_PAIR_DISTANCE_UM = 50       # daughter centroid ↔ parent centroid
MIN_TRACK_LENGTH = 5            # parent + daughter must each have at
                                 # least this many total frames
PRE_STATE_LOOKBACK = 3          # frames before split to inspect state
MIN_PRE_BALLED_FRAC = 0.30      # parent must have been balled/
                                 # transitional in at least this
                                 # fraction of pre-split frames


# ---------------------------------------------------------------------
# Per-track summary
# ---------------------------------------------------------------------
def build_track_table(labels):
    """For every track ID in `labels`, compute per-frame area,
    centroid, and cell state."""
    n_frames = len(labels)
    track_ids = np.unique(labels)
    track_ids = track_ids[track_ids > 0]
    out = {}
    for tid in track_ids:
        present = []
        areas, centroids, states = {}, {}, {}
        for f in range(n_frames):
            m = labels[f] == tid
            if not m.any():
                continue
            present.append(f)
            ys, xs = np.where(m)
            areas[f] = int(m.sum())
            centroids[f] = (float(xs.mean()), float(ys.mean()))
            metrics = shape_metrics_for_mask(m)
            states[f] = classify_state(metrics)
        if not present:
            continue
        out[int(tid)] = {
            "first_frame": present[0],
            "last_frame": present[-1],
            "frames_present": present,
            "area": areas,
            "centroid": centroids,
            "state": states,
        }
    return out


def _consecutive_persistence(frames_present, start_frame):
    if start_frame not in frames_present:
        return 0
    idx = frames_present.index(start_frame)
    persist = 1
    for j in range(idx + 1, len(frames_present)):
        if frames_present[j] == frames_present[j - 1] + 1:
            persist += 1
        else:
            break
    return persist


def _max_area_in_window(track, frames_list):
    if not frames_list:
        return 0, None
    pairs = [(track["area"].get(f, 0), f) for f in frames_list]
    a, f = max(pairs, key=lambda p: p[0])
    return a, f


# ---------------------------------------------------------------------
# Main candidate finder
# ---------------------------------------------------------------------
def find_candidates(labels, um_per_px=1.0):
    """Return (candidates, tracks). `candidates` is sorted by score
    descending. `tracks` is the per-track summary table."""
    tracks = build_track_table(labels)
    if not tracks:
        return [], tracks

    max_pair_px = MAX_PAIR_DISTANCE_UM / max(um_per_px, 0.01)
    candidates = []

    # Which tracks first-appear at each frame?
    first_at_frame = {}
    for tid, t in tracks.items():
        first_at_frame.setdefault(t["first_frame"], []).append(tid)

    for tid, t in tracks.items():
        if len(t["frames_present"]) < MIN_TRACK_LENGTH:
            continue
        present = t["frames_present"]
        for i in range(2, len(present)):
            f = present[i]
            a_now = t["area"][f]
            lookback = present[max(0, i - LOOKBACK_DELTA):i]
            peak_area, peak_frame = _max_area_in_window(t, lookback)
            if peak_area <= 0:
                continue
            if a_now / peak_area > AREA_RATIO_HALF:
                continue

            # Pre-swelling magnitude
            peak_idx = present.index(peak_frame)
            baseline_window = present[max(0, peak_idx
                                          - SWELLING_BASELINE_LOOKBACK):peak_idx]
            if baseline_window:
                baseline_area = float(np.percentile(
                    [t["area"][fp] for fp in baseline_window], 30))
            else:
                baseline_area = float(peak_area)
            swelling_ratio = peak_area / max(baseline_area, 1.0)

            # Pre-split state filter
            pre_states = [t["state"][fp]
                          for fp in present[max(0, i - PRE_STATE_LOOKBACK):i]]
            balled = sum(1 for s in pre_states
                         if s in (STATE_BALLED, STATE_TRANSITIONAL))
            pre_balled_frac = balled / max(len(pre_states), 1)
            if pre_balled_frac < MIN_PRE_BALLED_FRAC:
                continue

            # Parent must remain substantial after split
            min_sub = SUBSTANTIAL_FRAC * peak_area
            post_window = present[i:i + POST_GROWTH_WINDOW + 1]
            parent_post_peak, _ = _max_area_in_window(t, post_window)
            if parent_post_peak < min_sub:
                continue

            cx, cy = t["centroid"][f]

            for f_new in range(max(0, f - DAUGHTER_PRE_WINDOW),
                                f + DAUGHTER_FRAME_WINDOW + 1):
                for new_tid in first_at_frame.get(f_new, []):
                    if new_tid == tid:
                        continue
                    nt = tracks[new_tid]
                    if len(nt["frames_present"]) < MIN_TRACK_LENGTH:
                        continue
                    if f_new not in nt["centroid"]:
                        continue
                    nx, ny = nt["centroid"][f_new]
                    dist_px = float(np.hypot(nx - cx, ny - cy))
                    if dist_px > max_pair_px:
                        continue

                    # Daughter must grow into something substantial
                    d_window = nt["frames_present"][:POST_GROWTH_WINDOW + 1]
                    daughter_peak, _ = _max_area_in_window(nt, d_window)
                    if daughter_peak < min_sub:
                        continue

                    # Daughter must persist consecutively
                    persist = _consecutive_persistence(
                        nt["frames_present"], nt["first_frame"])
                    if persist < MIN_PERSIST_FRAMES:
                        continue

                    # Mass conservation against PRE-SPLIT PEAK
                    a_daughter_first = nt["area"][f_new]
                    mass_ratio = (parent_post_peak + daughter_peak) / peak_area

                    prox = 1.0 / (1.0 + dist_px / max_pair_px)
                    mass_score = (
                        1.0 if 0.7 <= mass_ratio <= 1.3 else
                        max(0.0, 1.0 - abs(mass_ratio - 1.0)))
                    persist_score = min(1.0, persist / 5.0)
                    balled_score = 0.5 + 0.5 * pre_balled_frac
                    swelling_score = min(
                        1.0, max(0.0, swelling_ratio - 1.0) / 0.5)
                    score = (prox * mass_score * persist_score *
                              balled_score * (0.5 + 0.5 * swelling_score))

                    candidates.append({
                        "frame": int(f),
                        "peak_frame": int(peak_frame),
                        "parent_track": int(tid),
                        "daughter_track": int(new_tid),
                        "daughter_first_frame": int(f_new),
                        "daughter_persistence_frames": int(persist),
                        "distance_px": dist_px,
                        "area_peak": int(peak_area),
                        "area_parent_at_split": int(a_now),
                        "area_parent_post_peak": int(parent_post_peak),
                        "area_daughter_first": int(a_daughter_first),
                        "area_daughter_post_peak": int(daughter_peak),
                        "area_baseline": int(baseline_area),
                        "swelling_ratio": float(swelling_ratio),
                        "mass_ratio": float(mass_ratio),
                        "pre_split_states": pre_states,
                        "pre_split_balled_frac": float(pre_balled_frac),
                        "mass_score": float(mass_score),
                        "persist_score": float(persist_score),
                        "balled_score": float(balled_score),
                        "swelling_score": float(swelling_score),
                        "prox_score": float(prox),
                        "score": float(score),
                        "parent_centroid": (cx, cy),
                        "daughter_centroid": (nx, ny),
                    })

    # Deduplicate per (parent, daughter)
    seen = {}
    for c in candidates:
        key = (c["parent_track"], c["daughter_track"])
        if key not in seen or c["score"] > seen[key]["score"]:
            seen[key] = c
    return sorted(seen.values(), key=lambda c: -c["score"]), tracks
