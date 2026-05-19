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
POST_STATE_WINDOW = 3           # frames AFTER split to also inspect —
                                 # cell may not be balled until cyto-
                                 # kinesis completes (see Pos51_Y1
                                 # where parent rounded F68-F70 with
                                 # split at F66)
MIN_PRE_BALLED_FRAC = 0.30      # parent must have been in balled/
                                 # transitional state in at least
                                 # this fraction of [pre, post]-split
                                 # frames combined
MAX_DAUGHTER_GAP_FRAMES = 2     # daughter may have track gaps of up
                                 # to this many frames in its first
                                 # POST_GROWTH_WINDOW (contact phase
                                 # often hides one daughter briefly)


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


def _consecutive_persistence(frames_present, start_frame,
                              max_gap=0):
    """Count frames in `frames_present` starting from `start_frame`,
    tolerating gaps of up to `max_gap` frames (a 1-frame gap is
    common during the contact phase where the tracker temporarily
    loses a daughter to the merged-pair mask)."""
    if start_frame not in frames_present:
        return 0
    idx = frames_present.index(start_frame)
    persist = 1
    for j in range(idx + 1, len(frames_present)):
        gap = frames_present[j] - frames_present[j - 1] - 1
        if gap <= max_gap:
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
def find_candidates(labels, um_per_px=1.0, include_rejected=False):
    """Return (candidates, tracks) — or (candidates, rejected, tracks)
    if include_rejected=True.

    candidates: list of dicts that passed all filters, sorted by
    score descending.

    rejected: list of dicts for area-halving events where the
    parent's mask sufficiently halved (so they're real "area-drop
    events" in the data) but at least one downstream filter failed.
    Each has `rejection_reason` ∈ {
      no_pre_balled, parent_not_substantial,
      no_nearby_new_track, daughter_not_substantial,
      daughter_transient
    }. Useful for verifying the filters are doing the right thing.
    """
    tracks = build_track_table(labels)
    if not tracks:
        if include_rejected:
            return [], [], tracks
        return [], tracks

    max_pair_px = MAX_PAIR_DISTANCE_UM / max(um_per_px, 0.01)
    candidates = []
    rejected = []

    first_at_frame = {}
    for tid, t in tracks.items():
        first_at_frame.setdefault(t["first_frame"], []).append(tid)

    def _make_event(tid, t, f, peak_area, peak_frame,
                     swelling_ratio, baseline_area,
                     pre_states, pre_balled_frac,
                     parent_post_peak, daughter_info=None,
                     reason=None):
        """Build an event dict (passed or rejected)."""
        a_now = t["area"][f]
        cx, cy = t["centroid"][f]
        d = {
            "frame": int(f),
            "peak_frame": int(peak_frame),
            "parent_track": int(tid),
            "area_peak": int(peak_area),
            "area_parent_at_split": int(a_now),
            "area_parent_post_peak": int(parent_post_peak),
            "area_baseline": int(baseline_area),
            "swelling_ratio": float(swelling_ratio),
            "pre_split_states": pre_states,
            "pre_split_balled_frac": float(pre_balled_frac),
            "parent_centroid": (cx, cy),
        }
        if daughter_info:
            nt, f_new, dist_px, daughter_peak, persist = daughter_info
            a_daughter_first = nt["area"][f_new]
            mass_ratio = (parent_post_peak + daughter_peak) / peak_area
            prox = 1.0 / (1.0 + dist_px / max_pair_px)
            mass_score = (1.0 if 0.7 <= mass_ratio <= 1.3 else
                          max(0.0, 1.0 - abs(mass_ratio - 1.0)))
            persist_score = min(1.0, persist / 5.0)
            balled_score = 0.5 + 0.5 * pre_balled_frac
            swelling_score = min(
                1.0, max(0.0, swelling_ratio - 1.0) / 0.5)
            score = (prox * mass_score * persist_score *
                      balled_score * (0.5 + 0.5 * swelling_score))
            nx, ny = nt["centroid"][f_new]
            d.update({
                "daughter_track": int(_track_id_of(tracks, nt)),
                "daughter_first_frame": int(f_new),
                "daughter_persistence_frames": int(persist),
                "distance_px": dist_px,
                "area_daughter_first": int(a_daughter_first),
                "area_daughter_post_peak": int(daughter_peak),
                "mass_ratio": float(mass_ratio),
                "mass_score": float(mass_score),
                "persist_score": float(persist_score),
                "balled_score": float(balled_score),
                "swelling_score": float(swelling_score),
                "prox_score": float(prox),
                "score": float(score),
                "daughter_centroid": (nx, ny),
            })
        if reason:
            d["rejection_reason"] = reason
        return d

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
            # Real area-halving event. Now apply downstream filters,
            # tracking the rejection reason for diagnostics.

            peak_idx = present.index(peak_frame)
            baseline_window = present[max(0, peak_idx
                                          - SWELLING_BASELINE_LOOKBACK):peak_idx]
            if baseline_window:
                baseline_area = float(np.percentile(
                    [t["area"][fp] for fp in baseline_window], 30))
            else:
                baseline_area = float(peak_area)
            swelling_ratio = peak_area / max(baseline_area, 1.0)

            # Peri-split state check — look both BEFORE (mitotic
            # prophase) and AT/AFTER the split (cytokinesis rounding).
            # Many divisions don't show pre-balling — the cell goes
            # from spread directly into halving + balling.
            pre_states = [t["state"][fp]
                          for fp in present[max(0, i - PRE_STATE_LOOKBACK):i]]
            post_states = [t["state"][fp]
                           for fp in present[i:i + POST_STATE_WINDOW + 1]]
            peri_states = pre_states + post_states
            balled = sum(1 for s in peri_states
                         if s in (STATE_BALLED, STATE_TRANSITIONAL))
            pre_balled_frac = balled / max(len(peri_states), 1)

            min_sub = SUBSTANTIAL_FRAC * peak_area
            post_window = present[i:i + POST_GROWTH_WINDOW + 1]
            parent_post_peak, _ = _max_area_in_window(t, post_window)

            base_kwargs = dict(
                tid=tid, t=t, f=f,
                peak_area=peak_area, peak_frame=peak_frame,
                swelling_ratio=swelling_ratio, baseline_area=baseline_area,
                pre_states=pre_states, pre_balled_frac=pre_balled_frac,
                parent_post_peak=parent_post_peak)

            if pre_balled_frac < MIN_PRE_BALLED_FRAC:
                if include_rejected:
                    rejected.append(_make_event(
                        **base_kwargs, reason="no_pre_balled"))
                continue
            if parent_post_peak < min_sub:
                if include_rejected:
                    rejected.append(_make_event(
                        **base_kwargs, reason="parent_not_substantial"))
                continue

            cx, cy = t["centroid"][f]
            best_daughter = None
            no_daughter_at_all = True

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
                    no_daughter_at_all = False

                    d_window = nt["frames_present"][:POST_GROWTH_WINDOW + 1]
                    daughter_peak, _ = _max_area_in_window(nt, d_window)
                    persist = _consecutive_persistence(
                        nt["frames_present"], nt["first_frame"],
                        max_gap=MAX_DAUGHTER_GAP_FRAMES)

                    if daughter_peak < min_sub:
                        if include_rejected and best_daughter is None:
                            best_daughter = (nt, f_new, dist_px,
                                             daughter_peak, persist,
                                             "daughter_not_substantial")
                        continue
                    if persist < MIN_PERSIST_FRAMES:
                        if include_rejected and (
                                best_daughter is None or
                                best_daughter[5] !=
                                "daughter_not_substantial"):
                            best_daughter = (nt, f_new, dist_px,
                                             daughter_peak, persist,
                                             "daughter_transient")
                        continue

                    # All criteria passed!
                    cand = _make_event(
                        **base_kwargs,
                        daughter_info=(nt, f_new, dist_px,
                                       daughter_peak, persist))
                    candidates.append(cand)

            if include_rejected and not any(
                    c["parent_track"] == int(tid) and c["frame"] == int(f)
                    for c in candidates):
                if no_daughter_at_all:
                    rejected.append(_make_event(
                        **base_kwargs, reason="no_nearby_new_track"))
                elif best_daughter is not None:
                    nt, f_new, dist_px, daughter_peak, persist, reason = (
                        best_daughter)
                    rejected.append(_make_event(
                        **base_kwargs,
                        daughter_info=(nt, f_new, dist_px,
                                        daughter_peak, persist),
                        reason=reason))

    # Deduplicate candidates per (parent, daughter)
    seen = {}
    for c in candidates:
        key = (c["parent_track"], c["daughter_track"])
        if key not in seen or c["score"] > seen[key]["score"]:
            seen[key] = c
    sorted_cands = sorted(seen.values(), key=lambda c: -c["score"])

    if include_rejected:
        # Temporal dedup: a track with multiple halvings within
        # REJECT_CLUSTER_FRAMES of each other is treated as one
        # event (keep the one with the largest peak area). Filter
        # out small/trivial halvings (parent peak below
        # SUBSTANTIAL_AREA_PX) — they're noise, not "near-misses".
        SUBSTANTIAL_AREA_PX = 500
        REJECT_CLUSTER_FRAMES = 5
        PASS_OVERLAP_FRAMES = 3   # also suppress rejected events
                                   # within ±this many frames of a
                                   # PASS in the same parent track —
                                   # they're the same biological
                                   # event flagged at a different
                                   # threshold crossing
        substantial = [r for r in rejected
                       if r.get("area_peak", 0) >= SUBSTANTIAL_AREA_PX]
        substantial.sort(
            key=lambda r: (r["parent_track"], r["frame"]))
        clustered = []
        for r in substantial:
            if (clustered and
                clustered[-1]["parent_track"] == r["parent_track"] and
                r["frame"] - clustered[-1]["frame"] <=
                REJECT_CLUSTER_FRAMES):
                if r["area_peak"] > clustered[-1]["area_peak"]:
                    clustered[-1] = r
            else:
                clustered.append(r)
        # Drop rejected events that overlap a PASS (same parent,
        # frame within ±PASS_OVERLAP_FRAMES). Those are noise from
        # the same biological event being flagged at multiple
        # threshold crossings.
        pass_index = {}
        for c in sorted_cands:
            pass_index.setdefault(c["parent_track"], []).append(
                c["frame"])
        deduped = []
        for r in clustered:
            pass_frames = pass_index.get(r["parent_track"], [])
            if any(abs(r["frame"] - pf) <= PASS_OVERLAP_FRAMES
                   for pf in pass_frames):
                continue
            deduped.append(r)
        deduped.sort(key=lambda r: -r.get("area_peak", 0))
        return sorted_cands, deduped, tracks
    return sorted_cands, tracks


def _track_id_of(tracks, track_dict):
    """Reverse-lookup the track ID given a track dict (linear scan)."""
    for tid, t in tracks.items():
        if t is track_dict:
            return tid
    return -1


# ---------------------------------------------------------------------
# Pipeline integration: annotate parent_id on a tracks LIST
# ---------------------------------------------------------------------
def annotate_track_lineage(tracks_list, labels, um_per_px=1.0):
    """Run division detection on a labels stack and write parent_id
    back to the tracks list.

    `tracks_list` is the list-of-dicts produced by
    `core.multi_cell.track_all_cells` (and friends). Convention:
    `tracks_list[i]` corresponds to label `i + 1` in the labels stack
    (matches `core.cy5_filter.rebuild_label_stack`).

    For each detected division, sets `tracks_list[daughter_idx]
    ["parent_id"] = parent_idx` (LIST indices, 0-based — matches the
    legacy Phase-16b convention in `track_all_cells`).

    Returns (candidates, n_lineage_set). Never raises — on internal
    error returns ([], 0) so the pipeline keeps going.

    Pre-existing `parent_id` values set by the legacy heuristic are
    PRESERVED (we only set parent_id where it isn't already set).
    """
    try:
        candidates, _ = find_candidates(labels, um_per_px=um_per_px)
    except Exception:
        return [], 0
    n_set = 0
    for c in candidates:
        parent_label = c["parent_track"]
        daughter_label = c["daughter_track"]
        # Convert label IDs (1-based, as stored in labels stack) to
        # tracks-list indices (0-based)
        p_idx = parent_label - 1
        d_idx = daughter_label - 1
        if not (0 <= p_idx < len(tracks_list)):
            continue
        if not (0 <= d_idx < len(tracks_list)):
            continue
        existing = tracks_list[d_idx].get("parent_id")
        if existing is None:
            tracks_list[d_idx]["parent_id"] = p_idx
            tracks_list[d_idx]["division_score"] = c["score"]
            tracks_list[d_idx]["division_frame"] = c["frame"]
            n_set += 1
    return candidates, n_set
