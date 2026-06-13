"""Shared per-cell track collector + cache for the IC295 motility analyses.

One pass over every recording's `masks.npz` builds a per-cell record with
enough per-frame detail to support BOTH the flower/MSD plots
(`ic295_flower_plots.py`) and the design-correct, confounder-aware motility
statistics (`ic295_motility_stats.py`). Collecting once and caching the
result keeps the ~1 h mask reload out of the iterate-on-plots loop.

Per-cell record (µm / µm² unless noted):
  label        recording id (for per-recording aggregation → arm stats)
  cond         condition (WT/KO/GOF/Y1/OT/DMSO)
  cell_id      integer label within the recording
  traj         (N,2) origin-centred track, NaN where absent
  cents        (N,2) absolute track µm, NaN where absent
  states       (N,) per-frame state code (rounded/spread/unknown/edge)
  n_neighbors  (N,) cells whose centroid is within NEIGHBOR_RADIUS_UM
  nn_dist      (N,) nearest-neighbour centroid distance µm (NaN if alone)
  area_frames  (N,) per-frame footprint µm² (NaN where absent)
  speed        scalar mean step speed µm/min (glitch-capped)
  distance     scalar total path length µm
  netdisp      scalar net displacement |last-first| µm
  area         scalar mean footprint µm² (name kept for flower-plot compat)

The condition dict also keeps the legacy 'all'/'rounded'/'spread' grouping
(whole-track single-state membership) the flower plots consume.

Density note: neighbours are CENTROID-based (cheap, robust). True membrane
contact (region-adjacency graph per frame) is a possible refinement; the
nearest-neighbour distance is a good contact proxy at this magnification.
"""
import os
import sys
import glob
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    RECORDINGS_ROOT, COMPARE_DIR, CONDITIONS, parse_condition)
import numpy as np  # noqa: E402

DEFAULT_UM = 0.6523                       # IC295 scope (single magnification)
DEFAULT_DT = 10.0                         # min / frame
SPEED_CAP = 15.0                          # µm/min — drop tracking glitches
NEIGHBOR_RADIUS_UM = 100.0                # local-density window (≈1.5 cell ⌀)

CACHE = os.path.join(COMPARE_DIR, "flower_plots", "_track_cache.pkl")
CACHE_VERSION = 2                          # bump when the record schema changes


def _recording_paths():
    return sorted(glob.glob(os.path.join(
        RECORDINGS_ROOT, "*", "*", "pipeline_results", "masks.npz")))


def _condition_of(mp):
    label = os.path.basename(os.path.dirname(os.path.dirname(mp)))
    cond = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(mp))))
    if cond not in CONDITIONS:
        cond = parse_condition(label)
    return label, cond


def _frame_centroid_table(all_cents, nframes):
    """{cid: (N,2) px} → per-frame array of present centroids (k,2) px."""
    pts = [[] for _ in range(nframes)]
    for cents in all_cents.values():
        finite = np.isfinite(cents[:, 0])
        for f in np.nonzero(finite)[0]:
            pts[f].append(cents[f])
    return [np.asarray(p) if p else np.empty((0, 2)) for p in pts]


def _density(cents, frame_pts, radius_px, um):
    """Per-frame (n_neighbors, nearest-neighbour dist µm) for one cell."""
    nframes = len(cents)
    nneigh = np.full(nframes, np.nan)
    nnd = np.full(nframes, np.nan)
    for f in range(nframes):
        if not np.isfinite(cents[f, 0]):
            continue
        pts = frame_pts[f]
        if len(pts) <= 1:
            nneigh[f] = 0.0
            continue
        d = np.linalg.norm(pts - cents[f], axis=1)
        d = d[d > 1e-6]                      # drop self
        if d.size:
            nneigh[f] = float(np.sum(d <= radius_px))
            nnd[f] = float(np.min(d) * um)
    return nneigh, nnd


def collect(um=DEFAULT_UM, dt=DEFAULT_DT):
    """Build the per-cell record dict, grouped by condition.

    {cond: {'cells': [rec,...], 'all'|'rounded'|'spread': [rec,...]}}
    'cells' is the full per-recording-labelled list; the three legacy keys
    point at the same rec objects for the flower groupings."""
    from core.cell_state import (classify_track_states, STATE_ROUNDED,
                                  STATE_SPREAD)
    from core.tracking import extract_centroids
    out = {c: {"cells": [], "all": [], "rounded": [], "spread": []}
           for c in CONDITIONS}
    radius_px = NEIGHBOR_RADIUS_UM / um
    paths = _recording_paths()
    for i, mp in enumerate(paths):
        label, cond = _condition_of(mp)
        if cond not in out:
            continue
        try:
            labels = np.load(mp)["labels"]
        except Exception as e:
            print(f"  WARN {mp}: {e}", flush=True)
            continue
        nframes = labels.shape[0]
        cids = [int(v) for v in np.unique(labels) if v > 0]
        all_cents = {cid: extract_centroids(labels == cid) for cid in cids}
        frame_pts = _frame_centroid_table(all_cents, nframes)
        for cid in cids:
            cents = all_cents[cid]
            valid = ~np.isnan(cents).any(axis=1)
            if valid.sum() < 2:
                continue
            cv = cents[valid]
            seg = np.linalg.norm(np.diff(cents, axis=0), axis=1) * um
            seg = seg[np.isfinite(seg)]
            spd = seg / dt
            spd = spd[spd <= SPEED_CAP]
            stack = labels == cid
            sd = classify_track_states(stack, um_per_px=um)
            ar = np.asarray(sd["metrics"]["area"], dtype=float)   # px²/frame
            nneigh, nnd = _density(cents, frame_pts, radius_px, um)
            rec = {
                "label": label, "cond": cond, "cell_id": cid,
                "traj": (cv - cv[0]) * um,
                "cents": cents * um,
                "states": np.asarray(sd["states"]),
                "n_neighbors": nneigh,
                "nn_dist": nnd,
                "area_frames": ar * um * um,
                "speed": float(np.mean(spd)) if spd.size else float("nan"),
                "distance": float(np.sum(seg)),
                "netdisp": float(np.linalg.norm(cv[-1] - cv[0]) * um),
                "area": (float(np.nanmean(ar)) * um * um
                         if np.isfinite(ar).any() else float("nan")),
            }
            out[cond]["cells"].append(rec)
            out[cond]["all"].append(rec)
            st = rec["states"]
            cls = st[(st == STATE_ROUNDED) | (st == STATE_SPREAD)]
            if cls.size:
                if np.all(cls == STATE_ROUNDED):
                    out[cond]["rounded"].append(rec)
                elif np.all(cls == STATE_SPREAD):
                    out[cond]["spread"].append(rec)
        print(f"  [{i+1}/{len(paths)}] {label} ({cond})", flush=True)
    return out


def load_or_build(um=DEFAULT_UM, dt=DEFAULT_DT, from_cache=False, rebuild=False):
    """Return the collect() dict, using the on-disk cache when valid.

    from_cache=True  → use the cache as-is (fast re-plot); errors if absent.
    rebuild=True     → ignore any cache and recollect.
    Otherwise        → use the cache only if present AND the right version."""
    if not rebuild and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            blob = pickle.load(f)
        if isinstance(blob, dict) and blob.get("version") == CACHE_VERSION:
            return blob["data"]
        if from_cache:
            # legacy (pre-v2) cache: hand back the raw dict for plot-only use
            return blob.get("data", blob) if isinstance(blob, dict) else blob
        print(f"  cache is stale (need v{CACHE_VERSION}) — rebuilding…",
              flush=True)
    elif from_cache:
        raise FileNotFoundError(f"--from-cache but {CACHE} missing")
    print(f"Collecting tracks (um/px={um}, dt={dt} min)…", flush=True)
    data = collect(um, dt)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({"version": CACHE_VERSION, "data": data}, f)
    print(f"  cached tracks → {CACHE}  (v{CACHE_VERSION})", flush=True)
    return data


if __name__ == "__main__":
    # Build / refresh the cache standalone (used by the rebuild step).
    load_or_build(rebuild="--rebuild" in sys.argv)
