"""Per-cell track collector + cache for the IC293 motility analysis.

Mirrors ic295_track_data but for the single-cell crops:
  * reads the IC293 by_condition tree,
  * tags every per-cell record with its POSITION (the cluster unit — see
    ic293_common.parse_position),
  * keeps the same per-frame schema (cents/states/n_neighbors/area_frames
    + scalar speed/distance/netdisp/area) so ic293_motility_stats can reuse
    the IC295 cell_features / MSD / Fürth machinery unchanged.

Density: the neighbour computation is reused verbatim from IC295. For a
1-cell crop it naturally yields n_neighbors=0 / nn_dist=NaN; a `-dividing`
crop with two lineage cells gets a real count. The stats layer drops
density as a treatment metric/confounder (no between-crop variation), but
collecting it costs nothing and keeps the record schema identical.
"""
import os
import sys
import glob
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic293_common import (  # noqa: E402
    RECORDINGS_ROOT, COMPARE_DIR, CONDITIONS, ANALYSIS_EXCLUDE,
    DEFAULT_UM_PER_PX, DEFAULT_DT_MIN, parse_condition, parse_position,
    select_primary_cell)
# Pure helpers (no IC295 state) — reuse so density/centroid logic can't drift.
from scripts.ic295_track_data import (  # noqa: E402
    _frame_centroid_table, _density, SPEED_CAP, NEIGHBOR_RADIUS_UM)
import numpy as np  # noqa: E402

DEFAULT_UM = DEFAULT_UM_PER_PX
DEFAULT_DT = DEFAULT_DT_MIN

CACHE = os.path.join(COMPARE_DIR, "flower_plots", "_track_cache.pkl")
CACHE_VERSION = 1


def _recording_paths():
    return sorted(glob.glob(os.path.join(
        RECORDINGS_ROOT, "*", "*", "pipeline_results", "masks.npz")))


def _label_cond(mp):
    label = os.path.basename(os.path.dirname(os.path.dirname(mp)))
    cond = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(mp))))
    if cond not in CONDITIONS:
        cond = parse_condition(label)
    return label, cond


def collect(um=DEFAULT_UM, dt=DEFAULT_DT):
    """Build the per-cell record dict, grouped by condition.

    {cond: {'cells': [rec,...], 'all'|'rounded'|'spread': [rec,...]}}
    Each rec also carries 'pos' (the field-of-view cluster unit)."""
    from core.cell_state import (classify_track_states, STATE_ROUNDED,
                                  STATE_SPREAD)
    from core.tracking import extract_centroids
    out = {c: {"cells": [], "all": [], "rounded": [], "spread": []}
           for c in CONDITIONS}
    radius_px = NEIGHBOR_RADIUS_UM / um
    paths = _recording_paths()
    for i, mp in enumerate(paths):
        label, cond = _label_cond(mp)
        if cond not in out or label in ANALYSIS_EXCLUDE:
            continue
        pos = parse_position(label)
        try:
            labels = np.load(mp)["labels"]
        except Exception as e:
            print(f"  WARN {mp}: {e}", flush=True)
            continue
        nframes = labels.shape[0]
        # Single-cell selection: keep only the one real cell (the rest are
        # artifacts in these hand-cropped single-cell recordings).
        primary_id, _cands, _amb = select_primary_cell(labels)
        if primary_id is None:
            print(f"  [{i+1}/{len(paths)}] {label}: no cell", flush=True)
            continue
        cids = [primary_id]
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
            ar = np.asarray(sd["metrics"]["area"], dtype=float)
            nneigh, nnd = _density(cents, frame_pts, radius_px, um)
            rec = {
                "label": label, "cond": cond, "pos": pos, "cell_id": cid,
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
    if not rebuild and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            blob = pickle.load(f)
        if isinstance(blob, dict) and blob.get("version") == CACHE_VERSION:
            return blob["data"]
        if from_cache:
            return blob.get("data", blob) if isinstance(blob, dict) else blob
        print(f"  cache stale (need v{CACHE_VERSION}) — rebuilding…", flush=True)
    elif from_cache:
        raise FileNotFoundError(f"--from-cache but {CACHE} missing")
    print(f"Collecting IC293 tracks (um/px={um}, dt={dt} min)…", flush=True)
    data = collect(um, dt)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({"version": CACHE_VERSION, "data": data}, f)
    print(f"  cached tracks → {CACHE}  (v{CACHE_VERSION})", flush=True)
    return data


if __name__ == "__main__":
    load_or_build(rebuild="--rebuild" in sys.argv)
