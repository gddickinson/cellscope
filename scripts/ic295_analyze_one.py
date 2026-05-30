"""Analyze one recording's masks. Idempotent.

Reads ic295_analysis/by_condition/<cond>/<label>/pipeline_results/masks.npz
(possibly user-edited after Phase-1 detection), and writes:
  analysis.json            — per-cell results (arrays stripped, ~few KB/cell)
  recording_summary.json   — single-row aggregate for treatment comparison
  per_cell.csv             — flat CSV of per-cell metrics
  RUN_METADATA.json        — refreshed with analysis timing

Skips if analysis.json already exists (unless --force).
"""
import os
import sys
import csv
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

import numpy as np
from scripts.ic295_common import (
    inventory_drive, recording_dir, atomic_write_json,
    write_cellscope_project,
)


def rebuild_tracks_from_labels(labels):
    if labels is None:
        return []
    ids = np.unique(labels)
    ids = ids[ids > 0]
    tracks = []
    for cid in ids.tolist():
        stack = (labels == int(cid))
        present = stack.any(axis=(1, 2))
        if not present.any():
            continue
        tracks.append({
            "stack":       stack,
            "first_frame": int(np.argmax(present)),
        })
    return tracks


def per_cell_row(c, ti):
    ss   = c.get("shape_summary", {}) or {}
    es   = c.get("edge_summary", {}) or {}
    area = ss.get("area_um2") or {}
    return {
        "cell_id":                  c.get("cell_id"),
        "first_frame":              ti.get("first_frame"),
        "frames_tracked":           ti.get("frames_tracked"),
        "parent_id":                ti.get("parent_id"),
        "division_frame":           ti.get("division_frame"),
        "division_score":           ti.get("division_score"),
        "mean_speed":               c.get("mean_speed"),
        "total_distance":           c.get("total_distance"),
        "net_displacement":         c.get("net_displacement"),
        "persistence":              c.get("persistence"),
        "mean_area_um2":            area.get("mean"),
        "median_area_um2":          area.get("median"),
        "mean_circularity":         (ss.get("circularity") or {}).get("mean"),
        "mean_solidity":            (ss.get("solidity") or {}).get("mean"),
        "mean_aspect_ratio":        (ss.get("aspect_ratio") or {}).get("mean"),
        "mean_eccentricity":        (ss.get("eccentricity") or {}).get("mean"),
        "mean_protrusion_velocity": es.get("mean_protrusion_velocity"),
        "mean_retraction_velocity": es.get("mean_retraction_velocity"),
        "mean_boundary_confidence": c.get("mean_boundary_confidence"),
        "state_frac_balled":        c.get("state_frac_balled"),
        "state_frac_attached":      c.get("state_frac_attached"),
        "mean_speed_balled":        c.get("mean_speed_balled"),
        "mean_speed_attached":      c.get("mean_speed_attached"),
    }


def aggregate_recording(rows, n_divisions):
    """Reduce across cells → one-row recording summary. Each cell is
    treated as an independent observation within this recording; the
    cross-recording phase treats each recording as one experiment."""
    n = len(rows)
    out = {"n_cells": n, "n_divisions": int(n_divisions),
           "division_rate": (float(n_divisions) / n) if n else 0.0}
    if not n:
        return out
    cols = [k for k in rows[0].keys()
            if k not in ("cell_id", "first_frame", "frames_tracked",
                          "parent_id", "division_frame", "division_score")]
    for k in cols:
        vals = [r[k] for r in rows
                if isinstance(r.get(k), (int, float))
                   and r[k] is not None and np.isfinite(r[k])]
        out[f"{k}_mean"]   = float(np.mean(vals))   if vals else None
        out[f"{k}_median"] = float(np.median(vals)) if vals else None
        out[f"{k}_std"]    = float(np.std(vals))    if len(vals) > 1 else None
        out[f"{k}_n"]      = len(vals)
    return out


def write_per_cell_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# Strip large per-frame arrays so analysis.json stays small.
_DROP_KEYS = {
    "frames", "masks", "centroids_px", "centroids_um", "trajectory",
    "speed", "msd", "autocorr", "shape_timeseries", "edge_angles",
    "edge_velocity", "boundary_confidence", "area_stability",
    "state_per_frame",
}


def _strip(d):
    return {k: v for k, v in d.items() if k not in _DROP_KEYS}


def _annotate_state(result, cell_stack, um, dt):
    """Cell-state classification + per-state mean speed."""
    from core.cell_state import (
        classify_track_states, state_fraction,
        STATE_BALLED, STATE_ATTACHED, STATE_TRANSITIONAL)
    try:
        from core.motility_state import state_speeds
        from core.tracking import extract_centroids
    except Exception:
        state_speeds = None
        extract_centroids = None
    sd = classify_track_states(cell_stack.astype(bool))
    states = np.asarray(sd["states"])
    result["state_per_frame"]         = states.tolist()
    result["state_frac_balled"]       = float(state_fraction(states, STATE_BALLED))
    result["state_frac_attached"]     = float(state_fraction(states, STATE_ATTACHED))
    result["state_frac_transitional"] = float(state_fraction(
        states, STATE_TRANSITIONAL))
    if state_speeds is None or extract_centroids is None:
        return
    cents = extract_centroids(cell_stack.astype(bool))
    for state, prefix in ((STATE_BALLED, "balled"),
                           (STATE_ATTACHED, "attached")):
        try:
            sp = state_speeds(cents, states, state, um, dt)
            vals = [v for v in sp if v is not None and np.isfinite(v)]
            result[f"mean_speed_{prefix}"] = (
                float(np.mean(vals)) if vals else None)
        except Exception:
            result[f"mean_speed_{prefix}"] = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-state", action="store_true",
                    help="skip cell-state classification (faster)")
    args = ap.parse_args()

    inv = inventory_drive()
    if args.label not in inv:
        print(f"ERROR: {args.label} not in drive inventory", file=sys.stderr)
        return 2
    info    = inv[args.label]
    rec_dir = recording_dir(args.label, info["condition"])
    if not os.path.isdir(rec_dir):
        print(f"ERROR: {rec_dir} missing — run detect first", file=sys.stderr)
        return 2
    analysis_out = os.path.join(rec_dir, "analysis.json")
    masks_path   = os.path.join(rec_dir, "pipeline_results", "masks.npz")
    if not os.path.exists(masks_path):
        print(f"ERROR: {masks_path} missing", file=sys.stderr); return 2
    if os.path.exists(analysis_out) and not args.force:
        print(f"[skip] {args.label}: analysis.json present"); return 0

    t0 = time.time()
    print(f"[load] {args.label}")
    from core.io import load_recording
    rec = load_recording(info["video_path"],
                          dic_channel=1, fluo_channel=0)
    um = float(rec.get("um_per_px") or 0.6523)
    dt = float(rec.get("time_interval_min") or 10.0)

    print(f"[masks] reading labels from {masks_path}")
    npz = np.load(masks_path)
    if "labels" in npz.files:
        labels = npz["labels"].astype(np.int32)
    else:
        # fall back to connected-components on the boolean mask
        from scipy.ndimage import label as _cc
        m = npz["masks"].astype(bool)
        labels = np.zeros_like(m, dtype=np.int32)
        for i in range(m.shape[0]):
            l, _ = _cc(m[i])
            labels[i] = l
    tracks = rebuild_tracks_from_labels(labels)
    print(f"  {len(tracks)} cells")

    # Cell-division lineage: fast path via divisions.json sidecar
    # (written by detect_one); recompute only when missing or unreadable.
    # Matches the focused GUI loader's logic — saves ~16 s/recording.
    divisions_path = os.path.join(rec_dir, "pipeline_results",
                                    "divisions.json")
    candidates = []
    loaded_from_sidecar = False
    if os.path.exists(divisions_path):
        try:
            import json as _json
            with open(divisions_path) as f:
                dj = _json.load(f)
            for entry in dj.get("track_lineage", []) or []:
                d = entry.get("track_index")
                if d is not None and 0 <= d < len(tracks):
                    tracks[d]["parent_id"]       = entry.get("parent_track_index")
                    tracks[d]["division_frame"]  = entry.get("division_frame")
                    tracks[d]["division_score"] = entry.get("division_score")
            candidates = dj.get("candidates", []) or []
            n_set = sum(1 for t in tracks
                        if t.get("parent_id") is not None)
            print(f"[divisions] loaded {len(candidates)} candidate(s) "
                  f"from divisions.json ({n_set} lineage links)")
            loaded_from_sidecar = True
        except Exception as e:
            print(f"[divisions] couldn't parse divisions.json ({e}); "
                  f"recomputing")
    if not loaded_from_sidecar:
        print(f"[divisions] annotate_track_lineage (no sidecar)")
        from core.division_annotator import annotate_track_lineage
        candidates, n_set = annotate_track_lineage(
            tracks, labels, um_per_px=um)
        print(f"  {len(candidates)} candidates, {n_set} lineage links")

    print(f"[analyze] per-cell ({len(tracks)} cells)")
    from core.pipeline import analyze_recording
    cell_results = []
    cell_rows    = []
    for i, t in enumerate(tracks):
        try:
            r = analyze_recording(rec, t["stack"])
        except Exception as e:
            print(f"  cell {i+1} analyze failed: {e}", file=sys.stderr)
            continue
        r["cell_id"]    = i + 1
        r["track_info"] = {
            "first_frame":     t["first_frame"],
            "frames_tracked":  int(t["stack"].any(axis=(1, 2)).sum()),
            "parent_id":       t.get("parent_id"),
            "division_frame":  t.get("division_frame"),
            "division_score":  t.get("division_score"),
        }
        if not args.no_state:
            try:
                _annotate_state(r, t["stack"], um, dt)
            except Exception as e:
                print(f"  cell {i+1} state failed: {e}", file=sys.stderr)
        cell_results.append(r)
        cell_rows.append(per_cell_row(r, r["track_info"]))

    summary = aggregate_recording(cell_rows, len(candidates))
    summary["label"]             = args.label
    summary["condition"]         = info["condition"]
    summary["um_per_px"]         = um
    summary["time_interval_min"] = dt
    summary["runtime_seconds"]   = time.time() - t0

    print(f"[save] outputs")
    atomic_write_json(analysis_out, {
        "label":     args.label,
        "condition": info["condition"],
        "n_cells":   len(cell_results),
        "cells":     [_strip(r) for r in cell_results],
    })
    atomic_write_json(
        os.path.join(rec_dir, "recording_summary.json"), summary)
    write_per_cell_csv(cell_rows,
                        os.path.join(rec_dir, "per_cell.csv"))

    # Refresh .cellscope project file (in case it's gone or stale)
    write_cellscope_project(rec_dir, info, {
        "um_per_px":         um,
        "time_interval_min": dt,
        "n_frames":          int(len(rec["frames"])),
    })

    print(f"[done] {args.label} in {time.time()-t0:.0f}s "
          f"({summary['n_cells']} cells, "
          f"{summary['n_divisions']} divisions)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
