"""Analyze one IC293 crop's masks. Idempotent.

Reads ic293_analysis/by_condition/<cond>/<label>/pipeline_results/masks.npz
(possibly user-edited after detection) and writes, in the recording dir:
  analysis.json            — per-cell results (arrays stripped)
  recording_summary.json   — single-row aggregate for the comparison
  per_cell.csv             — flat per-cell metrics
  RUN_METADATA.{md,json}   — analysis-phase metadata (rule 2)

All metrics are DIC-derived (no Cy5 needed). State (rounded/spread) uses
the IC295 keratinocyte-fit rule — these are endothelial cells, so the
state-split columns are EXPLORATORY (flagged in the comparison report);
the state-agnostic motility metrics are the primary read.

Skips if analysis.json already exists (unless --force). Run from cellpose4.
"""
import os
import sys
import time
import json
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

import numpy as np  # noqa: E402
from scripts.ic293_common import (  # noqa: E402
    inventory_cache, recording_dir, atomic_write_json,
    write_cellscope_project, read_sidecar_meta, select_primary_cell,
    PROJECT_ROOT,
)
from core.cell_metrics_table import (  # noqa: E402
    per_cell_row, aggregate_recording, write_per_cell_csv)
from core.state_analysis import annotate_state  # noqa: E402

_DROP_KEYS = {
    "frames", "masks", "centroids_px", "centroids_um", "trajectory",
    "speed", "msd", "autocorr", "shape_timeseries", "edge_angles",
    "edge_velocity", "boundary_confidence", "area_stability",
    "state_per_frame",
}


def _strip(d):
    return {k: v for k, v in d.items() if k not in _DROP_KEYS}


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
        tracks.append({"stack": stack,
                       "first_frame": int(np.argmax(present))})
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-state", action="store_true",
                    help="skip cell-state classification")
    args = ap.parse_args()

    inv = inventory_cache()
    if args.label not in inv:
        print(f"ERROR: {args.label} not in IC293 cache", file=sys.stderr)
        return 2
    info = inv[args.label]
    rec_dir = recording_dir(args.label, info["condition"])
    analysis_out = os.path.join(rec_dir, "analysis.json")
    masks_path = os.path.join(rec_dir, "pipeline_results", "masks.npz")
    if not os.path.exists(masks_path):
        print(f"ERROR: {masks_path} missing — run detect first",
              file=sys.stderr)
        return 2
    if os.path.exists(analysis_out) and not args.force:
        print(f"[skip] {args.label}: analysis.json present")
        return 0

    t0 = time.time()
    print(f"[load] {args.label}")
    from core.io import load_recording
    rec = load_recording(info["video_path"])          # single-channel
    meta = read_sidecar_meta(info)
    um = float(rec.get("um_per_px") or meta["um_per_px"])
    dt = float(rec.get("time_interval_min") or meta["time_interval_min"])

    npz = np.load(masks_path)
    if "labels" in npz.files:
        labels = npz["labels"].astype(np.int32)
    else:
        from scipy.ndimage import label as _cc
        m = npz["masks"].astype(bool)
        labels = np.zeros_like(m, dtype=np.int32)
        for i in range(m.shape[0]):
            l, _ = _cc(m[i])
            labels[i] = l
    # Single-cell selection: each crop is hand-cropped to ONE target cell;
    # any extra detections are artifacts (debris/fragments). Keep the one
    # real cell (persistent × substantial × central — see select_primary_cell)
    # and drop the rest. masks.npz is left untouched so the artifacts remain
    # visible/editable in the focused GUI review.
    primary_id, cands, ambiguous = select_primary_cell(labels)
    n_objects = len(cands)
    primary_presence = float(cands[0]["presence"]) if cands else 0.0
    if primary_id is None:
        print("  WARNING: no objects detected in masks", file=sys.stderr)
    elif n_objects > 1:
        labels = np.where(labels == primary_id, 1, 0).astype(np.int32)
        flag = "  [AMBIGUOUS — review in GUI]" if ambiguous else ""
        print(f"  single-cell select: kept object {primary_id} of {n_objects} "
              f"→ dropped {n_objects - 1} artifact(s){flag}")
    # Ground truth: the target cell is present in EVERY frame. If the kept
    # cell isn't (detection/gap-fill missed frames, or it left the FOV),
    # flag it — its track is incomplete and may need a GUI fix.
    if primary_id is not None and primary_presence < 0.95:
        print(f"  WARNING: kept cell present in only {primary_presence:.0%} of "
              f"frames (expected ~100%) — review", file=sys.stderr)
    tracks = rebuild_tracks_from_labels(labels)
    print(f"  {len(tracks)} cell after single-cell selection "
          f"(presence {primary_presence:.0%})")

    # Divisions are not analysed for these single-cell crops (one target
    # cell per crop; extra detections are artifacts, already dropped above).
    candidates = []

    print(f"[analyze] per-cell ({len(tracks)} cells)")
    from core.pipeline import analyze_recording
    cell_results, cell_rows = [], []
    for i, t in enumerate(tracks):
        try:
            r = analyze_recording(rec, t["stack"])
        except Exception as e:
            print(f"  cell {i+1} analyze failed: {e}", file=sys.stderr)
            continue
        r["cell_id"] = i + 1
        r["track_info"] = {
            "first_frame":    t["first_frame"],
            "frames_tracked": int(t["stack"].any(axis=(1, 2)).sum()),
            "parent_id":      t.get("parent_id"),
            "division_frame": t.get("division_frame"),
            "division_score": t.get("division_score"),
        }
        if not args.no_state:
            try:
                annotate_state(r, t["stack"], um, dt)
            except Exception as e:
                print(f"  cell {i+1} state failed: {e}", file=sys.stderr)
        cell_results.append(r)
        cell_rows.append(per_cell_row(r, r["track_info"]))

    summary = aggregate_recording(cell_rows, len(candidates))
    summary.update({"label": args.label, "condition": info["condition"],
                    "position": info["position"], "um_per_px": um,
                    "time_interval_min": dt,
                    "n_objects_detected": n_objects,
                    "n_artifacts_dropped": max(n_objects - 1, 0),
                    "selection_ambiguous": ambiguous,
                    "primary_cell_presence": primary_presence,
                    "runtime_seconds": time.time() - t0})

    print("[save] outputs")
    atomic_write_json(analysis_out, {
        "label": args.label, "condition": info["condition"],
        "position": info["position"], "n_cells": len(cell_results),
        "cells": [_strip(r) for r in cell_results]})
    atomic_write_json(os.path.join(rec_dir, "recording_summary.json"), summary)
    write_per_cell_csv(cell_rows, os.path.join(rec_dir, "per_cell.csv"))
    write_cellscope_project(rec_dir, info, {
        "um_per_px": um, "time_interval_min": dt,
        "n_frames": int(len(rec["frames"]))})

    # RUN_METADATA for the analysis phase (rule 2). Written in rec_dir so it
    # doesn't clobber the detect-phase metadata under pipeline_results/.
    try:
        from core.run_metadata import write_run_metadata
        from core.cell_state import DEFAULT_THRESHOLDS
        rec_for_meta = {"video_path": info["video_path"], "name": args.label,
                        "frames": rec["frames"], "cy5_frames": None,
                        "um_per_px": um, "time_interval_min": dt}
        write_run_metadata(
            rec_dir,
            recording=rec_for_meta,
            params={"state_thresholds": dict(DEFAULT_THRESHOLDS),
                    "um_per_px": um, "time_interval_min": dt},
            detect_result={"tracks": tracks},
            analysis_results=cell_results,
            pipeline_function="pipeline.analyze_recording + state_analysis.annotate_state",
            mode="single",
            runtime_seconds=time.time() - t0,
            rerun_command=(f"conda run -n cellpose4 python "
                           f"scripts/ic293_analyze_one.py {args.label}"),
            extra={"dataset": "IC293", "phase": "analyze",
                   "n_cells": len(cell_results),
                   "n_divisions": int(summary.get("n_divisions", 0)),
                   "state_rule": "IC295 keratinocyte-fit (EXPLORATORY on EC)"},
            project_root=PROJECT_ROOT)
    except Exception as e:
        print(f"  WARN: RUN_METADATA write failed: {e}", file=sys.stderr)

    print(f"[done] {args.label} in {time.time()-t0:.0f}s "
          f"({summary['n_cells']} cells, {summary['n_divisions']} divisions)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
