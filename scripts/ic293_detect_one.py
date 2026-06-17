"""Detect one IC293 single-cell crop. Idempotent.

These are DIC-only crops (no Cy5), so detection runs single-channel:
`load_recording(path)` with no channel args yields cy5_frames=None, and
`detect_recording(..., cy5_frames=None)` skips every Cy5 step (alignment,
annotation, persistence-guard) behind its `if has_cy5` guards. The
auto-selector picks cpsam_dic for these 1-cell scenes (p75 < 1.5 →
tighter boundaries on isolated cells).

Writes into ic293_analysis/by_condition/<cond>/<label>/:
  pipeline_results/masks.npz
  pipeline_results/divisions.json
  pipeline_results/RUN_METADATA.{md,json}   (rule 2 — canonical writer)
  <label>.cellscope                          (drag into focused GUI)

If pipeline_results/masks.npz already exists, exits 0 with no work.
Run from the cellpose4 env (cpsam_dic needs cellpose 4.x):
  conda run -n cellpose4 python scripts/ic293_detect_one.py Pos0-WT
"""
import os
import sys
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic293_common import (  # noqa: E402
    inventory_cache, setup_recording_folder, write_cellscope_project,
    read_sidecar_meta, atomic_write_json, PROJECT_ROOT,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="e.g. Pos0-WT")
    ap.add_argument("--force", action="store_true",
                    help="re-detect even if masks.npz exists")
    args = ap.parse_args()

    inv = inventory_cache()
    if args.label not in inv:
        print(f"ERROR: {args.label} not found in IC293 cache inventory",
              file=sys.stderr)
        return 2
    info = inv[args.label]

    rec_dir = setup_recording_folder(info)
    pr_dir = os.path.join(rec_dir, "pipeline_results")
    masks_out = os.path.join(pr_dir, "masks.npz")
    if os.path.exists(masks_out) and not args.force:
        print(f"[skip] {args.label}: masks.npz already present")
        return 0

    from core.io import load_recording
    from core.unified_detection import detect_recording
    from core.run_metadata import write_run_metadata
    import numpy as np

    t0 = time.time()
    print(f"[load] {info['video_path']}")
    # NO channel args → single-channel path: frames=(T,H,W), cy5_frames=None.
    rec = load_recording(info["video_path"])
    frames = rec["frames"]
    cy5 = rec.get("cy5_frames")            # None for these DIC-only crops
    meta = read_sidecar_meta(info)
    um = float(rec.get("um_per_px") or meta["um_per_px"])
    dt = float(rec.get("time_interval_min") or meta["time_interval_min"])
    nf = int(len(frames))
    rec.setdefault("name", info["label"])
    print(f"  {nf} frames, {um:.4f} um/px, dt={dt} min, "
          f"Cy5={'yes' if cy5 is not None else 'no (DIC-only)'}")
    if cy5 is not None:
        print("  WARNING: expected single-channel DIC but got a Cy5 channel; "
              "proceeding single-channel anyway", file=sys.stderr)
        cy5 = None

    print("[detect] unified detect_recording (auto backbone, no Cy5)...")
    t_det = time.time()
    result = detect_recording(
        frames, cy5_frames=None,
        um_per_px=um, time_interval_min=dt,
        pipeline_kind="auto",          # → cpsam_dic for single-cell scenes
        run_cy5_filter=False,          # no Cy5 to filter on
        # all other knobs default to DEFAULTS via None
    )
    det_s = time.time() - t_det
    n_tracks = len(result.get("tracks", []) or [])
    n_div = len(result.get("divisions", []) or [])
    auto = result.get("auto") or {}        # unified_detection stores it here
    print(f"  detection done in {det_s:.0f}s: {n_tracks} tracks, "
          f"{n_div} division candidates"
          + (f"; backbone={auto.get('pipeline_kind')}" if auto else ""))

    # ───────── save masks ─────────
    os.makedirs(pr_dir, exist_ok=True)
    save = {}
    if result.get("masks") is not None:
        save["masks"] = result["masks"].astype(bool)
    if result.get("labels") is not None:
        save["labels"] = result["labels"].astype(np.int32)
    if result.get("fusion_source_stack") is not None:
        save["fusion_source_stack"] = result["fusion_source_stack"].astype(np.uint8)
    np.savez_compressed(masks_out, **save)
    print(f"  wrote {masks_out}")

    # ───────── divisions.json ─────────
    tracks_out = result.get("tracks", []) or []
    lineage = [{
        "track_index":        i,
        "parent_track_index": t.get("parent_id"),
        "division_frame":     t.get("division_frame"),
        "division_score":     t.get("division_score"),
    } for i, t in enumerate(tracks_out) if t.get("parent_id") is not None]
    atomic_write_json(os.path.join(pr_dir, "divisions.json"), {
        "n_candidates":  len(result.get("divisions", []) or []),
        "candidates":    result.get("divisions", []) or [],
        "track_lineage": lineage,
    })

    # ───────── RUN_METADATA.{md,json} (rule 2, canonical writer) ─────────
    rec_for_meta = {"video_path": info["video_path"], "name": info["label"],
                    "frames": frames, "cy5_frames": None,
                    "um_per_px": um, "time_interval_min": dt}
    write_run_metadata(
        pr_dir,
        recording=rec_for_meta,
        params=result.get("params", {}) or {},
        detect_result=result,
        pipeline_function="unified_detection.detect_recording (auto)",
        mode="single",
        runtime_seconds=time.time() - t0,
        rerun_command=(f"conda run -n cellpose4 python "
                       f"scripts/ic293_detect_one.py {info['label']}"),
        extra={"dataset": "IC293", "auto_select": auto,
               "gap_fill_stats": result.get("gap_fill_stats"),
               "n_divisions": n_div},
        project_root=PROJECT_ROOT,
    )

    write_cellscope_project(rec_dir, info,
                            {"um_per_px": um, "time_interval_min": dt,
                             "n_frames": nf})
    print(f"[done] {info['label']} in {time.time()-t0:.0f}s "
          f"({n_tracks} tracks, {n_div} divisions)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
