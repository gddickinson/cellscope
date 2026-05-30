"""Detect one IC295 recording. Idempotent.

Writes into ic295_analysis/by_condition/<cond>/<label>/:
  pipeline_results/masks.npz          + fusion_diagnostic.png (when produced)
  divisions.json                       (lineage from annotate_track_lineage)
  RUN_METADATA.json                    (reproducibility — env + params + git)
  <label>.cellscope                    (project file: drag into focused GUI)

Plus, idempotent housekeeping (always re-runs cheaply):
  symlink to drive .ome.tif + _metadata.txt
  copy of .ome.json sidecar

If pipeline_results/masks.npz already exists, exits 0 with no work.
Called by ic295_batch.py per recording, but runnable standalone:
  python scripts/ic295_detect_one.py Pos21-KO
"""
import os
import sys
import json
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (
    inventory_drive, setup_recording_folder, write_cellscope_project,
    atomic_write_json,
)


def _read_sidecar_meta(info):
    """Return {um_per_px, time_interval_min, n_frames} from the .ome.json
    sidecar — sensible IC295 defaults if it's missing or partial."""
    out = {"um_per_px": 0.6523, "time_interval_min": 10.0, "n_frames": 97}
    sc = info.get("json_sidecar")
    if sc and os.path.exists(sc):
        try:
            with open(sc) as f:
                d = json.load(f)
            out["um_per_px"] = float(d.get("um_per_px") or out["um_per_px"])
            out["time_interval_min"] = float(
                d.get("time_interval_min") or out["time_interval_min"])
            if d.get("n_frames"):
                out["n_frames"] = int(d["n_frames"])
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="e.g. Pos21-KO")
    ap.add_argument("--force", action="store_true",
                    help="re-detect even if masks.npz exists")
    ap.add_argument("--detect-anyway", action="store_true",
                    help="run fresh detection even if the drive already "
                    "has masks (ignore the adopt-from-drive shortcut)")
    args = ap.parse_args()

    inv = inventory_drive()
    if args.label not in inv:
        print(f"ERROR: {args.label} not found in drive inventory",
              file=sys.stderr)
        return 2
    info = inv[args.label]

    rec_dir = setup_recording_folder(info)
    masks_out = os.path.join(rec_dir, "pipeline_results", "masks.npz")

    if os.path.exists(masks_out) and not args.force:
        print(f"[skip] {args.label}: masks.npz already present at "
              f"{masks_out}")
        return 0

    # Adopt path: drive already has detection (mini's run), copy it
    # into the analysis folder instead of re-detecting from scratch.
    # Copy (not symlink) so user mask edits don't overwrite the
    # canonical drive run.
    if (os.path.exists(info["drive_masks"]) and not args.force
            and not args.detect_anyway):
        import shutil
        pr_dir = os.path.join(rec_dir, "pipeline_results")
        os.makedirs(pr_dir, exist_ok=True)
        shutil.copy2(info["drive_masks"], masks_out)
        # Adjacent sidecars: copy everything the mini's batch wrote so
        # the recording folder is fully self-contained (the mini wrote
        # not just masks but also masks_unfiltered, filter_decisions,
        # per_cell.csv, analysis_summary, RUN_METADATA — keeping them
        # alongside is useful for QC + cross-check vs Phase-2 outputs).
        drive_pr = os.path.dirname(info["drive_masks"])
        for f in ("fusion_diagnostic.png", "divisions.json",
                  "RUN_METADATA.json", "RUN_METADATA.md",
                  "masks_unfiltered.npz", "filter_decisions.json",
                  "analysis_summary.json", "run_summary.txt",
                  "per_cell.csv"):
            src = os.path.join(drive_pr, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(pr_dir, f))
        # Read um_per_px / dt / n_frames from the .ome.json sidecar
        # so the .cellscope project has correct values.
        rec_meta = _read_sidecar_meta(info)
        write_cellscope_project(rec_dir, info, rec_meta)
        print(f"[adopt] {args.label}: copied existing drive masks "
              f"({os.path.getsize(masks_out)//1024} KB) — no re-detect")
        return 0

    # ───────── load + detect ─────────
    from core.io import load_recording
    from core.unified_detection import detect_recording
    from output.run_metadata import write_run_metadata

    t0 = time.time()
    print(f"[load] {info['video_path']}")
    rec = load_recording(info["video_path"],
                          dic_channel=1, fluo_channel=0)
    frames = rec["frames"]
    cy5    = rec.get("cy5_frames")
    um     = rec.get("um_per_px", 0.6523)
    dt     = rec.get("time_interval_min", 10.0)
    nf     = int(len(frames))
    print(f"  {nf} frames, {um:.4f} um/px, dt={dt} min, "
          f"Cy5={'yes' if cy5 is not None else 'no'}")

    print(f"[detect] running unified detect_recording with defaults...")
    t_det = time.time()
    result = detect_recording(
        frames, cy5_frames=cy5,
        um_per_px=um, time_interval_min=dt,
        pipeline_kind="auto",        # auto-select cpsam_dic vs raw cpsam
        run_cy5_filter=True,
        # All other knobs default to DEFAULTS via None
    )
    det_s = time.time() - t_det
    n_tracks = len(result.get("tracks", []) or [])
    n_div    = len(result.get("divisions", []) or [])
    print(f"  detection done in {det_s:.0f}s: "
          f"{n_tracks} tracks, {n_div} division candidates")

    # ───────── save outputs (atomic where possible) ─────────
    pr_dir = os.path.join(rec_dir, "pipeline_results")
    os.makedirs(pr_dir, exist_ok=True)

    import numpy as np
    masks_arr  = result.get("masks")
    labels_arr = result.get("labels")
    fusion_src = result.get("fusion_source_stack")

    save = {}
    if masks_arr is not None:
        save["masks"] = masks_arr.astype(bool)
    if labels_arr is not None:
        save["labels"] = labels_arr.astype(np.int32)
    if fusion_src is not None:
        save["fusion_source_stack"] = fusion_src.astype(np.uint8)
    np.savez_compressed(masks_out, **save)
    print(f"  wrote {masks_out}")

    # divisions.json (candidates + lineage table)
    divisions  = result.get("divisions", []) or []
    tracks_out = result.get("tracks", []) or []
    lineage = [{
        "track_index":         i,
        "parent_track_index":  t.get("parent_id"),
        "division_frame":      t.get("division_frame"),
        "division_score":      t.get("division_score"),
    } for i, t in enumerate(tracks_out)
       if t.get("parent_id") is not None]
    atomic_write_json(os.path.join(pr_dir, "divisions.json"), {
        "n_candidates":  len(divisions),
        "candidates":    divisions,
        "track_lineage": lineage,
    })

    # RUN_METADATA — captures env + git + params + diff vs DEFAULTS
    try:
        write_run_metadata(
            pr_dir,
            title=f"IC295 detect: {info['label']}",
            recording={"video_path": info["video_path"],
                       "n_frames": nf,
                       "um_per_px": um,
                       "time_interval_min": dt},
            params=result.get("params", {}),
            pipeline=f"unified_detection.detect_recording (auto)",
            runtime_seconds=time.time() - t0,
            rerun_cli=(f"python scripts/ic295_detect_one.py "
                       f"{info['label']}"),
        )
    except TypeError:
        # The helper's signature may differ from this guess; fall back
        # to a tiny JSON so the run is still self-describing.
        atomic_write_json(os.path.join(pr_dir, "RUN_METADATA.json"), {
            "label":           info["label"],
            "video_path":      info["video_path"],
            "n_frames":        nf,
            "um_per_px":       um,
            "time_interval_min": dt,
            "pipeline":        "unified_detection.detect_recording (auto)",
            "runtime_seconds": time.time() - t0,
            "n_tracks":        n_tracks,
            "n_divisions":     n_div,
        })

    # .cellscope project file — drag into focused GUI to review / edit
    rec_meta = {"um_per_px": um, "time_interval_min": dt, "n_frames": nf}
    write_cellscope_project(rec_dir, info, rec_meta)
    print(f"  wrote {info['label']}.cellscope")

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
