"""Apply the safe per-frame area filter to every detected IC295 recording.

Per-frame removes mask instances whose area is < MIN_AREA_PX or
> MAX_AREA_PX. The defaults come from cross-recording analysis on
4 reviewed recordings (71 cells: 16 REMOVED, 11 TRIMMED, 44 kept):
  * MIN_AREA_PX = 947 px — smallest real cell observed; never
    removes a real cell across the 4-recording calibration set
  * MAX_AREA_PX = 15000 px — above the largest real cell (12006 px)
    but below the only observed huge artefact (79855 px)

Backup policy:
  - if pipeline_results/masks_original.npz already exists (recording
    has been manually reviewed), leave it alone — it represents the
    pre-manual-edit baseline. The current masks.npz already has the
    user's edits; this filter applies ON TOP of those edits, removing
    additional area-outlier instances the user may have missed.
  - if no masks_original.npz exists yet, create one from the current
    masks.npz (the raw detection output) before filtering, so the
    user has a pre-filter backup to compare against / roll back to.

Usage:
    python scripts/ic295_apply_safe_filter.py             # all detected
    python scripts/ic295_apply_safe_filter.py --dry-run   # show plan
    python scripts/ic295_apply_safe_filter.py --label Pos2-WT  # one
    python scripts/ic295_apply_safe_filter.py --min 1500 --max 12000
"""
import argparse
import glob
import os
import shutil
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from ic295_common import RECORDINGS_ROOT  # noqa: E402

DEFAULT_MIN = 947
DEFAULT_MAX = 15000


def _filter_one(masks_path, min_area, max_area, dry_run=False):
    """Apply per-frame area filter to masks_path's labels.

    Returns dict with px_removed, n_instances_removed, n_frames_touched,
    n_cells_touched, before_state, after_state.
    """
    z = np.load(masks_path)
    labels = z["labels"].astype(np.int32)
    other = {k: z[k] for k in z.files if k != "labels"}
    n, H, W = labels.shape

    px_removed = 0
    n_instances = 0
    frames_touched = set()
    cells_touched = set()

    out = labels.copy() if not dry_run else None

    for i in range(n):
        frame = labels[i]
        ids = np.unique(frame)
        for cid in ids:
            if cid == 0:
                continue
            mask = (frame == cid)
            area = int(mask.sum())
            if area < min_area or area > max_area:
                px_removed += area
                n_instances += 1
                frames_touched.add(i)
                cells_touched.add(int(cid))
                if not dry_run:
                    out[i][mask] = 0

    # Regenerate the boolean "masks" foreground array if it was in the file
    masks_fg = None
    if "masks" in other and not dry_run:
        masks_fg = (out > 0).astype(bool)

    if not dry_run and px_removed > 0:
        tmp = masks_path + ".tmp.npz"
        kwargs = {"labels": out}
        if masks_fg is not None:
            kwargs["masks"] = masks_fg
        # Preserve other keys (fusion_source_stack, etc.)
        for k, v in other.items():
            if k not in kwargs:
                kwargs[k] = v
        np.savez_compressed(tmp, **kwargs)
        os.replace(tmp, masks_path)

    return {
        "px_removed": px_removed,
        "instances_removed": n_instances,
        "frames_touched": len(frames_touched),
        "cells_touched": len(cells_touched),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min", type=int, default=DEFAULT_MIN,
                    help=f"per-frame min area in px (default {DEFAULT_MIN})")
    ap.add_argument("--max", type=int, default=DEFAULT_MAX,
                    help=f"per-frame max area in px (default {DEFAULT_MAX})")
    ap.add_argument("--label", default=None,
                    help="single label (default: all detected)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would change without writing")
    args = ap.parse_args()

    # Discover targets
    targets = []
    for path in sorted(glob.glob(os.path.join(
            RECORDINGS_ROOT, "*", "*", "pipeline_results", "masks.npz"))):
        rec_dir = os.path.dirname(os.path.dirname(path))
        label = os.path.basename(rec_dir)
        if args.label and label != args.label:
            continue
        targets.append((label, rec_dir, path))

    if not targets:
        sys.exit("no recordings found")

    print(f"=== safe filter pass: min={args.min} max={args.max} ===")
    print(f"  {'recording':>12s}  {'backup':>8s}  "
          f"{'px removed':>11s}  {'inst':>5s}  {'frames':>6s}  "
          f"{'cells':>5s}")
    print("-" * 70)

    t0 = time.time()
    tot_px = tot_inst = tot_frames = 0
    n_with_changes = 0

    for label, rec_dir, masks_path in targets:
        pr = os.path.dirname(masks_path)
        orig_path = os.path.join(pr, "masks_original.npz")
        backup_action = "exists"
        if not os.path.exists(orig_path):
            if not args.dry_run:
                shutil.copy2(masks_path, orig_path)
            backup_action = "created"

        result = _filter_one(masks_path, args.min, args.max,
                             dry_run=args.dry_run)
        marker = " " if result["px_removed"] > 0 else "."
        print(f" {marker}{label:>12s}  {backup_action:>8s}  "
              f"{result['px_removed']:>11,d}  "
              f"{result['instances_removed']:>5d}  "
              f"{result['frames_touched']:>6d}  "
              f"{result['cells_touched']:>5d}")
        tot_px += result["px_removed"]
        tot_inst += result["instances_removed"]
        tot_frames += result["frames_touched"]
        if result["px_removed"] > 0:
            n_with_changes += 1

    print("-" * 70)
    print(f"  {'TOTAL':>12s}            "
          f"{tot_px:>11,d}  {tot_inst:>5d}  {tot_frames:>6d}")
    print(f"  recordings touched: {n_with_changes}/{len(targets)}")
    print(f"  elapsed: {time.time() - t0:.1f} s "
          f"{'(dry-run)' if args.dry_run else ''}")


if __name__ == "__main__":
    sys.exit(main())
