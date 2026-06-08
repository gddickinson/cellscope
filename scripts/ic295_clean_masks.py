"""Clean every recording's stored masks.npz: fill holes + remove specks.

Applies core.mask_cleanup.clean_cell_mask to each cell in each frame and
rewrites masks.npz, so the GUI and exports also see fixed masks (the
non-destructive measurement-time cleaning only fixed the metrics). Each
cell is reduced to its single largest connected component (every other
speck / mislabelled region removed) and its enclosed holes filled. Each
recording is backed up to masks_precleanup.npz first (idempotent — never
overwritten), and masks.npz is written atomically. Holes are only filled
into background (never stolen from a neighbouring cell).

Usage:
  conda run -n cellpose4 python scripts/ic295_clean_masks.py            # all
  conda run -n cellpose4 python scripts/ic295_clean_masks.py Pos7-WT    # one
  conda run -n cellpose4 python scripts/ic295_clean_masks.py --dry-run
"""
import os
import sys
import glob
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import RECORDINGS_ROOT  # noqa: E402
from core.mask_cleanup import clean_cell_mask  # noqa: E402
import numpy as np  # noqa: E402


def clean_recording(mp):
    """Return (new_labels, n_holes_filled, n_specks_removed, n_frames_touched)."""
    labels = np.load(mp)["labels"].astype(np.int32)
    new = labels.copy()
    holes = specks = touched = 0
    for fi in range(len(labels)):
        lf = labels[fi]
        ids = [int(v) for v in np.unique(lf) if v > 0]
        frame_changed = False
        for cid in ids:
            orig = lf == cid
            rr = np.any(orig, axis=1); cc = np.any(orig, axis=0)
            r0, r1 = np.where(rr)[0][[0, -1]]
            c0, c1 = np.where(cc)[0][[0, -1]]
            sub = orig[r0:r1 + 1, c0:c1 + 1]
            cleaned = clean_cell_mask(sub)
            if np.array_equal(cleaned, sub):
                continue
            sub_new = new[fi, r0:r1 + 1, c0:c1 + 1]
            removed = sub & ~cleaned            # specks → drop
            added = cleaned & ~sub              # hole fill (only into bg)
            sub_new[removed] = 0
            fill = added & (sub_new == 0)
            sub_new[fill] = cid
            specks += int(removed.sum()); holes += int(fill.sum())
            frame_changed = True
        touched += int(frame_changed)
    return new, holes, specks, touched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(
        RECORDINGS_ROOT, "*", "*", "pipeline_results", "masks.npz")))
    if args.label:
        paths = [p for p in paths
                 if os.path.basename(os.path.dirname(os.path.dirname(p)))
                 == args.label]
    tot_h = tot_s = tot_changed = 0
    for i, mp in enumerate(paths):
        label = os.path.basename(os.path.dirname(os.path.dirname(mp)))
        new, holes, specks, touched = clean_recording(mp)
        if holes or specks:
            tot_h += holes; tot_s += specks; tot_changed += 1
            print(f"  [{i+1}/{len(paths)}] {label}: filled {holes} hole-px, "
                  f"removed {specks} speck-px across {touched} frames"
                  + ("  (dry-run)" if args.dry_run else ""), flush=True)
            if not args.dry_run:
                bk = os.path.join(os.path.dirname(mp), "masks_precleanup.npz")
                if not os.path.exists(bk):
                    shutil.copy2(mp, bk)
                tmp = mp + ".tmp"   # savez_compressed appends ".npz"
                np.savez_compressed(tmp, labels=new.astype(np.int32))
                os.replace(tmp + ".npz", mp)
        else:
            print(f"  [{i+1}/{len(paths)}] {label}: clean already", flush=True)
    print(f"\n{tot_changed}/{len(paths)} recordings cleaned: "
          f"{tot_h} hole-px filled, {tot_s} speck-px removed"
          + ("  (dry-run — nothing written)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
