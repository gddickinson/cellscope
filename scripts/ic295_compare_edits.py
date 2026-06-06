"""Re-analyse the original (pre-edit) masks for any IC295 recording
that has a masks_original.npz alongside the current masks.npz, and
save the result as recording_summary_original.json beside the
existing (edited) recording_summary.json.

Lets us answer "how did the manual edits change the downstream
analysis numbers?" without disturbing the live pipeline state:
- backs up the current masks + summary + analysis.json
- swaps in masks_original.npz
- runs ic295_analyze_one.py (forces re-analysis)
- saves the resulting summary to *_original.json
- restores the edited state byte-for-byte

Usage:
    python scripts/ic295_compare_edits.py             # all flagged
    python scripts/ic295_compare_edits.py Pos60-DMSO  # single label

Skips a recording if either masks_original.npz is missing or the
*_original.json output already exists.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from ic295_common import RECORDINGS_ROOT  # noqa: E402


def _swap_and_analyse(label, rec_dir):
    pr = os.path.join(rec_dir, "pipeline_results")
    masks_current = os.path.join(pr, "masks.npz")
    masks_orig = os.path.join(pr, "masks_original.npz")
    summary_current = os.path.join(rec_dir, "recording_summary.json")
    summary_orig = os.path.join(rec_dir, "recording_summary_original.json")
    analysis_current = os.path.join(rec_dir, "analysis.json")
    per_cell_current = os.path.join(rec_dir, "per_cell.csv")

    if not os.path.exists(masks_orig):
        print(f"  skip: no masks_original.npz")
        return False
    if os.path.exists(summary_orig):
        print(f"  skip: recording_summary_original.json already present")
        return True

    backups = []

    def _backup(path):
        if os.path.exists(path):
            bk = path + ".reanalysis_backup"
            shutil.copy2(path, bk)
            backups.append((path, bk))

    try:
        # Backup edited state
        _backup(masks_current)
        _backup(summary_current)
        _backup(analysis_current)
        _backup(per_cell_current)

        # Swap original masks into place
        shutil.copy2(masks_orig, masks_current)

        # Run analyse (forces a fresh pass since summary exists)
        cmd = ["conda", "run", "-n", "cellpose4", "python",
               "scripts/ic295_analyze_one.py", label, "--force"]
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
        if rc != 0:
            print(f"  ✗ analyse failed (rc={rc})")
            return False

        # Save the original-summary to its side location
        if os.path.exists(summary_current):
            shutil.copy2(summary_current, summary_orig)
            print(f"  ✓ wrote {os.path.basename(summary_orig)}")
        return True
    finally:
        # Restore edited state
        for original, bk in backups:
            shutil.move(bk, original)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", default=None,
                    help="single label; default = all recordings "
                    "that have a masks_original.npz")
    args = ap.parse_args()

    todo = []
    if args.label:
        for cond in os.listdir(RECORDINGS_ROOT):
            cdir = os.path.join(RECORDINGS_ROOT, cond)
            if not os.path.isdir(cdir):
                continue
            ldir = os.path.join(cdir, args.label)
            if os.path.isdir(ldir):
                todo.append((args.label, cond, ldir))
                break
        if not todo:
            sys.exit(f"unknown label: {args.label}")
    else:
        for cond in sorted(os.listdir(RECORDINGS_ROOT)):
            cdir = os.path.join(RECORDINGS_ROOT, cond)
            if not os.path.isdir(cdir):
                continue
            for label in sorted(os.listdir(cdir)):
                ldir = os.path.join(cdir, label)
                pr = os.path.join(ldir, "pipeline_results")
                if (os.path.isdir(ldir)
                        and os.path.exists(
                            os.path.join(pr, "masks_original.npz"))
                        and os.path.exists(
                            os.path.join(pr, "masks.npz"))):
                    todo.append((label, cond, ldir))

    print(f"=== re-analysing {len(todo)} recording(s) on original masks ===")
    n_ok = 0
    for label, cond, rec_dir in todo:
        print(f"\n[{cond}] {label}")
        if _swap_and_analyse(label, rec_dir):
            n_ok += 1
    print(f"\n=== done: {n_ok}/{len(todo)} successful ===")
    return 0 if n_ok == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
