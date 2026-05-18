"""Create a tarball of every GT mask in the repo + the index.

Run periodically to protect against accidental deletion.

Output:
  data/gt_backups/gt_<YYYY-MM-DD>_<HHMMSS>.tar.gz

The tarball contains:
  data/GT_INDEX.md           ← so anyone unpacking it can see what they got
  data/gt_index.json
  data/ic295_gt_full/*/gt_masks/*.png
  data/ic295_gt_full/*/GT_FRAMES.txt
  data/ic295_gt_full/*/<rec>.ome.json    ← scale metadata
  data/legacy_gt/*/gt_masks/*.png
  data/legacy_gt/*/GT_FRAMES.txt
  data/legacy_gt/*/<rec>.json

NOT included: source .ome.tif files (too large + can be re-fetched
from /Volumes/GeorgeDrive). Existing pipeline_results/ + evaluation/
folders are also skipped — they can be regenerated.

Run:  python scripts/backup_gt.py
"""
import os
import sys
import time
import tarfile
import logging

CELLSCOPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("backup_gt")

INCLUDE_PATTERNS = [
    "data/GT_INDEX.md",
    "data/gt_index.json",
]
GT_ROOTS = ["data/ic295_gt_full", "data/legacy_gt"]
# Filename-suffix allow-list — only these go into the tarball.
INCLUDE_SUFFIXES = (".png", ".txt", ".json", ".md")


def iter_files():
    for root in GT_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            # Skip pipeline + evaluation outputs
            if ("pipeline_results" in dirpath
                    or "evaluation" in dirpath):
                continue
            for f in filenames:
                if not f.endswith(INCLUDE_SUFFIXES):
                    continue
                yield os.path.join(dirpath, f)
    for f in INCLUDE_PATTERNS:
        if os.path.exists(f):
            yield f


def main():
    # First, regenerate the index so the tarball ships with current
    # state. If GT_INDEX.md is missing it'd be a confusing backup.
    if not os.path.exists("data/GT_INDEX.md"):
        log.info("Generating fresh GT index …")
        os.system("python scripts/audit_gt.py")

    out_dir = "data/gt_backups"
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out_path = os.path.join(out_dir, f"gt_{stamp}.tar.gz")

    n = 0
    with tarfile.open(out_path, "w:gz") as tf:
        for f in sorted(set(iter_files())):
            tf.add(f)
            n += 1
            if n % 50 == 0:
                log.info("  packed %d files", n)
    size_kb = os.path.getsize(out_path) // 1024
    log.info("Wrote %s (%d files, %d KB)", out_path, n, size_kb)
    print(f"\n✓ Backup created: {out_path}")
    print(f"  {n} files, {size_kb} KB")


if __name__ == "__main__":
    main()
