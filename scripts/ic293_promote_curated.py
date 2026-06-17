"""Promote curated single-cell masks to canonical (idempotent).

For every analysis recording with a `pipeline_results/masks_curated.npz`,
back up the original detection ONCE to `masks_pre_curation.npz`, then copy
the curated masks over `masks.npz` so the analysis + GUI + .cellscope all
use the curated single-cell masks. Recordings in ANALYSIS_EXCLUDE are
skipped (kept untouched on disk, excluded from analysis).

Reverses cleanly: copy masks_pre_curation.npz back over masks.npz.

Usage: conda run -n cellpose4 python scripts/ic293_promote_curated.py
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic293_common import (  # noqa: E402
    inventory_cache, recording_dir, ANALYSIS_EXCLUDE)


def main():
    inv = inventory_cache()
    promoted = skipped_excluded = no_curated = 0
    for label, info in sorted(inv.items()):
        if label in ANALYSIS_EXCLUDE:
            skipped_excluded += 1
            continue
        pr = os.path.join(recording_dir(label, info["condition"]),
                          "pipeline_results")
        masks = os.path.join(pr, "masks.npz")
        curated = os.path.join(pr, "masks_curated.npz")
        backup = os.path.join(pr, "masks_pre_curation.npz")
        if not os.path.exists(curated):
            no_curated += 1
            continue
        if not os.path.exists(backup):          # preserve the ORIGINAL once
            shutil.copy2(masks, backup)
        shutil.copy2(curated, masks)
        promoted += 1
    print(f"Promoted {promoted} recordings (curated → masks.npz, "
          f"masks_pre_curation.npz backups kept).")
    print(f"Skipped {skipped_excluded} excluded-from-analysis "
          f"({', '.join(sorted(ANALYSIS_EXCLUDE))}); {no_curated} had no curated masks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
