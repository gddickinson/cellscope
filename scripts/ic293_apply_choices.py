"""Apply the reviewer's candidate picks to masks.npz.

Reads ic293_analysis/review/review_choices.csv (written by the candidate-
review GUI, main_review.py) and, for each recording, copies the chosen
candidate stack into pipeline_results/masks.npz:

  original      <- masks_pre_chunkadd.npz   (the curated baseline)
  moderate      <- masks_cand_moderate.npz
  conservative  <- masks_cand_conservative.npz
  sam2          <- masks_cand_sam2.npz

Recordings with no recorded pick keep Original (masks.npz is left as the
curated baseline). Cells flagged for manual brushing are listed so they can
be touched up in the focused GUI; their chosen candidate is still applied so
the brush starts from the best automatic outline.

Records provenance to review/applied_choices.json. NON-DESTRUCTIVE of the
curated masks (masks_pre_chunkadd.npz is never written).

Usage (from cellpose4):
  conda run -n cellpose4 python scripts/ic293_apply_choices.py
  conda run -n cellpose4 python scripts/ic293_apply_choices.py --dry-run
"""
import os
import sys
import csv
import json
import argparse
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic293_common import (  # noqa: E402
    inventory_cache, recording_dir, ANALYSIS_ROOT)

REVIEW_DIR = os.path.join(ANALYSIS_ROOT, "review")
SRC = {"original": "masks_pre_chunkadd.npz",
       "moderate": "masks_cand_moderate.npz",
       "conservative": "masks_cand_conservative.npz",
       "sam2": "masks_cand_sam2.npz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--choices", default=os.path.join(REVIEW_DIR, "review_choices.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.choices):
        print(f"no choices file at {args.choices} — review in main_review.py first")
        return 1
    # group per-frame picks by label; frame == -1 / choice 'manual' = manual flag
    frame_picks, manual = {}, set()
    with open(args.choices) as f:
        for r in csv.DictReader(f):
            label = r["label"]
            try:
                fr = int(r["frame"])
            except (TypeError, ValueError, KeyError):
                continue
            choice = (r.get("choice") or "original").strip()
            if fr < 0 or choice == "manual":
                manual.add(label)
            elif choice != "original":
                frame_picks.setdefault(label, {})[fr] = choice

    inv = inventory_cache()
    counts = Counter()
    applied, missing = [], []
    for label, picks in sorted(frame_picks.items()):
        if label not in inv:
            missing.append(label); continue
        pr = os.path.join(recording_dir(label, inv[label]["condition"]),
                          "pipeline_results")
        base_p = os.path.join(pr, SRC["original"])
        if not os.path.exists(base_p):
            missing.append(label); continue
        base = np.load(base_p)["labels"].copy()
        cand_cache, ok = {}, {}
        for fr, choice in picks.items():
            if choice not in cand_cache:
                cp = os.path.join(pr, SRC.get(choice, ""))
                cand_cache[choice] = (np.load(cp)["labels"]
                                      if os.path.exists(cp) else None)
            cand = cand_cache[choice]
            if cand is None or fr >= len(base):
                continue
            base[fr] = cand[fr]                  # graft this frame's chosen mask
            counts[choice] += 1
            ok[fr] = choice
        if not ok:
            continue
        if not args.dry_run:
            np.savez_compressed(os.path.join(pr, "masks.npz"),
                                labels=base, masks=(base > 0))
        applied.append({"label": label, "n_frames": len(ok),
                        "by_candidate": dict(Counter(ok.values())),
                        "manual": label in manual})

    nfr = sum(a["n_frames"] for a in applied)
    print(f"{'[dry-run] ' if args.dry_run else ''}assembled {len(applied)} "
          f"recordings from per-frame picks ({nfr} frames grafted): "
          + ", ".join(f"{k}={counts[k]}" for k in SRC if counts[k]))
    print(f"  {len(applied)} recordings changed from the curated baseline; "
          f"the rest keep Original (unchanged masks.npz)")
    if manual:
        print(f"  ✎ {len(manual)} flagged for manual brushing: "
              + ", ".join(sorted(manual)))
    if missing:
        print(f"  ⚠ {len(missing)} skipped (not in cache / file missing): "
              + ", ".join(missing))

    if not args.dry_run:
        os.makedirs(REVIEW_DIR, exist_ok=True)
        with open(os.path.join(REVIEW_DIR, "applied_choices.json"), "w") as f:
            json.dump({"applied": applied, "counts": dict(counts),
                       "manual": sorted(manual)}, f, indent=2)
        print("  wrote review/applied_choices.json")
        print("  next: conda run -n cellpose4 python scripts/ic293_batch.py "
              "--phase analyze --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
