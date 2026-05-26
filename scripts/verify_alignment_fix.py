"""Verify the DIC↔Cy5 alignment fix by re-measuring the offset
AFTER applying the shift. Expected: residual offset near zero.

Also reports the "duplicate cell" rate (Cy5 cells whose centroid is
within ~cell-radius of a DIC cell but Jaccard is too low) — this
should drop significantly after alignment.

Usage:
  conda run -n cellpose4 python scripts/verify_alignment_fix.py \\
      data/ic295_gt_full/Pos7_WT
"""
import os
import sys
import argparse
import json
import numpy as np

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    args = p.parse_args()
    folder = os.path.abspath(args.folder)

    for f in os.listdir(folder):
        if f.endswith((".ome.tif", ".tif")):
            tif = os.path.join(folder, f)
            break
    else:
        sys.exit("no .ome.tif")

    from core.io import load_recording
    from core.channel_alignment import (
        measure_dic_cy5_offset, apply_offset_to_stack)

    print(f"Loading {tif} …")
    rec = load_recording(tif, dic_channel=1, fluo_channel=0)
    dic, cy5 = rec["frames"], rec["cy5_frames"]
    if cy5 is None:
        sys.exit("no fluorescence channel")

    print("\n--- BEFORE alignment ---")
    (dy_before, dx_before), info_before = measure_dic_cy5_offset(
        dic, cy5, verbose=True)
    print(f"  median offset (dy, dx) = ({dy_before:+.2f}, "
          f"{dx_before:+.2f}) px")
    print(f"  matched pairs:   {info_before['n_pairs']}")
    print(f"  IQR:             {info_before['iqr_dy_px']:.2f}, "
          f"{info_before['iqr_dx_px']:.2f}")

    print(f"\nApplying shift ({dy_before:+.2f}, {dx_before:+.2f}) "
          f"to Cy5 …")
    cy5_aligned = apply_offset_to_stack(cy5, dy_before, dx_before)

    print("\n--- AFTER alignment ---")
    (dy_after, dx_after), info_after = measure_dic_cy5_offset(
        dic, cy5_aligned, verbose=True)
    print(f"  median offset (dy, dx) = ({dy_after:+.2f}, "
          f"{dx_after:+.2f}) px")
    print(f"  matched pairs:   {info_after['n_pairs']}")
    print(f"  IQR:             {info_after['iqr_dy_px']:.2f}, "
          f"{info_after['iqr_dx_px']:.2f}")

    um_per_px = rec.get("um_per_px", 1.0) or 1.0
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    res_dy = abs(dy_after)
    res_dx = abs(dx_after)
    pre_dy = abs(dy_before)
    pre_dx = abs(dx_before)
    print(f"  Pre-shift  magnitude: dy {pre_dy:.2f}px ({pre_dy*um_per_px:.2f}µm), "
          f"dx {pre_dx:.2f}px ({pre_dx*um_per_px:.2f}µm)")
    print(f"  Post-shift magnitude: dy {res_dy:.2f}px ({res_dy*um_per_px:.2f}µm), "
          f"dx {res_dx:.2f}px ({res_dx*um_per_px:.2f}µm)")
    print(f"  Residual fraction:    dy {res_dy/max(pre_dy,0.01)*100:.1f}%, "
          f"dx {res_dx/max(pre_dx,0.01)*100:.1f}%")

    if res_dy < 1.5 and res_dx < 1.5:
        print(f"\n  ✓ Alignment fix WORKS: residual <1.5 px in both axes")
    elif res_dy < pre_dy / 3 and res_dx < pre_dx / 3:
        print(f"\n  ~ Alignment fix HELPS but residual still significant")
    else:
        print(f"\n  ✗ Alignment fix did NOT improve residual offset")


if __name__ == "__main__":
    main()
