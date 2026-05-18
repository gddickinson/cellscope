"""Export a cellscope pipeline cache to TIFFs that Fiji/ImageJ can open.

Most biology labs live in Fiji, not Python. This bridge writes two
multipage 16-bit TIFFs that Fiji opens natively:

  <stem>_image.tif  — the source recording (uint8 → 16-bit grayscale)
  <stem>_labels.tif — the tracked label stack (int16; 0 = background,
                       1..N = consistent cell IDs across frames)

Pair them via the included `cellscope_load.ijm` macro: open both,
link as a hyperstack, and use the labels image as a colored overlay
on the source.

Usage:
  python scripts/cellscope_export_fiji.py path/to/pipeline_cache.npz \\
      --out-dir my_export/

  # Or batch a whole directory of cache files:
  python scripts/cellscope_export_fiji.py results/full_dataset/*.npz \\
      --out-dir fiji_export/
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

import numpy as np
import tifffile


def export_one(cache_path, out_dir):
    """Write <stem>_image.tif + <stem>_labels.tif from a cache .npz."""
    z = np.load(cache_path, allow_pickle=False)
    if "frames" not in z.files:
        print(f"  [skip] {cache_path} — no 'frames' key")
        return
    stem = os.path.splitext(os.path.basename(cache_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    frames = z["frames"]
    img_path = os.path.join(out_dir, f"{stem}_image.tif")
    # Save as 8-bit (Fiji handles natively, smaller files)
    tifffile.imwrite(img_path, frames, photometric="minisblack",
                     compression="zlib", metadata={"axes": "TYX"})

    if "labels" in z.files:
        labels = z["labels"]
        lab_path = os.path.join(out_dir, f"{stem}_labels.tif")
        # Compact label IDs (1..N) using existing values; cast to
        # int16 since Fiji's "Connected Components" / Glasbey LUT
        # show this nicely.
        max_id = int(labels.max())
        if max_id > 32767:
            print(f"  [warn] {stem} has > 32767 labels — clipping to int16")
            labels = np.where(labels > 32767, 0, labels)
        tifffile.imwrite(lab_path, labels.astype(np.int16),
                         photometric="minisblack",
                         compression="zlib",
                         metadata={"axes": "TYX"})
        print(f"  ✓ {img_path} + {lab_path}  "
              f"({labels.shape[0]} frames, {max_id} labels)")
    elif "masks" in z.files:
        masks = z["masks"].astype(np.uint8)
        mask_path = os.path.join(out_dir, f"{stem}_mask.tif")
        tifffile.imwrite(mask_path, masks,
                         photometric="minisblack",
                         compression="zlib",
                         metadata={"axes": "TYX"})
        print(f"  ✓ {img_path} + {mask_path}  ({masks.shape[0]} frames, "
              f"single-cell mask)")
    else:
        print(f"  ✓ {img_path}  (no masks/labels in cache)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+",
                    help="cache .npz file(s) or glob patterns")
    ap.add_argument("--out-dir", default="fiji_export",
                    help="directory to write TIFFs into "
                         "(default: ./fiji_export)")
    args = ap.parse_args()

    # Expand globs
    expanded = []
    for p in args.paths:
        if any(c in p for c in "*?["):
            expanded.extend(sorted(glob.glob(p)))
        else:
            expanded.append(p)
    if not expanded:
        print("No files found"); sys.exit(1)

    print(f"Exporting {len(expanded)} cache(s) → {args.out_dir}/")
    for p in expanded:
        if not os.path.exists(p):
            print(f"  [skip] {p} (not found)")
            continue
        print(f"\n• {p}")
        export_one(p, args.out_dir)
    print("\nNext: open Fiji → Plugins → Macros → Run… → "
          "select cellscope_load.ijm")


if __name__ == "__main__":
    main()
