"""Import legacy manual GT from the piezo1_analysis project on
GeorgeDrive into the standard data/legacy_gt/ layout.

Source: /Volumes/GeorgeDrive/cellscope_data/piezo1_analysis/data/manual_gt/
Each subfolder contains frame_NNNN_masks.png files + (sometimes) a
meta.json pointing at the source recording.

For each importable GT set we create:

  data/legacy_gt/<name>/
      <source_basename>           ← symlink to the .ome.tif (or .tif)
      <source_basename>.json      ← sidecar with um_per_px,
                                    time_interval_min, name
      gt_masks/
          mask_F0.png             ← renamed from frame_0000_masks.png
          mask_F1.png
          …

The Pos7_WT GT layout is unchanged. Skips GT sets whose source
recording can't be located.

Run:  python scripts/import_legacy_gt.py
"""
import os
import sys
import json
import shutil
import logging
import argparse
import numpy as np
from skimage import io as skio

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

LEGACY_GT_ROOT = (
    "/Volumes/GeorgeDrive/cellscope_data/piezo1_analysis/"
    "data/manual_gt")
DEST_ROOT = os.path.join(CELLSCOPE_ROOT, "data/legacy_gt")

# Where to look for source recordings referenced by meta.json
RECORDING_SEARCH_DIRS = [
    "/Volumes/GeorgeDrive/cellscope_data/piezo1_analysis/data/ignasi",
    "/Volumes/GeorgeDrive/cellscope_data/piezo1_analysis/data/examples",
]

# Default scale (no metadata sidecars exist for these crops). IC293
# cropped recordings are at the same pixel size as IC295 (0.6523 µm/px,
# 10 min/frame).
DEFAULT_UM_PER_PX = 0.6523
DEFAULT_TIME_MIN = 10.0

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("import_gt")


def find_source_recording(meta_path):
    """meta.json → absolute path to the source .ome.tif/.tif. Returns
    None if it can't be located."""
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return None
    vp = meta.get("video_path", "")
    basename = os.path.basename(vp)
    # 1) Try as-is (might be absolute and exist)
    if os.path.exists(vp):
        return vp
    # 2) Look for the basename in known dirs
    for d in RECORDING_SEARCH_DIRS:
        cand = os.path.join(d, basename)
        if os.path.exists(cand):
            return cand
    return None


def import_gt_set(src_folder, dest_folder, src_recording):
    """Import one legacy GT folder into the new layout."""
    os.makedirs(dest_folder, exist_ok=True)
    masks_dst = os.path.join(dest_folder, "gt_masks")
    os.makedirs(masks_dst, exist_ok=True)

    # Symlink the recording into the dest folder
    rec_basename = os.path.basename(src_recording)
    rec_link = os.path.join(dest_folder, rec_basename)
    if os.path.lexists(rec_link):
        os.remove(rec_link)
    os.symlink(src_recording, rec_link)

    # JSON sidecar
    sidecar = os.path.join(
        dest_folder,
        rec_basename.replace(".ome.tif", ".ome.json")
        if rec_basename.endswith(".ome.tif")
        else rec_basename.rsplit(".", 1)[0] + ".json")
    sidecar_data = {
        "name": os.path.basename(dest_folder),
        "um_per_px": DEFAULT_UM_PER_PX,
        "time_interval_min": DEFAULT_TIME_MIN,
        "source": src_recording,
    }
    with open(sidecar, "w") as f:
        json.dump(sidecar_data, f, indent=2)

    # Convert mask names: frame_NNNN_masks.png → mask_F<n>.png
    files = sorted(f for f in os.listdir(src_folder)
                   if f.startswith("frame_") and f.endswith("_masks.png"))
    n_copied = 0
    annotated_frames = []
    for f in files:
        try:
            fi = int(f[len("frame_"):-len("_masks.png")])
        except ValueError:
            continue
        src = os.path.join(src_folder, f)
        dst = os.path.join(masks_dst, f"mask_F{fi}.png")
        # Confirm the file actually has labelled content; some legacy
        # files exist but are all-zero placeholders.
        img = skio.imread(src)
        if img.max() == 0:
            continue
        skio.imsave(dst, img.astype(np.uint16), check_contrast=False)
        annotated_frames.append(fi)
        n_copied += 1

    # GT_FRAMES.txt — list of frames with labels
    with open(os.path.join(dest_folder, "GT_FRAMES.txt"), "w") as f:
        f.write("# Frames in this GT set (annotated frames only).\n")
        for fi in annotated_frames:
            f.write(f"{fi}\n")

    return n_copied, annotated_frames


def main():
    os.makedirs(DEST_ROOT, exist_ok=True)
    if not os.path.isdir(LEGACY_GT_ROOT):
        raise FileNotFoundError(LEGACY_GT_ROOT)

    summary = []
    for sub in sorted(os.listdir(LEGACY_GT_ROOT)):
        src = os.path.join(LEGACY_GT_ROOT, sub)
        if not os.path.isdir(src):
            continue
        meta_path = os.path.join(src, "meta.json")
        if not os.path.exists(meta_path):
            log.info("Skipping %s — no meta.json (cko/control "
                     "have no metadata pointer)", sub)
            summary.append((sub, "SKIP", "no meta.json", 0))
            continue
        rec = find_source_recording(meta_path)
        if rec is None:
            log.info("Skipping %s — source recording not found", sub)
            summary.append((sub, "SKIP", "source missing", 0))
            continue
        dest = os.path.join(DEST_ROOT, sub)
        try:
            n, frames = import_gt_set(src, dest, rec)
            log.info("Imported %s: %d masks from %s → %s",
                     sub, n, os.path.basename(rec), dest)
            summary.append((sub, "OK", os.path.basename(rec), n))
        except Exception as e:
            log.error("Failed to import %s: %s", sub, e)
            summary.append((sub, "ERROR", str(e), 0))

    print()
    print("=" * 60)
    print("Legacy GT import summary")
    print("=" * 60)
    print(f"{'Folder':<40} {'Status':<8} {'Masks':>6}  Source")
    for name, status, info, n in summary:
        print(f"{name:<40} {status:<8} {n:>6}  {info}")


if __name__ == "__main__":
    main()
