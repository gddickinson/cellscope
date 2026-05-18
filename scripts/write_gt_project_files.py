"""Drop a .cellscope project file in each GT folder so the GUI can
load the recording + pipeline labels in a single drag-and-drop.

For every folder with both `<rec>.ome.tif` (or .tif) and
`pipeline_results/masks.npz`, this writes:

  <folder>/<recording_name>.cellscope
  <folder>/<recording_name>_masks.npz   (sibling expected by load_project)

The .cellscope file is a JSON pointing at the .tif + this masks.npz.
Drag the .cellscope onto the GUI window to load everything at once.

Re-run safely — overwrites existing project files.

Usage:
  python scripts/write_gt_project_files.py
"""
import os
import sys
import json
import shutil
import numpy as np

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

from core.project import save_project, _tracks_from_labels


GT_ROOTS = ["data/ic295_gt_full", "data/legacy_gt"]


def find_recording(folder):
    for f in os.listdir(folder):
        if f.endswith((".ome.tif", ".tif", ".tiff")):
            return os.path.join(folder, f)
    return None


def find_projectable_folders():
    out = []
    for root in GT_ROOTS:
        if not os.path.isdir(root):
            continue
        for sub in sorted(os.listdir(root)):
            f = os.path.join(root, sub)
            if not os.path.isdir(f):
                continue
            if not os.path.exists(
                    os.path.join(f, "pipeline_results", "masks.npz")):
                continue
            if find_recording(f) is None:
                continue
            out.append(f)
    return out


def load_metadata(folder, tif_path):
    """Pull um_per_px + time_interval from the .ome.json sidecar."""
    for f in os.listdir(folder):
        if f.endswith(".ome.json") or f.endswith(".json"):
            try:
                with open(os.path.join(folder, f)) as fp:
                    return json.load(fp)
            except Exception:
                pass
    return {}


def write_one(folder):
    tif = find_recording(folder)
    masks_npz = os.path.join(folder, "pipeline_results", "masks.npz")
    data = np.load(masks_npz)
    labels = data["labels"]
    masks = data["masks"]
    source = (data["fusion_source_stack"]
              if "fusion_source_stack" in data.files else None)

    tracks = _tracks_from_labels(labels)

    detect_result = {
        "masks": masks,
        "labels": labels,
        "tracks": tracks,
    }
    if source is not None:
        detect_result["fusion_source_stack"] = source

    meta = load_metadata(folder, tif)
    name = os.path.basename(folder)
    rec_min = {
        "name": meta.get("name", name),
        "video_path": tif,
        "um_per_px": float(meta.get("um_per_px", 1.0)),
        "time_interval_min": float(meta.get("time_interval_min", 1.0)),
        # save_project only reads metadata keys (length) — it doesn't
        # need the actual frames in memory
        "frames": np.zeros((len(labels), 1, 1), dtype=np.uint8),
    }
    params = {"mode": "multi"}

    proj_path = os.path.join(folder, f"{name}.cellscope")
    save_project(proj_path, rec_min, detect_result,
                 analysis_result=None, params=params, mode="multi")
    return proj_path, len(tracks)


def main():
    folders = find_projectable_folders()
    print(f"Writing .cellscope project files for {len(folders)} folders:")
    written = []
    for f in folders:
        try:
            path, n = write_one(f)
            print(f"  ✓ {path}  ({n} tracks)")
            written.append((f, path, n))
        except Exception as e:
            print(f"  ✗ {f}  FAILED: {e}")
    print(f"\nWrote {len(written)} project files.")
    print("\nTo load in the GUI: drag any .cellscope file onto the "
          "main window OR File → Open Project.")


if __name__ == "__main__":
    main()
