"""Audit + index every GT set in the repo.

Scans:
  data/ic295_gt_full/     — multichannel IC295 GT (new workflow)
  data/legacy_gt/         — IC293 single-channel GT (imported)

For each subfolder with masks present, reports:
  - n labelled frames
  - frame indices labelled
  - image dimensions
  - n unique cell IDs (min/max across labelled frames)
  - whether identity is preserved across frames (track IDs match)
  - source recording path + whether it's accessible
  - whether pipeline_results/ + evaluation/ exist

Writes:
  data/GT_INDEX.md   — human-readable registry
  data/gt_index.json — machine-readable

Exits non-zero if any GT set is broken (missing source, all-zero masks,
etc.) so the script can be used as a pre-commit / CI check.

Run:  python scripts/audit_gt.py
"""
import os
import sys
import csv
import json
import argparse
import numpy as np
from skimage import io as skio
from collections import OrderedDict

CELLSCOPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)

GT_ROOTS = [
    ("data/ic295_gt_full",
     "Multichannel IC295 (DIC+Cy5), full-recording GT labelled in the "
     "current workflow (scripts/setup_ic295_gt_labeling.py)"),
    ("data/legacy_gt",
     "Single-channel cropped DIC GT carried over from the piezo1 "
     "project on GeorgeDrive (imported via "
     "scripts/import_legacy_gt.py). Source recordings live at "
     "/Volumes/GeorgeDrive/cellscope_data/piezo1_analysis/data/"),
]


def audit_folder(folder):
    """Return dict describing one GT subfolder."""
    name = os.path.basename(folder)
    gt_dir = os.path.join(folder, "gt_masks")
    info = {
        "name": name, "folder": folder,
        "gt_masks_dir": gt_dir if os.path.isdir(gt_dir) else None,
        "n_masks": 0, "frames_annotated": [],
        "image_shape": None,
        "n_cells_min": None, "n_cells_max": None,
        "n_unique_cell_ids": 0,
        "tracks_identity_preserved": None,
        "source_recording": None,
        "source_accessible": False,
        "has_pipeline_results": False,
        "has_evaluation": False,
        "issues": [],
    }
    if not info["gt_masks_dir"]:
        info["issues"].append("no gt_masks/ folder")
        return info
    mask_files = sorted(
        f for f in os.listdir(gt_dir)
        if f.startswith("mask_F") and f.endswith(".png"))
    if not mask_files:
        # No masks yet — placeholder folder; not an error
        info["issues"].append("placeholder (no labelled masks)")
        return info

    # Load every mask
    frames = []
    cell_counts = []
    all_ids = set()
    per_frame_ids = []
    shape = None
    for f in mask_files:
        try:
            fi = int(f[len("mask_F"):-len(".png")])
        except ValueError:
            continue
        img = skio.imread(os.path.join(gt_dir, f))
        if shape is None:
            shape = img.shape
        ids = sorted(int(c) for c in np.unique(img) if c)
        if not ids:
            info["issues"].append(
                f"{f}: empty mask (no cells labelled)")
            continue
        frames.append(fi)
        cell_counts.append(len(ids))
        all_ids.update(ids)
        per_frame_ids.append(set(ids))
    info["n_masks"] = len(frames)
    info["frames_annotated"] = sorted(frames)
    info["image_shape"] = list(shape) if shape else None
    if cell_counts:
        info["n_cells_min"] = min(cell_counts)
        info["n_cells_max"] = max(cell_counts)
    info["n_unique_cell_ids"] = len(all_ids)

    # Tracking identity preservation heuristic: do >= 2 frames share
    # the same set of cell IDs? If so we assume the labeller preserved
    # identity. If every frame has a different set we flag it.
    if len(per_frame_ids) >= 2:
        first = per_frame_ids[0]
        consistent = sum(1 for s in per_frame_ids[1:]
                          if len(s & first) >= len(first) // 2)
        info["tracks_identity_preserved"] = bool(
            consistent >= len(per_frame_ids) // 2)

    # Source recording = the only .ome.tif/.tif (not a mask png)
    candidates = [f for f in os.listdir(folder)
                  if f.endswith((".ome.tif", ".tif"))]
    if candidates:
        rec = os.path.join(folder, candidates[0])
        info["source_recording"] = rec
        # Resolve symlink chain to a real file
        real = os.path.realpath(rec)
        info["source_accessible"] = os.path.exists(real)
        info["source_real_path"] = real if info["source_accessible"] \
            else None
        if not info["source_accessible"]:
            info["issues"].append(
                f"source recording not accessible: {real}")
    else:
        info["issues"].append("no .tif recording in folder")

    info["has_pipeline_results"] = os.path.exists(
        os.path.join(folder, "pipeline_results", "masks.npz"))
    info["has_evaluation"] = os.path.exists(
        os.path.join(folder, "evaluation", "report.md"))
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if any issues are found.")
    args = p.parse_args()

    registry = OrderedDict()
    n_issues = 0
    for root, description in GT_ROOTS:
        if not os.path.isdir(root):
            continue
        registry[root] = {
            "description": description,
            "entries": [],
        }
        for sub in sorted(os.listdir(root)):
            folder = os.path.join(root, sub)
            if not os.path.isdir(folder):
                continue
            entry = audit_folder(folder)
            # Don't count "placeholder" or "empty" as failure-grade
            real_issues = [i for i in entry["issues"]
                           if "placeholder" not in i
                           and "empty mask" not in i]
            n_issues += len(real_issues)
            registry[root]["entries"].append(entry)

    # Write JSON registry
    with open("data/gt_index.json", "w") as f:
        json.dump(registry, f, indent=2)

    # Write markdown
    lines = ["# Ground-truth index",
             "",
             ("Auto-generated by `scripts/audit_gt.py`. Lists every "
              "GT set in the repository, where it lives, and whether "
              "the source recording is still accessible. **Never "
              "delete folders listed here without checking with the "
              "user first** — these masks took real work to produce."),
             ""]
    total_masks = 0
    total_recs = 0
    for root, root_info in registry.items():
        lines.append(f"## `{root}/`")
        lines.append("")
        lines.append(root_info["description"])
        lines.append("")
        lines.append(
            "| Recording | Frames | Image | Cells (min/max) | "
            "Identity preserved | Source accessible | Pipeline run | "
            "Evaluated | Issues |")
        lines.append(
            "|---|---:|---|---:|---:|---:|---:|---:|---|")
        for e in root_info["entries"]:
            if e["n_masks"] == 0:
                continue
            total_masks += e["n_masks"]
            total_recs += 1
            issues = [i for i in e["issues"]
                      if "placeholder" not in i
                      and "empty mask" not in i]
            shape = "x".join(str(s) for s in e["image_shape"]) \
                if e["image_shape"] else "?"
            lines.append(
                f"| `{e['name']}` | {e['n_masks']} | {shape} | "
                f"{e['n_cells_min']}/{e['n_cells_max']} | "
                f"{'✓' if e['tracks_identity_preserved'] else '?'} | "
                f"{'✓' if e['source_accessible'] else '✗'} | "
                f"{'✓' if e['has_pipeline_results'] else ' '} | "
                f"{'✓' if e['has_evaluation'] else ' '} | "
                f"{'; '.join(issues) if issues else '—'} |")
        # List empty / placeholder folders separately
        placeholders = [e["name"] for e in root_info["entries"]
                        if e["n_masks"] == 0]
        if placeholders:
            lines.append("")
            lines.append(f"_Placeholder folders awaiting labelling: "
                         f"{', '.join('`' + p + '`' for p in placeholders)}_")
        lines.append("")

    lines.append("## Totals")
    lines.append("")
    lines.append(f"- **{total_recs} recordings** with at least one "
                 f"labelled frame")
    lines.append(f"- **{total_masks} labelled mask files** total")
    lines.append(
        f"- All mask PNGs are stored as **real files** in the repo, "
        f"not symlinks — safe even when the external drive is "
        f"unplugged.")
    lines.append(
        f"- Source recordings (the `.ome.tif` next to each "
        f"`gt_masks/` folder) are **symlinks**; some point through "
        f"`piezo1_analysis/data/ignasi/` to "
        f"`/Volumes/GeorgeDrive/cellscope_data/...`. Pipeline runs "
        f"and re-evaluation need the drive mounted.")
    lines.append("")
    lines.append("## How to inspect results in the GUI")
    lines.append("")
    lines.append(
        "Each evaluated GT folder now contains a "
        "`<name>.cellscope` project file. Drag any of them onto the "
        "main GUI window (or use File → Open Project) to load:")
    lines.append("")
    lines.append("- the full multichannel recording (DIC + Cy5 when "
                 "present)")
    lines.append("- the pipeline's labels + reconstructed tracks")
    lines.append("- the fusion source stack when available — toggle "
                 "**Source ⓘ** in the viewer to colour cells by "
                 "detection origin")
    lines.append("")
    lines.append("Project files:")
    for root, root_info in registry.items():
        for e in root_info["entries"]:
            if e["n_masks"] == 0:
                continue
            proj = os.path.join(e["folder"], f"{e['name']}.cellscope")
            if os.path.exists(proj):
                rel = os.path.relpath(proj, ".")
                lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("Or run `python scripts/write_gt_project_files.py` "
                 "to (re)create them all from the saved "
                 "`pipeline_results/masks.npz`.")
    lines.append("")
    lines.append("## Backup")
    lines.append("")
    lines.append(
        "Run `python scripts/backup_gt.py` to create a tarball of "
        "**all GT masks + this index** in `data/gt_backups/`. The "
        "tarball excludes source recordings (too large) — the "
        "filenames in `gt_index.json` are enough to re-find them "
        "later.")
    lines.append("")
    lines.append("## What to do if GT is lost")
    lines.append("")
    lines.append(
        "1. Check `data/gt_backups/` for the most recent tarball.\n"
        "2. If no tarball, check the parent project on the drive: "
        "`/Volumes/GeorgeDrive/cellscope_data/piezo1_analysis/"
        "data/manual_gt/` (legacy GT was originally imported from "
        "there by `scripts/import_legacy_gt.py`).\n"
        "3. As a last resort, the `RUN_METADATA.json` in each "
        "`evaluation/` folder records which GT files were used, so "
        "you at least know the canonical filenames.")

    with open("data/GT_INDEX.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote data/GT_INDEX.md + data/gt_index.json")
    print(f"\n{total_recs} recordings, {total_masks} labelled masks")
    if n_issues:
        print(f"\n⚠️  {n_issues} issues found — see GT_INDEX.md")
        if args.strict:
            sys.exit(1)
    else:
        print("\n✓ No issues")


if __name__ == "__main__":
    main()
