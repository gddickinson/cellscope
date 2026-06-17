"""Test driver: run single-cell curation on one IC293 recording.

NON-DESTRUCTIVE — writes pipeline_results/masks_curated.npz +
curation_log.json + curation_montage.png; the original masks.npz is left
untouched until you approve. Uses Ignasi's curation priors (isolated,
flattened, not dividing, present every frame, centred); expected cell area
is inferred per recording unless --expected-area is given.

Usage:
  conda run -n cellpose4 python scripts/ic293_curate_one.py Pos19-KO
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

import numpy as np  # noqa: E402
from scripts.ic293_common import inventory_cache, recording_dir, read_sidecar_meta  # noqa: E402
from core.single_cell_curation import (  # noqa: E402
    SingleCellCurationParams, curate_single_cell)


def _montage(frames, before, after, log, out_path, ncols=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    aff = sorted({d["frame"] for d in log["frames_recovered"]}
                 | {d["frame"] for d in log["frames_repaired"]}
                 | {d["frame"] for d in log["frames_unresolved"]})
    if not aff:
        return None
    n = len(aff)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                            squeeze=False)
    kinds = {}
    for d in log["frames_recovered"]:
        kinds[d["frame"]] = "recovered"
    for d in log["frames_repaired"]:
        kinds[d["frame"]] = "repaired"
    for d in log["frames_unresolved"]:
        kinds[d["frame"]] = "UNRESOLVED"
    for i, f in enumerate(aff):
        ax = axs[i // ncols][i % ncols]
        ax.imshow(frames[f], cmap="gray")
        if (before[f] > 0).any():
            ax.contour(before[f] > 0, levels=[0.5], colors="red",
                       linewidths=1.0, linestyles="dashed")
        if (after[f] > 0).any():
            ax.contour(after[f] > 0, levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_title(f"F{f}  {kinds.get(f,'')}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    fig.suptitle("red dashed = before (detected)   |   green = after (curated)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110); plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--expected-area", type=float, default=None)
    args = ap.parse_args()

    inv = inventory_cache()
    if args.label not in inv:
        print(f"ERROR: {args.label} not in cache", file=sys.stderr)
        return 2
    info = inv[args.label]
    rec_dir = recording_dir(args.label, info["condition"])
    masks_path = os.path.join(rec_dir, "pipeline_results", "masks.npz")
    if not os.path.exists(masks_path):
        print(f"ERROR: {masks_path} missing", file=sys.stderr)
        return 2

    from core.io import load_recording
    rec = load_recording(info["video_path"])
    frames = rec["frames"]
    meta = read_sidecar_meta(info)
    um = float(rec.get("um_per_px") or meta["um_per_px"])
    before = np.load(masks_path)["labels"].astype(np.int32)

    params = SingleCellCurationParams(
        enabled=True, cell_present_every_frame=True, no_dividing=True,
        cell_isolated=True, roughly_centered=True,
        expected_cell_area_um2=args.expected_area)
    print(f"[curate] {args.label}: {before.shape[0]} frames, um/px={um:.4f}")
    after, log = curate_single_cell(before, frames, um, params)

    pr = os.path.join(rec_dir, "pipeline_results")
    np.savez_compressed(os.path.join(pr, "masks_curated.npz"),
                        labels=after, masks=(after > 0))
    with open(os.path.join(pr, "curation_log.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)
    montage = _montage(frames, before, after, log,
                       os.path.join(pr, "curation_montage.png"))

    print(f"  expected_area={log['expected_cell_area_um2']:.0f} µm²  "
          f"primary=id{log['primary_id']} (cellness {log['primary_cellness']})")
    print(f"  artifacts_removed={len(log['artifacts_removed'])}  "
          f"recovered={len(log['frames_recovered'])}  "
          f"repaired={len(log['frames_repaired'])}  "
          f"unresolved={len(log['frames_unresolved'])}")
    print(f"  final_presence={log['final_presence']}  "
          f"needs_manual_review={log['needs_manual_review']}")
    for e in log["exceptions"]:
        print(f"  EXCEPTION [{e['type']}]: {e['detail']}")
    print(f"  wrote masks_curated.npz + curation_log.json"
          + (f" + {os.path.basename(montage)}" if montage else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
