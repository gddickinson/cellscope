"""Create a small single-cell example recording by cropping ONE cell's
track out of a larger recording + its label stack.

Why: `scripts/test_focused_gui.py` needs a small, fast, single-cell
recording. The original (an IC293 cropped WT) lived on a drive that
failed; its local pointers are now dead symlinks. The IC295 review
(`ic295_analysis/`) produced real, user-edited `masks.npz` files whose
pixels are in `ic295_analysis/_cache/` — GT-quality labels we can crop.

This tool picks the cell whose whole-track bounding box fits inside
`--max-crop` and that is present in the most frames, crops a square
window (path bbox + `--margin`) of the DIC channel over the frame span
where the cell exists, keeps ONLY that cell's mask (relabelled to 1),
and writes a self-contained example:
    <out>/<name>.tif         single-channel uint8 (frames, H, W)
    <out>/<name>_masks.npz   labels (frames, H, W), the one cell = 1
    <out>/<name>.json        um_per_px, time_interval_min + provenance

Example:
    conda run -n cellpose4 python scripts/make_single_cell_example.py \
        --recording ic295_analysis/_cache/IC295__1_MMStack_Pos10-WT.ome.tif \
        --masks ic295_analysis/by_condition/WT/Pos10-WT/pipeline_results/masks.npz \
        --name single_cell_crop_wt
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _bbox_over_track(stack):
    """(ymin, ymax, xmin, xmax) union bbox over all frames a cell is in."""
    ys, xs = np.where(stack.any(axis=0))
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _longest_true_run(mask):
    """[start, end) of the longest contiguous run of True in a 1-D mask."""
    best_len = best_start = 0
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i > best_len:
                best_len, best_start = j - i, i
            i = j
        else:
            i += 1
    return best_start, best_start + best_len


def pick_cell(labels, max_crop):
    """Return (cell_id, present_mask, bbox) for the longest track whose
    whole-path bbox fits inside max_crop. None if nothing fits."""
    ids = np.unique(labels)
    ids = [int(i) for i in ids.tolist() if i > 0]
    best = None
    for cid in ids:
        stack = labels == cid
        present = stack.any(axis=(1, 2))
        nf = int(present.sum())
        if nf < 5:
            continue
        ymin, ymax, xmin, xmax = _bbox_over_track(stack)
        h, w = ymax - ymin + 1, xmax - xmin + 1
        if max(h, w) > max_crop:
            continue
        # Prefer longest track; tie-break the more compact (smaller) bbox.
        key = (nf, -max(h, w))
        if best is None or key > best[0]:
            best = (key, cid, present, (ymin, ymax, xmin, xmax))
    if best is None:
        return None
    return best[1], best[2], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True,
                    help="source .tif/.ome.tif (DIC pixels)")
    ap.add_argument("--masks", required=True,
                    help="masks.npz with a 'labels' (N,H,W) stack")
    ap.add_argument("--out-dir", default=None,
                    help="default: data/examples/<name>/")
    ap.add_argument("--name", default="single_cell_crop")
    ap.add_argument("--dic-channel", type=int, default=1)
    ap.add_argument("--fluo-channel", type=int, default=0)
    ap.add_argument("--max-crop", type=int, default=320,
                    help="reject cells whose track bbox exceeds this (px)")
    ap.add_argument("--margin", type=int, default=36,
                    help="padding (px) added around the track bbox")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join("data", "examples", args.name)
    os.makedirs(out_dir, exist_ok=True)

    # --- Load labels ---
    npz = np.load(args.masks)
    if "labels" in npz.files:
        labels = npz["labels"].astype(np.int32)
    else:
        from scipy.ndimage import label as _cc
        m = npz["masks"].astype(bool)
        labels = np.zeros_like(m, dtype=np.int32)
        for i in range(len(m)):
            labels[i], _ = _cc(m[i])
    n = len(labels)

    # --- Load source DIC frames (channel-aware) ---
    from core.io import load_recording, detect_channels
    rpath = args.recording
    n_ch = (detect_channels(rpath)
            if rpath.lower().endswith((".tif", ".tiff")) else 1)
    if n_ch > 1:
        rec = load_recording(rpath, dic_channel=args.dic_channel,
                             fluo_channel=args.fluo_channel)
    else:
        rec = load_recording(rpath)
    frames = rec["frames"]
    um = float(rec.get("um_per_px") or 0) or None
    dt = float(rec.get("time_interval_min") or 0) or None

    if frames.shape != labels.shape:
        raise SystemExit(
            f"frame/label shape mismatch: {frames.shape} vs {labels.shape}")

    # --- Pick the cell + crop window ---
    pick = pick_cell(labels, args.max_crop)
    if pick is None:
        raise SystemExit(
            f"no cell with a track bbox <= {args.max_crop}px; "
            f"raise --max-crop")
    cid, present, (ymin, ymax, xmin, xmax) = pick
    H, W = frames.shape[1:]
    y0 = max(0, ymin - args.margin)
    y1 = min(H, ymax + 1 + args.margin)
    x0 = max(0, xmin - args.margin)
    x1 = min(W, xmax + 1 + args.margin)
    # Use the LONGEST CONTIGUOUS run of present frames so the cropped
    # recording has the cell in every frame (deterministic single-cell
    # test — no internal gaps where detection could legitimately find
    # nothing).
    f0, f1 = _longest_true_run(present)

    crop_frames = np.ascontiguousarray(frames[f0:f1, y0:y1, x0:x1])
    crop_labels = (labels[f0:f1, y0:y1, x0:x1] == cid).astype(np.int32)

    # --- Write outputs ---
    import tifffile
    tif_path = os.path.join(out_dir, f"{args.name}.tif")
    tifffile.imwrite(tif_path, crop_frames, photometric="minisblack")
    np.savez_compressed(
        os.path.join(out_dir, f"{args.name}_masks.npz"), labels=crop_labels)
    meta = {
        "name": args.name,
        "um_per_px": um, "time_interval_min": dt,
        "cell_type": "keratinocyte (single-cell crop)",
        "n_frames": int(crop_frames.shape[0]),
        "source_recording": os.path.basename(rpath),
        "source_masks": args.masks,
        "source_cell_id": int(cid),
        "crop_window_yx": [y0, y1, x0, x1],
        "frame_span": [f0, f1],
        "provenance": ("single-cell crop generated by "
                       "scripts/make_single_cell_example.py from a "
                       "user-reviewed IC295 mask stack"),
    }
    with open(os.path.join(out_dir, f"{args.name}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    frac = float((crop_labels > 0).any(axis=(1, 2)).mean())
    print(f"[done] {tif_path}")
    print(f"  cell {cid}: {crop_frames.shape} "
          f"({crop_frames.nbytes/1e6:.1f} MB uncompressed), "
          f"cell present in {100*frac:.0f}% of cropped frames")
    print(f"  window y[{y0}:{y1}] x[{x0}:{x1}]  frames [{f0}:{f1}]")


if __name__ == "__main__":
    main()
