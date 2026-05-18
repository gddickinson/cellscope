"""Benchmark cpsam_dic against the new Ignasi GT (after labelling).

Reads `<name>.png` + `<name>_masks.png` pairs from
`data/ignasi_new_gt/candidates/` and reports IoU per condition
(WT / KO / GOF / Y1 / DMSO). Run only after at least some
`*_masks.png` files exist.

Run in the cellpose4 env (cpsam_dic is a CP4 model):

  conda run -n cellpose4 python scripts/bench_ignasi_new.py \\
      --model data/models/cpsam_dic

Why a separate script (vs `bench_cpsam_dic.py`):
  * That script's `detect_genotype` only knows WT/KO/GOF — adds
    Y1 and DMSO here.
  * The candidate frames are already flat-field corrected by
    `sample_gt_frames.py`, so no extra preprocessing is needed —
    just feed them straight to cpsam.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np
import tifffile  # noqa
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()


_COND_RE = re.compile(r"_(WT|KO|GOF|Y1|DMSO)_", re.I)


def parse_condition(name):
    m = _COND_RE.search(name)
    return m.group(1).upper() if m else "unknown"


def load_pair(img_path):
    img = np.array(Image.open(img_path).convert("L"))
    mask_path = img_path.replace(".png", "_masks.png")
    if not os.path.exists(mask_path):
        return img, None
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return img, (mask > 0)


def iou(pred, gt):
    p = pred.astype(bool)
    g = gt.astype(bool)
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",
                    default="data/ignasi_new_gt/candidates",
                    help="folder of <name>.png + <name>_masks.png pairs")
    ap.add_argument("--model", required=True,
                    help="cellpose model path (e.g. data/models/cpsam_dic)")
    ap.add_argument("--out", default="results/ignasi_new_eval/results.json",
                    help="JSON output path")
    ap.add_argument("--csv", default="results/ignasi_new_eval/results.csv")
    args = ap.parse_args()

    pngs = sorted(glob.glob(os.path.join(args.candidates, "*.png")))
    pngs = [p for p in pngs if "_masks" not in p]
    pairs = []
    for p in pngs:
        img, mask = load_pair(p)
        if mask is None:
            continue
        pairs.append((img, mask, os.path.basename(p),
                      parse_condition(os.path.basename(p))))
    if not pairs:
        print(f"No labelled pairs found in {args.candidates}.\n"
              f"Label some candidates first (see "
              f"data/ignasi_new_gt/LABELLING.md).")
        sys.exit(1)
    print(f"[bench] {len(pairs)} labelled pairs found")
    counts = {}
    for *_, c in pairs:
        counts[c] = counts.get(c, 0) + 1
    print(f"[bench] by condition: {counts}")

    print(f"[bench] loading model {args.model}")
    from cellpose import models
    model = models.CellposeModel(gpu=True, pretrained_model=args.model)

    rows = []
    t0 = time.time()
    for i, (img, gt, name, cond) in enumerate(pairs):
        out = model.eval(img)
        masks = out[0]
        pred = masks > 0
        rows.append({
            "name": name,
            "condition": cond,
            "iou": iou(pred, gt),
            "n_cells_pred": int(masks.max()),
            "pred_px": int(pred.sum()),
            "gt_px": int(gt.sum()),
        })
        if (i + 1) % 8 == 0 or i == len(pairs) - 1:
            print(f"[bench] {i+1}/{len(pairs)}  "
                  f"running mean IoU="
                  f"{np.mean([r['iou'] for r in rows]):.3f}")
    elapsed = time.time() - t0

    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r["iou"])
    summary = {
        "model": args.model,
        "n": len(rows),
        "mean_iou": float(np.mean([r["iou"] for r in rows])),
        "elapsed_s": elapsed,
        "by_condition": {
            k: {"n": len(v), "mean_iou": float(np.mean(v))}
            for k, v in by_cond.items()
        },
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.csv}")
    print(f"\n=== Summary ===")
    print(f"Overall mean IoU: {summary['mean_iou']:.3f}  (n={len(rows)})")
    for cond, stats in sorted(summary["by_condition"].items()):
        print(f"  {cond:<6} {stats['mean_iou']:.3f}  (n={stats['n']})")


if __name__ == "__main__":
    main()
