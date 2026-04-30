"""Bench one cellpose model on dic_splits_v3/test.

Runs in either env:
  conda run -n cellpose4 python scripts/bench_cpsam_dic.py \
      --model data/models/cpsam_dic --out results/dic_model_eval/cpsam_dic.json
  conda run -n cellpose  python scripts/bench_cpsam_dic.py \
      --model data/models/cellpose_dic_v3 \
      --out results/dic_model_eval/cellpose_dic_v3.json

Stratified subsample by genotype (control / cKO / GoF) so per-genotype
means are stable. Saves per-frame IoU + per-genotype + overall mean.

Run scripts/compare_cpsam_dic.py to summarise both JSONs side by side.
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import tifffile


def iou(pred, gt):
    p = pred.astype(bool)
    g = gt.astype(bool)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(p, g).sum() / union)


def detect_genotype(name):
    n = name.lower()
    if "gof" in n:
        return "gof"
    if "ko" in n:
        return "cko"
    if "wt" in n or "con" in n or "ctrl" in n:
        return "control"
    # our_ctrl_gt / our_cko_gt patterns
    if re.search(r"our_ctrl", n):
        return "control"
    if re.search(r"our_cko", n):
        return "cko"
    return "unknown"


def stratified_sample(pairs, per_genotype):
    """Return up to per_genotype frames per genotype."""
    by_g = {}
    for p in pairs:
        by_g.setdefault(p[3], []).append(p)
    out = []
    rng = np.random.default_rng(0)
    for g, lst in by_g.items():
        if len(lst) > per_genotype:
            idx = rng.choice(len(lst), per_genotype, replace=False)
            out.extend([lst[i] for i in sorted(idx)])
        else:
            out.extend(lst)
    return out


def load_test_pairs(test_dir, name_glob="*"):
    """Auto-detect *_img.tif/_masks.tif pairs OR <name_glob>.png/_masks.png."""
    pairs = []
    tif_pattern = os.path.join(test_dir, f"{name_glob}_img.tif")
    if name_glob == "*":
        tif_pattern = os.path.join(test_dir, "*_img.tif")
    tif_imgs = sorted(glob.glob(tif_pattern))
    if tif_imgs:
        for img_path in tif_imgs:
            msk_path = img_path.replace("_img.tif", "_masks.tif")
            if not os.path.exists(msk_path):
                continue
            img = tifffile.imread(img_path)
            msk = tifffile.imread(msk_path)
            if img.ndim == 3:
                img = img[0]
            if msk.ndim == 3:
                msk = msk[0]
            name = os.path.basename(img_path).replace("_img.tif", "")
            pairs.append((img, (msk > 0), name, detect_genotype(name)))
        return pairs

    # PNG mode: <name_glob>.png with paired *_masks.png
    import cv2
    png_pattern = os.path.join(test_dir, f"{name_glob}*.png")
    png_imgs = sorted(glob.glob(png_pattern))
    png_imgs = [p for p in png_imgs if "_masks" not in p]
    for img_path in png_imgs:
        msk_path = img_path.replace(".png", "_masks.png")
        if not os.path.exists(msk_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        name = os.path.basename(img_path).replace(".png", "")
        pairs.append((img, (msk > 0), name, detect_genotype(name)))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="path to cellpose model (CP3 or cpsam fine-tune)")
    ap.add_argument("--out", required=True,
                    help="JSON output path")
    ap.add_argument(
        "--test-dir",
        default="data/training/dic_splits_v3/test",
        help="directory with *_img.tif / *_masks.tif pairs")
    ap.add_argument("--per-genotype", type=int, default=30,
                    help="frames per genotype (control/cKO/GoF)")
    ap.add_argument("--name-glob", default="*",
                    help="filter filenames in PNG mode "
                         "(e.g. 'our_*_gt' for our full-frame GT)")
    args = ap.parse_args()

    print(f"[bench] loading test pairs from {args.test_dir} "
          f"(name_glob={args.name_glob!r})")
    pairs = load_test_pairs(args.test_dir, name_glob=args.name_glob)
    print(f"[bench] found {len(pairs)} pairs total")
    pairs = stratified_sample(pairs, args.per_genotype)
    counts = {}
    for _, _, _, g in pairs:
        counts[g] = counts.get(g, 0) + 1
    print(f"[bench] subsampled {len(pairs)} pairs: {counts}")

    print(f"[bench] loading model: {args.model}")
    from cellpose import models
    model = models.CellposeModel(gpu=True, pretrained_model=args.model)

    rows = []
    t0 = time.time()
    for i, (img, gt, name, geno) in enumerate(pairs):
        out = model.eval(img)
        # cellpose returns either (masks,flows,styles) or
        # (masks,flows,styles,diams) depending on version
        masks = out[0]
        pred_bool = (masks > 0)
        rows.append({
            "name": name,
            "genotype": geno,
            "iou": iou(pred_bool, gt),
            "pred_px": int(pred_bool.sum()),
            "gt_px": int(gt.sum()),
            "detected": bool(pred_bool.any()),
        })
        if (i + 1) % 10 == 0 or i == len(pairs) - 1:
            print(f"[bench] {i+1}/{len(pairs)}  "
                  f"running mean IoU={np.mean([r['iou'] for r in rows]):.3f}")
    elapsed = time.time() - t0

    ious = [r["iou"] for r in rows]
    by_g = {}
    for r in rows:
        by_g.setdefault(r["genotype"], []).append(r["iou"])

    summary = {
        "model": args.model,
        "test_dir": args.test_dir,
        "n": len(rows),
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious)),
        "det_rate": float(sum(r["detected"] for r in rows) / len(rows)),
        "elapsed_s": elapsed,
        "per_genotype": {g: {
            "n": len(v),
            "mean_iou": float(np.mean(v)),
            "std_iou": float(np.std(v)),
        } for g, v in by_g.items()},
        "rows": rows,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[bench] {args.model}")
    print(f"  n={summary['n']}  mean IoU={summary['mean_iou']:.3f}  "
          f"det_rate={summary['det_rate']:.0%}  "
          f"time={summary['elapsed_s']:.0f}s")
    for g, s in summary["per_genotype"].items():
        print(f"  {g:8s}  n={s['n']:3d}  IoU={s['mean_iou']:.3f}"
              f" ± {s['std_iou']:.3f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
