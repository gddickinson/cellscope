"""Benchmark cpsam(DIC) + Cy5 filter against hand-labelled GT.

Run after IC295 GT candidates have been hand-labelled (i.e.
`<name>_dic_masks.png` files exist next to the `<name>_dic.png`
sources).

Output (under results/ic295_eval/):
  results.json     per-frame: IoU(DIC-only vs GT), IoU(filtered vs GT)
  results.csv      same in tabular form
  by_condition.csv aggregated WT/KO/GOF/Y1/DMSO/OT
  RUN_METADATA.md  reproducibility info
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
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()


_COND_RE = re.compile(r"_(WT|KO|GOF|Y1|DMSO|OT)_", re.I)


def parse_condition(name):
    m = _COND_RE.search(name)
    return m.group(1).upper() if m else "unknown"


def iou(pred, gt):
    p = pred.astype(bool)
    g = gt.astype(bool)
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",
                    default="data/ic295_gt/candidates",
                    help="folder with <name>_dic.png + _cy5.png + "
                         "_dic_masks.png triples")
    ap.add_argument("--out-dir", default="results/ic295_eval")
    ap.add_argument("--min-score", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dic_pngs = sorted(glob.glob(os.path.join(args.candidates,
                                              "*_dic.png")))
    pairs = []
    for dp in dic_pngs:
        gt_path = dp.replace("_dic.png", "_dic_masks.png")
        cy5_path = dp.replace("_dic.png", "_cy5.png")
        if not os.path.exists(gt_path):
            continue
        if not os.path.exists(cy5_path):
            print(f"  [skip] no Cy5 for {os.path.basename(dp)}")
            continue
        dic = np.array(Image.open(dp).convert("L"))
        cy5 = np.array(Image.open(cy5_path).convert("L"))
        gt = np.array(Image.open(gt_path))
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt_bool = gt > 0
        cond = parse_condition(os.path.basename(dp))
        pairs.append((dic, cy5, gt_bool, os.path.basename(dp), cond))
    if not pairs:
        print(f"No labelled pairs in {args.candidates}.")
        sys.exit(1)
    print(f"[bench] {len(pairs)} labelled pairs")

    print(f"[bench] loading cpsam (cellpose4)…")
    from cellpose import models
    from core.multichannel import filter_dic_labels_by_cy5
    cpsam = models.CellposeModel(gpu=True)

    rows = []
    t0 = time.time()
    for i, (dic, cy5, gt, name, cond) in enumerate(pairs):
        out = cpsam.eval(dic)
        masks_dic = out[0].astype(np.int32)
        filtered, scores, kept = filter_dic_labels_by_cy5(
            masks_dic, cy5, min_score=args.min_score)
        iou_dic = iou(masks_dic > 0, gt)
        iou_filt = iou(filtered > 0, gt)
        rows.append({
            "name": name,
            "condition": cond,
            "n_dic": int(masks_dic.max()),
            "n_filtered": int(filtered.max()),
            "iou_dic": iou_dic,
            "iou_filtered": iou_filt,
            "delta_iou": iou_filt - iou_dic,
        })
        if (i + 1) % 4 == 0 or i == len(pairs) - 1:
            print(f"  {i+1}/{len(pairs)}  mean IoU "
                  f"DIC={np.mean([r['iou_dic'] for r in rows]):.3f}  "
                  f"filtered={np.mean([r['iou_filtered'] for r in rows]):.3f}")
    elapsed = time.time() - t0

    # Aggregate
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)
    summary = {
        "n": len(rows),
        "mean_iou_dic": float(np.mean([r["iou_dic"] for r in rows])),
        "mean_iou_filtered": float(np.mean([r["iou_filtered"] for r in rows])),
        "mean_delta": float(np.mean([r["delta_iou"] for r in rows])),
        "min_score_threshold": args.min_score,
        "elapsed_s": elapsed,
        "rows": rows,
        "by_condition": {
            c: {
                "n": len(rs),
                "iou_dic": float(np.mean([r["iou_dic"] for r in rs])),
                "iou_filtered": float(np.mean([r["iou_filtered"] for r in rs])),
            }
            for c, rs in by_cond.items()
        },
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "results.csv"), "w",
               newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(args.out_dir, "by_condition.csv"), "w",
               newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n", "iou_dic", "iou_filtered",
                    "delta"])
        for c, s in summary["by_condition"].items():
            w.writerow([c, s["n"], f"{s['iou_dic']:.3f}",
                        f"{s['iou_filtered']:.3f}",
                        f"{s['iou_filtered'] - s['iou_dic']:+.3f}"])

    from output.run_metadata import write_run_metadata
    write_run_metadata(
        out_path=os.path.join(args.out_dir, "RUN_METADATA.md"),
        title="IC295 multichannel benchmark — DIC vs DIC+Cy5 filter",
        sections={
            "Source": f"GT pairs from `{args.candidates}/`",
            "Method": (
                "For each labelled DIC frame:\n"
                "1. Run cpsam (cellpose 4 base) on DIC\n"
                "2. Compute Cy5 presence score per mask\n"
                f"3. Drop masks with score < {args.min_score}\n"
                "4. Compare both (raw, filtered) to GT mask via IoU"),
            "Results summary": (
                f"* {summary['n']} frames\n"
                f"* Mean IoU DIC-only:  {summary['mean_iou_dic']:.3f}\n"
                f"* Mean IoU filtered:  {summary['mean_iou_filtered']:.3f}\n"
                f"* Mean Δ:             {summary['mean_delta']:+.3f}"),
        },
        rerun_cli=(
            f"conda run -n cellpose4 python scripts/bench_multichannel.py "
            f"--candidates {args.candidates} "
            f"--out-dir {args.out_dir} --min-score {args.min_score}"),
        timing_seconds={"total": elapsed},
    )
    print(f"\n=== Summary ===")
    print(f"Mean IoU DIC-only : {summary['mean_iou_dic']:.3f}")
    print(f"Mean IoU filtered : {summary['mean_iou_filtered']:.3f}")
    print(f"Mean Δ            : {summary['mean_delta']:+.3f}")
    for cond, s in summary["by_condition"].items():
        print(f"  {cond}: dic={s['iou_dic']:.3f}  "
              f"filt={s['iou_filtered']:.3f}  "
              f"Δ={s['iou_filtered'] - s['iou_dic']:+.3f}  (n={s['n']})")
    print(f"\nResults: {args.out_dir}/")


if __name__ == "__main__":
    main()
