"""Benchmark gap-fill Phase 1 variants, scored against reviewed masks
as ground truth.

The reviewed `masks.npz` for a recording is GT. We delete a sample of
frames from complete cell tracks (synthetic gaps whose TRUE mask we
know), then run Phase-1 cpsam several ways on the SAME gaps and compare
fill-rate, IoU-vs-GT and time:

  full+aug   — full-frame, augment=True   (the original behaviour)
  crop+aug   — adaptive crop, augment=True (current default)
  crop+noaug — adaptive crop, augment=False (proposed: ~4× fewer
               forward passes; augment is the dominant cost at the
               downsampled production resolution where the crop alone
               doesn't help)

`--downsample N` reproduces production conditions (IC295 auto-downsamples
2048→1024, where cpsam is NOT pixel-bound so the augment 4× dominates).
Synthetic gaps are *easier* than real ones (they delete frames the
detector originally found), so a no-augment pass that holds up here
should still be paired with an augment fallback on miss in production.

Usage:
    conda run -n cellpose4 python scripts/bench_gap_fill.py \
        --label Pos7-WT --cond WT --downsample 2 --n-gaps 18
"""
import os
import sys
import time
import argparse

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# (name, use_crop, use_augment)
VARIANTS = [("full+aug", False, True),
            ("crop+aug", True, True),
            ("crop+noaug", True, False)]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(u) if u else 0.0


def downsample(frames, labels, n):
    if n <= 1:
        return frames, labels
    import cv2
    H, W = frames.shape[1:]
    h, w = H // n, W // n
    f2 = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
                   for f in frames])
    l2 = np.stack([cv2.resize(l, (w, h), interpolation=cv2.INTER_NEAREST)
                   for l in labels.astype(np.int32)])
    return f2, l2


def pick_gaps(stacks, n_gaps):
    gaps = []
    for cid, stack in stacks.items():
        present = np.where(stack.any(axis=(1, 2)))[0]
        if len(present) < 5:
            continue
        interior = [f for f in present[2:-2]
                    if (f - 1 in present or f - 2 in present)
                    and (f + 1 in present or f + 2 in present)]
        if not interior:
            continue
        k = max(1, min(4, len(interior)))
        for j in np.linspace(0, len(interior) - 1, k).round().astype(int):
            gaps.append((cid, int(interior[j])))
    if len(gaps) > n_gaps:
        gaps = [gaps[i] for i in
                np.linspace(0, len(gaps) - 1, n_gaps).round().astype(int)]
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="WT")
    ap.add_argument("--label", default="Pos7-WT")
    ap.add_argument("--n-gaps", type=int, default=18)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--search-radius", type=int, default=200)
    ap.add_argument("--min-area", type=int, default=200)
    args = ap.parse_args()

    rec_dir = f"ic295_analysis/by_condition/{args.cond}/{args.label}"
    masks_p = f"{rec_dir}/pipeline_results/masks.npz"
    tif = next((os.path.join(rec_dir, f) for f in os.listdir(rec_dir)
                if f.endswith(".ome.tif")), None)
    from core.io import load_recording
    rec = load_recording(tif, dic_channel=1, fluo_channel=0)
    frames = rec["frames"]
    labels = np.load(masks_p)["labels"].astype(np.int32)

    sr, ma = args.search_radius, args.min_area
    if args.downsample > 1:
        frames, labels = downsample(frames, labels, args.downsample)
        sr = max(20, sr // args.downsample)
        ma = max(20, ma // (args.downsample ** 2))
    n = len(frames)
    print(f"{n} frames @ {frames.shape[1]}x{frames.shape[2]} "
          f"(downsample {args.downsample}), search_radius={sr} "
          f"min_area={ma}", flush=True)

    ids = [int(i) for i in np.unique(labels).tolist() if i > 0]
    stacks = {c: (labels == c) for c in ids}
    gaps = pick_gaps(stacks, args.n_gaps)
    print(f"{len(ids)} cells, {len(gaps)} synthetic gaps\n", flush=True)

    from cellpose import models
    from core.track_gap_fill import (
        _try_primary_cpsam, _interpolate_centroid, _track_median_area)
    print("loading cpsam model…", flush=True)
    model = models.CellposeModel(gpu=True)

    rows = []
    for gi, (cid, fi) in enumerate(gaps):
        true_mask = stacks[cid][fi]
        if not true_mask.any():
            continue
        gappy = stacks[cid].copy()
        gappy[fi] = False
        cent = _interpolate_centroid({"stack": gappy}, fi, n)
        if cent is None:
            continue
        ea = _track_median_area({"stack": gappy}, n)
        row = {"cid": cid, "fi": fi}
        for name, crop, aug in VARIANTS:
            t0 = time.time()
            mask = _try_primary_cpsam(frames[fi], cent, sr, ma,
                                      cpsam_model=model, expected_area=ea,
                                      use_crop=crop, use_augment=aug)
            row[name] = {"fill": mask is not None,
                         "iou": iou(mask, true_mask),
                         "t": time.time() - t0,
                         "mask": mask}
        rows.append(row)
        s = "  ".join(f"{nm} {'Y' if row[nm]['fill'] else 'n'}"
                      f"/{row[nm]['iou']:.2f}/{row[nm]['t']:.1f}s"
                      for nm, _, _ in VARIANTS)
        print(f"  [{gi+1}/{len(gaps)}] C{cid} f{fi}: {s}", flush=True)

    _summary(rows)
    return 0


def _summary(rows):
    n = len(rows)
    print("\n" + "=" * 66)
    print(f"gaps: {n}\n")
    print(f"{'variant':12s} {'fill':>8s} {'meanIoU':>8s} "
          f"{'s/gap':>7s} {'total':>7s}")
    agg = {}
    for name, _, _ in VARIANTS:
        fills = sum(r[name]["fill"] for r in rows)
        ious = [r[name]["iou"] for r in rows if r[name]["fill"]] or [0]
        tt = sum(r[name]["t"] for r in rows)
        agg[name] = dict(fills=fills, iou=float(np.mean(ious)), t=tt)
        print(f"{name:12s} {fills:3d}/{n:<4d} {np.mean(ious):8.3f} "
              f"{tt/n:7.1f} {tt:7.0f}")

    base, prop = "crop+aug", "crop+noaug"
    print(f"\n{prop} vs {base} (the decision):")
    speed = agg[base]["t"] / max(agg[prop]["t"], 1e-6)
    print(f"  speed-up {speed:.1f}×   "
          f"({agg[base]['t']/60:.1f} → {agg[prop]['t']/60:.1f} min)")
    # good fills (IoU>=0.5 under crop+aug) that crop+noaug drops
    dropped = [r for r in rows
               if r[base]["fill"] and r[base]["iou"] >= 0.5
               and not r[prop]["fill"]]
    both = [r for r in rows if r[base]["fill"] and r[prop]["fill"]]
    di = (np.mean([r[prop]["iou"] - r[base]["iou"] for r in both])
          if both else 0.0)
    print(f"  good {base} fills dropped by {prop}: {len(dropped)}")
    print(f"  shared-fill IoU Δ (noaug − aug): {di:+.3f} (n={len(both)})")
    ok = len(dropped) == 0 and di >= -0.03
    print("\nVERDICT: " + (
        f"no-augment matches augment on the crop — adopt with an "
        f"augment fallback on miss ({speed:.1f}× Phase-1)"
        if ok else
        f"REVIEW — no-augment dropped {len(dropped)} good fills / "
        f"IoU Δ {di:+.3f}"))


if __name__ == "__main__":
    sys.exit(main())
