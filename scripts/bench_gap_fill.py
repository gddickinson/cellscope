"""Benchmark gap-fill Phase 1: full-frame vs adaptive-crop cpsam, scored
against reviewed masks as ground truth.

The reviewed `masks.npz` for a recording is treated as GT. We delete a
sample of frames from complete cell tracks (creating synthetic gaps
whose TRUE mask we know), then run Phase-1 cpsam BOTH ways on the same
gaps and compare:

  - fill rate   — did the method recover a cell at the gap?
  - IoU vs GT   — how well the recovered mask matches the deleted truth
  - time        — wall-clock per gap

If crop matches full-frame on fill-rate + IoU at a large speed-up, the
crop default (DEFAULTS.gap_fill_crop) is safe. Synthetic gaps are
*easier* than real ones (they delete frames the detector originally
found), but that is exactly the right test for "does cropping change
the answer" — Phase 1 only accepts a cell within search_radius, and the
crop is sized to contain that whole region.

Usage:
    conda run -n cellpose4 python scripts/bench_gap_fill.py \
        --cond WT --label Pos7-WT --n-gaps 24
"""
import os
import sys
import time
import argparse

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def iou(a, b):
    if a is None or b is None:
        return 0.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def build_tracks(labels):
    """Per-cell boolean stacks from a label movie."""
    ids = [int(i) for i in np.unique(labels).tolist() if i > 0]
    return {cid: (labels == cid) for cid in ids}


def pick_gaps(stacks, n_gaps, rng_seed=0):
    """Choose interior present frames (with present flanks) to delete."""
    gaps = []
    for cid, stack in stacks.items():
        present = np.where(stack.any(axis=(1, 2)))[0]
        if len(present) < 5:
            continue
        # interior frames that have a present neighbour on both sides
        interior = [f for f in present[2:-2]
                    if (f - 1 in present or f - 2 in present)
                    and (f + 1 in present or f + 2 in present)]
        if not interior:
            continue
        # spread a few per cell
        k = max(1, min(4, len(interior)))
        idxs = np.linspace(0, len(interior) - 1, k).round().astype(int)
        for j in idxs:
            gaps.append((cid, int(interior[j])))
    # even sample down to n_gaps across cells
    if len(gaps) > n_gaps:
        sel = np.linspace(0, len(gaps) - 1, n_gaps).round().astype(int)
        gaps = [gaps[i] for i in sel]
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="WT")
    ap.add_argument("--label", default="Pos7-WT")
    ap.add_argument("--n-gaps", type=int, default=24)
    ap.add_argument("--search-radius", type=int, default=200)
    ap.add_argument("--min-area", type=int, default=200)
    args = ap.parse_args()

    rec_dir = f"ic295_analysis/by_condition/{args.cond}/{args.label}"
    masks_p = f"{rec_dir}/pipeline_results/masks.npz"
    # source recording: the symlinked .ome.tif in the recording folder
    tif = None
    for f in os.listdir(rec_dir):
        if f.endswith(".ome.tif"):
            tif = os.path.join(rec_dir, f)
            break
    print(f"recording: {tif}\nmasks: {masks_p}", flush=True)

    from core.io import load_recording
    rec = load_recording(tif, dic_channel=1, fluo_channel=0)
    frames = rec["frames"]
    labels = np.load(masks_p)["labels"].astype(np.int32)
    n = len(frames)
    stacks = build_tracks(labels)
    print(f"{len(frames)} frames, {len(stacks)} cells", flush=True)

    gaps = pick_gaps(stacks, args.n_gaps)
    print(f"{len(gaps)} synthetic gaps\n", flush=True)

    from cellpose import models
    from core.track_gap_fill import (
        _try_primary_cpsam, _interpolate_centroid, _track_median_area)
    print("loading cpsam model…", flush=True)
    model = models.CellposeModel(gpu=True)

    rows = []
    for gi, (cid, frame_idx) in enumerate(gaps):
        true_mask = stacks[cid][frame_idx]
        if not true_mask.any():
            continue
        # gappy track copy: zero the gap frame so centroid is interpolated
        gappy = stacks[cid].copy()
        gappy[frame_idx] = False
        track = {"stack": gappy}
        cent = _interpolate_centroid(track, frame_idx, n)
        if cent is None:
            continue
        exp_area = _track_median_area(track, n)

        t0 = time.time()
        full = _try_primary_cpsam(frames[frame_idx], cent, args.search_radius,
                                  args.min_area, cpsam_model=model,
                                  expected_area=exp_area, use_crop=False)
        t_full = time.time() - t0
        t0 = time.time()
        crop = _try_primary_cpsam(frames[frame_idx], cent, args.search_radius,
                                  args.min_area, cpsam_model=model,
                                  expected_area=exp_area, use_crop=True)
        t_crop = time.time() - t0

        rows.append(dict(
            cid=cid, frame=frame_idx,
            full_fill=full is not None, crop_fill=crop is not None,
            iou_full=iou(full, true_mask), iou_crop=iou(crop, true_mask),
            iou_fc=iou(full, crop), t_full=t_full, t_crop=t_crop))
        r = rows[-1]
        print(f"  [{gi+1}/{len(gaps)}] C{cid} f{frame_idx}: "
              f"full {'Y' if r['full_fill'] else 'n'} IoU={r['iou_full']:.2f} "
              f"{r['t_full']:.1f}s | crop {'Y' if r['crop_fill'] else 'n'} "
              f"IoU={r['iou_crop']:.2f} {r['t_crop']:.1f}s", flush=True)

    if not rows:
        print("no usable gaps")
        return 1
    _summary(rows)
    return 0


def _summary(rows):
    n = len(rows)
    full_fill = sum(r["full_fill"] for r in rows)
    crop_fill = sum(r["crop_fill"] for r in rows)
    # IoU vs GT over gaps where THAT method filled
    iou_full = np.mean([r["iou_full"] for r in rows if r["full_fill"]] or [0])
    iou_crop = np.mean([r["iou_crop"] for r in rows if r["crop_fill"]] or [0])
    # agreement where both filled
    both = [r for r in rows if r["full_fill"] and r["crop_fill"]]
    iou_fc = np.mean([r["iou_fc"] for r in both] or [0])
    t_full = sum(r["t_full"] for r in rows)
    t_crop = sum(r["t_crop"] for r in rows)

    print("\n" + "=" * 60)
    print(f"gaps: {n}")
    print(f"fill rate   full {full_fill}/{n} ({100*full_fill/n:.0f}%)  "
          f"crop {crop_fill}/{n} ({100*crop_fill/n:.0f}%)")
    print(f"mean IoU vs GT   full {iou_full:.3f}   crop {iou_crop:.3f}")
    print(f"crop↔full agreement IoU (both filled, n={len(both)}): {iou_fc:.3f}")
    print(f"time   full {t_full:.0f}s ({t_full/n:.1f}s/gap)   "
          f"crop {t_crop:.0f}s ({t_crop/n:.1f}s/gap)   "
          f"speed-up {t_full/max(t_crop,1e-6):.1f}×")

    # Where the two disagree — does crop drop GOOD full fills, or only
    # poor ones? (A full fill with IoU<0.5 vs GT is a weak/wrong fill
    # that later cascade phases would handle anyway.)
    crop_miss = [r for r in rows if r["full_fill"] and not r["crop_fill"]]
    full_miss = [r for r in rows if r["crop_fill"] and not r["full_fill"]]
    if crop_miss:
        good = [r for r in crop_miss if r["iou_full"] >= 0.5]
        iou_list = ", ".join("%.2f" % r["iou_full"] for r in crop_miss)
        print(f"crop missed {len(crop_miss)} gaps full filled "
              f"(full IoU there: {iou_list}); "
              f"{len(good)} were GOOD fills (IoU>=0.5)")
    if full_miss:
        print(f"full missed {len(full_miss)} gaps crop filled")
    # On gaps BOTH filled, is crop's mask as good as full's vs GT?
    if both:
        d = np.mean([r["iou_crop"] - r["iou_full"] for r in both])
        print(f"on {len(both)} shared fills: mean IoU(crop)-IoU(full) "
              f"vs GT = {d:+.3f}")

    # Verdict: crop must not drop GOOD full fills, and shared-fill IoU
    # must not drop materially.
    dropped_good = sum(1 for r in crop_miss if r["iou_full"] >= 0.5)
    shared_iou_drop = (np.mean([r["iou_full"] - r["iou_crop"] for r in both])
                       if both else 0.0)
    ok = dropped_good == 0 and shared_iou_drop <= 0.03
    print("\nVERDICT: " + (
        "CROP MATCHES FULL on good fills — keep crop default ON"
        if ok else
        f"REVIEW — crop dropped {dropped_good} good fills / "
        f"shared-IoU drop {shared_iou_drop:+.3f}"))


if __name__ == "__main__":
    sys.exit(main())
