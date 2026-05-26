"""Spatial diff of the two Pos7_WT pipeline runs.

Compares:
  data/ic295_gt_full/Pos7_WT/pipeline_results_v2_cpsam_noalign/masks.npz
    (32 raw tracks → 17 kept, full-res, no alignment)
  data/ic295_gt_full/Pos7_WT/pipeline_results/masks.npz
    (21 raw → 11 kept, downsample=2 + alignment)

For each annotated frame, matches cells between the two runs and
reports which cells are in one but not the other + their areas + GT
match status. Tells us whether the "extra" tracks in the no-alignment
run were false positives (no GT match) or real cells (GT match).
"""
import os
import sys
import numpy as np
from skimage import io as skio
from scipy.optimize import linear_sum_assignment

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

FOLDER = "data/ic295_gt_full/Pos7_WT"
OLD = FOLDER + "/pipeline_results_v2_cpsam_noalign/masks.npz"
NEW = FOLDER + "/pipeline_results/masks.npz"
GT_DIR = FOLDER + "/gt_masks"


def per_cell_iou(a, b):
    gids = [i for i in range(1, int(a.max()) + 1) if (a == i).any()]
    pids = [i for i in range(1, int(b.max()) + 1) if (b == i).any()]
    if not gids or not pids:
        return [], gids, pids
    iou = np.zeros((len(gids), len(pids)))
    for i, g in enumerate(gids):
        gm = a == g
        for j, p in enumerate(pids):
            pm = b == p
            inter = (gm & pm).sum()
            if inter == 0:
                continue
            iou[i, j] = inter / (gm | pm).sum()
    rows, cols = linear_sum_assignment(-iou)
    matched = []
    for r, c in zip(rows, cols):
        if iou[r, c] > 0.1:
            matched.append((gids[r], pids[c], float(iou[r, c])))
    return matched, gids, pids


def main():
    old_labels = np.load(OLD)["labels"]
    new_labels = np.load(NEW)["labels"]
    print(f"OLD (full-res no-align): {old_labels.shape}, "
          f"max ID {old_labels.max()}")
    print(f"NEW (downsample+align):  {new_labels.shape}, "
          f"max ID {new_labels.max()}")

    # Per-annotated-frame analysis
    print()
    print(f"{'frame':>5} {'gt':>3} "
          f"{'old_cells':>9} {'new_cells':>9} "
          f"{'old_TP':>6} {'new_TP':>6} "
          f"{'in_old_only':>11} {'in_new_only':>11}")
    summary_old_extras = []   # (frame, area, gt_match_iou)
    summary_new_extras = []

    for f in sorted(os.listdir(GT_DIR)):
        if not (f.startswith("mask_F") and f.endswith(".png")):
            continue
        fi = int(f[len("mask_F"):-len(".png")])
        gt = skio.imread(os.path.join(GT_DIR, f)).astype(np.int32)
        old = old_labels[fi]
        new = new_labels[fi]

        n_gt = int(gt.max())
        n_old = sum(1 for i in range(1, int(old.max()) + 1)
                     if (old == i).any())
        n_new = sum(1 for i in range(1, int(new.max()) + 1)
                     if (new == i).any())

        # GT matches for each
        gt_old_matches, _, _ = per_cell_iou(gt, old)
        gt_new_matches, _, _ = per_cell_iou(gt, new)
        old_tp = sum(1 for g, p, i in gt_old_matches if i >= 0.5)
        new_tp = sum(1 for g, p, i in gt_new_matches if i >= 0.5)

        # cell-set diff between old and new
        oldnew_matches, old_pids, new_pids = per_cell_iou(old, new)
        matched_old = {m[0] for m in oldnew_matches}
        matched_new = {m[1] for m in oldnew_matches}
        old_only = [p for p in old_pids if p not in matched_old]
        new_only = [p for p in new_pids if p not in matched_new]

        print(f"{fi:>5} {n_gt:>3} "
              f"{n_old:>9} {n_new:>9} "
              f"{old_tp:>6} {new_tp:>6} "
              f"{len(old_only):>11} {len(new_only):>11}")

        # Detail per cell in each "only" set
        for cid in old_only:
            cell = old == cid
            area = int(cell.sum())
            # Best GT IoU
            best_gt = 0.0
            for g in range(1, int(gt.max()) + 1):
                gm = gt == g
                inter = (cell & gm).sum()
                if inter > 0:
                    iou_ = inter / (cell | gm).sum()
                    best_gt = max(best_gt, iou_)
            summary_old_extras.append((fi, cid, area, best_gt))
        for cid in new_only:
            cell = new == cid
            area = int(cell.sum())
            best_gt = 0.0
            for g in range(1, int(gt.max()) + 1):
                gm = gt == g
                inter = (cell & gm).sum()
                if inter > 0:
                    iou_ = inter / (cell | gm).sum()
                    best_gt = max(best_gt, iou_)
            summary_new_extras.append((fi, cid, area, best_gt))

    print()
    print("=" * 60)
    print(f"Cells PRESENT IN OLD BUT NOT IN NEW (would be 'lost' "
          f"by going to downsample+align)")
    print("=" * 60)
    if summary_old_extras:
        print(f"{'frame':>5} {'cell':>5} {'area_px':>9} "
              f"{'best_gt_iou':>11} {'status':>20}")
        n_real = 0
        n_fp = 0
        areas_real = []
        areas_fp = []
        for fi, cid, area, gt_iou in summary_old_extras:
            status = ("REAL (matches GT)" if gt_iou >= 0.3
                      else "FALSE POSITIVE")
            if gt_iou >= 0.3:
                n_real += 1
                areas_real.append(area)
            else:
                n_fp += 1
                areas_fp.append(area)
            print(f"{fi:>5} {cid:>5} {area:>9} "
                  f"{gt_iou:>11.3f} {status:>20}")
        print(f"\n  Total old-only: {len(summary_old_extras)} cells")
        print(f"    REAL (lost by going to new): {n_real} "
              f"(median area {int(np.median(areas_real)) if areas_real else 0} px)")
        print(f"    FALSE POSITIVES (good we lost them): "
              f"{n_fp} (median area "
              f"{int(np.median(areas_fp)) if areas_fp else 0} px)")
    else:
        print("  (none — new run contains everything from old)")

    print()
    print("=" * 60)
    print(f"Cells PRESENT IN NEW BUT NOT IN OLD (cells GAINED)")
    print("=" * 60)
    if summary_new_extras:
        n_real = 0; n_fp = 0
        for fi, cid, area, gt_iou in summary_new_extras:
            status = ("REAL (matches GT)" if gt_iou >= 0.3
                      else "FALSE POSITIVE")
            if gt_iou >= 0.3:
                n_real += 1
            else:
                n_fp += 1
            print(f"{fi:>5} {cid:>5} {area:>9} "
                  f"{gt_iou:>11.3f} {status:>20}")
        print(f"\n  Total new-only: {len(summary_new_extras)} cells")
        print(f"    REAL (gained): {n_real}")
        print(f"    FALSE POSITIVES (gained but bad): {n_fp}")
    else:
        print("  (none — old run contains everything from new)")


if __name__ == "__main__":
    main()
