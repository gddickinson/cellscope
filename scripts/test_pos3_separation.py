"""Test what would separate the merged cells in IC293 Pos3.

Tries multiple detection strategies on a few problem frames:
  A. cpsam_dic default (current pipeline) — baseline merge problem
  B. cpsam_dic with TTA (4-rotation augmentation)
  C. cpsam (vit_h, no domain bias) at default diameter
  D. cpsam with diameter=30 (smaller — biases toward separation)
  E. cpsam with diameter=15 (even smaller)
  F. cpsam_dic + watershed split on the merged blobs

Reports cells found per strategy + which match GT.
"""
import os
import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import io as skio, measure
from scipy.optimize import linear_sum_assignment

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

OUT_DIR = "/tmp/fluo_investigation"
REC = "data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/"\
      "IC293__1_MMStack_Pos3-WT.ome-cropped.tif"
GT_DIR = "data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/gt_masks"
FRAMES_TO_TEST = [0, 24, 48, 72]
MIN_AREA = 200


def filter_small(labels, min_area):
    out = np.zeros_like(labels)
    nid = 0
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if m.sum() >= min_area:
            nid += 1
            out[m] = nid
    return out, nid


def watershed_split(labels, min_distance=15):
    """Split large blobs using distance-transform watershed."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    out = np.zeros_like(labels)
    nid = 0
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if not m.any():
            continue
        dist = ndi.distance_transform_edt(m)
        coords = peak_local_max(dist, min_distance=min_distance,
                                 labels=m.astype(int))
        if len(coords) <= 1:
            nid += 1
            out[m] = nid
            continue
        markers = np.zeros_like(m, dtype=int)
        for k, (y, x) in enumerate(coords, start=1):
            markers[y, x] = k
        markers = ndi.label(markers > 0)[0]
        ws = watershed(-dist, markers, mask=m)
        for k in range(1, ws.max() + 1):
            sub = ws == k
            if sub.sum() < MIN_AREA:
                continue
            nid += 1
            out[sub] = nid
    return out, nid


def best_iou_matches(gt, pred):
    """Hungarian matching, returns list of (gt_id, pred_id, iou)."""
    gids = list(range(1, int(gt.max()) + 1))
    pids = list(range(1, int(pred.max()) + 1))
    if not gids or not pids:
        return []
    iou = np.zeros((len(gids), len(pids)))
    for i, g in enumerate(gids):
        gm = gt == g
        for j, p in enumerate(pids):
            pm = pred == p
            inter = (gm & pm).sum()
            if inter == 0:
                continue
            iou[i, j] = inter / (gm | pm).sum()
    rows, cols = linear_sum_assignment(-iou)
    out = []
    for r, c in zip(rows, cols):
        if iou[r, c] > 0.1:
            out.append((gids[r], pids[c], float(iou[r, c])))
    return out


def main():
    from cellpose import models
    print("Loading models …")
    cpsam_dic = models.CellposeModel(
        gpu=True, pretrained_model="data/models/cpsam_dic")
    cpsam = models.CellposeModel(gpu=True)

    from core.io import load_video
    print("Loading frames …")
    frames = load_video(REC)

    print(f"\n{'frame':>5} {'strategy':>16} {'n_cells':>8} "
          f"{'matched':>8} {'avg_iou':>8}")
    print("-" * 60)
    rows = []
    for fi in FRAMES_TO_TEST:
        gt = skio.imread(os.path.join(GT_DIR, f"mask_F{fi}.png"))
        n_gt = int(gt.max())
        print(f"\n--- F{fi}: GT has {n_gt} cells ---")

        # A. cpsam_dic default
        labA = cpsam_dic.eval(frames[fi], augment=False)[0].astype(np.int32)
        labA, nA = filter_small(labA, MIN_AREA)
        mA = best_iou_matches(gt, labA)
        avgA = np.mean([m[2] for m in mA]) if mA else 0
        print(f"  A default cpsam_dic:  {nA} cells, "
              f"matched {len(mA)}/{n_gt}, avg IoU {avgA:.3f}")

        # B. cpsam_dic + TTA
        labB = cpsam_dic.eval(frames[fi], augment=True)[0].astype(np.int32)
        labB, nB = filter_small(labB, MIN_AREA)
        mB = best_iou_matches(gt, labB)
        avgB = np.mean([m[2] for m in mB]) if mB else 0
        print(f"  B cpsam_dic + TTA:    {nB} cells, "
              f"matched {len(mB)}/{n_gt}, avg IoU {avgB:.3f}")

        # C. raw cpsam default
        labC = cpsam.eval(frames[fi], augment=False)[0].astype(np.int32)
        labC, nC = filter_small(labC, MIN_AREA)
        mC = best_iou_matches(gt, labC)
        avgC = np.mean([m[2] for m in mC]) if mC else 0
        print(f"  C raw cpsam (vit_h):  {nC} cells, "
              f"matched {len(mC)}/{n_gt}, avg IoU {avgC:.3f}")

        # D. cpsam diameter=30
        labD = cpsam.eval(frames[fi], augment=False,
                           diameter=30)[0].astype(np.int32)
        labD, nD = filter_small(labD, MIN_AREA)
        mD = best_iou_matches(gt, labD)
        avgD = np.mean([m[2] for m in mD]) if mD else 0
        print(f"  D cpsam diameter=30:  {nD} cells, "
              f"matched {len(mD)}/{n_gt}, avg IoU {avgD:.3f}")

        # E. cpsam diameter=50
        labE = cpsam.eval(frames[fi], augment=False,
                           diameter=50)[0].astype(np.int32)
        labE, nE = filter_small(labE, MIN_AREA)
        mE = best_iou_matches(gt, labE)
        avgE = np.mean([m[2] for m in mE]) if mE else 0
        print(f"  E cpsam diameter=50:  {nE} cells, "
              f"matched {len(mE)}/{n_gt}, avg IoU {avgE:.3f}")

        # F. cpsam_dic + watershed split
        labF, nF = watershed_split(labA.copy(), min_distance=15)
        mF = best_iou_matches(gt, labF)
        avgF = np.mean([m[2] for m in mF]) if mF else 0
        print(f"  F A + watershed split: {nF} cells, "
              f"matched {len(mF)}/{n_gt}, avg IoU {avgF:.3f}")

        rows.append({
            "frame": fi, "n_gt": n_gt,
            "default_n": nA, "default_matched": len(mA), "default_iou": avgA,
            "tta_n": nB, "tta_matched": len(mB), "tta_iou": avgB,
            "cpsam_n": nC, "cpsam_matched": len(mC), "cpsam_iou": avgC,
            "d30_n": nD, "d30_matched": len(mD), "d30_iou": avgD,
            "d50_n": nE, "d50_matched": len(mE), "d50_iou": avgE,
            "ws_n": nF, "ws_matched": len(mF), "ws_iou": avgF,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (4 frames, 3 GT cells each = 12 GT cells total)")
    print("=" * 60)
    for k, label in [
            ("default", "A. cpsam_dic default (current pipeline)"),
            ("tta",     "B. cpsam_dic + TTA"),
            ("cpsam",   "C. raw cpsam (vit_h)"),
            ("d30",     "D. cpsam diameter=30"),
            ("d50",     "E. cpsam diameter=50"),
            ("ws",      "F. cpsam_dic + watershed split"),
    ]:
        total_n = sum(r[f"{k}_n"] for r in rows)
        total_matched = sum(r[f"{k}_matched"] for r in rows)
        avg_iou = np.mean([r[f"{k}_iou"] for r in rows
                            if r[f"{k}_iou"] > 0])
        print(f"  {label:<45} cells={total_n:>3} "
              f"matched={total_matched:>2}/12 "
              f"avg_IoU={avg_iou:.3f}")


if __name__ == "__main__":
    main()
