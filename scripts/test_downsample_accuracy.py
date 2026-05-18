"""Quick test: does downsampled cpsam give the same cells as full-res?

Picks 5 frames from IC293 Pos3 (multi-cell, cropped, has GT), runs
cpsam at full-res and 2× downsample on each, and reports per-cell
IoU against GT for both runs.

If the IoU drop is < 0.03 on average, downsampling is safe to
recommend as default for evaluation runs.
"""
import os
import sys
import time
import numpy as np
from skimage import io as skio
from scipy.optimize import linear_sum_assignment

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

FOLDER = "data/legacy_gt/ignasi_3_cells_control_IC293_Pos3"
TIF = (FOLDER + "/IC293__1_MMStack_Pos3-WT.ome-cropped.tif")
GT_DIR = FOLDER + "/gt_masks"
FRAMES = [0, 24, 48, 72, 96]
MIN_AREA = 200


def filter_small(labels, min_area=MIN_AREA):
    out = np.zeros_like(labels)
    nid = 0
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if m.sum() >= min_area:
            nid += 1
            out[m] = nid
    return out


def cpsam_on(stack, model):
    """Per-frame cpsam → int32 label stack."""
    out = np.zeros(stack.shape, dtype=np.int32)
    t0 = time.time()
    for i in range(len(stack)):
        out[i] = model.eval(stack[i], augment=False)[0].astype(np.int32)
    return out, time.time() - t0


def upscale_labels(arr, target_shape):
    """Nearest-neighbour upscale a label stack to target HxW."""
    import cv2
    H, W = target_shape
    out = np.empty((len(arr), H, W), dtype=arr.dtype)
    for i in range(len(arr)):
        out[i] = cv2.resize(arr[i], (W, H),
                             interpolation=cv2.INTER_NEAREST)
    return out


def downsample(stack, factor):
    import cv2
    H, W = stack.shape[1:]
    new_h, new_w = H // factor, W // factor
    out = np.empty((len(stack), new_h, new_w), dtype=stack.dtype)
    for i in range(len(stack)):
        out[i] = cv2.resize(stack[i], (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
    return out


def per_cell_iou_vs_gt(gt, pred):
    """Hungarian-matched per-cell IoU."""
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
    return [float(iou[r, c]) for r, c in zip(rows, cols)
            if iou[r, c] > 0]


def main():
    from core.io import load_video
    from cellpose import models

    print("Loading frames + GT …")
    full = load_video(TIF)
    print(f"  Full-res shape: {full.shape}")

    sample_full = full[FRAMES]
    sample_half = downsample(sample_full, 2)
    print(f"  Sample full: {sample_full.shape}, "
          f"half: {sample_half.shape}")

    gt_stacks = [skio.imread(os.path.join(GT_DIR, f"mask_F{fi}.png"))
                 for fi in FRAMES]

    cpsam = models.CellposeModel(gpu=True)

    print("\nRunning cpsam at FULL resolution …")
    labels_full, t_full = cpsam_on(sample_full, cpsam)
    labels_full = np.array([filter_small(l) for l in labels_full])

    print("Running cpsam at 2× DOWNSAMPLED resolution …")
    labels_half_small, t_half = cpsam_on(sample_half, cpsam)
    # Adjust min_area for downsampled frames (¼ the px)
    labels_half_small = np.array(
        [filter_small(l, min_area=MIN_AREA // 4) for l in labels_half_small])
    # Upscale labels back to full resolution
    labels_half = upscale_labels(labels_half_small, sample_full.shape[1:])

    print()
    print("=" * 60)
    print(f"Timing: full={t_full:.1f}s  half={t_half:.1f}s  "
          f"speedup={t_full/t_half:.1f}×")
    print("=" * 60)

    print()
    print(f"{'frame':>5} {'gt_cells':>8} "
          f"{'full_cells':>10} {'half_cells':>10} "
          f"{'full_iou':>9} {'half_iou':>9} {'diff':>8}")
    full_ious, half_ious = [], []
    for j, fi in enumerate(FRAMES):
        gt = gt_stacks[j].astype(np.int32)
        full_iou = per_cell_iou_vs_gt(gt, labels_full[j])
        half_iou = per_cell_iou_vs_gt(gt, labels_half[j])
        full_avg = float(np.mean(full_iou)) if full_iou else 0
        half_avg = float(np.mean(half_iou)) if half_iou else 0
        n_full = int(labels_full[j].max())
        n_half = int(labels_half[j].max())
        print(f"{fi:>5} {int(gt.max()):>8} {n_full:>10} {n_half:>10} "
              f"{full_avg:>9.3f} {half_avg:>9.3f} "
              f"{half_avg - full_avg:>+8.3f}")
        full_ious.extend(full_iou)
        half_ious.extend(half_iou)

    print()
    print(f"Mean IoU full:  {np.mean(full_ious):.3f} "
          f"({len(full_ious)} matched cells)")
    print(f"Mean IoU half:  {np.mean(half_ious):.3f} "
          f"({len(half_ious)} matched cells)")
    print(f"ΔIoU (half − full): {np.mean(half_ious) - np.mean(full_ious):+.3f}")
    print(f"Speedup: {t_full / t_half:.2f}×")


if __name__ == "__main__":
    main()
