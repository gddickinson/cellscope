"""Investigate fluorescence vs DIC detection on a single frame.

Loads the DMSO_busy multichannel demo, runs cpsam separately on DIC
and on Cy5, plus a simple Cy5 threshold pass, and compares cell counts
and spatial coverage. Output:

  /tmp/fluo_investigation/01_dic_vs_cy5_cpsam.png  side-by-side overlays
  /tmp/fluo_investigation/01_summary.txt           counts + IoU stats

Run from cellscope4 env (has cpsam). Single frame only.
"""
import os
import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage import measure
from scipy.ndimage import label as cc_label, binary_fill_holes

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

OUT_DIR = "/tmp/fluo_investigation"
os.makedirs(OUT_DIR, exist_ok=True)

TIFF = ("data/examples/multichannel_DIC_Cy5_DMSO_busy/"
        "multichannel_DIC_Cy5_DMSO_busy.ome.tif")
FRAME_IDX = 20  # mid-recording, busy scene
MIN_AREA = 250  # px


def load_frame(idx):
    """Channels: page[2*i]=Cy5 (ch0), page[2*i+1]=DIC (ch1)."""
    with tifffile.TiffFile(TIFF) as tf:
        cy5 = tf.pages[2 * idx].asarray()
        dic = tf.pages[2 * idx + 1].asarray()
    return dic, cy5


def filter_small(labels, min_area):
    out = np.zeros_like(labels)
    nid = 0
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if m.sum() >= min_area:
            nid += 1
            out[m] = nid
    return out, nid


def cpsam_detect(image, augment=False):
    from cellpose import models
    m = models.CellposeModel(gpu=True)
    masks, _, _ = m.eval(image, augment=augment)
    return masks.astype(np.int32)


def cy5_threshold_detect(cy5, min_area=MIN_AREA):
    """Simple binary threshold + connected components on Cy5."""
    med = float(np.median(cy5))
    mad = max(1.0, float(np.median(np.abs(cy5.astype(np.float32) - med))))
    thr = med + 3.0 * mad
    binary = cy5 > thr
    binary = binary_fill_holes(binary)
    cc, _ = cc_label(binary)
    return filter_small(cc.astype(np.int32), min_area)


def jaccard_overlap_per_label(target_labels, reference_labels):
    """For each label in target, return its best Jaccard with any
    reference label."""
    out = {}
    for t in range(1, int(target_labels.max()) + 1):
        t_mask = target_labels == t
        if not t_mask.any():
            continue
        best = 0.0
        for r in range(1, int(reference_labels.max()) + 1):
            r_mask = reference_labels == r
            inter = (t_mask & r_mask).sum()
            if inter == 0:
                continue
            union = (t_mask | r_mask).sum()
            j = inter / union
            if j > best:
                best = j
        out[t] = best
    return out


def count_unique_cells(target_labels, reference_labels,
                       min_iou=0.3):
    """How many target labels have NO reference match (≥min_iou)?
    These are cells unique to target."""
    js = jaccard_overlap_per_label(target_labels, reference_labels)
    return sum(1 for j in js.values() if j < min_iou)


def draw_contours(ax, labels, color, lw=1.0):
    n = 0
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if not m.any():
            continue
        for c in measure.find_contours(m.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=color, lw=lw)
        n += 1
    return n


def main():
    print(f"Loading frame {FRAME_IDX} from {TIFF}")
    dic, cy5 = load_frame(FRAME_IDX)
    print(f"  DIC: {dic.shape} dtype={dic.dtype} "
          f"range=[{dic.min()}, {dic.max()}] mean={dic.mean():.1f}")
    print(f"  Cy5: {cy5.shape} dtype={cy5.dtype} "
          f"range=[{cy5.min()}, {cy5.max()}] mean={cy5.mean():.1f}")

    # Run three detectors
    print("\nRunning cpsam on DIC ...")
    dic_labels = cpsam_detect(dic)
    dic_labels, n_dic = filter_small(dic_labels, MIN_AREA)
    print(f"  cpsam(DIC) → {n_dic} cells (after area>={MIN_AREA})")

    print("\nRunning cpsam on Cy5 ...")
    cy5_labels = cpsam_detect(cy5)
    cy5_labels, n_cy5 = filter_small(cy5_labels, MIN_AREA)
    print(f"  cpsam(Cy5) → {n_cy5} cells (after area>={MIN_AREA})")

    print("\nRunning simple Cy5 threshold ...")
    thr_labels, n_thr = cy5_threshold_detect(cy5)
    print(f"  thresh(Cy5) → {n_thr} cells")

    # Overlap analysis
    print("\nOverlap analysis (Jaccard ≥ 0.3 considered same cell):")
    cy5_unique = count_unique_cells(cy5_labels, dic_labels, min_iou=0.3)
    thr_unique = count_unique_cells(thr_labels, dic_labels, min_iou=0.3)
    dic_unique = count_unique_cells(dic_labels, cy5_labels, min_iou=0.3)
    print(f"  cpsam(Cy5) detections NOT covered by DIC: {cy5_unique}")
    print(f"  thresh(Cy5) detections NOT covered by DIC: {thr_unique}")
    print(f"  DIC detections NOT covered by cpsam(Cy5): {dic_unique}")

    # Union estimate
    union_labels = dic_labels.copy()
    nxt = int(union_labels.max()) + 1
    for lab in range(1, int(cy5_labels.max()) + 1):
        cm = cy5_labels == lab
        if not cm.any():
            continue
        # Drop if heavily overlapping an existing DIC label
        overlap = (cm & (union_labels > 0)).sum() / max(cm.sum(), 1)
        if overlap > 0.5:
            continue
        union_labels[cm & (union_labels == 0)] = nxt
        nxt += 1
    n_union = int(union_labels.max())
    print(f"\nDIC ∪ Cy5 (cpsam) union → {n_union} cells "
          f"(+{n_union - n_dic} over DIC alone)")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(dic, cmap="gray")
    axes[0, 0].set_title(f"DIC channel  frame {FRAME_IDX}")
    n0 = draw_contours(axes[0, 0], dic_labels, "red", 1.0)
    axes[0, 0].text(0.02, 0.98,
                    f"cpsam(DIC): {n0} cells",
                    transform=axes[0, 0].transAxes, color="red",
                    fontsize=12, va="top",
                    bbox=dict(facecolor="white", alpha=0.7))

    axes[0, 1].imshow(cy5, cmap="inferno", vmin=0, vmax=80)
    axes[0, 1].set_title(f"Cy5 (F-actin) channel  frame {FRAME_IDX}")
    n1 = draw_contours(axes[0, 1], cy5_labels, "cyan", 1.0)
    axes[0, 1].text(0.02, 0.98,
                    f"cpsam(Cy5): {n1} cells",
                    transform=axes[0, 1].transAxes, color="cyan",
                    fontsize=12, va="top",
                    bbox=dict(facecolor="black", alpha=0.7))

    axes[0, 2].imshow(cy5, cmap="inferno", vmin=0, vmax=80)
    axes[0, 2].set_title("Cy5 + threshold contours")
    n2 = draw_contours(axes[0, 2], thr_labels, "yellow", 1.0)
    axes[0, 2].text(0.02, 0.98,
                    f"thresh(Cy5): {n2} cells",
                    transform=axes[0, 2].transAxes, color="yellow",
                    fontsize=12, va="top",
                    bbox=dict(facecolor="black", alpha=0.7))

    # Bottom row: overlays + unique-to-Cy5 highlighted
    axes[1, 0].imshow(dic, cmap="gray")
    axes[1, 0].set_title(f"DIC + cpsam(DIC) red, cpsam(Cy5) cyan")
    draw_contours(axes[1, 0], dic_labels, "red", 1.2)
    draw_contours(axes[1, 0], cy5_labels, "cyan", 0.8)

    # Highlight cells that are unique to Cy5 (i.e. DIC missed them)
    axes[1, 1].imshow(dic, cmap="gray")
    axes[1, 1].set_title("Cells DIC MISSED but cpsam(Cy5) caught")
    js = jaccard_overlap_per_label(cy5_labels, dic_labels)
    miss_count = 0
    for lab in range(1, int(cy5_labels.max()) + 1):
        if js.get(lab, 0) < 0.3:
            m = cy5_labels == lab
            for c in measure.find_contours(m.astype(float), 0.5):
                axes[1, 1].plot(c[:, 1], c[:, 0], color="lime", lw=2.0)
            miss_count += 1
    axes[1, 1].text(0.02, 0.98,
                    f"Missed by DIC: {miss_count} cells",
                    transform=axes[1, 1].transAxes, color="lime",
                    fontsize=12, va="top",
                    bbox=dict(facecolor="black", alpha=0.7))

    axes[1, 2].imshow(dic, cmap="gray")
    axes[1, 2].set_title(f"DIC ∪ Cy5 union: {n_union} cells")
    draw_contours(axes[1, 2], union_labels, "magenta", 1.0)

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    out = f"{OUT_DIR}/01_dic_vs_cy5_cpsam.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nSaved {out}")

    # Summary file
    summary = f"""DMSO_busy frame {FRAME_IDX}  cells with area >= {MIN_AREA} px

  cpsam(DIC)  : {n_dic}
  cpsam(Cy5)  : {n_cy5}
  thresh(Cy5) : {n_thr}

  cells DIC missed but cpsam(Cy5) caught : {cy5_unique}
  cells DIC missed but thresh(Cy5) caught: {thr_unique}
  cells cpsam(Cy5) missed but DIC caught : {dic_unique}

  DIC ∪ cpsam(Cy5) union total           : {n_union}  (+{n_union - n_dic} vs DIC alone)
"""
    print(summary)
    with open(f"{OUT_DIR}/01_summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
