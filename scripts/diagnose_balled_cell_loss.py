"""Trace where balled cells get lost in the multichannel pipeline.

For a single frame:
  1. Show DIC + raw cpsam_dic(DIC) result    (column 1)
  2. Show Cy5 + raw cpsam(Cy5) result (NO area filter, every label)  (column 2)
  3. Show Cy5 + cpsam(Cy5) result AFTER min_area_px=500 filter  (column 3)
  4. Show pipeline's final saved labels overlaid on DIC  (column 4)
  5. Highlight `balled candidate' regions identified by simple
     Cy5 thresholding + circularity (col 4)

Output: /tmp/fluo_investigation/07_balled_diagnosis.png + text report.

Compares cell counts at each stage; whichever stage drops most balled
candidates is the culprit.
"""
import os
import sys
import json
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import (
    label as cc_label, binary_fill_holes, binary_opening)

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

OUT_DIR = "/tmp/fluo_investigation"
os.makedirs(OUT_DIR, exist_ok=True)

REC = os.path.join(
    CELLSCOPE_ROOT,
    "data/examples/multichannel_DIC_Cy5_DMSO_busy/"
    "multichannel_DIC_Cy5_DMSO_busy.ome.tif")
SAVED_LABELS = os.path.join(
    CELLSCOPE_ROOT,
    "data/examples/multichannel_DIC_Cy5_DMSO_busy/results/masks.npz")

# Test a few frames so we can see frame-by-frame variation
FRAMES_TO_TEST = [10, 20, 30]
MIN_AREA_PX = 500          # pipeline default
CY5_BALLED_K_MAD = 4.0     # threshold for "bright spot" candidates


def load_frame(idx):
    with tifffile.TiffFile(REC) as tf:
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


def balled_candidates(cy5, k_mad=CY5_BALLED_K_MAD, min_area=80,
                       max_area=2000, min_circularity=0.6):
    """Find bright, compact (likely balled) regions in Cy5."""
    med = float(np.median(cy5))
    mad = max(1.0, float(np.median(
        np.abs(cy5.astype(np.float32) - med))))
    bright = cy5 > med + k_mad * mad
    bright = binary_fill_holes(binary_opening(bright, iterations=1))
    cc, _ = cc_label(bright)
    out = []
    for cid in range(1, int(cc.max()) + 1):
        m = cc == cid
        a = int(m.sum())
        if not (min_area <= a <= max_area):
            continue
        # Circularity = 4πA / P²
        peri = measure.perimeter(m)
        if peri == 0:
            continue
        circ = (4 * np.pi * a) / (peri ** 2)
        if circ < min_circularity:
            continue
        ys, xs = np.where(m)
        out.append({
            "mask": m,
            "centroid": (float(ys.mean()), float(xs.mean())),
            "area": a,
            "circularity": float(circ),
        })
    return out


def count_overlap_with_labels(candidate_mask, labels):
    """Count how many unique non-zero label IDs overlap with this
    candidate mask (with overlap fraction > 30% of candidate)."""
    if labels is None:
        return 0
    overlap = labels[candidate_mask]
    unique = set(int(v) for v in overlap if v > 0)
    matched = []
    for lab in unique:
        full = (labels == lab) & candidate_mask
        if full.sum() / max(candidate_mask.sum(), 1) > 0.3:
            matched.append(lab)
    return len(matched)


def main():
    print("Loading cpsam …")
    from cellpose import models
    cpsam = models.CellposeModel(gpu=True)
    cpsam_dic_path = "data/models/cpsam_dic"
    print(f"Loading cpsam_dic from {cpsam_dic_path} …")
    cpsam_dic = models.CellposeModel(gpu=True,
                                       pretrained_model=cpsam_dic_path)

    saved = np.load(SAVED_LABELS, allow_pickle=True)
    final_labels = saved["labels"]
    print(f"Saved final labels: {final_labels.shape}")

    fig, axes = plt.subplots(len(FRAMES_TO_TEST), 5,
                              figsize=(28, 5.5 * len(FRAMES_TO_TEST)))
    report_lines = []

    for ri, fi in enumerate(FRAMES_TO_TEST):
        dic, cy5 = load_frame(fi)
        print(f"\n--- Frame {fi} ---")

        # cpsam_dic on DIC (raw, no area filter)
        dic_raw = cpsam_dic.eval(dic, augment=False)[0].astype(np.int32)
        dic_filt, n_dic = filter_small(dic_raw, MIN_AREA_PX)
        n_dic_raw = int(dic_raw.max())
        # cpsam on Cy5 (raw, no area filter)
        cy5_raw = cpsam.eval(cy5, augment=False)[0].astype(np.int32)
        cy5_filt, n_cy5 = filter_small(cy5_raw, MIN_AREA_PX)
        n_cy5_raw = int(cy5_raw.max())

        # Cell areas: distribution of all labels (raw, all sizes)
        dic_areas = [int((dic_raw == lab).sum())
                     for lab in range(1, n_dic_raw + 1)]
        cy5_areas = [int((cy5_raw == lab).sum())
                     for lab in range(1, n_cy5_raw + 1)]

        # Balled candidates from Cy5 thresholding
        candidates = balled_candidates(cy5)
        n_cand = len(candidates)
        # Final pipeline labels at this frame
        final = final_labels[fi]
        n_final = len(set(int(c) for c in np.unique(final) if c))

        # For each balled candidate, where does it survive?
        cand_caught_by_dic_raw = 0
        cand_caught_by_cy5_raw = 0
        cand_caught_by_dic_filt = 0
        cand_caught_by_cy5_filt = 0
        cand_in_final = 0
        for c in candidates:
            if count_overlap_with_labels(c["mask"], dic_raw):
                cand_caught_by_dic_raw += 1
            if count_overlap_with_labels(c["mask"], cy5_raw):
                cand_caught_by_cy5_raw += 1
            if count_overlap_with_labels(c["mask"], dic_filt):
                cand_caught_by_dic_filt += 1
            if count_overlap_with_labels(c["mask"], cy5_filt):
                cand_caught_by_cy5_filt += 1
            if count_overlap_with_labels(c["mask"], final):
                cand_in_final += 1

        msg = (f"Frame {fi:>3} | cpsam_dic raw: {n_dic_raw:>3} cells "
               f"(areas {sorted(dic_areas)[:5]}...) | "
               f"cpsam_dic ≥500px: {n_dic} | "
               f"cpsam(Cy5) raw: {n_cy5_raw:>3} "
               f"(areas {sorted(cy5_areas)[:5]}...) | "
               f"cpsam(Cy5) ≥500px: {n_cy5} | "
               f"Final saved: {n_final}")
        print(msg)
        cand_msg = (f"           Balled candidates: {n_cand} | "
                    f"caught by cpsam_dic raw: {cand_caught_by_dic_raw} | "
                    f"by cpsam_dic ≥500: {cand_caught_by_dic_filt} | "
                    f"by cpsam(Cy5) raw: {cand_caught_by_cy5_raw} | "
                    f"by cpsam(Cy5) ≥500: {cand_caught_by_cy5_filt} | "
                    f"in FINAL pipeline output: {cand_in_final}")
        print(cand_msg)
        report_lines.append(msg)
        report_lines.append(cand_msg)

        # Per-frame plots
        ax = axes[ri, 0]
        ax.imshow(dic, cmap="gray")
        ax.set_title(f"F{fi}  DIC + cpsam_dic raw ({n_dic_raw} cells)")
        for lab in range(1, n_dic_raw + 1):
            m = dic_raw == lab
            if not m.any():
                continue
            for c in measure.find_contours(m.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color="red", lw=0.9)

        ax = axes[ri, 1]
        ax.imshow(cy5, cmap="inferno", vmin=0,
                  vmax=np.percentile(cy5, 99))
        ax.set_title(f"F{fi}  Cy5 + cpsam(Cy5) RAW (no filter): "
                     f"{n_cy5_raw} cells")
        for lab in range(1, n_cy5_raw + 1):
            m = cy5_raw == lab
            if not m.any():
                continue
            a = m.sum()
            color = "cyan" if a >= MIN_AREA_PX else "yellow"
            for c in measure.find_contours(m.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color=color, lw=0.9)

        ax = axes[ri, 2]
        ax.imshow(cy5, cmap="inferno", vmin=0,
                  vmax=np.percentile(cy5, 99))
        ax.set_title(f"F{fi}  Cy5 + cpsam(Cy5) ≥{MIN_AREA_PX}px: "
                     f"{n_cy5} cells (cyan)")
        for lab in range(1, int(cy5_filt.max()) + 1):
            m = cy5_filt == lab
            if not m.any():
                continue
            for c in measure.find_contours(m.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color="cyan", lw=1.2)

        ax = axes[ri, 3]
        ax.imshow(dic, cmap="gray")
        ax.set_title(f"F{fi}  Final pipeline output: {n_final} cells")
        if n_final > 0:
            for lab in range(1, int(final.max()) + 1):
                m = final == lab
                if not m.any():
                    continue
                for c in measure.find_contours(m.astype(float), 0.5):
                    ax.plot(c[:, 1], c[:, 0], color="magenta", lw=1.2)

        ax = axes[ri, 4]
        ax.imshow(dic, cmap="gray")
        ax.set_title(f"F{fi}  Balled cand ({n_cand}); in final: "
                     f"{cand_in_final}")
        for c in candidates:
            in_final = bool(count_overlap_with_labels(c["mask"], final))
            color = "lime" if in_final else "yellow"
            for cont in measure.find_contours(c["mask"].astype(float), 0.5):
                ax.plot(cont[:, 1], cont[:, 0], color=color, lw=2.0)
            cy_, cx = c["centroid"]
            ax.text(cx, cy_, f"A={c['area']}\nC={c['circularity']:.2f}",
                    color="white", fontsize=7, ha="center", va="center",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1))

        for a in axes[ri]:
            a.axis("off")

    plt.tight_layout()
    out = f"{OUT_DIR}/07_balled_diagnosis.png"
    plt.savefig(out, dpi=85, bbox_inches="tight")
    print(f"\nSaved {out}")

    with open(f"{OUT_DIR}/07_balled_diagnosis.txt", "w") as f:
        f.write("\n".join(report_lines))


if __name__ == "__main__":
    main()
