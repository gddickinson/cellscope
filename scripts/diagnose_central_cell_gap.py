"""Diagnose why cpsam_dic missed the central cell in frames 20-29.

Shows the DIC + Cy5 crop around the cell's last known position, then
tries 3 alternative detection strategies to find what would have
caught it:
  A. cpsam_dic default (the current pipeline)
  B. cpsam_dic with augment=True (TTA — 4-rotation vote)
  C. cpsam_dic at multiple diameters (auto, 1.5×, 0.7×) merged

Output: /tmp/fluo_investigation/09_gap_diagnosis.png + counts table.
"""
import os
import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import measure

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

OUT_DIR = "/tmp/fluo_investigation"
os.makedirs(OUT_DIR, exist_ok=True)

REC = os.path.join(
    CELLSCOPE_ROOT,
    "data/examples/multichannel_DIC_Cy5_DMSO_busy/"
    "multichannel_DIC_Cy5_DMSO_busy.ome.tif")

GAP_FRAMES = [18, 21, 23, 25, 27, 29]    # span of the gap
EXPECTED_CY = 665                          # last-known centroid y
EXPECTED_CX = 530                          # last-known centroid x
CROP_HALF = 150                            # 300x300 crop


def load_frame(idx):
    with tifffile.TiffFile(REC) as tf:
        cy5 = tf.pages[2 * idx].asarray()
        dic = tf.pages[2 * idx + 1].asarray()
    return dic, cy5


def detect_dic(model, img, augment=False, diameter=None):
    if diameter is None:
        out = model.eval(img, augment=augment)
    else:
        out = model.eval(img, augment=augment, diameter=diameter)
    return out[0].astype(np.int32)


def filter_by_area(labels, min_area=200):
    out = np.zeros_like(labels)
    nid = 0
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if m.sum() >= min_area:
            nid += 1
            out[m] = nid
    return out, nid


def in_central_box(labels, cy, cx, half=80):
    """Return True if any label has its centroid within `half` px
    of (cy, cx)."""
    H, W = labels.shape
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if not m.any():
            continue
        ys, xs = np.where(m)
        c_y, c_x = ys.mean(), xs.mean()
        if abs(c_y - cy) < half and abs(c_x - cx) < half:
            return True, lab, (c_y, c_x), int(m.sum())
    return False, 0, (0, 0), 0


def main():
    from core.multichannel import to_uint8_dic, to_uint8_fluorescence
    from cellpose import models

    print("Loading cpsam_dic …")
    mdic = models.CellposeModel(gpu=True,
                                 pretrained_model="data/models/cpsam_dic")
    print("Loading cpsam (vit_h) …")
    mcpsam = models.CellposeModel(gpu=True)

    fig, axes = plt.subplots(len(GAP_FRAMES), 5,
                              figsize=(22, 4 * len(GAP_FRAMES)))
    rows = []

    for ri, fi in enumerate(GAP_FRAMES):
        dic_raw, cy5_raw = load_frame(fi)
        dic = to_uint8_dic(dic_raw)
        cy5 = to_uint8_fluorescence(cy5_raw)

        H, W = dic.shape
        cy0 = max(0, EXPECTED_CY - CROP_HALF)
        cy1 = min(H, EXPECTED_CY + CROP_HALF)
        cx0 = max(0, EXPECTED_CX - CROP_HALF)
        cx1 = min(W, EXPECTED_CX + CROP_HALF)
        dic_crop = dic[cy0:cy1, cx0:cx1]
        cy5_crop = cy5[cy0:cy1, cx0:cx1]
        local_cy = EXPECTED_CY - cy0
        local_cx = EXPECTED_CX - cx0

        # A: default cpsam_dic (full frame)
        labA = detect_dic(mdic, dic, augment=False)
        labA, nA = filter_by_area(labA, 200)
        hitA, idA, centA, areaA = in_central_box(labA, EXPECTED_CY,
                                                   EXPECTED_CX)

        # B: TTA cpsam_dic
        labB = detect_dic(mdic, dic, augment=True)
        labB, nB = filter_by_area(labB, 200)
        hitB, idB, centB, areaB = in_central_box(labB, EXPECTED_CY,
                                                   EXPECTED_CX)

        # C: multi-scale cpsam_dic — diameters [auto, 1.5x, 0.7x]
        # Auto diameter is None (cellpose decides). For multi-scale
        # we hint actual sizes (px). On VAMPIRE crops cpsam_dic was
        # trained for ~30-px diameter cells; the central spread cell
        # has effective diameter ~50px. Try 30, 50, 80 px.
        labels_multi = []
        for d in (30, 50, 80):
            lab = detect_dic(mdic, dic, augment=False, diameter=d)
            lab, _ = filter_by_area(lab, 200)
            labels_multi.append(lab)
        # Merge by union: assign new IDs as we go
        labC = labels_multi[0].copy()
        nxt = int(labC.max()) + 1
        for lab2 in labels_multi[1:]:
            for lab in range(1, int(lab2.max()) + 1):
                m = lab2 == lab
                if not m.any():
                    continue
                # Skip if substantially overlaps an existing label
                cov = (m & (labC > 0)).sum() / max(m.sum(), 1)
                if cov > 0.3:
                    continue
                labC[m & (labC == 0)] = nxt
                nxt += 1
        labC, nC = filter_by_area(labC, 200)
        hitC, idC, centC, areaC = in_central_box(labC, EXPECTED_CY,
                                                   EXPECTED_CX)

        # D: cpsam (vit_h) on Cy5 — compare for reference
        labD = detect_dic(mcpsam, cy5, augment=False)
        labD, nD = filter_by_area(labD, 200)
        hitD, idD, centD, areaD = in_central_box(labD, EXPECTED_CY,
                                                   EXPECTED_CX)

        msg = (f"F{fi:>2}: A_default={'✓' if hitA else '✗'} "
               f"({nA} cells) | "
               f"B_TTA={'✓' if hitB else '✗'} ({nB}) | "
               f"C_multiscale={'✓' if hitC else '✗'} ({nC}) | "
               f"D_cpsamCy5={'✓' if hitD else '✗'} ({nD})")
        print(msg)
        rows.append({"frame": fi, "default": hitA, "tta": hitB,
                      "multiscale": hitC, "cpsam_cy5": hitD,
                      "area_default": areaA, "area_tta": areaB,
                      "area_multi": areaC, "area_cy5": areaD})

        # Crop visualizations — 5 columns: DIC crop alone, then 4 strategies
        axes[ri, 0].imshow(dic_crop, cmap="gray")
        axes[ri, 0].plot(local_cx, local_cy, "x", color="red",
                          markersize=12, mew=2)
        axes[ri, 0].set_title(f"F{fi}  DIC crop (X = expected centroid)")

        for col, (lab, n, hit, name) in enumerate([
                (labA, nA, hitA, "A: default"),
                (labB, nB, hitB, "B: TTA"),
                (labC, nC, hitC, "C: multi-scale"),
                (labD, nD, hitD, "D: cpsam(Cy5)")]):
            ax = axes[ri, col + 1]
            base = cy5_crop if "Cy5" in name else dic_crop
            cmap = "inferno" if "Cy5" in name else "gray"
            vmax = (np.percentile(cy5_crop, 99)
                    if "Cy5" in name else None)
            ax.imshow(base, cmap=cmap, vmax=vmax)
            # Draw any contours falling inside this crop
            for L in range(1, int(lab.max()) + 1):
                m = lab == L
                if not m.any():
                    continue
                m_crop = m[cy0:cy1, cx0:cx1]
                if not m_crop.any():
                    continue
                color = "lime" if hit else "yellow"
                # Specifically mark the central cell if found
                ys, xs = np.where(m)
                c_y, c_x = ys.mean(), xs.mean()
                if (abs(c_y - EXPECTED_CY) < 80
                        and abs(c_x - EXPECTED_CX) < 80):
                    color = "lime"
                else:
                    color = "cyan" if "Cy5" in name else "red"
                for c in measure.find_contours(m_crop.astype(float),
                                                0.5):
                    ax.plot(c[:, 1], c[:, 0], color=color, lw=1.5)
            ax.set_title(
                f"F{fi}  {name}  found:{'✓' if hit else '✗'}")
        for ax in axes[ri]:
            ax.axis("off")

    plt.tight_layout()
    out = f"{OUT_DIR}/09_central_cell_gap.png"
    plt.savefig(out, dpi=85, bbox_inches="tight")
    print(f"\nSaved {out}")

    # Summary
    print("\n=== Detection rate on the central cell across "
          f"{len(GAP_FRAMES)} gap frames ===")
    for k in ("default", "tta", "multiscale", "cpsam_cy5"):
        n_hit = sum(1 for r in rows if r[k])
        print(f"  {k:>14}: {n_hit}/{len(rows)} frames")


if __name__ == "__main__":
    main()
