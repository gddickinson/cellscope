"""Prototype + benchmark three Cy5 fusion strategies on single frames.

  A. cellpose_dic(DIC) alone  (status quo)
  B. cellpose_dic(DIC) ∪ cpsam(Cy5)  (user's idea: stage-1 fusion)
  C. cellpose_dic(DIC) ∪ Cy5_threshold_seeded_cpsam_on_DIC_crop
     (Cy5 threshold suggests positions, then run cpsam on DIC crop)

For each frame, output:
  - count per strategy
  - cells unique to fusion (i.e. that pure DIC would miss)
  - figure with overlays

Run from cellpose4 env.
"""
import os
import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import label as cc_label, binary_fill_holes, \
    binary_dilation

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

OUT_DIR = "/tmp/fluo_investigation"
os.makedirs(OUT_DIR, exist_ok=True)

TIFF = ("data/examples/multichannel_DIC_Cy5_DMSO_busy/"
        "multichannel_DIC_Cy5_DMSO_busy.ome.tif")
FRAMES = [0, 15, 25, 39]
MIN_AREA = 250


def load_frame(idx):
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


def jaccard(a_mask, b_mask):
    inter = (a_mask & b_mask).sum()
    if inter == 0:
        return 0.0
    return inter / (a_mask | b_mask).sum()


def merge_into_union(union_labels, new_labels, jaccard_thresh=0.3,
                     min_area=MIN_AREA, max_overlap_frac=0.5):
    """Add new_labels into union_labels, skipping ones that overlap
    heavily with existing labels."""
    out = union_labels.copy()
    nxt = int(out.max()) + 1
    added = []
    for lab in range(1, int(new_labels.max()) + 1):
        nm = new_labels == lab
        if nm.sum() < min_area:
            continue
        # Best Jaccard with any existing label
        best_j = 0.0
        for ex in range(1, int(out.max()) + 1):
            em = out == ex
            j = jaccard(nm, em)
            best_j = max(best_j, j)
        if best_j >= jaccard_thresh:
            continue
        # Also reject if the new mask is mostly covered by union
        # (avoids stacking labels)
        cov = (nm & (out > 0)).sum() / max(nm.sum(), 1)
        if cov > max_overlap_frac:
            continue
        out[nm & (out == 0)] = nxt
        added.append(nxt)
        nxt += 1
    return out, added


def cy5_threshold_seeds(cy5, k_mad=3.0, min_area=80,
                         dilate_px=8):
    """Threshold Cy5 into seed regions for ROI cropping."""
    med = float(np.median(cy5))
    mad = max(1.0, float(np.median(
        np.abs(cy5.astype(np.float32) - med))))
    bright = cy5 > med + k_mad * mad
    bright = binary_fill_holes(bright)
    cc, _ = cc_label(bright)
    seeds = []
    for cid in range(1, int(cc.max()) + 1):
        m = cc == cid
        if m.sum() < min_area:
            continue
        ys, xs = np.where(m)
        seeds.append({
            "centroid": (float(ys.mean()), float(xs.mean())),
            "mask": m,
            "bbox": (int(ys.min()), int(ys.max()) + 1,
                     int(xs.min()), int(xs.max()) + 1),
            "cy5_max": int(cy5[m].max()),
        })
    return seeds


def run_cpsam_on_dic_crop_around_seed(dic, seed, cpsam_model,
                                       pad_px=64, use_tta=True,
                                       existing_labels=None):
    """Run cpsam on a DIC crop centred on seed['centroid']."""
    H, W = dic.shape
    r0, r1, c0, c1 = seed["bbox"]
    r0p = max(0, r0 - pad_px); r1p = min(H, r1 + pad_px)
    c0p = max(0, c0 - pad_px); c1p = min(W, c1 + pad_px)
    if (r1p - r0p) < 64 or (c1p - c0p) < 64:
        return None
    crop = dic[r0p:r1p, c0p:c1p]
    try:
        out = cpsam_model.eval(crop, augment=use_tta)
        crop_labels = out[0].astype(np.int32)
    except Exception:
        return None
    if crop_labels.max() == 0:
        return None
    seed_crop = seed["mask"][r0p:r1p, c0p:c1p]
    best_lab, best_ovl = 0, 0
    for lab in range(1, int(crop_labels.max()) + 1):
        ovl = int(np.logical_and(crop_labels == lab, seed_crop).sum())
        if ovl > best_ovl:
            best_ovl, best_lab = ovl, lab
    if best_lab == 0:
        return None
    recovered = np.zeros_like(dic, dtype=bool)
    recovered[r0p:r1p, c0p:c1p] = (crop_labels == best_lab)
    if existing_labels is not None:
        # Don't overwrite existing cells
        recovered &= (existing_labels == 0)
    if recovered.sum() < MIN_AREA:
        return None
    return recovered


def run_cellpose_dic_on_full_frame(frames, model_path):
    """Run cellpose_dic on a stack via cellpose subprocess (CP3 env)."""
    # Easier: run in-process from the cellpose env? We're in cellpose4.
    # Need subprocess to cellpose env.
    import subprocess, tempfile
    n = len(frames)
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.npz")
        outp = os.path.join(tmp, "out.npz")
        np.savez_compressed(inp, frames=frames)
        script = f"""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "{CELLSCOPE_ROOT}")
from core.detection import detect_cellpose_labels
data = np.load("{inp}", allow_pickle=True)
out = detect_cellpose_labels(
    data["frames"], gpu=True,
    model_path="{model_path}",
    flow_threshold=0.0, cellprob_threshold=0.0,
    min_area_px={MIN_AREA})
np.savez_compressed("{outp}", labels=out)
print("CP3_OK")
"""
        res = subprocess.run(
            ["conda", "run", "-n", "cellpose", "python", "-c", script],
            capture_output=True, text=True, timeout=600,
            cwd=CELLSCOPE_ROOT)
        if "CP3_OK" not in res.stdout:
            print("CP3 subprocess stderr:\n", res.stderr[-800:])
            raise RuntimeError("cellpose_dic subprocess failed")
        return np.load(outp)["labels"]


def draw(ax, labels, color, lw=1.0, label_size=8):
    n = 0
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if not m.any():
            continue
        for c in measure.find_contours(m.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=color, lw=lw)
        n += 1


def main():
    from cellpose import models
    print("Loading cpsam (vit_h) ...")
    cpsam_model = models.CellposeModel(gpu=True)

    # Run cellpose_dic on all needed frames in one subprocess call
    print(f"Loading {len(FRAMES)} frames + Cy5 channel ...")
    dics, cy5s = [], []
    for fi in FRAMES:
        d, c = load_frame(fi)
        dics.append(d); cy5s.append(c)
    dic_stack = np.array(dics)
    cy5_stack = np.array(cy5s)

    print("Running cellpose_dic on DIC stack via cellpose subprocess ...")
    dic_labels_all = run_cellpose_dic_on_full_frame(
        dic_stack, model_path="data/models/cellpose_dic")
    print(f"  cellpose_dic labels shape: {dic_labels_all.shape}")

    print("\nRunning cpsam on Cy5 stack ...")
    cy5_labels_all = []
    for ci, c in enumerate(cy5s):
        out = cpsam_model.eval(c, augment=False)
        labels = out[0].astype(np.int32)
        cy5_labels_all.append(labels)

    rows = []
    fig, axes = plt.subplots(len(FRAMES), 5, figsize=(26, 5*len(FRAMES)))

    for ri, fi in enumerate(FRAMES):
        dic, cy5 = dics[ri], cy5s[ri]
        print(f"\n=== Frame {fi} ===")

        dic_labels = dic_labels_all[ri]
        dic_labels, n_dic = filter_small(dic_labels, MIN_AREA)
        print(f"  A. cellpose_dic(DIC):       {n_dic} cells")

        cy5_labels = cy5_labels_all[ri]
        cy5_labels, n_cy5 = filter_small(cy5_labels, MIN_AREA)
        print(f"     cpsam(Cy5):              {n_cy5} cells")

        # Strategy B: union of cellpose_dic + cpsam(Cy5)
        union_B, added_B = merge_into_union(dic_labels, cy5_labels)
        n_B = int(union_B.max())
        print(f"  B. cellpose_dic ∪ cpsam(Cy5):"
              f" {n_B} cells (+{len(added_B)} new from Cy5)")

        # Strategy C: Cy5 threshold seeds → cpsam on DIC crop around seed
        seeds = cy5_threshold_seeds(cy5)
        n_seeds = len(seeds)
        print(f"     Cy5 threshold seeds:     {n_seeds}")
        union_C = dic_labels.copy()
        nxt = int(union_C.max()) + 1
        added_C = []
        for s in seeds:
            # Skip seeds already covered by DIC
            cov = (s["mask"] & (union_C > 0)).sum() / max(s["mask"].sum(), 1)
            if cov > 0.5:
                continue
            recov = run_cpsam_on_dic_crop_around_seed(
                dic, s, cpsam_model, pad_px=64, use_tta=True,
                existing_labels=union_C)
            if recov is None:
                continue
            union_C[recov] = nxt
            added_C.append(nxt)
            nxt += 1
        n_C = int(union_C.max())
        print(f"  C. cellpose_dic + Cy5-seeded cpsam(DIC):"
              f" {n_C} cells (+{len(added_C)} new)")

        # Visualisations
        axes[ri, 0].imshow(dic, cmap="gray")
        axes[ri, 0].set_title(f"F{fi}  A. cellpose_dic(DIC): {n_dic}")
        draw(axes[ri, 0], dic_labels, "red", 1.1)

        axes[ri, 1].imshow(cy5, cmap="inferno",
                            vmin=0, vmax=np.percentile(cy5, 99))
        axes[ri, 1].set_title(f"F{fi}  Cy5 + cpsam(Cy5): {n_cy5}")
        draw(axes[ri, 1], cy5_labels, "cyan", 1.0)

        axes[ri, 2].imshow(dic, cmap="gray")
        axes[ri, 2].set_title(
            f"F{fi}  B. fused (+{len(added_B)}): {n_B}")
        draw(axes[ri, 2], dic_labels, "red", 1.0)
        # Highlight cells added from Cy5
        added_mask = np.isin(union_B, added_B)
        for lab in added_B:
            m = union_B == lab
            for c in measure.find_contours(m.astype(float), 0.5):
                axes[ri, 2].plot(c[:, 1], c[:, 0], color="lime", lw=2.0)

        axes[ri, 3].imshow(dic, cmap="gray")
        axes[ri, 3].set_title(
            f"F{fi}  C. Cy5-seeded crops (+{len(added_C)}): {n_C}")
        draw(axes[ri, 3], dic_labels, "red", 1.0)
        for lab in added_C:
            m = union_C == lab
            for c in measure.find_contours(m.astype(float), 0.5):
                axes[ri, 3].plot(c[:, 1], c[:, 0], color="yellow", lw=2.0)

        axes[ri, 4].imshow(cy5, cmap="inferno",
                            vmin=0, vmax=np.percentile(cy5, 99))
        axes[ri, 4].set_title(
            f"F{fi}  Cy5 + threshold seeds: {n_seeds}")
        for s in seeds:
            for c in measure.find_contours(s["mask"].astype(float), 0.5):
                axes[ri, 4].plot(c[:, 1], c[:, 0],
                                  color="yellow", lw=1.0)

        for ax in axes[ri]:
            ax.axis("off")

        rows.append({
            "frame": fi, "A_dic": n_dic, "cy5_cpsam": n_cy5,
            "B_fused": n_B, "B_new": len(added_B),
            "C_seeded": n_C, "C_new": len(added_C),
            "n_seeds": n_seeds
        })

    plt.tight_layout()
    out = f"{OUT_DIR}/03_fusion_prototypes.png"
    plt.savefig(out, dpi=85, bbox_inches="tight")
    print(f"\nSaved {out}")

    print("\n=== Summary table ===")
    print(f"{'frame':>5} {'A_dic':>6} {'cy5_cp':>6} "
          f"{'B':>4} {'B+':>4} {'C':>4} {'C+':>4} {'seeds':>6}")
    for r in rows:
        print(f"{r['frame']:>5} {r['A_dic']:>6} {r['cy5_cpsam']:>6} "
              f"{r['B_fused']:>4} +{r['B_new']:>2} "
              f"{r['C_seeded']:>4} +{r['C_new']:>2} {r['n_seeds']:>6}")

    with open(f"{OUT_DIR}/03_summary.txt", "w") as f:
        f.write(f"{'frame':>5} {'A_dic':>6} {'cy5_cp':>6} "
                f"{'B':>4} {'B+':>4} {'C':>4} {'C+':>4} {'seeds':>6}\n")
        for r in rows:
            f.write(f"{r['frame']:>5} {r['A_dic']:>6} {r['cy5_cpsam']:>6} "
                    f"{r['B_fused']:>4} +{r['B_new']:>2} "
                    f"{r['C_seeded']:>4} +{r['C_new']:>2} {r['n_seeds']:>6}\n")


if __name__ == "__main__":
    main()
