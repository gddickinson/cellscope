"""Check 6 different frames — including ones where saved pipeline
found few cells — to identify worst-case mismatches between DIC and
Cy5 detection counts."""
import os
import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import label as cc_label, binary_fill_holes

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

OUT_DIR = "/tmp/fluo_investigation"
os.makedirs(OUT_DIR, exist_ok=True)

TIFF = ("data/examples/multichannel_DIC_Cy5_DMSO_busy/"
        "multichannel_DIC_Cy5_DMSO_busy.ome.tif")
FRAMES_TO_TEST = [0, 5, 15, 25, 35, 39]
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


def jaccard_per_label(target, reference):
    out = {}
    for t in range(1, int(target.max()) + 1):
        tm = target == t
        if not tm.any():
            continue
        best = 0.0
        for r in range(1, int(reference.max()) + 1):
            rm = reference == r
            i = (tm & rm).sum()
            if i == 0:
                continue
            j = i / (tm | rm).sum()
            best = max(best, j)
        out[t] = best
    return out


def draw_contours(ax, labels, color, lw=1.0):
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if not m.any():
            continue
        for c in measure.find_contours(m.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=color, lw=lw)


def main():
    from cellpose import models
    m = models.CellposeModel(gpu=True)

    # Also load saved labels for comparison
    saved = np.load(
        "data/examples/multichannel_DIC_Cy5_DMSO_busy/"
        "results/masks.npz", allow_pickle=True)
    saved_labels = saved["labels"]

    rows = []
    fig, axes = plt.subplots(len(FRAMES_TO_TEST), 4,
                              figsize=(22, 5 * len(FRAMES_TO_TEST)))

    for ri, fi in enumerate(FRAMES_TO_TEST):
        dic, cy5 = load_frame(fi)
        print(f"\n--- Frame {fi} ---")

        out = m.eval(dic, augment=False)
        dic_labels = out[0]
        dic_labels, n_dic = filter_small(dic_labels.astype(np.int32),
                                          MIN_AREA)

        out = m.eval(cy5, augment=False)
        cy5_labels = out[0]
        cy5_labels, n_cy5 = filter_small(cy5_labels.astype(np.int32),
                                          MIN_AREA)

        # Saved-pipeline count
        saved_frame = saved_labels[fi]
        n_saved = len(set(int(c) for c in np.unique(saved_frame) if c))

        # Cells caught by Cy5 NOT in DIC
        js_cy5 = jaccard_per_label(cy5_labels, dic_labels)
        cy5_unique = sum(1 for v in js_cy5.values() if v < 0.3)

        # DIC cells not in Cy5
        js_dic = jaccard_per_label(dic_labels, cy5_labels)
        dic_unique = sum(1 for v in js_dic.values() if v < 0.3)

        # Saved cells not in DIC (anomaly check)
        js_saved = jaccard_per_label(saved_frame.astype(np.int32),
                                     dic_labels)
        saved_unique = sum(1 for v in js_saved.values() if v < 0.3)

        # Build union
        union = dic_labels.copy()
        nxt = int(union.max()) + 1
        for lab in range(1, int(cy5_labels.max()) + 1):
            if js_cy5.get(lab, 0) >= 0.3:
                continue
            cm = cy5_labels == lab
            if cm.sum() < MIN_AREA:
                continue
            union[cm & (union == 0)] = nxt
            nxt += 1
        n_union = int(union.max())

        print(f"  cpsam(DIC): {n_dic}  cpsam(Cy5): {n_cy5}  "
              f"saved-pipeline: {n_saved}  union: {n_union}")
        print(f"  Cy5 not in DIC: {cy5_unique}  "
              f"DIC not in Cy5: {dic_unique}")

        rows.append({
            "frame": fi, "n_dic": n_dic, "n_cy5": n_cy5,
            "n_saved": n_saved, "n_union": n_union,
            "cy5_unique": cy5_unique, "dic_unique": dic_unique
        })

        # Visualisations
        axes[ri, 0].imshow(dic, cmap="gray")
        axes[ri, 0].set_title(
            f"F{fi}  DIC + cpsam(DIC) red: {n_dic} cells  "
            f"saved: {n_saved}")
        draw_contours(axes[ri, 0], dic_labels, "red", 1.0)

        axes[ri, 1].imshow(cy5, cmap="inferno",
                            vmin=0, vmax=np.percentile(cy5, 99))
        axes[ri, 1].set_title(f"F{fi}  Cy5 + cpsam(Cy5) cyan: "
                               f"{n_cy5} cells")
        draw_contours(axes[ri, 1], cy5_labels, "cyan", 1.0)

        axes[ri, 2].imshow(dic, cmap="gray")
        axes[ri, 2].set_title(
            f"F{fi}  Cy5-unique (DIC missed): {cy5_unique}")
        for lab in range(1, int(cy5_labels.max()) + 1):
            if js_cy5.get(lab, 0) < 0.3:
                cm = cy5_labels == lab
                for c in measure.find_contours(cm.astype(float), 0.5):
                    axes[ri, 2].plot(c[:, 1], c[:, 0],
                                      color="lime", lw=2.0)

        axes[ri, 3].imshow(dic, cmap="gray")
        axes[ri, 3].set_title(
            f"F{fi}  Union DIC ∪ cpsam(Cy5): {n_union} cells")
        draw_contours(axes[ri, 3], union, "magenta", 1.0)

        for ax in axes[ri]:
            ax.axis("off")

    plt.tight_layout()
    out = f"{OUT_DIR}/02_multi_frame_comparison.png"
    plt.savefig(out, dpi=85, bbox_inches="tight")
    print(f"\nSaved {out}")

    # Summary table
    print("\n=== Summary table ===")
    print(f"{'frame':>5} {'DIC':>5} {'Cy5':>5} {'saved':>5} "
          f"{'union':>5} {'Cy5_only':>9} {'DIC_only':>9}")
    for r in rows:
        print(f"{r['frame']:>5} {r['n_dic']:>5} {r['n_cy5']:>5} "
              f"{r['n_saved']:>5} {r['n_union']:>5} "
              f"{r['cy5_unique']:>9} {r['dic_unique']:>9}")

    with open(f"{OUT_DIR}/02_summary.txt", "w") as f:
        f.write(f"{'frame':>5} {'DIC':>5} {'Cy5':>5} {'saved':>5} "
                f"{'union':>5} {'Cy5_only':>9} {'DIC_only':>9}\n")
        for r in rows:
            f.write(f"{r['frame']:>5} {r['n_dic']:>5} {r['n_cy5']:>5} "
                    f"{r['n_saved']:>5} {r['n_union']:>5} "
                    f"{r['cy5_unique']:>9} {r['dic_unique']:>9}\n")


if __name__ == "__main__":
    main()
