"""Side-by-side: cpsam_dic (current default) vs raw cpsam on the
Pos3 problem frames. Shows that raw cpsam separates the touching
cells cpsam_dic merges."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio, measure

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

REC = "data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/"\
      "IC293__1_MMStack_Pos3-WT.ome-cropped.tif"
GT_DIR = "data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/gt_masks"
FRAMES = [0, 24, 48, 72]


def filter_small(labels, min_area=200):
    out = np.zeros_like(labels)
    nid = 0
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if m.sum() >= min_area:
            nid += 1
            out[m] = nid
    return out


def draw(ax, labels, color):
    for cid in range(1, int(labels.max()) + 1):
        m = labels == cid
        if not m.any(): continue
        for c in measure.find_contours(m.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=color, lw=1.5)


def main():
    from cellpose import models
    from core.io import load_video
    cpsam_dic = models.CellposeModel(
        gpu=True, pretrained_model="data/models/cpsam_dic")
    cpsam = models.CellposeModel(gpu=True)
    frames = load_video(REC)

    fig, axes = plt.subplots(len(FRAMES), 3, figsize=(15, 5 * len(FRAMES)))

    for ri, fi in enumerate(FRAMES):
        gt = skio.imread(os.path.join(GT_DIR, f"mask_F{fi}.png"))
        dic = frames[fi]
        labA = filter_small(cpsam_dic.eval(
            dic, augment=False)[0].astype(np.int32))
        labC = filter_small(cpsam.eval(
            dic, augment=False)[0].astype(np.int32))

        for ax in axes[ri]:
            ax.imshow(dic, cmap="gray")
            ax.axis("off")
        draw(axes[ri, 0], gt, "lime")
        axes[ri, 0].set_title(
            f"F{fi} GT ({int(gt.max())} cells)", color="lime")
        draw(axes[ri, 1], labA, "magenta")
        axes[ri, 1].set_title(
            f"F{fi} cpsam_dic (current default): "
            f"{int(labA.max())} cells", color="magenta")
        draw(axes[ri, 2], labC, "cyan")
        axes[ri, 2].set_title(
            f"F{fi} raw cpsam: {int(labC.max())} cells",
            color="cyan")

    plt.tight_layout()
    out = "/tmp/fluo_investigation/12_pos3_cpsam_vs_dic.png"
    plt.savefig(out, dpi=85, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
