"""Diagnose IC293 Pos3 cell merging.

GT shows 3 cells per frame; pipeline outputs 2-3 cells but their
areas suggest cell merging (pipeline cells are 2-3× larger than
individual GT cells). Render side-by-side overlays + raw cpsam_dic
output (no postprocess) to see if the merging happens at detection
or in tracking/post-processing.
"""
import os
import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import io as skio, measure

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

OUT_DIR = "/tmp/fluo_investigation"
os.makedirs(OUT_DIR, exist_ok=True)

REC = "data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/"\
      "IC293__1_MMStack_Pos3-WT.ome-cropped.tif"
GT_DIR = "data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/gt_masks"
PIPE = ("data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/"
        "pipeline_results/masks.npz")
FRAMES_TO_TEST = [0, 12, 24, 48, 60, 84]


def main():
    pipeline = np.load(PIPE)["labels"]
    print(f"Pipeline labels shape: {pipeline.shape}, "
          f"max IDs: {[int(pipeline[fi].max()) for fi in FRAMES_TO_TEST]}")

    # Load DIC frames (single-channel for legacy IC293)
    from core.io import load_video
    frames = load_video(REC)
    print(f"DIC frames: {frames.shape}")

    fig, axes = plt.subplots(len(FRAMES_TO_TEST), 3,
                              figsize=(18, 5 * len(FRAMES_TO_TEST)))

    for ri, fi in enumerate(FRAMES_TO_TEST):
        gt = skio.imread(os.path.join(GT_DIR, f"mask_F{fi}.png"))
        pipe = pipeline[fi]
        dic = frames[fi]

        # Col 0: GT
        ax = axes[ri, 0]
        ax.imshow(dic, cmap="gray")
        for cid in range(1, int(gt.max()) + 1):
            m = gt == cid
            if not m.any(): continue
            for c in measure.find_contours(m.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color="lime", lw=1.5)
            ys, xs = np.where(m)
            ax.text(xs.mean(), ys.mean(), str(cid),
                    color="white", fontsize=11, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2))
        ax.set_title(f"F{fi}  GT ({int(gt.max())} cells, "
                     f"areas {[int((gt==i).sum()) for i in range(1, gt.max()+1)]})")

        # Col 1: pipeline
        ax = axes[ri, 1]
        ax.imshow(dic, cmap="gray")
        for cid in range(1, int(pipe.max()) + 1):
            m = pipe == cid
            if not m.any(): continue
            for c in measure.find_contours(m.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color="magenta", lw=1.5)
            ys, xs = np.where(m)
            ax.text(xs.mean(), ys.mean(), str(cid),
                    color="white", fontsize=11, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2))
        n_pipe = sum(1 for i in range(1, pipe.max()+1) if (pipe==i).any())
        ax.set_title(f"F{fi}  Pipeline ({n_pipe} cells, "
                     f"areas {[int((pipe==i).sum()) for i in range(1,pipe.max()+1) if (pipe==i).sum()>0]})")

        # Col 2: overlay both
        ax = axes[ri, 2]
        ax.imshow(dic, cmap="gray")
        for cid in range(1, int(gt.max()) + 1):
            m = gt == cid
            for c in measure.find_contours(m.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color="lime", lw=1.0,
                         label="GT" if cid == 1 else None)
        for cid in range(1, int(pipe.max()) + 1):
            m = pipe == cid
            if not m.any(): continue
            for c in measure.find_contours(m.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color="magenta", lw=1.0,
                         label="Pipeline" if cid == 1 else None)
        ax.set_title(f"F{fi}  Overlay (lime=GT, magenta=pipeline)")
        if ri == 0:
            ax.legend(fontsize=8, loc="upper right")

        for a in axes[ri]:
            a.axis("off")

    plt.tight_layout()
    out = f"{OUT_DIR}/11_pos3_merging.png"
    plt.savefig(out, dpi=85, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
