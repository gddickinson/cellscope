"""Test SAM2 video propagation on the central cell gap (F19→F30)
that all single-frame detectors missed.

Anchor: cell 9's mask at F19. Gap to fill: F20–F29.

Expected: SAM2 follows the cell through the wisp/retraction frames
where cpsam_dic and CP3 fallback both failed.

Output: /tmp/fluo_investigation/10_sam2_central_cell.png with
contours overlaid on the DIC frames.
"""
import os
import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import measure

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

ANCHOR_FRAME = 19      # last frame cell 9 was correctly tracked
GAP_FRAMES = list(range(20, 30))   # F20-F29


def main():
    from core.multichannel import to_uint8_dic
    from core.sam2_video import propagate_through_gap, _build_predictor

    # Load DIC stack
    print("Loading DIC stack ...")
    with tifffile.TiffFile(REC) as tf:
        n = len(tf.pages) // 2
        dic_raw = np.array([tf.pages[2*i + 1].asarray()
                            for i in range(n)])
    dic = np.array([to_uint8_dic(f) for f in dic_raw])
    print(f"  {dic.shape}")

    # Use the SAVED cell-9 mask at F19 as anchor (so we replicate
    # what the pipeline would do)
    saved = np.load(SAVED_LABELS)
    labels = saved["labels"]
    anchor_mask = labels[ANCHOR_FRAME] == 9
    if not anchor_mask.any():
        print(f"!! Saved labels have no cell 9 at F{ANCHOR_FRAME}, "
              f"trying cell 8 …")
        anchor_mask = labels[ANCHOR_FRAME] == 8
    print(f"Anchor mask area at F{ANCHOR_FRAME}: {anchor_mask.sum()} px")
    ys, xs = np.where(anchor_mask)
    print(f"  centroid: ({ys.mean():.0f}, {xs.mean():.0f})")

    # Build SAM2 once
    print("\nLoading SAM2 video predictor (~5s) ...")
    predictor = _build_predictor()

    # Propagate through gap
    print(f"\nPropagating through F{GAP_FRAMES[0]}..F{GAP_FRAMES[-1]} "
          f"(10 frames) ...")
    import time; t0 = time.time()
    propagated = propagate_through_gap(
        dic, ANCHOR_FRAME, anchor_mask, GAP_FRAMES,
        predictor=predictor, min_area=20)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s ({elapsed/len(GAP_FRAMES)*1000:.0f} "
          f"ms/frame)")
    print(f"  filled {len(propagated)}/{len(GAP_FRAMES)} gap frames")
    for fi in GAP_FRAMES:
        m = propagated.get(fi)
        msg = (f"  F{fi}: AREA={m.sum():>5} px"
               if m is not None else f"  F{fi}: NOT FOUND")
        if m is not None:
            ys, xs = np.where(m)
            msg += f", centroid=({ys.mean():.0f},{xs.mean():.0f})"
        print(msg)

    # Visualize: anchor + each gap frame
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    show_frames = [ANCHOR_FRAME] + GAP_FRAMES + [30]
    for ax, fi in zip(axes.flat, show_frames):
        ax.imshow(dic[fi], cmap="gray")
        ax.set_title(f"F{fi}", fontsize=11)
        ax.axis("off")
        # Crop view to the cell area
        ax.set_xlim(350, 750)
        ax.set_ylim(820, 480)   # inverted y for image
        if fi == ANCHOR_FRAME:
            for c in measure.find_contours(anchor_mask.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color="lime", lw=2.0,
                         label="anchor")
            ax.set_title(f"F{fi} ANCHOR (saved cell 9)", fontsize=11,
                          color="lime")
        elif fi in propagated:
            for c in measure.find_contours(
                    propagated[fi].astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color="cyan", lw=1.8)
            ax.set_title(f"F{fi}  SAM2 propagated "
                          f"({propagated[fi].sum()} px)", fontsize=11)
        elif fi in GAP_FRAMES:
            ax.set_title(f"F{fi}  no fill", fontsize=11, color="orange")
        elif fi == 30:
            saved_30 = labels[30]
            for lab in range(1, int(saved_30.max()) + 1):
                m = saved_30 == lab
                if not m.any():
                    continue
                for c in measure.find_contours(m.astype(float), 0.5):
                    ax.plot(c[:, 1], c[:, 0], color="magenta", lw=1.2)
            ax.set_title(f"F{fi} (re-detection, saved labels)",
                          fontsize=11, color="magenta")

    plt.tight_layout()
    out = f"{OUT_DIR}/10_sam2_central_cell.png"
    plt.savefig(out, dpi=85, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
