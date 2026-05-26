"""End-to-end validation of Cy5 fusion on the DMSO_busy demo.

Runs detect_hybrid_dic_multi with use_cy5_fusion=False then =True on
all 40 frames. Compares:
  - max cells/frame
  - total tracks
  - cells unique to fusion run

Saves a 6-frame overlay grid showing both runs side-by-side.

Must be run from cellpose env (worker default). cpsam subprocess
is spawned automatically.
"""
import os
import sys
import time
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

TIFF = ("data/examples/multichannel_DIC_Cy5_DMSO_busy/"
        "multichannel_DIC_Cy5_DMSO_busy.ome.tif")


def load_channels(tif_path):
    """Return (dic, cy5) stacks, both (N, H, W) uint8."""
    with tifffile.TiffFile(tif_path) as tf:
        n = len(tf.pages) // 2
        h, w = tf.pages[0].shape
        cy5 = np.empty((n, h, w), dtype=np.uint8)
        dic = np.empty((n, h, w), dtype=np.uint8)
        for i in range(n):
            cy5[i] = tf.pages[2 * i].asarray()
            dic[i] = tf.pages[2 * i + 1].asarray()
    return dic, cy5


def draw(ax, labels, colormap_lut=None, lw=1.0, default_color="red"):
    """Draw contours, one per unique label > 0."""
    for lab in range(1, int(labels.max()) + 1):
        m = labels == lab
        if not m.any():
            continue
        color = (colormap_lut[lab] if colormap_lut is not None
                 else default_color)
        for c in measure.find_contours(m.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=color, lw=lw)


def main():
    from core.hybrid_dic import detect_hybrid_dic_multi

    print(f"Loading {TIFF} ...")
    dic, cy5 = load_channels(TIFF)
    n = len(dic)
    print(f"  loaded {n} frames; DIC {dic.shape}, Cy5 {cy5.shape}")

    # ---- Run A: no fusion (baseline)
    print("\n[A] Running hybrid_dic_multi (no fusion) ...")
    t0 = time.time()
    res_A = detect_hybrid_dic_multi(
        dic, progress_fn=None,
        min_area_px=200,
        use_preprocess=True,
        use_deepsea=True,
        use_retry=True,
        use_gap_fill=True,
        model_path="data/models/cellpose_dic",
        cy5_frames=None,
        use_cy5_fusion=False,
    )
    t_A = time.time() - t0
    cells_A = [int(res_A["labels"][i].max()) for i in range(n)]
    print(f"  done in {t_A:.0f}s; tracks={len(res_A['tracks'])}, "
          f"max cells/frame={max(cells_A)}, mean={np.mean(cells_A):.1f}")

    # ---- Run B: with fusion
    print("\n[B] Running hybrid_dic_multi (Cy5 fusion ON) ...")
    t0 = time.time()
    res_B = detect_hybrid_dic_multi(
        dic, progress_fn=None,
        min_area_px=200,
        use_preprocess=True,
        use_deepsea=True,
        use_retry=True,
        use_gap_fill=True,
        model_path="data/models/cellpose_dic",
        cy5_frames=cy5,
        use_cy5_fusion=True,
    )
    t_B = time.time() - t0
    cells_B = [int(res_B["labels"][i].max()) for i in range(n)]
    print(f"  done in {t_B:.0f}s; tracks={len(res_B['tracks'])}, "
          f"max cells/frame={max(cells_B)}, mean={np.mean(cells_B):.1f}")
    print(f"  Cy5 fusion added {res_B.get('n_cy5_fusion_added', 0)} "
          f"cells across all frames (pre-tracking)")

    # ---- Per-frame compare
    print("\nPer-frame cells (label-count) — A: no fusion, "
          "B: fusion ON")
    print(f"{'frame':>5} {'A':>4} {'B':>4} {'delta':>6}")
    for i in range(n):
        print(f"{i:>5} {cells_A[i]:>4} {cells_B[i]:>4} "
              f"{cells_B[i] - cells_A[i]:>+6}")

    # ---- 6-frame visual grid
    sample_frames = [0, 7, 15, 22, 31, 39]
    fig, axes = plt.subplots(len(sample_frames), 3,
                              figsize=(18, 5 * len(sample_frames)))

    rng = np.random.default_rng(7)
    palette = [tuple(rng.uniform(0.2, 1, size=3)) for _ in range(200)]

    for ri, fi in enumerate(sample_frames):
        axes[ri, 0].imshow(dic[fi], cmap="gray")
        lut_A = {lab: palette[lab % len(palette)]
                 for lab in range(1, int(res_A['labels'][fi].max()) + 1)}
        axes[ri, 0].set_title(
            f"F{fi}  no fusion: {cells_A[fi]} cells")
        draw(axes[ri, 0], res_A["labels"][fi], colormap_lut=lut_A, lw=1.3)

        axes[ri, 1].imshow(dic[fi], cmap="gray")
        lut_B = {lab: palette[lab % len(palette)]
                 for lab in range(1, int(res_B['labels'][fi].max()) + 1)}
        axes[ri, 1].set_title(
            f"F{fi}  Cy5 fusion ON: {cells_B[fi]} cells "
            f"(+{cells_B[fi] - cells_A[fi]})")
        draw(axes[ri, 1], res_B["labels"][fi], colormap_lut=lut_B, lw=1.3)

        axes[ri, 2].imshow(cy5[fi], cmap="inferno",
                            vmin=0, vmax=np.percentile(cy5[fi], 99))
        axes[ri, 2].set_title(f"F{fi}  Cy5 + fusion contours")
        draw(axes[ri, 2], res_B["labels"][fi], colormap_lut=lut_B, lw=1.0)

        for ax in axes[ri]:
            ax.axis("off")

    plt.tight_layout()
    out = f"{OUT_DIR}/04_fusion_e2e_grid.png"
    plt.savefig(out, dpi=85, bbox_inches="tight")
    print(f"\nSaved {out}")

    # Summary
    summary = f"""End-to-end Cy5 fusion validation on DMSO_busy ({n} frames)

  Baseline (no fusion):
    max cells/frame: {max(cells_A)}
    mean cells/frame: {np.mean(cells_A):.1f}
    total tracks: {len(res_A['tracks'])}
    runtime: {t_A:.0f}s

  With Cy5 fusion:
    max cells/frame: {max(cells_B)}
    mean cells/frame: {np.mean(cells_B):.1f}
    total tracks: {len(res_B['tracks'])}
    cells added (pre-tracking): {res_B.get('n_cy5_fusion_added', 0)}
    runtime: {t_B:.0f}s  (+{t_B - t_A:.0f}s vs baseline)

  Net gain:
    Δ max cells/frame: {max(cells_B) - max(cells_A):+d}
    Δ mean cells/frame: {np.mean(cells_B) - np.mean(cells_A):+.1f}
    Δ tracks: {len(res_B['tracks']) - len(res_A['tracks']):+d}
"""
    print(summary)
    with open(f"{OUT_DIR}/04_summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
