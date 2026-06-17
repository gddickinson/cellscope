"""Guarded outline refinement for flagged single-cell recordings.

For each recording in review_flags.csv, prompt SAM2 with the existing mask
(per frame) and ACCEPT the refined mask only when it genuinely improves
the outline without degrading it:

  * keeps ≥92% of the original cell (doesn't lose the cell),
  * grows 1.03–1.6× (recovers missed parts, no explosion),
  * the ADDED region is cell-like textured (≥0.5× the cell's DIC texture,
    i.e. real cell, not flat background).

Otherwise the original mask is kept — so the pass can only improve, never
degrade. NON-DESTRUCTIVE: writes pipeline_results/masks_refined.npz +
a before/after montage; masks.npz is untouched until you promote. Only the
flagged recordings are touched (the rest are already good).

Usage: conda run -n cellpose4 python scripts/ic293_refine_flagged.py
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

import numpy as np  # noqa: E402
from scripts.ic293_common import (  # noqa: E402
    inventory_cache, recording_dir, ANALYSIS_ROOT)

KEEP_MIN = 0.92          # fraction of original cell retained
GROW_LO, GROW_HI = 1.03, 1.6
ADDED_TEX_FRAC = 0.5     # added region texture ≥ this × cell texture


def _texture(frame, mask):
    px = frame[mask]
    return float(px.std()) if px.size else 0.0


def _guarded_refine(frame, seed, predictor):
    """Return (mask, accepted). Accept SAM2's mask only if it's a clean,
    textured growth of the seed."""
    from core.sam_refine import refine_with_sam2
    if not seed.any():
        return seed, False
    refined = refine_with_sam2(frame, seed, predictor=predictor).astype(bool)
    a0, a1 = int(seed.sum()), int(refined.sum())
    if a0 == 0 or a1 == 0:
        return seed, False
    keep = (seed & refined).sum() / a0
    ratio = a1 / a0
    added = refined & ~seed
    cell_tex = _texture(frame, seed)
    add_tex = _texture(frame, added) if added.any() else 0.0
    ok = (keep >= KEEP_MIN and GROW_LO <= ratio <= GROW_HI
          and add_tex >= ADDED_TEX_FRAC * max(cell_tex, 1e-6))
    return (refined if ok else seed), ok


def _montage(items, out_path, ncols=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure
    n = len(items)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.8 * nrows),
                            squeeze=False)
    for i, (label, f, img, seed, refined, changed) in enumerate(items):
        ax = axs[i // ncols][i % ncols]
        ax.imshow(img, cmap="gray")
        for ct in measure.find_contours(seed.astype(float), 0.5):
            ax.plot(ct[:, 1], ct[:, 0], "-", color="red", lw=1.0)
        if changed:
            for ct in measure.find_contours(refined.astype(float), 0.5):
                ax.plot(ct[:, 1], ct[:, 0], "-", color="lime", lw=1.2)
        ys, xs = np.where(seed | refined)
        if len(ys):
            ax.set_xlim(xs.min() - 20, xs.max() + 20)
            ax.set_ylim(ys.max() + 20, ys.min() - 20)
        ax.set_title(f"{label} F{f}" + ("  ✎" if changed else "  (kept)"),
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    fig.suptitle("red = original   |   green = refined (only where accepted)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", default=os.path.join(
        ANALYSIS_ROOT, "review", "review_flags.csv"))
    args = ap.parse_args()
    if not os.path.exists(args.flags):
        print(f"no flags file at {args.flags}"); return 1
    with open(args.flags) as f:
        flagged = [r for r in csv.DictReader(f)
                   if str(r.get("flagged", "")).lower() in ("1", "true", "yes")]
    inv = inventory_cache()
    from core.io import load_recording
    from core.sam_refine import load_sam2
    predictor = load_sam2("hiera_t")

    montage_items = []
    total_changed = total_frames = 0
    for r in flagged:
        label = r["label"]
        if label not in inv:
            continue
        info = inv[label]
        pr = os.path.join(recording_dir(label, info["condition"]), "pipeline_results")
        labels = np.load(os.path.join(pr, "masks.npz"))["labels"].astype(np.int32)
        frames = load_recording(info["video_path"])["frames"]
        out = np.zeros_like(labels)
        n_changed = 0
        present = np.where((labels > 0).any(axis=(1, 2)))[0]
        for fi in present:
            seed = labels[fi] > 0
            mask, acc = _guarded_refine(frames[fi], seed, predictor)
            out[fi][mask] = 1
            n_changed += int(acc)
        np.savez_compressed(os.path.join(pr, "masks_refined.npz"),
                            labels=out, masks=(out > 0))
        total_changed += n_changed; total_frames += len(present)
        # montage panel at the frame the reviewer flagged
        ff = int(r.get("frame") or present[0])
        ff = ff if ff in present else int(present[0])
        seed0 = labels[ff] > 0
        montage_items.append((label, ff, frames[ff], seed0, out[ff] > 0,
                              bool((out[ff] > 0).sum() != seed0.sum())))
        print(f"  {label}: {n_changed}/{len(present)} frames refined", flush=True)

    _montage(montage_items, os.path.join(ANALYSIS_ROOT, "review",
                                         "refine_before_after.png"))
    print(f"\nRefined {len(flagged)} flagged recordings → masks_refined.npz "
          f"(non-destructive). {total_changed}/{total_frames} frames improved.")
    print("Review review/refine_before_after.png; promote with "
          "ic293_promote_refined.py if good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
