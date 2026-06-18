"""Validate + demo the boundary RF (from ic293_train_boundary_rf.py).

Hold-out test on clean masks (recordings NOT used in training): perturb
each good mask — erode (simulate under-seg), dilate (simulate over-seg),
none (degradation check) — refine via the RF probability isoline, and
measure IoU recovery vs the true mask across thresholds. Then a visual
before/after on the flagged cells. Writes review/rf_validation.png +
prints the IoU table.

Usage: conda run -n cellpose4 python scripts/ic293_rf_validate.py
"""
import os
import sys
import csv
import json
import argparse
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

import numpy as np  # noqa: E402
from scipy.ndimage import binary_erosion, binary_dilation  # noqa: E402
from core.io import load_recording  # noqa: E402
from core.boundary_rf import (  # noqa: E402
    load_rf_model, predict_cell_probability, _refine_with_isoline)
from scripts.ic293_common import (  # noqa: E402
    inventory_cache, recording_dir, ANALYSIS_ROOT)

THRESHOLDS = [0.4, 0.5, 0.6, 0.7]


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def _flagged_rows():
    p = os.path.join(ANALYSIS_ROOT, "review", "review_flags.csv")
    out = []
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                if str(r.get("flagged", "")).lower() in ("1", "true", "yes"):
                    out.append((r["label"], int(r.get("frame") or 0)))
    return out


def _demo(inv, rf, config, thresh, out_path, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure
    rows = _flagged_rows()[:n]
    if not rows:
        return
    ncols = 4
    nrows = (len(rows) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
                            squeeze=False)
    for i, (label, f) in enumerate(rows):
        ax = axs[i // ncols][i % ncols]
        if label not in inv:
            ax.axis("off"); continue
        info = inv[label]
        frames = load_recording(info["video_path"])["frames"]
        labels = np.load(os.path.join(recording_dir(label, info["condition"]),
                                      "pipeline_results", "masks.npz"))["labels"]
        f = f if f < len(frames) else 0
        img, seed = frames[f], labels[f] > 0
        prob = predict_cell_probability(img, rf, config)
        ref = _refine_with_isoline(prob, seed, thresh, 200)
        ax.imshow(img, cmap="gray")
        for ct in measure.find_contours(seed.astype(float), 0.5):
            ax.plot(ct[:, 1], ct[:, 0], "-", color="red", lw=1.1)
        for ct in measure.find_contours(ref.astype(float), 0.5):
            ax.plot(ct[:, 1], ct[:, 0], "-", color="lime", lw=1.2)
        ys, xs = np.where(seed | ref)
        if len(ys):
            ax.set_xlim(xs.min() - 18, xs.max() + 18)
            ax.set_ylim(ys.max() + 18, ys.min() - 18)
        ax.set_title(f"{label} F{f}  {seed.sum()}->{int(ref.sum())}px", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(rows), nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"red = current mask | green = RF-refined (isoline t={thresh})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="/tmp/rf_holdout.json")
    ap.add_argument("--max-frames", type=int, default=30)
    args = ap.parse_args()
    rf, config = load_rf_model()
    if rf is None:
        print("no RF model — run ic293_train_boundary_rf.py first"); return 1
    inv = inventory_cache()
    holdout = json.load(open(args.holdout)) if os.path.exists(args.holdout) else []

    frames_list = []
    for label in holdout:
        if label not in inv or len(frames_list) >= args.max_frames:
            continue
        info = inv[label]
        labels = np.load(os.path.join(recording_dir(label, info["condition"]),
                                      "pipeline_results", "masks.npz"))["labels"]
        present = np.where((labels > 0).any(axis=(1, 2)))[0]
        if len(present) < 3:
            continue
        vid = load_recording(info["video_path"])["frames"]
        for f in (present[len(present) // 3], present[2 * len(present) // 3]):
            frames_list.append((label, int(f), vid[f], labels[f] > 0))
    print(f"hold-out frames: {len(frames_list)}", flush=True)

    res = collections.defaultdict(list)
    perts = [("erode3", lambda m: binary_erosion(m, iterations=3)),
             ("dilate3", lambda m: binary_dilation(m, iterations=3)),
             ("none", lambda m: m)]
    for k, (label, f, img, M) in enumerate(frames_list):
        prob = predict_cell_probability(img, rf, config)
        for pert, fn in perts:
            seed = fn(M)
            res[("before", pert)].append(_iou(seed, M))
            for t in THRESHOLDS:
                res[("after", pert, t)].append(_iou(_refine_with_isoline(prob, seed, t, 200), M))
        if (k + 1) % 5 == 0:
            print(f"  {k+1}/{len(frames_list)}", flush=True)

    print(f"\n=== hold-out IoU vs true good mask (mean over {len(frames_list)} frames) ===")
    for pert in ("erode3", "dilate3", "none"):
        before = np.mean(res[("before", pert)])
        afters = "  ".join(f"t{t}={np.mean(res[('after', pert, t)]):.3f}"
                           for t in THRESHOLDS)
        print(f"  {pert:8s} before={before:.3f}   after: {afters}")

    # Pick threshold: best erode-recovery while degradation (none) stays >=0.95
    best_t, best_rec = THRESHOLDS[0], -1
    for t in THRESHOLDS:
        rec = np.mean(res[("after", "erode3", t)])
        deg = np.mean(res[("after", "none", t)])
        if deg >= 0.95 and rec > best_rec:
            best_rec, best_t = rec, t
    print(f"\nchosen threshold (max erode-recovery with none-degradation>=0.95): "
          f"t={best_t} (recovery {best_rec:.3f})")
    _demo(inv, rf, config, best_t,
          os.path.join(ANALYSIS_ROOT, "review", "rf_validation.png"))
    print("wrote review/rf_validation.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
