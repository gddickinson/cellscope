"""Per-recording review montages for the curated IC293 set.

For every recording, render the cell's mask outline on the DIC at evenly
sampled frames (one montage PNG per recording) + a REVIEW_INDEX.md that
lists frames / presence / number of cell IDs and FLAGS anything whose ID
count isn't exactly 1 (the curated single-cell invariant). Lets you scan
all recordings quickly and jump straight to anything off.

Outputs under ic293_analysis/review/<cond>/<label>.png + REVIEW_INDEX.md.
Usage: conda run -n cellpose4 python scripts/ic293_review_montages.py
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

import numpy as np  # noqa: E402
from scripts.ic293_common import (  # noqa: E402
    inventory_cache, recording_dir, ANALYSIS_EXCLUDE, ANALYSIS_ROOT)

REVIEW_DIR = os.path.join(ANALYSIS_ROOT, "review")
_COLORS = ["lime", "red", "cyan", "yellow", "magenta", "orange"]


def _montage(frames, labels, ids, title, flag, out_path, n=12, ncols=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = labels.shape[0]
    sample = np.unique(np.linspace(0, T - 1, min(n, T)).round().astype(int))
    nrows = (len(sample) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows),
                            squeeze=False)
    for i, f in enumerate(sample):
        ax = axs[i // ncols][i % ncols]
        ax.imshow(frames[f], cmap="gray")
        for k, cid in enumerate(ids):
            m = labels[f] == cid
            if m.any():
                ax.contour(m, levels=[0.5], colors=_COLORS[k % len(_COLORS)],
                           linewidths=1.0)
        ax.set_title(f"F{f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(sample), nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    fig.suptitle(title, fontsize=12, color=("red" if flag else "black"))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=95); plt.close(fig)


def main():
    argparse.ArgumentParser().parse_args()
    inv = inventory_cache()
    rows = []
    for label, info in sorted(inv.items(), key=lambda kv: (kv[1]["condition"], kv[0])):
        cond = info["condition"]
        mp = os.path.join(recording_dir(label, cond), "pipeline_results", "masks.npz")
        if not os.path.exists(mp):
            continue
        labels = np.load(mp)["labels"]
        T = labels.shape[0]
        ids = [int(v) for v in np.unique(labels) if v > 0]
        present = int((labels > 0).any(axis=(1, 2)).sum())
        excluded = label in ANALYSIS_EXCLUDE
        # invariant: exactly one cell ID, present every frame (unless excluded)
        flag = (len(ids) != 1 or present != T) and not excluded
        tag = "  [EXCLUDED]" if excluded else ("  ⚑" if flag else "")
        title = (f"{label} ({cond}){tag}\n{T} frames · presence "
                 f"{present}/{T} · {len(ids)} cell ID(s)")
        _montage(_load_frames(info), labels, ids, title, flag,
                 os.path.join(REVIEW_DIR, cond, f"{label}.png"))
        rows.append((label, cond, T, present, len(ids), excluded, flag))

    rows.sort(key=lambda r: (not r[6], r[5], r[1], r[0]))   # flagged first
    L = ["# IC293 review index", "",
         f"{len(rows)} recordings. Montages in `review/<cond>/<label>.png`. "
         "Invariant: exactly **1 cell ID**, present every frame. "
         "⚑ = needs a look; rows are sorted flagged-first.", "",
         "| recording | cond | frames | presence | cell IDs | status |",
         "|---|---|--:|--:|--:|---|"]
    for label, cond, T, present, nid, exc, flag in rows:
        status = "EXCLUDED (kept on disk)" if exc else ("⚑ REVIEW" if flag else "ok")
        L.append(f"| {label} | {cond} | {T} | {present}/{T} | {nid} | {status} |")
    n_flag = sum(1 for r in rows if r[6])
    L.insert(3, f"**{n_flag} flagged** (ID count ≠ 1 or gaps); "
             f"{sum(1 for r in rows if not r[6] and not r[5])} clean single-cell.")
    with open(os.path.join(REVIEW_DIR, "REVIEW_INDEX.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {len(rows)} montages + REVIEW_INDEX.md → {REVIEW_DIR}/")
    print(f"  flagged (ID≠1 or gaps): {n_flag}")
    return 0


def _load_frames(info):
    from core.io import load_recording
    return load_recording(info["video_path"])["frames"]


if __name__ == "__main__":
    sys.exit(main())
