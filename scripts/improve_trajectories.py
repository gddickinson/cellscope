"""Regenerate multi_trajectories.png with high-contrast track lines.

Reads existing focused_multi_detected.png, extracts the track lines
from the previously-saved pipeline output (cached in results/), and
overlays them with thicker / brighter strokes against a dimmed image.

If the cached pipeline output isn't available, falls back to a fresh
hybrid_dic_multi run.

Usage:
  conda run -n cellpose python scripts/improve_trajectories.py
"""
import os
import sys
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports, benchmark_data_root  # noqa
setup_imports()

import numpy as np
import matplotlib.pyplot as plt
import tifffile

OUT = "docs/figures"


def imnorm(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    return np.clip((img - p1) / max(p99 - p1, 1e-6), 0, 1)


def safe_read_tiff(path):
    try:
        return tifffile.imread(path)
    except AttributeError as e:
        if "newbyteorder" not in str(e):
            raise
    pages = []
    with tifffile.TiffFile(path) as tf:
        with open(path, "rb") as fh:
            for pg in tf.pages:
                bps = pg.bitspersample; sf = pg.sampleformat
                bo = pg.parent.byteorder
                offsets = pg.dataoffsets; bcs = pg.databytecounts
                chunks = []
                for off, n in zip(offsets, bcs):
                    fh.seek(off); chunks.append(fh.read(n))
                raw = b"".join(chunks)
                kind = {1: "u", 2: "i", 3: "f"}.get(sf, "u")
                dt = np.dtype(f"{bo}{kind}{bps // 8}")
                arr = np.frombuffer(raw, dtype=dt).reshape(
                    pg.imagelength, pg.imagewidth)
                if arr.dtype.byteorder not in ("=", "|"):
                    arr = arr.astype(arr.dtype.newbyteorder("=")).copy()
                pages.append(arr)
    return np.stack(pages) if len(pages) > 1 else pages[0]


def main():
    bd = benchmark_data_root()
    pos17 = bd / "data" / "examples" / "jesse_wt" / "pos17_wt.ome.tif"
    s = safe_read_tiff(str(pos17))
    if s.dtype != np.uint8:
        p1, p99 = np.percentile(s, [1, 99])
        s = np.clip((s.astype(np.float32) - p1) /
                    max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
    stack = s[:30]

    print("Running hybrid_dic_multi (no gap fill, ~3 min)…")
    from core.hybrid_dic import detect_hybrid_dic_multi
    result = detect_hybrid_dic_multi(
        stack, model_path="data/models/cpsam_dic",
        min_area_px=500, use_preprocess=False,
        use_deepsea=True, use_retry=False, use_gap_fill=False)
    tracks = result["tracks"]
    print(f"  → {len(tracks)} tracks")

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
    bg = imnorm(stack[len(stack) // 2])
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, alpha=0.45)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(tracks), 10)))
    for i, t in enumerate(tracks):
        cents = t.get("centroids", [])
        if not cents:
            continue
        ys = [c[0] for c in cents]; xs = [c[1] for c in cents]
        col = cmap[i % 10]
        # thick line, big endpoints
        ax.plot(xs, ys, "-", color=col, linewidth=3.5, alpha=0.95,
                label=f"Cell {i + 1}",
                path_effects=None)
        ax.plot(xs[0], ys[0], "o", color=col, markersize=10,
                markeredgecolor="white", markeredgewidth=2)
        ax.plot(xs[-1], ys[-1], "s", color=col, markersize=10,
                markeredgecolor="white", markeredgewidth=2)
    ax.set_title(f"Multi-cell tracking ({len(tracks)} tracks "
                 f"over {len(stack)} frames)", fontsize=11,
                 color="white")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85,
              labelcolor="black")
    fig.tight_layout()
    out = f"{OUT}/multi_trajectories.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
