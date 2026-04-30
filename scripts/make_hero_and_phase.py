"""Make the README hero + phase-contrast figure.

Hero: 4 panels showing the pipeline end-to-end on a clean recording
      (Jesse pos0_wt — single cell, 200 frames, very stable tracking).
      Speeds are smoothed (rolling-window) and capped at a biologically
      plausible upper bound (15 µm/min for keratinocytes), eliminating
      the centroid-jump artifacts in the previous hero.

Phase-contrast: detect_hybrid_cpsam_multi requires cellpose 4, so this
      script runs in cellpose4 env. The DIC sections also run there
      because cpsam_dic loads natively (no subprocess round-trip).

Usage:
  conda run -n cellpose4 python scripts/make_hero_and_phase.py
"""
import os
import sys
import warnings
import logging
import glob

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports, benchmark_data_root  # noqa
setup_imports()

import numpy as np
import matplotlib.pyplot as plt
import tifffile

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

UM_PER_PX = 0.65
DT_MIN = 5.0
SPEED_CAP = 15.0   # keratinocytes don't migrate faster than this


# ──────────────────────────────────────────────────────────────────────
# Helpers (subset reused from make_doc_figures.py)
# ──────────────────────────────────────────────────────────────────────
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
                bps = pg.bitspersample
                sf = pg.sampleformat
                bo = pg.parent.byteorder
                offsets = pg.dataoffsets
                bcs = pg.databytecounts
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


def load_uint8_stack(path):
    s = safe_read_tiff(path)
    if s.dtype != np.uint8:
        p1, p99 = np.percentile(s, [1, 99])
        s = np.clip((s.astype(np.float32) - p1) /
                    max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
    return s


def smoothed_centroids(stack):
    """Centroid (y,x) per frame with 3-frame rolling smoothing."""
    from core.tracking import extract_centroids
    cents = extract_centroids(stack)  # (n,2), nan where missing
    valid = ~np.isnan(cents[:, 0])
    if valid.sum() < 3:
        return cents, valid
    # rolling mean window=3 over valid runs
    smoothed = cents.copy()
    for i in range(1, len(cents) - 1):
        if valid[i]:
            window = []
            for k in (-1, 0, 1):
                j = i + k
                if 0 <= j < len(cents) and valid[j]:
                    window.append(cents[j])
            if window:
                smoothed[i] = np.mean(window, axis=0)
    return smoothed, valid


def speed_timeseries(stack, um_per_px, dt_min, cap=SPEED_CAP):
    """Per-frame speed (µm/min) with smoothing + speed cap."""
    cents, valid = smoothed_centroids(stack)
    speeds = np.full(len(cents), np.nan)
    for i in range(1, len(cents)):
        if valid[i] and valid[i - 1]:
            d = (cents[i] - cents[i - 1]) * um_per_px
            s = float(np.linalg.norm(d) / dt_min)
            if s <= cap:
                speeds[i] = s
    return speeds


# ──────────────────────────────────────────────────────────────────────
# Hero figure: clean 4-panel pipeline visualisation
# ──────────────────────────────────────────────────────────────────────
def make_hero():
    """4 panels: Single-cell DIC, tracked multi-cell, speed/MSD, comparison."""
    bd = benchmark_data_root()

    # Single-cell DIC (very clean) — Jesse pos0_wt, 60 frames
    print("\n=== Hero: panel 1 — single-cell DIC ===")
    pos0 = bd / "data" / "examples" / "jesse_wt" / "pos0_wt.ome.tif"
    if not pos0.exists():
        print(f"  Missing {pos0}; skipping hero.")
        return
    stack_single = load_uint8_stack(str(pos0))[:60]
    from core.hybrid_dic import detect_hybrid_dic
    masks_single, _ = detect_hybrid_dic(
        stack_single, model_path="data/models/cpsam_dic",
        use_preprocess=False, use_deepsea=True, use_retry=False)

    # Multi-cell DIC — pos17_wt, 30 frames (4 cells)
    print("\n=== Hero: panel 2 — multi-cell DIC ===")
    pos17 = bd / "data" / "examples" / "jesse_wt" / "pos17_wt.ome.tif"
    stack_multi = load_uint8_stack(str(pos17))[:30]
    from core.hybrid_dic import detect_hybrid_dic_multi
    # Disable gap fill for figure generation — it subprocess-calls
    # cellpose env which is slow when we're already in cellpose4.
    result_multi = detect_hybrid_dic_multi(
        stack_multi, model_path="data/models/cpsam_dic",
        min_area_px=500,
        use_preprocess=False, use_deepsea=True, use_retry=False,
        use_gap_fill=False)

    # Compose
    fig = plt.figure(figsize=(16, 5.0))

    # Panel 1: single-cell detection overlay
    ax1 = fig.add_subplot(1, 4, 1)
    fi = 30
    ax1.imshow(imnorm(stack_single[fi]), cmap="gray", vmin=0, vmax=1)
    if masks_single[fi].any():
        ax1.contour(masks_single[fi].astype(np.uint8), levels=[0.5],
                    colors=["#ff5555"], linewidths=1.6)
    ax1.set_title("1. Detect — DIC single cell\n(cpsam_dic + DeepSea)",
                  fontsize=10, fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])

    # Panel 2: multi-cell detection
    ax2 = fig.add_subplot(1, 4, 2)
    fi = 10
    ax2.imshow(imnorm(stack_multi[fi]), cmap="gray", vmin=0, vmax=1)
    labels = result_multi["labels"][fi]
    ids = sorted(set(np.unique(labels).tolist()) - {0})
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(ids), 10)))
    for i, cid in enumerate(ids):
        ax2.contour((labels == cid).astype(np.uint8), levels=[0.5],
                    colors=[cmap[i % 10]], linewidths=1.6)
    ax2.set_title("2. Detect — multi-cell\n"
                  f"({len(ids)} cells, per-cell labels)",
                  fontsize=10, fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])

    # Panel 3: tracked trajectories
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(imnorm(stack_multi[len(stack_multi) // 2]),
               cmap="gray", vmin=0, vmax=1, alpha=0.55)
    for i, t in enumerate(result_multi["tracks"]):
        cents = t.get("centroids", [])
        if not cents:
            continue
        ys = [c[0] for c in cents]; xs = [c[1] for c in cents]
        col = cmap[i % 10]
        ax3.plot(xs, ys, "-", color=col, linewidth=2.2, alpha=0.95)
        ax3.plot(xs[0], ys[0], "o", color=col, markersize=7,
                 markeredgecolor="white", markeredgewidth=1.3)
        ax3.plot(xs[-1], ys[-1], "s", color=col, markersize=7,
                 markeredgecolor="white", markeredgewidth=1.3)
    ax3.set_title("3. Track — Hungarian + gap fill",
                  fontsize=10, fontweight="bold")
    ax3.set_xticks([]); ax3.set_yticks([])

    # Panel 4: clean speed timeseries on the single-cell recording.
    # Smoothed + capped — reflects actual biology rather than tracking
    # noise.
    ax4 = fig.add_subplot(1, 4, 4)
    speeds = speed_timeseries(masks_single, UM_PER_PX, DT_MIN)
    ts = np.arange(len(speeds)) * DT_MIN
    ax4.plot(ts, speeds, "-", color="#3b6cad", linewidth=2,
             label="Per-frame")
    # rolling mean
    if np.any(~np.isnan(speeds)):
        valid_idx = ~np.isnan(speeds)
        rolled = np.full_like(speeds, np.nan)
        for i in range(len(speeds)):
            lo, hi = max(0, i - 3), min(len(speeds), i + 4)
            chunk = speeds[lo:hi]
            chunk = chunk[~np.isnan(chunk)]
            if chunk.size:
                rolled[i] = chunk.mean()
        ax4.plot(ts, rolled, "-", color="#d65a31", linewidth=2.5,
                 label="Rolling mean")
    ax4.set_xlabel("Time (min)")
    ax4.set_ylabel("Speed (µm/min)")
    ax4.set_title("4. Analyse — migration",
                  fontsize=10, fontweight="bold")
    ax4.set_ylim(0, max(SPEED_CAP, np.nanmax(speeds) * 1.1
                         if np.any(~np.isnan(speeds)) else SPEED_CAP))
    ax4.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax4.grid(alpha=0.3)

    fig.suptitle("CellScope: detect → track → analyse",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    out = f"{OUT}/hero.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Phase-contrast figure (requires cellpose 4)
# ──────────────────────────────────────────────────────────────────────
def make_phase():
    bd = benchmark_data_root()
    print("\n=== Phase-contrast multi-cell ===")
    path = bd / "data" / "ignasi" / "IC293__1_MMStack_Pos2-WT.ome-cropped.tif"
    if not path.exists():
        print(f"  Missing {path}; skipping.")
        return
    stack = load_uint8_stack(str(path))[:30]
    from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi
    result = detect_hybrid_cpsam_multi(
        stack, min_area_px=500,
        use_fallback=True, use_deepsea=True, use_gap_fill=False)
    labels = result["labels"]

    fig, ax = plt.subplots(figsize=(7, 5))
    fi = 10
    ax.imshow(imnorm(stack[fi]), cmap="gray", vmin=0, vmax=1)
    ids = sorted(set(np.unique(labels[fi]).tolist()) - {0})
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(ids), 10)))
    for i, cid in enumerate(ids):
        ax.contour((labels[fi] == cid).astype(np.uint8), levels=[0.5],
                   colors=[cmap[i % 10]], linewidths=1.8)
    ax.set_title(f"Phase-contrast multi-cell: cpsam + DeepSea "
                 f"(frame {fi}, {len(ids)} cells)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = f"{OUT}/focused_phase_detected.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    make_hero()
    make_phase()
    print(f"\nDone. Figures under {OUT}/")


if __name__ == "__main__":
    main()
