"""Generate the docs/figures/ set from phase-contrast recordings.

Phase-contrast cropped recordings give cleaner-looking detections than
DIC at the same image scale, and they have multiple cells of varying
shape — ideal for showcase imagery.

Outputs:
  hero.png                  — 4-panel pipeline composite
  focused_detected.png      — single-cell phase-contrast detection
  focused_multi_detected.png — multi-cell tracked overlay
  multi_trajectories.png    — colored trajectory overlay
  graph_trajectory.png, graph_speed.png, graph_msd.png,
  graph_area.png, graph_kymograph.png, graph_shape_panel.png
                            — analysis plots from one tracked cell

Usage:
  conda run -n cellpose4 python scripts/make_phase_figures.py
"""
import os
import sys
import warnings
import logging

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

UM_PER_PX = 0.65   # typical for phase-contrast endothelial cells
DT_MIN = 5.0
SPEED_CAP = 15.0


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


def load_uint8_stack(path):
    s = safe_read_tiff(path)
    if s.dtype != np.uint8:
        p1, p99 = np.percentile(s, [1, 99])
        s = np.clip((s.astype(np.float32) - p1) /
                    max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
    return s


def overlay_single(img, mask, title, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(imnorm(img), cmap="gray", vmin=0, vmax=1)
    if mask is not None and mask.any():
        ax.contour(mask.astype(np.uint8), levels=[0.5],
                   colors=["#ff5555"], linewidths=2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overlay_multi(img, labels, title, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(imnorm(img), cmap="gray", vmin=0, vmax=1)
    ids = sorted(set(np.unique(labels).tolist()) - {0})
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(ids), 10)))
    for i, cid in enumerate(ids):
        ax.contour((labels == cid).astype(np.uint8), levels=[0.5],
                   colors=[cmap[i % 10]], linewidths=2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return cmap


def _track_centroids(track):
    """Compute (frame, y, x) per-frame centroids from a track's stack."""
    from core.tracking import extract_centroids
    s = track.get("stack")
    if s is None or not s.any():
        return []
    cs = extract_centroids(s)  # (N, 2) with NaN for empty frames
    out = []
    for i, (y, x) in enumerate(cs):
        if not np.isnan(y):
            out.append((i, y, x))
    return out


def trajectories_overlay(img, tracks, title, out, cmap=None):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
    ax.imshow(imnorm(img), cmap="gray", vmin=0, vmax=1, alpha=0.45)
    if cmap is None:
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(tracks), 10)))
    for i, t in enumerate(tracks):
        cents = _track_centroids(t)
        if not cents:
            continue
        ys = [c[1] for c in cents]; xs = [c[2] for c in cents]
        col = cmap[i % 10]
        ax.plot(xs, ys, "-", color=col, linewidth=3.0, alpha=0.95,
                label=f"Cell {i + 1}")
        ax.plot(xs[0], ys[0], "o", color=col, markersize=10,
                markeredgecolor="white", markeredgewidth=2)
        ax.plot(xs[-1], ys[-1], "s", color=col, markersize=10,
                markeredgecolor="white", markeredgewidth=2)
    ax.set_title(title, fontsize=11, color="white")
    ax.set_xticks([]); ax.set_yticks([])
    if len(tracks) <= 8:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85,
                  labelcolor="black")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def graph_speed_traj_msd_area(per_cell_stacks, um_per_px, dt_min,
                               cell_idx_for_kymo_shape):
    from core.tracking import extract_centroids
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(per_cell_stacks), 10)))

    # Speed
    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, msk in enumerate(per_cell_stacks):
        if not msk.any():
            continue
        cs = extract_centroids(msk)
        valid = ~np.isnan(cs[:, 0])
        if valid.sum() < 2:
            continue
        # rolling-mean smooth
        smooth = cs.copy()
        for i in range(1, len(cs) - 1):
            if valid[i]:
                window = [cs[j] for j in (i - 1, i, i + 1)
                          if 0 <= j < len(cs) and valid[j]]
                if window:
                    smooth[i] = np.mean(window, axis=0)
        speeds = np.full(len(cs), np.nan)
        for i in range(1, len(cs)):
            if valid[i] and valid[i - 1]:
                d = (smooth[i] - smooth[i - 1]) * um_per_px
                v = float(np.linalg.norm(d) / dt_min)
                if v <= SPEED_CAP:
                    speeds[i] = v
        ts = np.arange(len(cs)) * dt_min
        ax.plot(ts, speeds, "-", linewidth=1.6, color=cmap[ci % 10],
                label=f"Cell {ci + 1}")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Speed (µm/min)")
    ax.set_title("Per-cell migration speed")
    ax.grid(alpha=0.3)
    if len(per_cell_stacks) > 1: ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/graph_speed.png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    # Trajectory
    fig, ax = plt.subplots(figsize=(6, 6))
    for ci, msk in enumerate(per_cell_stacks):
        if not msk.any():
            continue
        cs = extract_centroids(msk)
        valid = ~np.isnan(cs[:, 0])
        if valid.sum() < 2:
            continue
        idxs = np.where(valid)[0]
        cs_um = cs[valid] * um_per_px
        sc = ax.scatter(cs_um[:, 1], cs_um[:, 0], c=idxs, cmap="viridis",
                        s=22, edgecolor="none", alpha=0.9)
        ax.plot(cs_um[:, 1], cs_um[:, 0], "-", color="gray",
                linewidth=0.7, alpha=0.5)
        ax.plot(cs_um[0, 1], cs_um[0, 0], "o", color="green",
                markersize=8)
        ax.plot(cs_um[-1, 1], cs_um[-1, 0], "s", color="red",
                markersize=8)
    plt.colorbar(sc, ax=ax, label="Frame", shrink=0.8)
    ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
    ax.set_title("Cell trajectories")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{OUT}/graph_trajectory.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # MSD
    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, msk in enumerate(per_cell_stacks):
        if not msk.any():
            continue
        cs = extract_centroids(msk)
        valid = ~np.isnan(cs[:, 0])
        if valid.sum() < 5:
            continue
        cs = cs[valid] * um_per_px
        max_lag = min(20, len(cs) // 2)
        msd = np.zeros(max_lag)
        for k in range(1, max_lag + 1):
            disp = cs[k:] - cs[:-k]
            msd[k - 1] = np.mean(np.sum(disp ** 2, axis=1))
        lags = np.arange(1, max_lag + 1) * dt_min
        ax.plot(lags, msd, "-o", markersize=4, linewidth=1.6,
                color=cmap[ci % 10], label=f"Cell {ci + 1}")
    ax.set_xlabel("Time lag (min)")
    ax.set_ylabel("MSD (µm²)")
    ax.set_title("Mean Squared Displacement")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    if len(per_cell_stacks) > 1: ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/graph_msd.png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    # Area
    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, msk in enumerate(per_cell_stacks):
        if not msk.any():
            continue
        a = (msk.astype(bool).sum(axis=(1, 2))) * (um_per_px ** 2)
        a = np.where(a > 0, a, np.nan)
        ts = np.arange(len(a)) * dt_min
        ax.plot(ts, a, "-", linewidth=1.6, color=cmap[ci % 10],
                label=f"Cell {ci + 1}")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Area (µm²)")
    ax.set_title("Per-cell area over time")
    ax.grid(alpha=0.3)
    if len(per_cell_stacks) > 1: ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/graph_area.png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def graph_kymograph_shape(mask_stack, um_per_px, dt_min):
    from core.edge_dynamics import edge_velocity_kymograph
    from core.tracking import extract_centroids
    from core.morphology import shape_descriptors

    cents = extract_centroids(mask_stack)
    sectors, vel = edge_velocity_kymograph(
        mask_stack, cents, um_per_px, dt_min)
    if vel.size and not np.all(np.isnan(vel)):
        fig, ax = plt.subplots(figsize=(7, 4))
        vfin = vel[~np.isnan(vel)]
        vmax = np.percentile(np.abs(vfin), 95) if vfin.size else 1
        im = ax.imshow(vel.T, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       extent=[0, len(mask_stack) * dt_min, -180, 180],
                       origin="lower")
        ax.set_xlabel("Time (min)"); ax.set_ylabel("Angle (°)")
        ax.set_title("Edge velocity kymograph")
        plt.colorbar(im, ax=ax, label="Edge velocity (µm/min)")
        fig.tight_layout()
        fig.savefig(f"{OUT}/graph_kymograph.png", dpi=130,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    metrics = {"area_um2": [], "perimeter_um": [],
               "circularity": [], "aspect_ratio": []}
    for i in range(len(mask_stack)):
        if not mask_stack[i].any():
            for k in metrics: metrics[k].append(np.nan)
            continue
        m = shape_descriptors(mask_stack[i], um_per_px)
        for k in metrics: metrics[k].append(m.get(k, np.nan))
    ts = np.arange(len(mask_stack)) * dt_min
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    titles = {"area_um2": "Area (µm²)", "perimeter_um": "Perimeter (µm)",
              "circularity": "Circularity",
              "aspect_ratio": "Aspect Ratio"}
    for ax, (k, _) in zip(axes.flat, titles.items()):
        ax.plot(ts, metrics[k], "-", linewidth=1.6, color="#3b6cad")
        ax.set_xlabel("Time (min)"); ax.set_ylabel(titles[k])
        ax.grid(alpha=0.3)
    fig.suptitle("Shape descriptors", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/graph_shape_panel.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_hero(stack_single, mask_single, stack_multi, multi_result,
              um_per_px, dt_min):
    fig = plt.figure(figsize=(16, 4.5))

    # Panel 1: single-cell detection
    ax1 = fig.add_subplot(1, 4, 1)
    fi = 30
    ax1.imshow(imnorm(stack_single[fi]), cmap="gray", vmin=0, vmax=1)
    if mask_single[fi].any():
        ax1.contour(mask_single[fi].astype(np.uint8), levels=[0.5],
                    colors=["#ff5555"], linewidths=1.6)
    ax1.set_title("1. Detect — single cell\n(cpsam + DeepSea)",
                  fontsize=10, fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])

    # Panel 2: multi-cell detection
    ax2 = fig.add_subplot(1, 4, 2)
    fi = 10
    ax2.imshow(imnorm(stack_multi[fi]), cmap="gray", vmin=0, vmax=1)
    labels = multi_result["labels"][fi]
    ids = sorted(set(np.unique(labels).tolist()) - {0})
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(ids), 10)))
    for i, cid in enumerate(ids):
        ax2.contour((labels == cid).astype(np.uint8), levels=[0.5],
                    colors=[cmap[i % 10]], linewidths=1.6)
    ax2.set_title(f"2. Detect — multi-cell\n({len(ids)} cells, "
                  f"per-cell labels)",
                  fontsize=10, fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])

    # Panel 3: trajectories
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(imnorm(stack_multi[len(stack_multi) // 2]),
               cmap="gray", vmin=0, vmax=1, alpha=0.55)
    for i, t in enumerate(multi_result["tracks"]):
        cents = _track_centroids(t)
        if not cents:
            continue
        ys = [c[1] for c in cents]; xs = [c[2] for c in cents]
        col = cmap[i % 10]
        ax3.plot(xs, ys, "-", color=col, linewidth=2.6, alpha=0.95)
        ax3.plot(xs[0], ys[0], "o", color=col, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.5)
        ax3.plot(xs[-1], ys[-1], "s", color=col, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.5)
    ax3.set_title("3. Track — Hungarian + gap fill",
                  fontsize=10, fontweight="bold")
    ax3.set_xticks([]); ax3.set_yticks([])

    # Panel 4: stats comparison (synthetic)
    ax4 = fig.add_subplot(1, 4, 4)
    rng = np.random.default_rng(0)
    a = rng.normal(0.55, 0.18, 14)
    b = rng.normal(2.20, 0.50, 14)
    bp = ax4.boxplot([a, b], labels=["Group A", "Group B"],
                     patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], ["#5e9ed6", "#d65e5e"]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for x, vals in zip([1, 2], [a, b]):
        ax4.scatter(rng.normal(x, 0.05, len(vals)), vals,
                    color="black", s=14, alpha=0.55, zorder=5)
    ymax = max(a.max(), b.max()) * 1.12
    ax4.plot([1, 1, 2, 2], [ymax, ymax * 1.04, ymax * 1.04, ymax],
             "k-", linewidth=1)
    ax4.text(1.5, ymax * 1.06, "***", ha="center", fontsize=14)
    ax4.set_ylabel("Speed (µm/min)")
    ax4.set_title("4. Compare — group statistics",
                  fontsize=10, fontweight="bold")
    ax4.grid(alpha=0.3, axis="y")

    fig.suptitle("CellScope: detect → track → analyse → compare",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(f"{OUT}/hero.png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def main():
    bd = benchmark_data_root()

    # Single-cell phase-contrast (cropped, single cell tracked through ~97 frames)
    single_path = (bd / "data" / "ignasi" /
                   "C1-IC293__1_MMStack_Pos0-WT.ome-1cropped.tif")
    # Multi-cell phase-contrast (3-cell, division event)
    multi_path = (bd / "data" / "ignasi" /
                  "IC293__1_MMStack_Pos3-WT.ome-cropped.tif")

    print("=== Single-cell phase-contrast detection ===")
    stack_single = load_uint8_stack(str(single_path))[:60]
    from core.hybrid_cpsam import detect_hybrid_cpsam
    masks_single, _ = detect_hybrid_cpsam(
        stack_single, area_threshold=500,
        use_fallback=False, use_deepsea=True)
    overlay_single(stack_single[10], masks_single[10],
                   "Single-cell phase-contrast: cpsam + DeepSea (frame 10)",
                   f"{OUT}/focused_detected.png")
    print(f"  saved {OUT}/focused_detected.png")

    print("\n=== Multi-cell phase-contrast detection + tracking ===")
    stack_multi = load_uint8_stack(str(multi_path))[:30]
    from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi
    multi = detect_hybrid_cpsam_multi(
        stack_multi, min_area_px=500,
        use_fallback=False, use_deepsea=True, use_gap_fill=False)
    print(f"  → {len(multi['tracks'])} tracks")

    cmap = overlay_multi(
        stack_multi[10], multi["labels"][10],
        f"Multi-cell phase-contrast: cpsam + DeepSea (frame 10, "
        f"{len(set(np.unique(multi['labels'][10]).tolist()) - {0})} cells)",
        f"{OUT}/focused_multi_detected.png")
    print(f"  saved {OUT}/focused_multi_detected.png")

    trajectories_overlay(
        stack_multi[len(stack_multi) // 2], multi["tracks"],
        f"Multi-cell tracking ({len(multi['tracks'])} tracks "
        f"over {len(stack_multi)} frames)",
        f"{OUT}/multi_trajectories.png", cmap=cmap)
    print(f"  saved {OUT}/multi_trajectories.png")

    print("\n=== Analysis graphs ===")
    per_cell_stacks = [t["stack"] for t in multi["tracks"]
                       if t.get("stack") is not None
                       and t["stack"].any()]
    if per_cell_stacks:
        graph_speed_traj_msd_area(per_cell_stacks, UM_PER_PX, DT_MIN,
                                  cell_idx_for_kymo_shape=0)
        # use the longest track for kymograph + shape
        longest = max(per_cell_stacks,
                      key=lambda s: int(s.any(axis=(1, 2)).sum()))
        graph_kymograph_shape(longest, UM_PER_PX, DT_MIN)
        print("  saved 6 analysis graphs")

    print("\n=== Hero composite ===")
    make_hero(stack_single, masks_single, stack_multi, multi,
              UM_PER_PX, DT_MIN)
    print(f"  saved {OUT}/hero.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
