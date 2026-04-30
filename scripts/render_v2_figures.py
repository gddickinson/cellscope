"""Render docs/figures/ from cached pipeline outputs.

Reads results/figure_pipelines/*.npz (built by cache_pipelines.py) and
writes high-quality figures with:
  • Track-length filter (drop short/phantom tracks)
  • Temporal smoothing (3-frame rolling mean) + speed cap
  • Larger contour widths, clearer legends, proper axis units

Usage:
  conda run -n cellpose python scripts/render_v2_figures.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

import numpy as np
import matplotlib.pyplot as plt

CACHE = "results/figure_pipelines"
OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

UM_PER_PX = 0.65
DT_MIN = 5.0
SPEED_CAP = 15.0
TRACK_KEEP_FRACTION = 0.5  # drop tracks shorter than 50% of window


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def imnorm(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    return np.clip((img - p1) / max(p99 - p1, 1e-6), 0, 1)


def load_multi(path):
    """Return (frames, labels, tracks_list-of-dicts)."""
    z = np.load(path, allow_pickle=False)
    frames = z["frames"]
    labels = z["labels"]
    n_tracks = int(z["tracks_n"])
    tracks = []
    for i in range(50):  # safe upper bound
        if f"track_{i}_stack" not in z:
            continue
        tracks.append({
            "id": i,
            "stack": z[f"track_{i}_stack"],
            "centroids": z[f"track_{i}_centroids"],
            "first": int(z[f"track_{i}_first"]),
            "last": int(z[f"track_{i}_last"]),
        })
    return frames, labels, tracks


def load_single(path):
    z = np.load(path, allow_pickle=False)
    return z["frames"], z["masks"]


def filter_tracks(tracks, n_frames, keep_fraction=TRACK_KEEP_FRACTION):
    """Keep only tracks present in ≥ keep_fraction × n_frames frames."""
    min_frames = int(keep_fraction * n_frames)
    kept = []
    for t in tracks:
        n_present = int(t["stack"].any(axis=(1, 2)).sum())
        if n_present >= min_frames:
            t = dict(t, n_present=n_present)
            kept.append(t)
    return kept


def smoothed_speeds(centroids, um_per_px, dt_min, cap=SPEED_CAP, win=3):
    """Per-frame speed (µm/min) with rolling-window centroid smoothing
    + speed cap to remove tracking jitter spikes.
    """
    valid = ~np.isnan(centroids[:, 0])
    smooth = centroids.copy()
    half = win // 2
    for i in range(len(centroids)):
        if valid[i]:
            chunks = []
            for k in range(-half, half + 1):
                j = i + k
                if 0 <= j < len(centroids) and valid[j]:
                    chunks.append(centroids[j])
            if chunks:
                smooth[i] = np.mean(chunks, axis=0)
    speeds = np.full(len(centroids), np.nan)
    for i in range(1, len(centroids)):
        if valid[i] and valid[i - 1]:
            d = (smooth[i] - smooth[i - 1]) * um_per_px
            v = float(np.linalg.norm(d) / dt_min)
            if v <= cap:
                speeds[i] = v
    return speeds


# ──────────────────────────────────────────────────────────────────────
# Figure generators
# ──────────────────────────────────────────────────────────────────────
def fig_focused_detected(frames, masks, fi=20, title=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(imnorm(frames[fi]), cmap="gray", vmin=0, vmax=1)
    if masks[fi].any():
        ax.contour(masks[fi].astype(np.uint8), levels=[0.5],
                   colors=["#ff5555"], linewidths=2)
    if title is None:
        title = f"Single-cell detection: cpsam + DeepSea (frame {fi})"
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{OUT}/focused_detected.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {OUT}/focused_detected.png")


def crop_to_track(frames, track, pad=20):
    """Tight-crop frames + track stack to the track's union bbox + padding."""
    s = track["stack"]
    ys, xs = np.where(s.any(axis=0))
    H, W = s.shape[1:]
    y0 = max(0, ys.min() - pad); y1 = min(H, ys.max() + pad)
    x0 = max(0, xs.min() - pad); x1 = min(W, xs.max() + pad)
    return (frames[:, y0:y1, x0:x1].copy(),
            s[:, y0:y1, x0:x1].copy())


def _best_multi_frame(tracks, default):
    """Pick a frame where the maximum number of tracks have a mask."""
    if not tracks:
        return default
    counts = np.zeros(tracks[0]["stack"].shape[0], dtype=int)
    for t in tracks:
        counts += t["stack"].any(axis=(1, 2)).astype(int)
    if counts.max() == 0:
        return default
    return int(counts.argmax())


def fig_focused_multi_detected(frames, labels, tracks, fi=None,
                               title=None):
    if fi is None:
        fi = _best_multi_frame(tracks, len(frames) // 3)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(imnorm(frames[fi]), cmap="gray", vmin=0, vmax=1)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(tracks), 10)))
    n_drawn = 0
    for i, t in enumerate(tracks):
        m = t["stack"][fi]
        if not m.any():
            continue
        ax.contour(m.astype(np.uint8), levels=[0.5],
                   colors=[cmap[i % 10]], linewidths=2)
        n_drawn += 1
    if title is None:
        title = (f"Multi-cell detection: cpsam + DeepSea "
                 f"(frame {fi}, {n_drawn} cells)")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{OUT}/focused_multi_detected.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {OUT}/focused_multi_detected.png")


def fig_trajectories(frames, tracks, n_frames):
    """Trajectories overlay, auto-cropped to the convex hull of all
    centroids + padding so motion is visible."""
    fi = n_frames // 2
    H, W = frames.shape[1:]

    # Compute the bounding box of every centroid across every track,
    # plus padding, and crop the backdrop to that region.
    all_y, all_x = [], []
    for t in tracks:
        cents = t["centroids"]
        ok = ~np.isnan(cents[:, 0])
        all_y.extend(cents[ok, 0].tolist())
        all_x.extend(cents[ok, 1].tolist())
    if not all_y:
        return
    pad = 80
    y0 = max(0, int(min(all_y)) - pad); y1 = min(H, int(max(all_y)) + pad)
    x0 = max(0, int(min(all_x)) - pad); x1 = min(W, int(max(all_x)) + pad)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="black")
    ax.imshow(imnorm(frames[fi, y0:y1, x0:x1]),
              cmap="gray", vmin=0, vmax=1, alpha=0.45,
              extent=[x0, x1, y1, y0])
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(tracks), 10)))
    for i, t in enumerate(tracks):
        cents = t["centroids"]
        ok = ~np.isnan(cents[:, 0])
        if ok.sum() < 2:
            continue
        ys = cents[ok, 0]; xs = cents[ok, 1]
        col = cmap[i % 10]
        ax.plot(xs, ys, "-", color=col, linewidth=3.0, alpha=0.95,
                label=f"Cell {i + 1}")
        ax.plot(xs[0], ys[0], "o", color=col, markersize=10,
                markeredgecolor="white", markeredgewidth=2)
        ax.plot(xs[-1], ys[-1], "s", color=col, markersize=10,
                markeredgecolor="white", markeredgewidth=2)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)  # invert so image-y matches
    ax.set_title(f"Multi-cell tracking ({len(tracks)} tracks, "
                 f"{n_frames} frames)", fontsize=11, color="white")
    ax.set_xticks([]); ax.set_yticks([])
    if tracks:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85,
                  labelcolor="black")
    fig.tight_layout()
    fig.savefig(f"{OUT}/multi_trajectories.png", dpi=130,
                bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  saved {OUT}/multi_trajectories.png")


def fig_speed(track_stacks_with_centroids, um_per_px, dt_min, label_prefix):
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    all_speeds = []
    for i, (stack, centroids) in enumerate(track_stacks_with_centroids):
        speeds = smoothed_speeds(centroids, um_per_px, dt_min)
        ts = np.arange(len(speeds)) * dt_min
        ax.plot(ts, speeds, "-", linewidth=1.6, color=cmap[i % 10],
                label=f"{label_prefix} {i + 1}")
        all_speeds.extend(speeds[~np.isnan(speeds)].tolist())
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Speed (µm/min)")
    ax.set_title("Per-cell migration speed")
    # Autoscale: set y-axis to data range with a 30% headroom, capped at SPEED_CAP
    if all_speeds:
        ymax = min(max(all_speeds) * 1.3, SPEED_CAP)
        ymax = max(ymax, 0.5)  # floor at 0.5 so a stationary plot isn't blank
        ax.set_ylim(0, ymax)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/graph_speed.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {OUT}/graph_speed.png")


def fig_trajectory_xy(track_stacks_with_centroids, um_per_px):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (stack, centroids) in enumerate(track_stacks_with_centroids):
        ok = ~np.isnan(centroids[:, 0])
        if ok.sum() < 2:
            continue
        idxs = np.where(ok)[0]
        cs = centroids[ok] * um_per_px
        sc = ax.scatter(cs[:, 1], cs[:, 0], c=idxs, cmap="viridis",
                        s=22, edgecolor="none", alpha=0.9)
        ax.plot(cs[:, 1], cs[:, 0], "-", color="gray",
                linewidth=0.7, alpha=0.5)
        ax.plot(cs[0, 1], cs[0, 0], "o", color="green", markersize=8)
        ax.plot(cs[-1, 1], cs[-1, 0], "s", color="red", markersize=8)
    plt.colorbar(sc, ax=ax, label="Frame", shrink=0.8)
    ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
    ax.set_title("Cell trajectory")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{OUT}/graph_trajectory.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {OUT}/graph_trajectory.png")


def fig_msd(track_stacks_with_centroids, um_per_px, dt_min):
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (stack, centroids) in enumerate(track_stacks_with_centroids):
        ok = ~np.isnan(centroids[:, 0])
        if ok.sum() < 5:
            continue
        cs = centroids[ok] * um_per_px
        max_lag = min(20, len(cs) // 2)
        msd = np.zeros(max_lag)
        for k in range(1, max_lag + 1):
            disp = cs[k:] - cs[:-k]
            msd[k - 1] = np.mean(np.sum(disp ** 2, axis=1))
        lags = np.arange(1, max_lag + 1) * dt_min
        ax.plot(lags, msd, "-o", markersize=4, linewidth=1.6,
                color=cmap[i % 10], label=f"Cell {i + 1}")
    ax.set_xlabel("Time lag (min)")
    ax.set_ylabel("MSD (µm²)")
    ax.set_title("Mean Squared Displacement")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/graph_msd.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {OUT}/graph_msd.png")


def fig_area(track_stacks_with_centroids, um_per_px, dt_min):
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (stack, _) in enumerate(track_stacks_with_centroids):
        a = stack.astype(bool).sum(axis=(1, 2)) * (um_per_px ** 2)
        a = np.where(a > 0, a, np.nan)
        ts = np.arange(len(a)) * dt_min
        ax.plot(ts, a, "-", linewidth=1.6, color=cmap[i % 10],
                label=f"Cell {i + 1}")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Area (µm²)")
    ax.set_title("Per-cell area over time")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/graph_area.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {OUT}/graph_area.png")


def fig_kymograph_and_shape(stack, um_per_px, dt_min):
    """Kymograph + 4-panel shape descriptors for a single track."""
    if not stack.any():
        print("  [skip] kymograph + shape — empty stack")
        return
    from core.edge_dynamics import edge_velocity_kymograph
    from core.tracking import extract_centroids
    from core.morphology import shape_descriptors

    centroids = extract_centroids(stack)
    sectors, vel = edge_velocity_kymograph(stack, centroids, um_per_px,
                                           dt_min)
    if vel.size and not np.all(np.isnan(vel)):
        fig, ax = plt.subplots(figsize=(7, 4))
        vfin = vel[~np.isnan(vel)]
        vmax = np.percentile(np.abs(vfin), 95) if vfin.size else 1
        im = ax.imshow(vel.T, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       extent=[0, len(stack) * dt_min, -180, 180],
                       origin="lower")
        ax.set_xlabel("Time (min)"); ax.set_ylabel("Angle (°)")
        ax.set_title("Edge velocity kymograph")
        plt.colorbar(im, ax=ax, label="Edge velocity (µm/min)")
        fig.tight_layout()
        fig.savefig(f"{OUT}/graph_kymograph.png", dpi=130,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved {OUT}/graph_kymograph.png")

    keys = ["area_um2", "perimeter_um", "circularity", "aspect_ratio"]
    metrics = {k: [] for k in keys}
    for i in range(len(stack)):
        if not stack[i].any():
            for k in keys: metrics[k].append(np.nan)
            continue
        m = shape_descriptors(stack[i], um_per_px)
        for k in keys: metrics[k].append(m.get(k, np.nan))
    ts = np.arange(len(stack)) * dt_min
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    titles = {"area_um2": "Area (µm²)",
              "perimeter_um": "Perimeter (µm)",
              "circularity": "Circularity",
              "aspect_ratio": "Aspect Ratio"}
    for ax, k in zip(axes.flat, keys):
        ax.plot(ts, metrics[k], "-", linewidth=1.6, color="#3b6cad")
        ax.set_xlabel("Time (min)"); ax.set_ylabel(titles[k])
        ax.grid(alpha=0.3)
    fig.suptitle("Shape descriptors", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/graph_shape_panel.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {OUT}/graph_shape_panel.png")


def fig_hero(frames_single, masks_single, frames_multi, tracks_multi):
    """4-panel composite for README header."""
    fig = plt.figure(figsize=(16, 4.5))

    # Panel 1: single-cell — pick the middle frame of presence
    present = np.where(masks_single.any(axis=(1, 2)))[0]
    fi_s = int(present[len(present) // 2]) if len(present) else 0
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(imnorm(frames_single[fi_s]), cmap="gray", vmin=0, vmax=1)
    if masks_single[fi_s].any():
        ax1.contour(masks_single[fi_s].astype(np.uint8), levels=[0.5],
                    colors=["#ff5555"], linewidths=1.6)
    ax1.set_title("1. Detect — single cell\n(cpsam + DeepSea)",
                  fontsize=10, fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])

    # Panel 2: multi-cell detection — pick frame with most tracks
    fi_m = _best_multi_frame(tracks_multi, len(frames_multi) // 3)
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(imnorm(frames_multi[fi_m]), cmap="gray", vmin=0, vmax=1)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(tracks_multi), 10)))
    n_drawn = 0
    for i, t in enumerate(tracks_multi):
        m = t["stack"][fi_m]
        if not m.any():
            continue
        ax2.contour(m.astype(np.uint8), levels=[0.5],
                    colors=[cmap[i % 10]], linewidths=1.6)
        n_drawn += 1
    ax2.set_title(f"2. Detect — multi-cell\n({n_drawn} cells with "
                  f"per-cell labels)",
                  fontsize=10, fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])

    # Panel 3: trajectories
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(imnorm(frames_multi[len(frames_multi) // 2]),
               cmap="gray", vmin=0, vmax=1, alpha=0.55)
    for i, t in enumerate(tracks_multi):
        cents = t["centroids"]
        ok = ~np.isnan(cents[:, 0])
        if ok.sum() < 2:
            continue
        ys = cents[ok, 0]; xs = cents[ok, 1]
        col = cmap[i % 10]
        ax3.plot(xs, ys, "-", color=col, linewidth=2.6, alpha=0.95)
        ax3.plot(xs[0], ys[0], "o", color=col, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.5)
        ax3.plot(xs[-1], ys[-1], "s", color=col, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.5)
    ax3.set_title("3. Track — Hungarian",
                  fontsize=10, fontweight="bold")
    ax3.set_xticks([]); ax3.set_yticks([])

    # Panel 4: stats
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
    print(f"  saved {OUT}/hero.png")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    multi_path = f"{CACHE}/pos17_wt_60f.npz"

    if not os.path.exists(multi_path):
        print(f"MISSING: {multi_path}; cannot render.")
        return

    frames_m, labels_m, tracks_m = load_multi(multi_path)
    n_m = len(frames_m)
    kept = filter_tracks(tracks_m, n_m, TRACK_KEEP_FRACTION)
    print(f"=== {n_m}-frame window: {len(tracks_m)} tracks → "
          f"{len(kept)} kept (≥{int(TRACK_KEEP_FRACTION*100)}% present) ===")
    for t in kept:
        print(f"    Cell {t['id']}: {t['n_present']} frames present")

    if not kept:
        print("ERROR: no tracks pass length filter; aborting.")
        return

    # Pick the longest track for single-cell-style figures
    longest = max(kept, key=lambda t: t["n_present"])
    print(f"\n=== Longest track ({longest['n_present']} frames) → "
          f"single-cell figures ===")
    frames_s, masks_s = crop_to_track(frames_m, longest, pad=20)
    print(f"  cropped to {frames_s.shape[1:]}")
    fig_focused_detected(frames_s, masks_s, fi=n_m // 3)
    from core.tracking import extract_centroids
    cents_s = extract_centroids(masks_s)
    single_track = [(masks_s, cents_s)]
    fig_speed(single_track, UM_PER_PX, DT_MIN, label_prefix="Cell")
    fig_trajectory_xy(single_track, UM_PER_PX)
    fig_msd(single_track, UM_PER_PX, DT_MIN)
    fig_area(single_track, UM_PER_PX, DT_MIN)
    fig_kymograph_and_shape(masks_s, UM_PER_PX, DT_MIN)

    print(f"\n=== Multi-cell figures (full frame) ===")
    fig_focused_multi_detected(frames_m, labels_m, kept, fi=n_m // 3)
    fig_trajectories(frames_m, kept, n_m)

    print(f"\n=== Hero composite ===")
    fig_hero(frames_s, masks_s, frames_m, kept)

    print(f"\nDone. Figures under {OUT}/")


if __name__ == "__main__":
    main()
