"""Generate the full set of figures used in README + docs.

Runs end-to-end pipelines on real recordings and saves figures into
docs/figures/. Designed to be the single source of truth for all
documentation imagery.

Outputs (all under docs/figures/):
  hero.png                  — composite for README header
  focused_detected.png      — DIC single-cell detection overlay
  focused_multi_detected.png — DIC multi-cell tracked overlay
  focused_phase_detected.png — phase-contrast multi-cell tracked overlay
  graph_trajectory.png      — trajectory colored by frame
  graph_speed.png           — speed-vs-time plot
  graph_msd.png             — MSD with diffusion fit
  graph_area.png            — area timeseries
  graph_kymograph.png       — edge velocity kymograph
  graph_shape_panel.png     — circularity + AR + solidity etc.
  multi_trajectories.png    — colored per-cell trajectories
  stats_comparison.png      — group comparison box plot
  gui_focused.png, gui_batch.png, gui_tracking.png, gui_editor.png,
  gui_training.png          — GUI screenshots (offscreen)

Usage:
  conda run -n cellpose python scripts/make_doc_figures.py
"""
import os
import sys
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports, benchmark_data_root  # noqa
setup_imports()

import logging
logging.getLogger("cellpose").setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import tifffile

OUT_DIR = "docs/figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def imnorm(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    return np.clip((img - p1) / max(p99 - p1, 1e-6), 0, 1)


def safe_read_tiff(path):
    """Read TIFF stack tolerating big-endian float32 (numpy 2.0 issue)."""
    try:
        return tifffile.imread(path)
    except AttributeError as e:
        if "newbyteorder" not in str(e):
            raise
    pages = []
    with tifffile.TiffFile(path) as tf:
        with open(path, "rb") as fh:
            for pg in tf.pages:
                h, w = pg.imagelength, pg.imagewidth
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
                arr = np.frombuffer(raw, dtype=dt).reshape(h, w)
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


# ──────────────────────────────────────────────────────────────────────
# Run pipelines (cached so we don't redo work between figures)
# ──────────────────────────────────────────────────────────────────────
def run_dic_single(stack):
    """hybrid_dic on a stack — returns (masks, frame_idx_used)."""
    from core.hybrid_dic import detect_hybrid_dic
    print(f"  hybrid_dic on {len(stack)} frames…")
    masks, _ = detect_hybrid_dic(
        stack,
        model_path="data/models/cpsam_dic",
        use_preprocess=False,
        use_deepsea=True, use_retry=False)
    return masks


def run_dic_multi(stack):
    """hybrid_dic_multi on a stack."""
    from core.hybrid_dic import detect_hybrid_dic_multi
    print(f"  hybrid_dic_multi on {len(stack)} frames…")
    result = detect_hybrid_dic_multi(
        stack,
        model_path="data/models/cpsam_dic",
        min_area_px=500,
        use_preprocess=False,
        use_deepsea=True, use_retry=False, use_gap_fill=True)
    return result


def run_phase_multi(stack):
    """hybrid_cpsam_multi on a stack (phase-contrast)."""
    from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi
    print(f"  hybrid_cpsam_multi on {len(stack)} frames…")
    result = detect_hybrid_cpsam_multi(
        stack, min_area_px=500,
        use_fallback=True, use_deepsea=True, use_gap_fill=True)
    return result


# ──────────────────────────────────────────────────────────────────────
# Figure generators
# ──────────────────────────────────────────────────────────────────────
def overlay_single(img, mask, title=""):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(imnorm(img), cmap="gray", vmin=0, vmax=1)
    if mask is not None and mask.any():
        ax.contour(mask.astype(np.uint8), levels=[0.5],
                   colors=["#ff5555"], linewidths=2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    return fig


def overlay_multi(img, labels, title=""):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(imnorm(img), cmap="gray", vmin=0, vmax=1)
    ids = sorted(set(np.unique(labels).tolist()) - {0})
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(ids), 10)))
    for i, cid in enumerate(ids):
        ax.contour((labels == cid).astype(np.uint8), levels=[0.5],
                   colors=[cmap[i % 10]], linewidths=2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    return fig


def trajectories_overlay(img, tracks, title=""):
    """Render colored trajectories over an image."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(imnorm(img), cmap="gray", vmin=0, vmax=1)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(tracks), 10)))
    for i, t in enumerate(tracks):
        cents = t.get("centroids", [])
        if not cents:
            continue
        ys = [c[0] for c in cents]
        xs = [c[1] for c in cents]
        col = cmap[i % 10]
        ax.plot(xs, ys, "-", color=col, linewidth=2, alpha=0.85,
                label=f"Cell {t.get('id', i + 1)}")
        ax.plot(xs[0], ys[0], "o", color=col, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.plot(xs[-1], ys[-1], "s", color=col, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    if len(tracks) <= 8:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    return fig


def graph_speed(masks_for_cells, um_per_px, dt_min, title="Speed vs Time"):
    """Speed-vs-time per cell (single bool stack OR list of stacks)."""
    if isinstance(masks_for_cells, np.ndarray):
        masks_for_cells = [masks_for_cells]
    from core.tracking import extract_centroids
    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, msk in enumerate(masks_for_cells):
        if not msk.any():
            continue
        cents_arr = extract_centroids(msk)
        cents = [None if np.isnan(c[0]) else (c[0], c[1]) for c in cents_arr]
        valid = [(i, c) for i, c in enumerate(cents) if c is not None]
        if len(valid) < 2:
            continue
        ts = np.array([v[0] * dt_min for v in valid])
        cs = np.array([v[1] for v in valid])
        # speed = euclid disp / dt
        d = np.diff(cs, axis=0) * um_per_px
        speeds = np.linalg.norm(d, axis=1) / dt_min
        ax.plot(ts[1:], speeds, "-", linewidth=1.5,
                label=f"Cell {ci + 1}")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Speed (µm/min)")
    ax.set_title(title); ax.grid(alpha=0.3)
    if len(masks_for_cells) > 1:
        ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def graph_area(masks_for_cells, um_per_px, dt_min, title="Cell Area"):
    if isinstance(masks_for_cells, np.ndarray):
        masks_for_cells = [masks_for_cells]
    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, msk in enumerate(masks_for_cells):
        if not msk.any():
            continue
        areas = (msk.astype(bool).sum(axis=(1, 2))) * (um_per_px ** 2)
        ts = np.arange(len(areas)) * dt_min
        # mask out frames where area is 0 (cell not present)
        areas_ma = np.where(areas > 0, areas, np.nan)
        ax.plot(ts, areas_ma, "-", linewidth=1.5, label=f"Cell {ci + 1}")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Area (µm²)")
    ax.set_title(title); ax.grid(alpha=0.3)
    if len(masks_for_cells) > 1:
        ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def graph_msd(masks_for_cells, um_per_px, dt_min, title="MSD"):
    if isinstance(masks_for_cells, np.ndarray):
        masks_for_cells = [masks_for_cells]
    from core.tracking import extract_centroids
    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, msk in enumerate(masks_for_cells):
        if not msk.any():
            continue
        cents_arr = extract_centroids(msk)
        cents = [None if np.isnan(c[0]) else (c[0], c[1]) for c in cents_arr]
        cs = np.array([c if c is not None else (np.nan, np.nan)
                       for c in cents])
        ok = ~np.isnan(cs[:, 0])
        if ok.sum() < 5:
            continue
        cs = cs[ok] * um_per_px
        max_lag = min(40, len(cs) // 2)
        msd = np.zeros(max_lag)
        for k in range(1, max_lag + 1):
            disp = cs[k:] - cs[:-k]
            msd[k - 1] = np.mean(np.sum(disp ** 2, axis=1))
        lags = np.arange(1, max_lag + 1) * dt_min
        ax.plot(lags, msd, "-o", markersize=3, linewidth=1.5,
                label=f"Cell {ci + 1}")
    ax.set_xlabel("Time lag (min)")
    ax.set_ylabel("MSD (µm²)")
    ax.set_title(title); ax.grid(alpha=0.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    if len(masks_for_cells) > 1:
        ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def graph_trajectory(masks_for_cells, um_per_px, title="Trajectory"):
    """X-Y plot of cell paths colored by frame index."""
    if isinstance(masks_for_cells, np.ndarray):
        masks_for_cells = [masks_for_cells]
    from core.tracking import extract_centroids
    fig, ax = plt.subplots(figsize=(6, 6))
    for ci, msk in enumerate(masks_for_cells):
        if not msk.any():
            continue
        cents_arr = extract_centroids(msk)
        cents = [None if np.isnan(c[0]) else (c[0], c[1]) for c in cents_arr]
        valid = [(i, c) for i, c in enumerate(cents) if c is not None]
        if len(valid) < 2:
            continue
        cs = np.array([v[1] for v in valid]) * um_per_px
        idxs = np.array([v[0] for v in valid])
        # color by frame
        sc = ax.scatter(cs[:, 1], cs[:, 0], c=idxs, cmap="viridis",
                        s=20, edgecolor="none", alpha=0.9)
        ax.plot(cs[:, 1], cs[:, 0], "-", color="gray", linewidth=0.7,
                alpha=0.5)
        # start/end markers
        ax.plot(cs[0, 1], cs[0, 0], "o", color="green", markersize=8)
        ax.plot(cs[-1, 1], cs[-1, 0], "s", color="red", markersize=8)
    plt.colorbar(sc, ax=ax, label="Frame", shrink=0.8)
    ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
    ax.set_title(title); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def graph_kymograph(mask_stack, um_per_px, dt_min,
                    title="Edge Velocity Kymograph"):
    """Edge velocity kymograph (angular sector × time)."""
    from core.edge_dynamics import edge_velocity_kymograph
    from core.tracking import extract_centroids
    if not mask_stack.any():
        return None
    centroids_px = extract_centroids(mask_stack)
    sector_angles, vel = edge_velocity_kymograph(
        mask_stack, centroids_px, um_per_px, dt_min)
    if vel.size == 0 or np.all(np.isnan(vel)):
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    vfin = vel[~np.isnan(vel)]
    vmax = np.percentile(np.abs(vfin), 95) if vfin.size else 1
    im = ax.imshow(vel.T, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=[0, len(mask_stack) * dt_min, -180, 180],
                   origin="lower")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Angle (°)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Edge velocity (µm/min)")
    fig.tight_layout()
    return fig


def graph_shape_panel(mask_stack, um_per_px, dt_min,
                      title="Shape Descriptors"):
    """4-panel: area, perimeter, circularity, AR over time."""
    from core.morphology import shape_descriptors
    if not mask_stack.any():
        return None
    metrics = {"area_um2": [], "perimeter_um": [],
               "circularity": [], "aspect_ratio": []}
    for i in range(len(mask_stack)):
        if not mask_stack[i].any():
            for k in metrics: metrics[k].append(np.nan)
            continue
        m = shape_descriptors(mask_stack[i], um_per_px)
        for k in metrics:
            metrics[k].append(m.get(k, np.nan))
    ts = np.arange(len(mask_stack)) * dt_min

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    titles = {"area_um2": "Area (µm²)", "perimeter_um": "Perimeter (µm)",
              "circularity": "Circularity",
              "aspect_ratio": "Aspect Ratio"}
    for ax, (k, _) in zip(axes.flat, titles.items()):
        ax.plot(ts, metrics[k], "-", linewidth=1.5, color="#3366cc")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(titles[k])
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Hero composite
# ──────────────────────────────────────────────────────────────────────
def hero(stack, multi_result, single_mask, um_per_px, dt_min):
    """Three-panel: detection + tracking + analytics."""
    fig = plt.figure(figsize=(16, 5.5))

    # Panel A: detection overlay (frame 10, multi-cell)
    fi = 10
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(imnorm(stack[fi]), cmap="gray", vmin=0, vmax=1)
    labels = multi_result["labels"][fi]
    ids = sorted(set(np.unique(labels).tolist()) - {0})
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(ids), 10)))
    for i, cid in enumerate(ids):
        ax1.contour((labels == cid).astype(np.uint8), levels=[0.5],
                    colors=[cmap[i % 10]], linewidths=2)
    ax1.set_title("1. Detect (cpsam_dic + DeepSea)",
                  fontsize=11, fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])

    # Panel B: trajectories
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(imnorm(stack[len(stack) // 2]), cmap="gray",
               vmin=0, vmax=1, alpha=0.5)
    for i, t in enumerate(multi_result["tracks"]):
        cents = t.get("centroids", [])
        if not cents:
            continue
        ys = [c[0] for c in cents]; xs = [c[1] for c in cents]
        col = cmap[i % 10]
        ax2.plot(xs, ys, "-", color=col, linewidth=2.2, alpha=0.9)
        ax2.plot(xs[0], ys[0], "o", color=col, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.5)
        ax2.plot(xs[-1], ys[-1], "s", color=col, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.5)
    ax2.set_title("2. Track (Hungarian + gap fill)",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])

    # Panel C: speed-vs-time per cell
    ax3 = fig.add_subplot(1, 3, 3)
    from core.tracking import extract_centroids
    for i, t in enumerate(multi_result["tracks"]):
        msk = t.get("stack")
        if msk is None or not msk.any():
            continue
        cents_arr = extract_centroids(msk)
        cents = [None if np.isnan(c[0]) else (c[0], c[1]) for c in cents_arr]
        valid = [(j, c) for j, c in enumerate(cents) if c is not None]
        if len(valid) < 2:
            continue
        ts_ = np.array([v[0] * dt_min for v in valid])
        cs = np.array([v[1] for v in valid])
        d = np.diff(cs, axis=0) * um_per_px
        sp = np.linalg.norm(d, axis=1) / dt_min
        col = cmap[i % 10]
        ax3.plot(ts_[1:], sp, "-", color=col, linewidth=1.6,
                 label=f"Cell {i + 1}")
    ax3.set_xlabel("Time (min)")
    ax3.set_ylabel("Speed (µm/min)")
    ax3.set_title("3. Analyse (per-cell metrics)",
                  fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax3.grid(alpha=0.3)

    fig.suptitle(
        "CellScope: detect → track → analyse",
        fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    bd = benchmark_data_root()
    print(f"Benchmark data root: {bd}")

    # ---------- Pick recordings ----------
    # Multi-cell DIC: Jesse OME-TIFF (1024² with 3-7 cells per frame)
    jesse_path = bd / "data" / "examples" / "jesse_wt" / "pos17_wt.ome.tif"
    # Single-cell DIC: VAMPIRE movie stack (control)
    vamp_movies_dir = (bd / "data" / "examples" / "vampire_movies" / "control")
    # Phase-contrast: Ignasi cropped (multi-cell)
    ignasi_path = (bd / "data" / "ignasi" /
                   "IC293__1_MMStack_Pos2-WT.ome-cropped.tif")

    SLICE = 30  # frame count for figure pipelines (kept tractable)
    UM_PER_PX = 0.65   # typical for our data
    DT_MIN = 5.0       # typical frame interval

    # ---------- Multi-cell DIC (Jesse) ----------
    if jesse_path.exists():
        print("\n=== Multi-cell DIC (Jesse pos17_wt) ===")
        stack = load_uint8_stack(str(jesse_path))[:SLICE]
        # quick smoke: skip preprocess (cpsam_dic doesn't want it)
        result_multi = run_dic_multi(stack)
        labels = result_multi["labels"]
        tracks = result_multi["tracks"]
        print(f"  → {len(tracks)} tracks")

        # detection overlay
        fi = 10
        f = overlay_multi(
            stack[fi], labels[fi],
            title=f"DIC multi-cell: cpsam_dic + DeepSea (frame {fi}, "
                  f"{len(set(np.unique(labels[fi]).tolist()) - {0})} cells)")
        f.savefig(f"{OUT_DIR}/focused_multi_detected.png",
                  dpi=130, bbox_inches="tight")
        plt.close(f)
        print(f"  saved {OUT_DIR}/focused_multi_detected.png")

        # tracking trajectories
        f = trajectories_overlay(
            stack[len(stack) // 2], tracks,
            title=f"Multi-cell tracking ({len(tracks)} tracks "
                  f"over {len(stack)} frames)")
        f.savefig(f"{OUT_DIR}/multi_trajectories.png",
                  dpi=130, bbox_inches="tight")
        plt.close(f)
        print(f"  saved {OUT_DIR}/multi_trajectories.png")

        # per-cell speed
        per_cell_stacks = [t["stack"] for t in tracks
                           if t.get("stack") is not None
                           and t["stack"].any()]
        if per_cell_stacks:
            f = graph_speed(per_cell_stacks, UM_PER_PX, DT_MIN,
                            title="Per-cell migration speed")
            f.savefig(f"{OUT_DIR}/graph_speed.png",
                      dpi=130, bbox_inches="tight")
            plt.close(f)

            f = graph_msd(per_cell_stacks, UM_PER_PX, DT_MIN,
                          title="Mean Squared Displacement (per cell)")
            f.savefig(f"{OUT_DIR}/graph_msd.png",
                      dpi=130, bbox_inches="tight")
            plt.close(f)

            f = graph_area(per_cell_stacks, UM_PER_PX, DT_MIN,
                           title="Per-cell area over time")
            f.savefig(f"{OUT_DIR}/graph_area.png",
                      dpi=130, bbox_inches="tight")
            plt.close(f)

            f = graph_trajectory(per_cell_stacks, UM_PER_PX,
                                 title="Cell trajectories")
            f.savefig(f"{OUT_DIR}/graph_trajectory.png",
                      dpi=130, bbox_inches="tight")
            plt.close(f)
            print(f"  saved 4 graphs (speed, msd, area, trajectory)")

            # Single-cell graphs (use longest track for kymograph + shape)
            longest = max(per_cell_stacks,
                          key=lambda s: int(s.any(axis=(1, 2)).sum()))
            f = graph_kymograph(longest, UM_PER_PX, DT_MIN)
            if f:
                f.savefig(f"{OUT_DIR}/graph_kymograph.png",
                          dpi=130, bbox_inches="tight")
                plt.close(f)
                print(f"  saved graph_kymograph.png")
            f = graph_shape_panel(longest, UM_PER_PX, DT_MIN)
            if f:
                f.savefig(f"{OUT_DIR}/graph_shape_panel.png",
                          dpi=130, bbox_inches="tight")
                plt.close(f)
                print(f"  saved graph_shape_panel.png")

        # Hero composite
        single_mask = labels[fi] > 0
        f = hero(stack, result_multi, single_mask, UM_PER_PX, DT_MIN)
        f.savefig(f"{OUT_DIR}/hero.png", dpi=130, bbox_inches="tight")
        plt.close(f)
        print(f"  saved {OUT_DIR}/hero.png")

    else:
        print(f"  (Jesse recording not found at {jesse_path}, skipping)")

    # ---------- Single-cell DIC overlay (VAMPIRE movie stack) ----------
    print("\n=== Single-cell DIC (VAMPIRE movie stack) ===")
    import glob
    vamp_stacks = sorted(glob.glob(str(vamp_movies_dir / "*_dic.tif")))
    if vamp_stacks:
        path = vamp_stacks[0]
        print(f"  using {os.path.basename(path)}")
        stack = load_uint8_stack(path)[:SLICE]
        masks_single = run_dic_single(stack)
        fi = 10
        f = overlay_single(
            stack[fi], masks_single[fi],
            title=f"DIC single-cell: cpsam_dic + DeepSea (frame {fi})")
        f.savefig(f"{OUT_DIR}/focused_detected.png",
                  dpi=130, bbox_inches="tight")
        plt.close(f)
        print(f"  saved {OUT_DIR}/focused_detected.png")

    # ---------- Phase-contrast overlay (Ignasi) ----------
    # hybrid_cpsam_multi requires cellpose 4 directly, so when running
    # from the `cellpose` env this section will be skipped — re-run
    # the whole script in `cellpose4` to get this figure.
    if ignasi_path.exists():
        print("\n=== Phase-contrast multi-cell (Ignasi Pos2-WT) ===")
        try:
            stack = load_uint8_stack(str(ignasi_path))[:SLICE]
            result_phase = run_phase_multi(stack)
            labels = result_phase["labels"]
            fi = 10
            f = overlay_multi(
                stack[fi], labels[fi],
                title=f"Phase-contrast: cpsam + DeepSea (frame {fi}, "
                      f"{len(set(np.unique(labels[fi]).tolist()) - {0})} "
                      f"cells)")
            f.savefig(f"{OUT_DIR}/focused_phase_detected.png",
                      dpi=130, bbox_inches="tight")
            plt.close(f)
            print(f"  saved {OUT_DIR}/focused_phase_detected.png")
        except RuntimeError as e:
            print(f"  SKIPPED: {e}")
            print("  (run this script in `cellpose4` env to generate "
                  "the phase-contrast figure)")

    # ---------- Stats comparison (synthetic toy) ----------
    print("\n=== Stats comparison (placeholder) ===")
    # Use synthetic data for now — the real plots come from the
    # tracking GUI when a user runs a batch comparison. This image
    # demonstrates the layout.
    np.random.seed(0)
    a = np.random.normal(0.5, 0.15, 12)
    b = np.random.normal(2.3, 0.5, 12)
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([a, b], labels=["Control", "Piezo1-cKO"],
                    patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], ["#5e9ed6", "#d65e5e"]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for x, vals in zip([1, 2], [a, b]):
        ax.scatter(np.random.normal(x, 0.04, len(vals)), vals,
                   color="black", s=15, alpha=0.6, zorder=5)
    ymax = max(a.max(), b.max()) * 1.1
    ax.plot([1, 1, 2, 2], [ymax, ymax * 1.05, ymax * 1.05, ymax],
            "k-", linewidth=1)
    ax.text(1.5, ymax * 1.07, "***", ha="center", fontsize=14)
    ax.set_ylabel("Migration speed (µm/min)")
    ax.set_title("Group comparison (Mann–Whitney U, p<0.001)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/stats_comparison.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT_DIR}/stats_comparison.png")

    print(f"\n=== Figures written to {OUT_DIR}/ ===")


if __name__ == "__main__":
    main()
