"""Render publication-quality figures for the README from recent
successful pipeline outputs.

Reads masks.npz + multichannel/single-channel recordings, produces:
  - focused_detected.png       — single-cell phase-contrast detection
  - focused_multi_detected.png — multi-cell phase-contrast (3 cells)
  - focused_phase_detected.png — single-cell DIC + per-cell DeepSea
  - multichannel_detected.png  — DIC + Cy5 fusion overlay
  - multi_trajectories.png     — per-cell paths over a 60-frame window
  - hero.png                   — 4-panel composite

GUI screenshots + analysis graph plots are copied separately from
results/comprehensive_gui_tests/screenshots/.

Usage:
  conda run -n cellpose4 python scripts/render_readme_figures.py
"""
import os
import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "docs/figures"
DPI = 150
os.makedirs(OUT_DIR, exist_ok=True)

# Colours for per-cell overlays (perceptually distinct)
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
    "#9a6324", "#e6beff",
]


def _pct_normalize(img, p_lo=1, p_hi=99):
    """Stretch intensity to [0,1] by percentile clip."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def _read_page_raw(page):
    """Read a TIFF page's pixel data without going through
    page.asarray() — bypasses the tifffile newbyteorder() bug under
    numpy 2.0. Handles uint8/uint16/float32, little- and big-endian."""
    # Get the raw byte offsets/lengths
    fh = page.parent.filehandle
    dtype = page.dtype
    h, w = page.shape[-2:]

    # Read all strips and concatenate
    offsets = page.dataoffsets
    bytecounts = page.databytecounts
    if not offsets:
        return page.asarray()  # fallback if no strips
    buf = bytearray()
    for off, n in zip(offsets, bytecounts):
        fh.seek(off)
        buf.extend(fh.read(n))
    # Force little-endian dtype regardless of what's in the file —
    # we'll byteswap if needed below.
    base_dtype = np.dtype(dtype.kind + str(dtype.itemsize))
    arr = np.frombuffer(bytes(buf), dtype=base_dtype)
    arr = arr.reshape(h, w)
    # Byteswap if the file is big-endian
    if dtype.byteorder == ">" or (
            dtype.byteorder == "=" and dtype.str.startswith(">")):
        arr = arr.byteswap()
    return arr.copy()


def _load_recording_frame(tif_path, frame_idx, channel=None):
    """Read a single frame (and channel if multi-channel) without
    loading the whole stack into memory."""
    with tifffile.TiffFile(tif_path) as t:
        series = t.series[0]
        shape = series.shape
        if len(shape) == 4:    # (T, C, H, W)
            n_ch = shape[1]
            page_idx = frame_idx * n_ch + (channel or 0)
        elif len(shape) == 3:  # (T, H, W) single-channel
            page_idx = frame_idx
        else:
            page_idx = 0
        page = t.pages[page_idx]
        try:
            return page.asarray()
        except AttributeError:
            # tifffile + numpy-2.0 newbyteorder bug — use raw read
            return _read_page_raw(page)


def _to_uint8(img):
    """Normalise to uint8 for matplotlib imshow as grayscale."""
    n = _pct_normalize(img)
    return (n * 255).astype(np.uint8)


def _contours(label_img, cell_id):
    """Polygon contours for one cell ID."""
    mask = label_img == cell_id
    return measure.find_contours(mask.astype(float), 0.5)


def _cell_bbox(label_stack, pad=80):
    """Bounding box that contains ALL non-zero labels across the
    entire stack, plus padding."""
    union = label_stack.any(axis=0)
    if not union.any():
        return None
    ys, xs = np.where(union)
    H, W = union.shape
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + pad)
    return y0, y1, x0, x1


def _cell_bbox_frame(label, pad=80):
    """Bounding box for a single frame's cells."""
    if not label.any():
        return None
    ys, xs = np.where(label > 0)
    H, W = label.shape
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + pad)
    return y0, y1, x0, x1


def _pick_good_frame(labels, prefer_n_cells=None):
    """Find a frame with the most cells (or the closest to a target)."""
    counts = np.array([np.unique(f).max() if f.any() else 0 for f in labels])
    if prefer_n_cells is None:
        return int(counts.argmax())
    diff = np.abs(counts - prefer_n_cells)
    return int(diff.argmin())


def _cached_detect(tif_path, mask_cache, mode="hybrid_cpsam_multi"):
    """Run detection if cache doesn't exist; otherwise load cached masks."""
    if os.path.exists(mask_cache):
        d = np.load(mask_cache)
        return d["labels"] if "labels" in d.files else d["masks"]
    print(f"  running detection on {os.path.basename(tif_path)}…")
    from core.io import load_recording
    from core.pipeline import detect
    rec = load_recording(tif_path)
    det = detect(rec["frames"], mode=mode)
    labels = det.get("labels", det["masks"].astype(np.int32))
    if labels.dtype == bool:
        labels = labels.astype(np.int32)
    os.makedirs(os.path.dirname(mask_cache), exist_ok=True)
    np.savez_compressed(mask_cache, labels=labels)
    return labels


def render_single_cell_phase():
    """Single-cell phase-contrast detection on the curated demo
    single_cell_phase_WT (uint8, 50 frames, 438×759). Picks a frame
    where the cell shows good spread morphology."""
    print("Rendering single-cell phase…")
    tif = "data/examples/single_cell_phase_WT/single_cell_phase_WT.tif"
    cache = "results/figure_cache/single_cell_phase_WT_masks.npz"
    masks = _cached_detect(tif, cache, mode="hybrid_cpsam")
    frame_idx = 25 if len(masks) > 25 else len(masks) // 2
    img = _load_recording_frame(tif, frame_idx)
    label = masks[frame_idx]
    if label.dtype == bool:
        label = label.astype(np.int32)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    ax.imshow(img, cmap="gray", aspect="equal",
              vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    for cnts in _contours(label, 1):
        ax.plot(cnts[:, 1], cnts[:, 0], color="#e6194b", lw=2.2)
    ax.set_title(
        f"Single-cell phase-contrast detection\n"
        f"cpsam + DeepSea (frame {frame_idx})", fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "focused_detected.png"),
                bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def render_multi_cell_phase():
    """Multi-cell detection using the IC295 Pos20-KO multichannel GT
    pipeline output. Shows 11 tracked cells with per-cell colours on
    the DIC channel, cropped to the cell region for clarity."""
    print("Rendering multi-cell phase…")
    tif = ("data/ic295_gt_full/Pos20_KO/"
           "IC295__1_MMStack_Pos20-KO.ome.tif")
    masks = np.load(
        "data/ic295_gt_full/Pos20_KO/pipeline_results/masks.npz"
        )["labels"]
    frame_idx = _pick_good_frame(masks)
    img = _load_recording_frame(tif, frame_idx, channel=1)
    label = masks[frame_idx]
    n_cells = int(label.max())

    bbox = _cell_bbox(masks, pad=120)   # crop tightly to all cells
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        img = img[y0:y1, x0:x1]
        label = label[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=DPI)
    ax.imshow(img, cmap="gray", aspect="equal",
              vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    for cid in range(1, n_cells + 1):
        col = PALETTE[(cid - 1) % len(PALETTE)]
        for cnts in _contours(label, cid):
            ax.plot(cnts[:, 1], cnts[:, 0], color=col, lw=1.8)
    ax.set_title(
        f"Multi-cell detection — Hungarian tracker\n"
        f"cpsam + DeepSea + per-cell ID (frame {frame_idx}, "
        f"{n_cells} cells)", fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "focused_multi_detected.png"),
                bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def render_multichannel():
    """Multichannel DIC + Cy5 fusion overlay on IC295 Pos20-KO,
    cropped to cell region for clarity. Uses the fusion_source_stack
    to colour cells by source:
      red    = DIC only
      green  = both channels
      yellow = Cy5 only"""
    print("Rendering multichannel detection…")
    tif = ("data/ic295_gt_full/Pos20_KO/"
           "IC295__1_MMStack_Pos20-KO.ome.tif")
    npz = np.load(
        "data/ic295_gt_full/Pos20_KO/pipeline_results/masks.npz")
    labels = npz["labels"]
    fusion_src = npz["fusion_source_stack"]
    frame_idx = _pick_good_frame(labels)
    dic = _load_recording_frame(tif, frame_idx, channel=1)
    label = labels[frame_idx]
    src = fusion_src[frame_idx]
    n_cells = int(label.max())

    bbox = _cell_bbox(labels, pad=120)
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        dic = dic[y0:y1, x0:x1]
        label = label[y0:y1, x0:x1]
        src = src[y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), dpi=DPI)
    # Panel 1: DIC + per-cell colored contours
    axes[0].imshow(dic, cmap="gray", aspect="equal",
                    vmin=np.percentile(dic, 1),
                    vmax=np.percentile(dic, 99))
    for cid in range(1, n_cells + 1):
        col = PALETTE[(cid - 1) % len(PALETTE)]
        for cnts in _contours(label, cid):
            axes[0].plot(cnts[:, 1], cnts[:, 0], color=col, lw=1.8)
    axes[0].set_title(
        f"Multichannel detection — DIC + Cy5 fusion\n"
        f"IC295 Pos20-KO frame {frame_idx} ({n_cells} cells tracked)",
        fontsize=11)
    axes[0].set_axis_off()

    # Panel 2: source breakdown
    axes[1].imshow(dic, cmap="gray", aspect="equal", alpha=0.7,
                    vmin=np.percentile(dic, 1),
                    vmax=np.percentile(dic, 99))
    src_color = np.zeros((*src.shape, 4), dtype=np.float32)
    src_color[src == 1] = [1.0, 0.2, 0.2, 0.55]   # DIC only
    src_color[src == 2] = [0.2, 0.85, 0.2, 0.55]  # both
    src_color[src == 3] = [1.0, 1.0, 0.2, 0.55]   # Cy5 only
    axes[1].imshow(src_color, aspect="equal")
    axes[1].set_title(
        "Detection source breakdown\n"
        "red = DIC only · green = both · yellow = Cy5 rescue",
        fontsize=11)
    axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "multichannel_detected.png"),
                bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def render_dic_keratinocyte():
    """DIC keratinocyte detection — Pos30-GOF multichannel pipeline,
    cropped to cell region."""
    print("Rendering DIC keratinocyte…")
    tif = ("data/ic295_gt_full/Pos30_GOF/"
           "IC295__1_MMStack_Pos30-GOF.ome.tif")
    masks = np.load(
        "data/ic295_gt_full/Pos30_GOF/pipeline_results/masks.npz"
        )["labels"]
    frame_idx = _pick_good_frame(masks)
    img = _load_recording_frame(tif, frame_idx, channel=1)
    label = masks[frame_idx]
    n_cells = int(label.max())

    bbox = _cell_bbox(masks, pad=120)
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        img = img[y0:y1, x0:x1]
        label = label[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=DPI)
    ax.imshow(img, cmap="gray", aspect="equal",
              vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    for cid in range(1, n_cells + 1):
        col = PALETTE[(cid - 1) % len(PALETTE)]
        for cnts in _contours(label, cid):
            ax.plot(cnts[:, 1], cnts[:, 0], color=col, lw=1.8)
    ax.set_title(
        f"DIC + multichannel pipeline\n"
        f"cpsam_dic + per-cell DeepSea + Cy5 filter "
        f"(frame {frame_idx}, {n_cells} cells)", fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "focused_phase_detected.png"),
                bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def render_multi_trajectories():
    """Multi-cell trajectories on Pos20_KO multichannel (11 cells over
    97 frames), cropped to cell region. Background = DIC channel
    mid-frame for context."""
    print("Rendering multi-cell trajectories…")
    masks = np.load(
        "data/ic295_gt_full/Pos20_KO/pipeline_results/masks.npz"
        )["labels"]
    tif = ("data/ic295_gt_full/Pos20_KO/"
           "IC295__1_MMStack_Pos20-KO.ome.tif")
    n_frames = len(masks)
    n_cells = int(masks.max())

    centroids = {cid: [] for cid in range(1, n_cells + 1)}
    for f in range(n_frames):
        for cid in range(1, n_cells + 1):
            mask = masks[f] == cid
            if not mask.any():
                continue
            yx = np.argwhere(mask).mean(axis=0)
            centroids[cid].append((yx[1], yx[0]))   # (x, y)

    bg = _load_recording_frame(tif, n_frames // 2, channel=1)
    bbox = _cell_bbox(masks, pad=120)
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        bg = bg[y0:y1, x0:x1]
        # Translate centroids into crop coordinates
        for cid in centroids:
            centroids[cid] = [(x - x0, y - y0) for x, y in centroids[cid]]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=DPI)
    ax.imshow(bg, cmap="gray", aspect="equal", alpha=0.45,
              vmin=np.percentile(bg, 1), vmax=np.percentile(bg, 99))
    for cid, points in centroids.items():
        if len(points) < 2:
            continue
        pts = np.array(points)
        col = PALETTE[(cid - 1) % len(PALETTE)]
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2.0, alpha=0.9)
        ax.plot(pts[0, 0], pts[0, 1], "o", color=col, ms=8,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.plot(pts[-1, 0], pts[-1, 1], "s", color=col, ms=8,
                markeredgecolor="white", markeredgewidth=1.4)
    ax.set_title(
        f"Multi-cell trajectories — {n_cells} cells over "
        f"{n_frames} frames\nHungarian tracker on IC295 multichannel · "
        "circle = start · square = end", fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "multi_trajectories.png"),
                bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def render_hero():
    """3-panel composite: multichannel detect → multi-cell trajectories
    → per-cell shape metrics. No group comparisons — those are for
    publication, not the README."""
    print("Rendering hero composite…")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)

    # === Panel 1: single-cell detection on the clean uint8 demo ===
    tif = "data/examples/single_cell_phase_WT/single_cell_phase_WT.tif"
    cache = "results/figure_cache/single_cell_phase_WT_masks.npz"
    sc_masks = _cached_detect(tif, cache, mode="hybrid_cpsam")
    frame_idx = 25 if len(sc_masks) > 25 else len(sc_masks) // 2
    sc_img = _load_recording_frame(tif, frame_idx)
    sc_label = sc_masks[frame_idx]
    if sc_label.dtype == bool:
        sc_label = sc_label.astype(np.int32)
    axes[0].imshow(sc_img, cmap="gray", aspect="equal",
                    vmin=np.percentile(sc_img, 1),
                    vmax=np.percentile(sc_img, 99))
    for cnts in _contours(sc_label, 1):
        axes[0].plot(cnts[:, 1], cnts[:, 0], color="#e6194b", lw=2.2)
    axes[0].set_title(
        "1. Detect — cpsam + DeepSea\n"
        "phase-contrast keratinocyte",
        fontsize=11, fontweight="bold")
    axes[0].set_axis_off()

    # === Panel 2: multi-cell trajectories ===
    masks_p3 = np.load(
        "data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/"
        "pipeline_results/masks.npz")["labels"]
    n_frames = min(60, len(masks_p3))
    n_cells3 = int(masks_p3[:n_frames].max())
    centroids = {cid: [] for cid in range(1, n_cells3 + 1)}
    for f in range(n_frames):
        for cid in range(1, n_cells3 + 1):
            m = masks_p3[f] == cid
            if not m.any():
                continue
            yx = np.argwhere(m).mean(axis=0)
            centroids[cid].append((f, yx[1], yx[0]))
    # Use multi_cell_DIC_WT example (uint8, clean) for background
    bg_tif = "data/examples/multi_cell_DIC_WT/multi_cell_DIC_WT.tif"
    bg = _load_recording_frame(bg_tif, 25)
    # Pos3 frames are 1028×828 — different from the bg. Use a neutral
    # gray background instead so trajectories are clearly visible.
    H = max(int(np.array(c)[:, 2].max()) for c in centroids.values()
            if len(c) > 0) + 50
    W = max(int(np.array(c)[:, 1].max()) for c in centroids.values()
            if len(c) > 0) + 50
    axes[1].set_facecolor("#f4f4f4")
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)   # invert so up is up in image coords
    axes[1].set_aspect("equal")
    for cid, pts in centroids.items():
        if len(pts) < 2:
            continue
        pts = np.array(pts)
        col = PALETTE[(cid - 1) % len(PALETTE)]
        axes[1].plot(pts[:, 1], pts[:, 2], color=col, lw=1.8)
        axes[1].plot(pts[0, 1], pts[0, 2], "o", color=col, ms=8,
                     markeredgecolor="white", markeredgewidth=1.4,
                     label=f"Cell {cid} start")
        axes[1].plot(pts[-1, 1], pts[-1, 2], "s", color=col, ms=8,
                     markeredgecolor="white", markeredgewidth=1.4)
    axes[1].set_title(
        f"2. Track — Hungarian\n{n_cells3} cells · {n_frames} frames "
        f"· circle=start, square=end",
        fontsize=11, fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # === Panel 3: per-cell shape timeseries (area + circularity for
    # cell 1) ===
    from skimage import measure as _mm
    first_cell = sorted(centroids.keys())[0]
    areas = []
    circs = []
    for f in range(n_frames):
        m = masks_p3[f] == first_cell
        if not m.any():
            areas.append(np.nan)
            circs.append(np.nan)
            continue
        props = _mm.regionprops(m.astype(np.int32))
        if not props:
            areas.append(np.nan)
            circs.append(np.nan)
            continue
        a = props[0].area
        p = props[0].perimeter or 1
        areas.append(a)
        circs.append(4 * np.pi * a / (p * p))
    t = np.arange(n_frames) * 10.0   # min
    # Two-axis plot: area on left (µm²), circularity on right
    ax_a = axes[2]
    px_to_um2 = 0.65 ** 2
    a_arr = np.array(areas) * px_to_um2
    ax_a.plot(t, a_arr, color="#e6194b", lw=2.0, label="Area (µm²)")
    ax_a.set_xlabel("Time (min)", fontsize=10)
    ax_a.set_ylabel("Area (µm²)", color="#e6194b", fontsize=10)
    ax_a.tick_params(axis="y", labelcolor="#e6194b")
    ax_a.grid(alpha=0.3)

    ax_c = ax_a.twinx()
    ax_c.plot(t, circs, color="#4363d8", lw=2.0, label="Circularity")
    ax_c.set_ylabel("Circularity", color="#4363d8", fontsize=10)
    ax_c.tick_params(axis="y", labelcolor="#4363d8")
    ax_c.set_ylim(0, 1)
    ax_a.set_title(
        f"3. Analyse — per-cell shape\n"
        f"Cell {first_cell}: area + circularity over time",
        fontsize=11, fontweight="bold")

    fig.suptitle(
        "CellScope: detect → track → analyse",
        fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "hero.png"),
                bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def render_analysis_graphs():
    """Render standalone analysis plots (no GUI chrome) from the
    single_cell_phase_WT detection cache. Uses the same GRAPH_REGISTRY
    plot functions the GUI uses."""
    print("Rendering analysis graphs…")
    from core.io import load_recording
    from core.pipeline import analyze_recording
    from gui_focused.analysis_plots import GRAPH_REGISTRY

    tif = "data/examples/single_cell_phase_WT/single_cell_phase_WT.tif"
    cache = "results/figure_cache/single_cell_phase_WT_masks.npz"
    masks = _cached_detect(tif, cache, mode="hybrid_cpsam")
    rec = load_recording(tif)
    result = analyze_recording(rec, masks > 0)

    plots = {
        "graph_trajectory.png":     "Trajectory",
        "graph_speed.png":          "Speed vs Time",
        "graph_msd.png":            "MSD with Diffusion Fit",
        "graph_area.png":           "Area vs Time",
        "graph_kymograph.png":      "Edge Kymograph",
        "graph_shape_panel.png":    "Shape Panel (6 metrics)",
    }
    for out_name, graph_name in plots.items():
        fn, requires_multi = GRAPH_REGISTRY[graph_name]
        if requires_multi:
            continue
        fig = plt.figure(figsize=(7, 5), dpi=DPI)
        try:
            fn(fig, result, gap_interp_max=0)
        except Exception as e:
            print(f"  skipped {graph_name}: {e}")
            plt.close(fig)
            continue
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, out_name),
                    bbox_inches="tight", dpi=DPI)
        plt.close(fig)


def main():
    render_single_cell_phase()
    render_multi_cell_phase()
    render_dic_keratinocyte()
    render_multichannel()
    render_multi_trajectories()
    render_analysis_graphs()
    render_hero()
    print("\nDone. Figures in", OUT_DIR)


if __name__ == "__main__":
    main()
