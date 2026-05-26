"""Render GT vs pipeline overlays for every evaluable GT recording.

For each folder with `gt_masks/` + `pipeline_results/masks.npz`, writes
to `<folder>/evaluation/overlays/`:

  contact_sheet.png       — 12-tile grid of sample annotated frames,
                            DIC + GT (lime) + pipeline (magenta)
  overlay.tif             — multi-page TIF (RGB), one page per
                            ANNOTATED frame, openable in Fiji/ImageJ
  per_frame/F<idx>.png    — individual full-resolution overlays per
                            annotated frame (handy for slides)

With --all-frames, also produces (slower; 10-100× more output):

  all_frames/F<idx>.png   — pipeline contour on every frame;
                            GT overlay added where annotation exists
  overlay_all.tif         — same data, one TIF page per frame

Usage:
  python scripts/render_gt_overlays.py                       # ann only
  python scripts/render_gt_overlays.py --all-frames          # all
  python scripts/render_gt_overlays.py data/ic295_gt_full/Pos7_WT --all-frames
"""
import os
import sys
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import io as skio, measure

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

GT_ROOTS = ["data/ic295_gt_full", "data/legacy_gt"]
N_TILES = 12  # contact sheet target count


def find_evaluable_folders():
    out = []
    for root in GT_ROOTS:
        if not os.path.isdir(root):
            continue
        for sub in sorted(os.listdir(root)):
            f = os.path.join(root, sub)
            if not os.path.isdir(f):
                continue
            gt = os.path.join(f, "gt_masks")
            pp = os.path.join(f, "pipeline_results", "masks.npz")
            if (os.path.isdir(gt)
                    and any(x.startswith("mask_F")
                            and x.endswith(".png")
                            for x in os.listdir(gt))
                    and os.path.exists(pp)):
                out.append(f)
    return out


def annotated_frames(gt_dir):
    """Sorted list of (frame_idx, gt_mask) tuples."""
    out = []
    for f in sorted(os.listdir(gt_dir)):
        if not (f.startswith("mask_F") and f.endswith(".png")):
            continue
        try:
            fi = int(f[len("mask_F"):-len(".png")])
        except ValueError:
            continue
        m = skio.imread(os.path.join(gt_dir, f))
        out.append((fi, m.astype(np.int32)))
    return sorted(out, key=lambda x: x[0])


def find_source_recording(folder):
    """Return the .ome.tif path next to the GT folder."""
    for f in os.listdir(folder):
        if f.lower().endswith((".ome.tif", ".tif", ".tiff")):
            return os.path.join(folder, f)
    return None


_DIC_STACK_CACHE = {}


def get_dic_stack(tif_path, n_channels):
    """Cached load of the full DIC channel as a (N, H, W) uint8 stack.

    Uses core.io.load_recording / load_video which already handles the
    tifffile / numpy-2.0 big-endian-float32 incompatibility that
    affects the legacy IC293 cropped recordings.
    """
    if tif_path in _DIC_STACK_CACHE:
        return _DIC_STACK_CACHE[tif_path]
    from core.io import load_recording, load_video
    if n_channels >= 2:
        rec = load_recording(tif_path, dic_channel=1, fluo_channel=0)
        stack = rec["frames"]
    else:
        stack = load_video(tif_path)
    _DIC_STACK_CACHE[tif_path] = stack
    return stack


def load_dic_frame(tif_path, frame_idx, n_channels):
    """Return one DIC frame as uint8."""
    stack = get_dic_stack(tif_path, n_channels)
    idx = min(frame_idx, len(stack) - 1)
    return stack[idx]


def draw_contours(ax, labels, color, lw=1.2):
    n = 0
    for cid in range(1, int(labels.max()) + 1):
        m = labels == cid
        if not m.any():
            continue
        for c in measure.find_contours(m.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=color, lw=lw)
        n += 1
    return n


def draw_overlay_rgb(dic, gt, pred, gt_color=(80, 255, 80),
                      pred_color=(255, 80, 255), thickness=2):
    """Render an RGB image with GT contours in lime + pred contours
    in magenta. Used for the TIF stack pages + per-frame PNGs."""
    import cv2
    if dic.ndim == 2:
        rgb = np.stack([dic, dic, dic], axis=-1).astype(np.uint8)
    else:
        rgb = dic.copy()
    for cid in range(1, int(gt.max()) + 1):
        m = (gt == cid).astype(np.uint8)
        if not m.any():
            continue
        contours, _ = cv2.findContours(
            m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, gt_color, thickness)
    for cid in range(1, int(pred.max()) + 1):
        m = (pred == cid).astype(np.uint8)
        if not m.any():
            continue
        contours, _ = cv2.findContours(
            m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, pred_color, thickness)
    return rgb


def render_all_frames(folder, n_channels):
    """Produce per-frame PNGs + TIF stack for EVERY frame in the
    recording. Pipeline contours always shown; GT contours overlaid
    where annotation exists. Called after `render_recording` so the
    annotated-frame artefacts are already produced.
    """
    gt_lookup = {fi: gt for fi, gt in annotated_frames(
        os.path.join(folder, "gt_masks"))}
    pipeline = np.load(os.path.join(folder, "pipeline_results",
                                     "masks.npz"))["labels"].astype(
                                         np.int32)
    n_frames = len(pipeline)
    tif = find_source_recording(folder)
    if tif is None:
        return None

    out_dir = os.path.join(folder, "evaluation", "overlays")
    all_dir = os.path.join(out_dir, "all_frames")
    os.makedirs(all_dir, exist_ok=True)

    dic_stack = get_dic_stack(tif, n_channels)
    print(f"    Rendering {n_frames} frames "
          f"(GT overlay on {len(gt_lookup)} of them)")
    tif_pages = []
    h, w = pipeline.shape[1:]
    for fi in range(n_frames):
        dic = dic_stack[min(fi, len(dic_stack) - 1)]
        if dic.shape != (h, w):
            from scipy.ndimage import zoom
            sy = h / dic.shape[0]; sx = w / dic.shape[1]
            dic = zoom(dic, (sy, sx), order=1).astype(np.uint8)
        gt = gt_lookup.get(fi)
        if gt is not None and gt.shape != (h, w):
            from scipy.ndimage import zoom
            sy = h / gt.shape[0]; sx = w / gt.shape[1]
            gt = zoom(gt, (sy, sx), order=0).astype(np.int32)
        rgb = draw_overlay_rgb(
            dic, gt if gt is not None else np.zeros_like(pipeline[fi]),
            pipeline[fi],
            thickness=max(1, h // 600))
        skio.imsave(os.path.join(all_dir, f"F{fi:03d}.png"),
                    rgb, check_contrast=False)
        tif_pages.append(rgb)
        if fi % 20 == 0:
            print(f"      F{fi}/{n_frames - 1}")

    if tif_pages:
        max_h = max(p.shape[0] for p in tif_pages)
        max_w = max(p.shape[1] for p in tif_pages)
        padded = []
        for p in tif_pages:
            if p.shape[:2] == (max_h, max_w):
                padded.append(p)
            else:
                tmp = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                tmp[:p.shape[0], :p.shape[1]] = p
                padded.append(tmp)
        stack = np.stack(padded, axis=0)
        tifffile.imwrite(
            os.path.join(out_dir, "overlay_all.tif"),
            stack, photometric="rgb")
        print(f"    Wrote overlay_all.tif ({stack.nbytes // (1024*1024)}"
              f" MB)")


def render_recording(folder, contact_sheet_dpi=85):
    """Produce all overlay artifacts for one recording."""
    gt_list = annotated_frames(os.path.join(folder, "gt_masks"))
    if not gt_list:
        return None
    pipeline = np.load(os.path.join(folder, "pipeline_results",
                                     "masks.npz"))["labels"].astype(
                                         np.int32)
    n_pipe = len(pipeline)
    tif = find_source_recording(folder)
    if tif is None:
        print(f"  no recording in {folder}, skipping")
        return None

    # Figure out channel layout (multichannel TIFFs have 2 pages/frame)
    from core.io import detect_channels
    n_ch = detect_channels(tif) if tif.lower().endswith(
        (".tif", ".tiff")) else 1

    out_dir = os.path.join(folder, "evaluation", "overlays")
    os.makedirs(out_dir, exist_ok=True)
    per_frame_dir = os.path.join(out_dir, "per_frame")
    os.makedirs(per_frame_dir, exist_ok=True)

    name = os.path.basename(folder)
    n_gt = len(gt_list)

    # === Contact sheet ===
    indices_for_grid = np.linspace(
        0, n_gt - 1, min(N_TILES, n_gt), dtype=int)
    n_show = len(indices_for_grid)
    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]

    # === Per-frame PNGs + TIF pages ===
    tif_pages = []   # for the TIF stack
    for i, (fi, gt) in enumerate(gt_list):
        if fi >= n_pipe:
            continue
        pred = pipeline[fi]
        # Handle scale mismatch (pipeline may be at half-res, GT full)
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom
            sy = gt.shape[0] / pred.shape[0]
            sx = gt.shape[1] / pred.shape[1]
            pred = zoom(pred, (sy, sx), order=0).astype(np.int32)

        dic = load_dic_frame(tif, fi, n_ch)
        if dic.shape != gt.shape:
            # Tightly align to gt — usually DIC is the same size
            from scipy.ndimage import zoom as _z
            sy = gt.shape[0] / dic.shape[0]
            sx = gt.shape[1] / dic.shape[1]
            dic = _z(dic, (sy, sx), order=1).astype(np.uint8)

        # Full-resolution PNG + TIF page
        rgb = draw_overlay_rgb(dic, gt, pred,
                                thickness=max(1, gt.shape[0] // 600))
        skio.imsave(os.path.join(per_frame_dir, f"F{fi:03d}.png"),
                    rgb, check_contrast=False)
        tif_pages.append(rgb)

    # === TIF stack (Fiji-readable) ===
    if tif_pages:
        # Pad to common shape (in case dimensions varied)
        max_h = max(p.shape[0] for p in tif_pages)
        max_w = max(p.shape[1] for p in tif_pages)
        padded = []
        for p in tif_pages:
            if p.shape[:2] == (max_h, max_w):
                padded.append(p)
            else:
                tmp = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                tmp[:p.shape[0], :p.shape[1]] = p
                padded.append(tmp)
        stack = np.stack(padded, axis=0)
        tifffile.imwrite(
            os.path.join(out_dir, "overlay.tif"),
            stack, photometric="rgb")

    # === Contact sheet ===
    for cell_idx, (panel_idx, _) in enumerate(zip(indices_for_grid,
                                                   range(n_show))):
        gi = int(indices_for_grid[cell_idx])
        fi, gt = gt_list[gi]
        ax = axes[cell_idx // ncols, cell_idx % ncols]
        if fi >= n_pipe:
            ax.axis("off")
            continue
        pred = pipeline[fi]
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom
            sy = gt.shape[0] / pred.shape[0]
            sx = gt.shape[1] / pred.shape[1]
            pred = zoom(pred, (sy, sx), order=0).astype(np.int32)
        dic = load_dic_frame(tif, fi, n_ch)
        if dic.shape != gt.shape:
            from scipy.ndimage import zoom as _z
            sy = gt.shape[0] / dic.shape[0]
            sx = gt.shape[1] / dic.shape[1]
            dic = _z(dic, (sy, sx), order=1).astype(np.uint8)
        ax.imshow(dic, cmap="gray")
        n_gt_cells = draw_contours(ax, gt, "lime", lw=1.3)
        n_pred_cells = draw_contours(ax, pred, "magenta", lw=1.0)
        ax.set_title(f"F{fi}  GT={n_gt_cells} pipeline={n_pred_cells}",
                     fontsize=11)
        ax.axis("off")
    # Hide any unused subplot
    for j in range(n_show, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(
        f"{name} — GT (lime) vs pipeline (magenta)\n"
        f"{n_gt} annotated frames | "
        f"recording: {os.path.basename(tif)}",
        fontsize=13, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(os.path.join(out_dir, "contact_sheet.png"),
                dpi=contact_sheet_dpi, bbox_inches="tight")
    plt.close(fig)
    return out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", nargs="?",
                   help="Single recording folder, else all evaluable")
    p.add_argument("--all-frames", action="store_true",
                   help="Also render every frame in the recording "
                   "(not just the GT-annotated ones).")
    args = p.parse_args()

    if args.folder:
        folders = [args.folder]
    else:
        folders = find_evaluable_folders()

    print(f"Rendering overlays for {len(folders)} recordings"
          f"{' (incl. all frames)' if args.all_frames else ''}:")
    for f in folders:
        print(f"\n  {f}")
        try:
            out = render_recording(f)
            if out:
                print(f"    → {out}")
            if args.all_frames:
                from core.io import detect_channels
                tif = find_source_recording(f)
                n_ch = (detect_channels(tif)
                        if tif and tif.lower().endswith(
                            (".tif", ".tiff")) else 1)
                render_all_frames(f, n_ch)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
        # Free the cached DIC stack between recordings
        _DIC_STACK_CACHE.clear()


if __name__ == "__main__":
    main()
