"""Inspection figures: cpsam_dic (and cpsam) detections across recording types.

Run in the cellpose4 env so cpsam* models load. Loads each model once and
re-uses it for all panels in that figure.

Outputs:
  results/best_detections/01_our_gt_dic.png
  results/best_detections/02_vampire_ood.png
  results/best_detections/03_jesse_1024.png
  results/best_detections/04_ignasi_phase.png

Usage:
  conda run -n cellpose4 python scripts/make_overlay_figures.py
"""
import json
import os
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile

# Helpers (resolves project root + benchmark data location)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import benchmark_data_root  # noqa

OUT_DIR = "results/best_detections"
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def safe_read_tiff(path):
    """Read a TIFF stack tolerating big-endian float32 (numpy 2.0 issue).

    tifffile internally calls .newbyteorder() which was removed in numpy
    2.0. Catch that and fall back to raw-bytes per page (mirrors
    piezo1_analysis/scripts/prepare_vampire_gof.py:read_image_uint8).
    """
    try:
        return tifffile.imread(path)
    except AttributeError as e:
        if "newbyteorder" not in str(e):
            raise
    pages_arrs = []
    with tifffile.TiffFile(path) as tf:
        with open(path, "rb") as fh:
            for pg in tf.pages:
                h, w = pg.imagelength, pg.imagewidth
                bps = pg.bitspersample
                sf = pg.sampleformat  # 1=uint, 2=int, 3=float
                byteorder = pg.parent.byteorder  # '<' or '>'
                offsets = pg.dataoffsets
                byte_counts = pg.databytecounts
                chunks = []
                for off, n in zip(offsets, byte_counts):
                    fh.seek(off); chunks.append(fh.read(n))
                raw = b"".join(chunks)
                np_kind = {1: "u", 2: "i", 3: "f"}.get(sf, "u")
                nbytes = bps // 8
                np_dtype = np.dtype(f"{byteorder}{np_kind}{nbytes}")
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(h, w)
                if arr.dtype.byteorder not in ("=", "|"):
                    arr = arr.astype(arr.dtype.newbyteorder("=")).copy()
                pages_arrs.append(arr)
    return np.stack(pages_arrs) if len(pages_arrs) > 1 else pages_arrs[0]


def imnorm(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip((img - p1) / max(p99 - p1, 1e-6), 0, 1)
    return img


def overlay_panel(ax, img, gt=None, pred=None, pred_label=None,
                  title="", cmap="gray"):
    """Render one panel: image + optional GT (yellow) + prediction(s).

    pred can be bool (single colour) or int32 label stack (per-cell colour).
    """
    ax.imshow(imnorm(img), cmap=cmap, vmin=0, vmax=1)

    if gt is not None and gt.any():
        ax.contour(gt, levels=[0.5], colors=["yellow"], linewidths=1.5)

    if pred is not None:
        if pred.dtype.kind in "iub" and pred.max() > 1:
            # int32 label stack — colour per cell
            ids = sorted(set(np.unique(pred)) - {0})
            cmap_cells = plt.cm.tab10(np.linspace(0, 1, max(len(ids), 10)))
            for idx, cid in enumerate(ids):
                ax.contour((pred == cid), levels=[0.5],
                           colors=[cmap_cells[idx % 10]], linewidths=1.5)
        else:
            ax.contour((pred > 0).astype(np.uint8), levels=[0.5],
                       colors=["red"], linewidths=1.5)

    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def bench_rows(json_path):
    with open(json_path) as f:
        return json.load(f)["rows"]


def pick_frames_for_genotype(rows, geno, n=3, mode="best"):
    """Pick top-N (or worst-N) named frames for a genotype."""
    sub = [r for r in rows if r["genotype"] == geno]
    sub.sort(key=lambda r: r["iou"], reverse=(mode == "best"))
    return [r["name"] for r in sub[:n]]


# ──────────────────────────────────────────────────────────────────────
# Figure 1: our-GT DIC (526² full-frame, in-domain)
# ──────────────────────────────────────────────────────────────────────
def figure_our_gt(model_cpsam_dic):
    print("[fig 1] our-GT DIC (526² in-domain)")
    rows = bench_rows(f"{OUT_DIR}/../dic_model_eval/cpsam_dic_ourgt.json")
    ctrl_names = pick_frames_for_genotype(rows, "control", n=3, mode="best")
    cko_names = pick_frames_for_genotype(rows, "cko", n=3, mode="best")

    base = str(benchmark_data_root() / "data" / "training")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("our-GT DIC (526² in-domain) — cpsam_dic predictions",
                 fontsize=12, y=0.995)

    for col, name in enumerate(ctrl_names):
        img = cv2.imread(f"{base}/{name}.png", cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(f"{base}/{name}_masks.png", cv2.IMREAD_UNCHANGED) > 0
        pred = model_cpsam_dic.eval(img)[0] > 0
        iou = float(np.logical_and(pred, gt).sum() /
                    max(np.logical_or(pred, gt).sum(), 1))
        overlay_panel(axes[0, col], img, gt=gt, pred=pred,
                      title=f"control  ({name[-4:]})  IoU={iou:.3f}")

    for col, name in enumerate(cko_names):
        img = cv2.imread(f"{base}/{name}.png", cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(f"{base}/{name}_masks.png", cv2.IMREAD_UNCHANGED) > 0
        pred = model_cpsam_dic.eval(img)[0] > 0
        iou = float(np.logical_and(pred, gt).sum() /
                    max(np.logical_or(pred, gt).sum(), 1))
        overlay_panel(axes[1, col], img, gt=gt, pred=pred,
                      title=f"cKO  ({name[-4:]})  IoU={iou:.3f}")

    fig.text(0.5, 0.01,
             "yellow = GT manual annotation   red = cpsam_dic prediction",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = f"{OUT_DIR}/01_our_gt_dic.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Figure 2: VAMPIRE held-out test (3 genotypes)
# ──────────────────────────────────────────────────────────────────────
def figure_vampire(model_cpsam_dic):
    print("[fig 2] VAMPIRE held-out test (control / cKO / GoF)")
    rows = bench_rows(f"{OUT_DIR}/../dic_model_eval/cpsam_dic.json")
    base = "data/training/dic_splits_v3/test"
    rows_per_geno = {
        "control": pick_frames_for_genotype(rows, "control", n=3, mode="best"),
        "cko": pick_frames_for_genotype(rows, "cko", n=3, mode="best"),
        "gof": pick_frames_for_genotype(rows, "gof", n=3, mode="best"),
    }

    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    fig.suptitle("VAMPIRE held-out test (out-of-domain crops) — "
                 "cpsam_dic predictions", fontsize=12, y=0.995)

    for row, (geno, names) in enumerate(rows_per_geno.items()):
        for col, name in enumerate(names):
            img = tifffile.imread(f"{base}/{name}_img.tif")
            if img.ndim == 3:
                img = img[0]
            gt = tifffile.imread(f"{base}/{name}_masks.tif")
            if gt.ndim == 3:
                gt = gt[0]
            gt_bool = gt > 0
            pred = model_cpsam_dic.eval(img)[0] > 0
            iou = float(np.logical_and(pred, gt_bool).sum() /
                        max(np.logical_or(pred, gt_bool).sum(), 1))
            overlay_panel(axes[row, col], img, gt=gt_bool, pred=pred,
                          title=f"{geno:>7s}  IoU={iou:.3f}")

    fig.text(0.5, 0.01,
             "yellow = GT   red = cpsam_dic prediction   "
             "(held-out cell sequences cpsam_dic never saw)",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = f"{OUT_DIR}/02_vampire_ood.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Figure 3: Jesse OME-TIFF 1024² (no per-frame GT)
# ──────────────────────────────────────────────────────────────────────
def figure_jesse(model_cpsam_dic):
    print("[fig 3] Jesse OME-TIFF (1024², no GT)")
    base = str(benchmark_data_root() / "data" / "examples")
    recs = [
        ("jesse_wt/pos0_wt.ome.tif", "WT pos0"),
        ("jesse_wt/pos17_wt.ome.tif", "WT pos17"),
        ("jesse_ko/pos59_ko.ome.tif", "KO pos59"),
        ("jesse_ko/pos65_ko.ome.tif", "KO pos65"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle("Jesse OME-TIFF 1024² (no manual GT) — "
                 "cpsam_dic predictions on raw recordings",
                 fontsize=12, y=0.995)

    for col, (rel, label) in enumerate(recs):
        path = f"{base}/{rel}"
        if not os.path.exists(path):
            for row in range(2):
                axes[row, col].text(0.5, 0.5, f"missing\n{rel}",
                                    ha="center", va="center",
                                    transform=axes[row, col].transAxes)
                axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            continue
        stack = safe_read_tiff(path)
        # rescale uint16 → uint8 if needed
        if stack.dtype == np.uint16:
            p1, p99 = np.percentile(stack, [1, 99])
            stack = np.clip((stack.astype(np.float32) - p1) /
                            max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
        # 2 frames per recording: one early, one late
        frame_idx = [10, len(stack) // 2]
        for row, fi in enumerate(frame_idx):
            img = stack[fi]
            pred_labels = model_cpsam_dic.eval(img)[0].astype(np.int32)
            n_cells = int(pred_labels.max())
            overlay_panel(axes[row, col], img, gt=None, pred=pred_labels,
                          title=f"{label}  frame {fi}  n={n_cells}")

    fig.text(0.5, 0.01,
             "coloured = per-cell prediction (cpsam_dic, no fine-tune)   "
             "no GT to compare against",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = f"{OUT_DIR}/03_jesse_1024.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Figure 4: Ignasi phase-contrast (multi-cell)
# ──────────────────────────────────────────────────────────────────────
def figure_ignasi(model_cpsam):
    print("[fig 4] Ignasi phase-contrast (multi-cell)")
    base = str(benchmark_data_root() / "data" / "ignasi")
    recs = [
        ("C1-IC293__1_MMStack_Pos0-WT.ome-1cropped.tif", "Pos0 WT"),
        ("IC293__1_MMStack_Pos2-WT.ome-cropped.tif", "Pos2 WT"),
        ("IC293__1_MMStack_Pos17-KO.ome-cropped.tif", "Pos17 KO"),
        ("IC293__1_MMStack_Pos3-WT.ome-cropped.tif", "Pos3 WT"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle("Ignasi phase-contrast cropped — base cpsam (default)",
                 fontsize=12, y=0.995)

    for col, (rel, label) in enumerate(recs):
        path = f"{base}/{rel}"
        if not os.path.exists(path):
            for row in range(2):
                axes[row, col].text(0.5, 0.5, f"missing\n{rel}",
                                    ha="center", va="center",
                                    transform=axes[row, col].transAxes)
                axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            continue
        stack = safe_read_tiff(path)
        # Normalise to uint8 for any dtype (uint16, float32 shade-corrected)
        if stack.dtype != np.uint8:
            p1, p99 = np.percentile(stack, [1, 99])
            stack = np.clip((stack.astype(np.float32) - p1) /
                            max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
        frame_idx = [10, len(stack) // 2]
        for row, fi in enumerate(frame_idx):
            img = stack[fi]
            pred_labels = model_cpsam.eval(img)[0].astype(np.int32)
            n_cells = int(pred_labels.max())
            overlay_panel(axes[row, col], img, gt=None, pred=pred_labels,
                          title=f"{label}  frame {fi}  n={n_cells}")

    fig.text(0.5, 0.01,
             "coloured = per-cell prediction (default cpsam, no DIC fine-tune)   "
             "expected best for phase-contrast endothelial cells",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = f"{OUT_DIR}/04_ignasi_phase.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def figure_ignasi_production(model_cpsam):
    """Same recordings + frames as figure_ignasi, but with production
    pipeline: cpsam → min_area filter → per-cell DeepSea refinement.
    Tracking is skipped (only 2 frames per recording, not meaningful)."""
    print("[fig 4p] Ignasi phase-contrast — PRODUCTION pipeline "
          "(cpsam + min_area + DeepSea)")
    from core.hybrid_cpsam_multi import _filter_labels
    from core.deepsea_multicell import refine_labels_with_deepsea

    base = str(benchmark_data_root() / "data" / "ignasi")
    recs = [
        ("C1-IC293__1_MMStack_Pos0-WT.ome-1cropped.tif", "Pos0 WT"),
        ("IC293__1_MMStack_Pos2-WT.ome-cropped.tif", "Pos2 WT"),
        ("IC293__1_MMStack_Pos17-KO.ome-cropped.tif", "Pos17 KO"),
        ("IC293__1_MMStack_Pos3-WT.ome-cropped.tif", "Pos3 WT"),
    ]
    MIN_AREA = 500

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle("Ignasi phase-contrast cropped — PRODUCTION pipeline "
                 "(cpsam + min_area=500 + per-cell DeepSea)",
                 fontsize=12, y=0.995)

    for col, (rel, label) in enumerate(recs):
        path = f"{base}/{rel}"
        if not os.path.exists(path):
            for row in range(2):
                axes[row, col].text(0.5, 0.5, f"missing\n{rel}",
                                    ha="center", va="center",
                                    transform=axes[row, col].transAxes)
                axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            continue
        stack = safe_read_tiff(path)
        if stack.dtype != np.uint8:
            p1, p99 = np.percentile(stack, [1, 99])
            stack = np.clip((stack.astype(np.float32) - p1) /
                            max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)

        # Take 2 frames; bundle them so DeepSea (per-frame) runs on the pair.
        frame_idx = [10, len(stack) // 2]
        frames = np.stack([stack[i] for i in frame_idx])

        # Step 1: cpsam labels
        raw_labels = np.zeros(frames.shape, dtype=np.int32)
        for j in range(len(frames)):
            raw_labels[j] = model_cpsam.eval(frames[j])[0].astype(np.int32)

        # Step 2: debris filter (min_area)
        for j in range(len(frames)):
            raw_labels[j], _ = _filter_labels(raw_labels[j], MIN_AREA)
        n_after_filter = [int(raw_labels[j].max()) for j in range(len(frames))]

        # Step 3: per-cell DeepSea refinement (preserves cell identity)
        refined = refine_labels_with_deepsea(frames, raw_labels, expand_px=20)

        for row, fi in enumerate(frame_idx):
            n_cells = int(refined[row].max())
            note = (f"  (cpsam→{n_after_filter[row]}→DeepSea→{n_cells})")
            overlay_panel(axes[row, col], frames[row], gt=None,
                          pred=refined[row].astype(np.int32),
                          title=f"{label}  frame {fi}  n={n_cells}{note}")

    fig.text(0.5, 0.01,
             "coloured = per-cell prediction (default cpsam → debris "
             "filter → DeepSea refinement)   "
             "title shows cell counts at each pipeline stage",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = f"{OUT_DIR}/04_ignasi_phase_production.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _paths import setup_imports, benchmark_data_root  # noqa
    setup_imports()
    from cellpose import models

    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    if only is None or "1" in only or "2" in only or "3" in only:
        print("Loading cpsam_dic (DIC fine-tune)…")
        model_cpsam_dic = models.CellposeModel(
            gpu=True, pretrained_model="data/models/cpsam_dic")
        if only is None or "1" in only:
            figure_our_gt(model_cpsam_dic)
        if only is None or "2" in only:
            figure_vampire(model_cpsam_dic)
        if only is None or "3" in only:
            figure_jesse(model_cpsam_dic)

    if only is None or "4" in only or "4p" in only:
        print("Loading cpsam (default for phase-contrast)…")
        model_cpsam = models.CellposeModel(gpu=True)
        if only is None or "4" in only:
            figure_ignasi(model_cpsam)
        if only is None or "4p" in only:
            figure_ignasi_production(model_cpsam)

    print(f"\nFigures saved under {OUT_DIR}/")


if __name__ == "__main__":
    main()
