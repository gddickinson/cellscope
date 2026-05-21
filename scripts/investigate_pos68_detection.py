"""Detection-method sweep on Pos68_DMSO GT frames.

Goal: rank candidate detection variants by F1 + edge/interior recall
on the 11 GT frames of Pos68_DMSO. Detection-only (no Cy5 fusion,
tracking, or per-cell refinement) so the variants are directly
comparable.

Variants:
  E0  cpsam @ 1024² (ds=2) — baseline matching current pipeline
  E1  cpsam @ 2048² (ds=1, full res)
  E2  cpsam tiled 2x2 (overlap=128) @ 1024²
  E3  cpsam tiled 2x2 @ 2048²
  E4  cpsam @ 1024² with 50 px mirror-pad on all sides
  E5  cpsam @ 1024² with TTA (augment=True)
  E6  cpsam tiled 2x2 + TTA @ 1024²
  E7  cpsam tiled 2x2 + mirror-pad @ 1024²
  E8  micro-sam vit_b_lm @ 1024² (subprocess to microsam env)
  E9  E0 + lower min_area filter (200 → 100)

Run in cellpose4 env:
  conda run -n cellpose4 python scripts/investigate_pos68_detection.py
"""
import os
import sys
import json
import time
import glob
import re
import argparse
import logging
import traceback
import numpy as np
import cv2
from PIL import Image
from skimage import measure

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

DEFAULT_REC = "data/ic295_gt_full/Pos68_DMSO"
DEFAULT_OUT = "results/pos68_investigation"

EDGE_PX = 50      # GT cell "at edge" if bbox within this many px of FoV
IOU_MATCH = 0.5   # IoU threshold for matched (TP)
MIN_AREA = 200    # default cell-area filter (matches DEFAULTS.min_area_px)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pos68_inv")


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_dic_and_gt(rec_folder=DEFAULT_REC):
    """Returns dic_full (N, H, W) uint8, gt_label_stack list, indices.

    Works for both multichannel IC295 recordings (DIC=ch1, Cy5=ch0)
    and single-channel legacy ignasi recordings — load_recording
    handles channel detection.
    """
    from core.io import load_recording
    tifs = [f for f in os.listdir(rec_folder)
            if f.endswith(".ome.tif") or f.endswith(".tif")]
    # Prefer .ome.tif when present
    ome = [f for f in tifs if f.endswith(".ome.tif")]
    tif = ome[0] if ome else tifs[0]
    tif_path = os.path.join(rec_folder, tif)
    try:
        rec = load_recording(tif_path, dic_channel=1, fluo_channel=0)
    except (TypeError, KeyError):
        rec = load_recording(tif_path)
    dic_full = rec["frames"]
    gt_dir = os.path.join(rec_folder, "gt_masks")
    gt_files = sorted(
        glob.glob(os.path.join(gt_dir, "mask_F*.png")),
        key=lambda p: int(re.search(r"F(\d+)", p).group(1)))
    frame_indices = [int(re.search(r"F(\d+)", p).group(1))
                     for p in gt_files]
    gt_masks_list = []
    for gtf in gt_files:
        g = np.array(Image.open(gtf))
        if g.ndim == 3:
            g = g[..., 0]
        gt_masks_list.append(measure.label(g > 0, connectivity=2))
    return dic_full[frame_indices], gt_masks_list, frame_indices


# ---------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------
def downsample(frames, factor):
    if factor == 1:
        return frames
    H, W = frames.shape[1:]
    new = (H // factor, W // factor)
    out = np.empty((len(frames), *new), dtype=np.uint8)
    for i in range(len(frames)):
        out[i] = cv2.resize(
            frames[i], (new[1], new[0]),
            interpolation=cv2.INTER_AREA)
    return out


def upsample_labels(labels, factor, target_shape):
    if factor == 1:
        return labels
    out = np.empty((len(labels), *target_shape), dtype=labels.dtype)
    for i in range(len(labels)):
        out[i] = cv2.resize(
            labels[i].astype(np.int32),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST)
    return out


# ---------------------------------------------------------------------
# Detection variants (E0-E7, E9)
# ---------------------------------------------------------------------
def run_cpsam(frames, augment=False):
    from cellpose import models
    m = models.CellposeModel(gpu=True)
    out = np.zeros(frames.shape, dtype=np.int32)
    for i in range(len(frames)):
        labs, _, _ = m.eval(frames[i], augment=augment)
        out[i] = labs
    return out


def run_cpsam_tiled(frames, n_tiles=(2, 2), overlap=128, augment=False):
    from core.cpsam_tiled import detect_cpsam_tiled
    return detect_cpsam_tiled(
        frames, n_tiles=n_tiles, overlap=overlap,
        min_area=50, augment=augment)


def run_cpsam_padded(frames, pad=50, augment=False):
    from cellpose import models
    m = models.CellposeModel(gpu=True)
    H, W = frames.shape[1:]
    out = np.zeros(frames.shape, dtype=np.int32)
    for i in range(len(frames)):
        padded = np.pad(frames[i], ((pad, pad), (pad, pad)),
                        mode="reflect")
        labs, _, _ = m.eval(padded, augment=augment)
        out[i] = labs[pad:pad + H, pad:pad + W]
    return out


def run_cpsam_tiled_padded(frames, pad=50, n_tiles=(2, 2), overlap=128,
                            augment=False):
    H, W = frames.shape[1:]
    padded = np.empty(
        (len(frames), H + 2 * pad, W + 2 * pad), dtype=frames.dtype)
    for i in range(len(frames)):
        padded[i] = np.pad(frames[i], ((pad, pad), (pad, pad)),
                            mode="reflect")
    labs = run_cpsam_tiled(padded, n_tiles=n_tiles, overlap=overlap,
                            augment=augment)
    return labs[:, pad:pad + H, pad:pad + W]


# ---------------------------------------------------------------------
# micro-sam — separate env, subprocess
# ---------------------------------------------------------------------
def run_microsam(frames, model_type="vit_b_lm"):
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "frames.npz")
        outp = os.path.join(tmp, "labels.npz")
        np.savez_compressed(inp, frames=frames)
        script = f'''
import sys, numpy as np
from micro_sam.automatic_segmentation import (
    automatic_instance_segmentation, get_predictor_and_segmenter)
data = np.load("{inp}")
frames = data["frames"]
predictor, segmenter = get_predictor_and_segmenter(
    model_type="{model_type}", device="mps")
out_list = []
for i in range(len(frames)):
    seg = automatic_instance_segmentation(
        predictor=predictor, segmenter=segmenter,
        input_path=frames[i], verbose=False)
    out_list.append(seg.astype("int32"))
out = np.stack(out_list, axis=0)
np.savez_compressed("{outp}", labels=out)
print("MICROSAM_OK", out.shape)
'''
        proc = subprocess.run(
            ["conda", "run", "-n", "microsam", "python", "-c", script],
            capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0 or not os.path.exists(outp):
            raise RuntimeError(
                f"micro-sam subprocess failed:\nSTDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr[-1500:]}")
        return np.load(outp)["labels"]


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------
def score_frame(pred_lab, gt_lab, min_area=MIN_AREA):
    H, W = gt_lab.shape
    gt_regions = [r for r in measure.regionprops(gt_lab)
                  if r.area >= min_area]
    pred_regions = [r for r in measure.regionprops(pred_lab)
                    if r.area >= min_area]

    n_g, n_p = len(gt_regions), len(pred_regions)
    iou = np.zeros((n_g, n_p))
    gt_masks_cache = [(gt_lab == gr.label) for gr in gt_regions]
    pr_masks_cache = [(pred_lab == pr.label) for pr in pred_regions]
    for i, gm in enumerate(gt_masks_cache):
        for j, pm in enumerate(pr_masks_cache):
            inter = np.logical_and(gm, pm).sum()
            if inter == 0:
                continue
            union = np.logical_or(gm, pm).sum()
            iou[i, j] = inter / union

    # Greedy match by IoU
    matched_gt = set()
    matched_pred = set()
    matched_ious = []
    if iou.size:
        pairs = sorted(
            [(iou[i, j], i, j)
             for i in range(n_g) for j in range(n_p)],
            reverse=True)
        for v, i, j in pairs:
            if v < IOU_MATCH:
                break
            if i in matched_gt or j in matched_pred:
                continue
            matched_gt.add(i)
            matched_pred.add(j)
            matched_ious.append(v)

    tp = len(matched_gt)
    fn = n_g - tp
    fp = n_p - len(matched_pred)

    # Edge/interior split (computed on GT only — same for every variant)
    edge_t = edge_c = int_t = int_c = 0
    for i, gr in enumerate(gt_regions):
        r0, c0, r1, c1 = gr.bbox
        at_edge = (r0 < EDGE_PX or c0 < EDGE_PX
                   or r1 > H - EDGE_PX or c1 > W - EDGE_PX)
        if at_edge:
            edge_t += 1
            if i in matched_gt:
                edge_c += 1
        else:
            int_t += 1
            if i in matched_gt:
                int_c += 1

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "n_gt": n_g, "n_pred": n_p,
        "mean_iou_matched": (float(np.mean(matched_ious))
                              if matched_ious else 0.0),
        "edge_total": edge_t, "edge_caught": edge_c,
        "interior_total": int_t, "interior_caught": int_c,
    }


def aggregate(per_frame_results):
    tot_tp = sum(r["tp"] for r in per_frame_results)
    tot_fp = sum(r["fp"] for r in per_frame_results)
    tot_fn = sum(r["fn"] for r in per_frame_results)
    e_t = sum(r["edge_total"] for r in per_frame_results)
    e_c = sum(r["edge_caught"] for r in per_frame_results)
    i_t = sum(r["interior_total"] for r in per_frame_results)
    i_c = sum(r["interior_caught"] for r in per_frame_results)
    matched = [r["mean_iou_matched"] for r in per_frame_results
               if r["mean_iou_matched"] > 0]
    return {
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "precision": tot_tp / max(1, tot_tp + tot_fp),
        "recall": tot_tp / max(1, tot_tp + tot_fn),
        "f1": 2 * tot_tp / max(1, 2 * tot_tp + tot_fp + tot_fn),
        "mean_iou_matched": float(np.mean(matched)) if matched else 0.0,
        "edge_recall": e_c / max(1, e_t),
        "interior_recall": i_c / max(1, i_t),
        "edge_total": e_t, "edge_caught": e_c,
        "interior_total": i_t, "interior_caught": i_c,
    }


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", nargs="+", default=None,
        help="Run only the named experiments (e.g. E0 E2 E8)")
    ap.add_argument(
        "--recording", default=DEFAULT_REC,
        help="GT recording folder to evaluate")
    ap.add_argument(
        "--out-dir", default=None,
        help="Output dir (default: results/<recording_basename>_inv)")
    args = ap.parse_args()

    out_dir = args.out_dir or (
        DEFAULT_OUT if args.recording == DEFAULT_REC
        else f"results/{os.path.basename(args.recording)}_inv")
    os.makedirs(out_dir, exist_ok=True)
    global OUT_DIR
    OUT_DIR = out_dir

    log.info("Recording: %s", args.recording)
    log.info("Out dir:   %s", out_dir)
    log.info("Loading DIC + GT frames…")
    dic_full, gt_masks, frame_indices = load_dic_and_gt(args.recording)
    log.info("  %d GT frames at indices %s",
             len(dic_full), frame_indices)
    log.info("  full-res shape: %s", dic_full.shape)
    H, W = dic_full.shape[1:]
    # Use the same auto-downsample as the production pipeline so the
    # detection-only comparison reflects what users actually run.
    from core.pipeline_defaults import resolve_downsample
    ds_factor, ds_reason = resolve_downsample("auto", (H, W))
    log.info("  auto-downsample: %d× (%s)", ds_factor, ds_reason)
    dic_half = downsample(dic_full, ds_factor)
    log.info("  detection-res shape: %s", dic_half.shape)

    def with_eval(label_stack, min_area=MIN_AREA):
        # Apply edge-sliver filter (same logic as production)
        from core.project import _tracks_from_labels
        from core.cy5_filter import rebuild_label_stack
        from core.track_postprocess import (
            reject_edge_sliver_detections)
        tracks = _tracks_from_labels(label_stack)
        reject_edge_sliver_detections(tracks, (H, W))
        cleaned = rebuild_label_stack(
            tracks, label_stack.shape, dtype=np.int32)
        per_frame = [
            score_frame(cleaned[i], gt_masks[i], min_area=min_area)
            for i in range(len(cleaned))]
        return cleaned, per_frame, aggregate(per_frame)

    # Names retain "_ds2" suffix for back-compat (sweep on Pos68_DMSO);
    # in practice the downsample factor is whatever resolve_downsample
    # decides per recording.
    experiments = [
        ("E0_baseline_ds2",
         lambda: upsample_labels(run_cpsam(dic_half), ds_factor, (H, W))),
        ("E1_cpsam_fullres",
         lambda: run_cpsam(dic_full)),
        ("E2_tiled2x2_ds2",
         lambda: upsample_labels(
             run_cpsam_tiled(dic_half, n_tiles=(2, 2), overlap=128),
             ds_factor, (H, W))),
        ("E3_tiled2x2_fullres",
         lambda: run_cpsam_tiled(dic_full, n_tiles=(2, 2),
                                  overlap=128)),
        ("E4_padded_ds2",
         lambda: upsample_labels(
             run_cpsam_padded(dic_half, pad=50), ds_factor, (H, W))),
        ("E5_TTA_ds2",
         lambda: upsample_labels(
             run_cpsam(dic_half, augment=True), ds_factor, (H, W))),
        ("E6_tiled_TTA_ds2",
         lambda: upsample_labels(
             run_cpsam_tiled(dic_half, augment=True),
             ds_factor, (H, W))),
        ("E7_tiled_padded_ds2",
         lambda: upsample_labels(
             run_cpsam_tiled_padded(dic_half), ds_factor, (H, W))),
        ("E8_microsam_vit_b_lm_ds2",
         lambda: upsample_labels(
             run_microsam(dic_half), ds_factor, (H, W))),
    ]

    if args.only:
        experiments = [e for e in experiments
                        if any(o in e[0] for o in args.only)]
        log.info("Filter to: %s", [e[0] for e in experiments])

    summary = []
    for name, fn in experiments:
        log.info("=" * 60)
        log.info("Running %s", name)
        t0 = time.time()
        try:
            label_stack = fn()
        except Exception as e:
            log.error("  FAILED: %s\n%s", e, traceback.format_exc())
            summary.append({"experiment": name, "error": str(e),
                            "runtime_sec": time.time() - t0})
            continue
        runtime = time.time() - t0
        cleaned, per_frame, agg = with_eval(label_stack)
        out_path = os.path.join(OUT_DIR, f"{name}_labels.npz")
        np.savez_compressed(out_path, labels=cleaned)
        log.info("  saved %s", out_path)
        log.info(
            "  TP=%d FP=%d FN=%d  F1=%.3f  IoU=%.3f  "
            "edge_recall=%.2f  interior_recall=%.2f  (%.1fs)",
            agg["tp"], agg["fp"], agg["fn"], agg["f1"],
            agg["mean_iou_matched"], agg["edge_recall"],
            agg["interior_recall"], runtime)
        summary.append({
            "experiment": name,
            "runtime_sec": runtime,
            **agg,
        })
        with open(os.path.join(OUT_DIR, f"{name}_per_frame.json"), "w") as f:
            json.dump({"frames": frame_indices, "per_frame": per_frame,
                        "agg": agg}, f, indent=2)

    # E9: re-evaluate E0 with min_area=100 (sanity check)
    e0_path = os.path.join(OUT_DIR, "E0_baseline_ds2_labels.npz")
    if os.path.exists(e0_path):
        log.info("=" * 60)
        log.info("E9_min_area_100 (re-evaluate E0 with smaller filter)")
        labs = np.load(e0_path)["labels"]
        per_frame = [score_frame(labs[i], gt_masks[i], min_area=100)
                     for i in range(len(labs))]
        agg = aggregate(per_frame)
        log.info(
            "  TP=%d FP=%d FN=%d  F1=%.3f  IoU=%.3f  "
            "edge_recall=%.2f  interior_recall=%.2f",
            agg["tp"], agg["fp"], agg["fn"], agg["f1"],
            agg["mean_iou_matched"], agg["edge_recall"],
            agg["interior_recall"])
        summary.append({"experiment": "E9_E0_min_area_100",
                        "runtime_sec": 0.0, **agg})

    # Final ranking table
    summary.sort(key=lambda r: -r.get("f1", -1))
    print()
    print("=" * 95)
    print("DETECTION SWEEP RESULTS — Pos68_DMSO 11 GT frames "
          "(155 GT cells; 76 at edge, 79 interior)")
    print("=" * 95)
    print(f"{'experiment':<28} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'F1':>6} {'IoU':>6} {'edge':>6} {'inter':>6} {'time':>6}")
    for r in summary:
        if "error" in r:
            print(f"{r['experiment']:<28}  ERROR: {r['error'][:50]}")
            continue
        print(f"{r['experiment']:<28} {r['tp']:>4d} {r['fp']:>4d} "
              f"{r['fn']:>4d} {r['f1']:>6.3f} {r['mean_iou_matched']:>6.3f} "
              f"{r['edge_recall']:>6.2f} {r['interior_recall']:>6.2f} "
              f"{r['runtime_sec']:>5.0f}s")
    print()

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"summary": summary,
                    "frames": frame_indices}, f, indent=2)
    log.info("Wrote summary to %s/summary.json", OUT_DIR)


if __name__ == "__main__":
    main()
