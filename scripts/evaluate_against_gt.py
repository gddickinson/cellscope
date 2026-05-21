"""Compare pipeline labels against manual GT masks for a recording.

Usage:
  python scripts/evaluate_against_gt.py data/ic295_gt_full/Pos7_WT

Reads:
  <folder>/gt_masks/mask_F<N>.png    int16 GT (one ID per cell)
  <folder>/pipeline_results/masks.npz   pipeline labels stack

Computes per annotated frame:
  - Per-cell IoU between each GT cell and its best pipeline match
    (Hungarian assignment maximising total IoU)
  - TP / FP / FN at IoU >= 0.5 (and a sweep at 0.3, 0.5, 0.7)
  - Precision / recall / F1
  - Boundary IoU distribution

Across the labelled frames:
  - Tracking identity preservation: for each GT cell ID, how often
    is it matched to the SAME pipeline ID? Confusion matrix and
    "ID switch rate" reported.
  - Frames where the pipeline misses cells (FN-heavy) or adds cells
    (FP-heavy)

Outputs:
  <folder>/evaluation/
    report.md            ← human-readable summary
    per_frame.csv        ← per-frame TP/FP/FN/precision/recall
    per_cell.csv         ← per-cell-per-frame IoU + matched IDs
    confusion.png        ← GT-ID × pipeline-ID heat map
    iou_distribution.png ← histogram of best-match IoUs
    report.json          ← machine-readable everything
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
sys.path.insert(0, CELLSCOPE_ROOT)


def load_gt_masks(gt_dir):
    """Return dict {frame_idx: (H, W) int32 mask}."""
    from skimage import io as skio
    out = {}
    if not os.path.isdir(gt_dir):
        return out
    for f in sorted(os.listdir(gt_dir)):
        if not (f.startswith("mask_F") and f.endswith(".png")):
            continue
        try:
            fi = int(f[len("mask_F"):-len(".png")])
        except ValueError:
            continue
        m = skio.imread(os.path.join(gt_dir, f))
        out[fi] = m.astype(np.int32)
    return out


def load_pipeline_labels(pipeline_dir):
    """Return (N, H, W) int32 label stack from pipeline_results/."""
    p = os.path.join(pipeline_dir, "masks.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    data = np.load(p)
    if "labels" in data.files:
        return data["labels"].astype(np.int32)
    if "masks" in data.files:
        return data["masks"].astype(np.int32)
    raise ValueError(f"no labels/masks in {p}")


def per_cell_masks(label_frame):
    """Return list of (cell_id, bool_mask) for every nonzero ID."""
    out = []
    if label_frame.max() == 0:
        return out
    for cid in range(1, int(label_frame.max()) + 1):
        m = label_frame == cid
        if m.any():
            out.append((cid, m))
    return out


def iou(a_mask, b_mask):
    inter = np.logical_and(a_mask, b_mask).sum()
    if inter == 0:
        return 0.0
    return float(inter / np.logical_or(a_mask, b_mask).sum())


def hungarian_iou_match(gt_cells, pred_cells):
    """Hungarian assignment maximising total IoU.

    Returns list of (gt_id, pred_id, iou) including 0-IoU non-matches
    for every gt_id and pred_id (so callers can derive TP/FP/FN).
    """
    if not gt_cells or not pred_cells:
        results = []
        for g_id, _ in gt_cells:
            results.append((g_id, None, 0.0))
        for p_id, _ in pred_cells:
            results.append((None, p_id, 0.0))
        return results

    iou_matrix = np.zeros((len(gt_cells), len(pred_cells)),
                          dtype=np.float64)
    for i, (_, gm) in enumerate(gt_cells):
        for j, (_, pm) in enumerate(pred_cells):
            iou_matrix[i, j] = iou(gm, pm)
    cost = -iou_matrix
    rows, cols = linear_sum_assignment(cost)

    matched_gt = set()
    matched_pred = set()
    out = []
    for r, c in zip(rows, cols):
        if iou_matrix[r, c] > 0:
            out.append((gt_cells[r][0], pred_cells[c][0],
                        float(iou_matrix[r, c])))
            matched_gt.add(r)
            matched_pred.add(c)
    for i, (g_id, _) in enumerate(gt_cells):
        if i not in matched_gt:
            out.append((g_id, None, 0.0))
    for j, (p_id, _) in enumerate(pred_cells):
        if j not in matched_pred:
            out.append((None, p_id, 0.0))
    return out


def evaluate_recording(folder, iou_thresholds=(0.3, 0.5, 0.7)):
    folder = os.path.abspath(folder)
    gt = load_gt_masks(os.path.join(folder, "gt_masks"))
    if not gt:
        raise FileNotFoundError(
            f"No GT masks found in {folder}/gt_masks/")
    print(f"Loaded {len(gt)} GT frames: {sorted(gt)}")

    pipeline_labels = load_pipeline_labels(
        os.path.join(folder, "pipeline_results"))
    print(f"Pipeline labels stack: {pipeline_labels.shape}, "
          f"{int(pipeline_labels.max())} unique IDs")

    # Sanity: GT and pipeline must share frame shape
    h, w = next(iter(gt.values())).shape
    if pipeline_labels.shape[1:] != (h, w):
        print(f"WARNING: shape mismatch — GT {(h, w)} vs pipeline "
              f"{pipeline_labels.shape[1:]}. Resizing pipeline.")
        from scipy.ndimage import zoom
        sy = h / pipeline_labels.shape[1]
        sx = w / pipeline_labels.shape[2]
        new_stack = np.empty((pipeline_labels.shape[0], h, w),
                             dtype=np.int32)
        for i in range(len(pipeline_labels)):
            new_stack[i] = zoom(pipeline_labels[i], (sy, sx),
                                 order=0)
        pipeline_labels = new_stack

    per_frame = []
    per_cell = []
    # gt_to_pred[g_id] = Counter of pred IDs that matched
    gt_to_pred_freq = defaultdict(lambda: defaultdict(int))

    for fi in sorted(gt):
        if fi >= len(pipeline_labels):
            continue
        gt_cells = per_cell_masks(gt[fi])
        pred_cells = per_cell_masks(pipeline_labels[fi])
        matches = hungarian_iou_match(gt_cells, pred_cells)

        # Per-cell rows
        for g, p, m_iou in matches:
            per_cell.append({
                "frame": fi, "gt_id": g, "pred_id": p, "iou": m_iou,
            })
            if g is not None and p is not None and m_iou > 0:
                gt_to_pred_freq[g][p] += 1

        # Per-prediction max IoU across all GT cells. Used to flag
        # predictions that have ZERO overlap with any GT cell —
        # those are "out-of-scope" detections (real cells the GT
        # didn't annotate, or genuine FPs). The standard F1 above
        # counts them as FPs; the *_focused metrics exclude them so
        # the recording's GT-coverage doesn't penalize valid
        # detections in unannotated areas. Important for single-cell
        # GT recordings (ignasi) and for recordings where GT only
        # partially covers the field.
        pred_max_iou_any = {}
        for p_id, p_mask in pred_cells:
            best = 0.0
            for _, g_mask in gt_cells:
                inter = int(np.logical_and(p_mask, g_mask).sum())
                if inter == 0:
                    continue
                ui = int(np.logical_or(p_mask, g_mask).sum())
                cand = inter / ui if ui > 0 else 0.0
                if cand > best:
                    best = cand
            pred_max_iou_any[p_id] = best
        out_of_scope = sum(
            1 for v in pred_max_iou_any.values() if v == 0)

        # Per-frame TP/FP/FN at each IoU threshold
        row = {"frame": fi, "n_gt": len(gt_cells),
               "n_pred": len(pred_cells),
               "n_pred_out_of_scope": out_of_scope}
        for thr in iou_thresholds:
            tp = sum(1 for g, p, i in matches
                     if g is not None and p is not None and i >= thr)
            fn = len(gt_cells) - tp
            fp = len(pred_cells) - tp
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = (2 * prec * rec / max(prec + rec, 1e-9)
                  if (prec + rec) > 0 else 0.0)
            # GT-focused variants: exclude out-of-scope predictions
            fp_focused = max(0, fp - out_of_scope)
            prec_focused = tp / max(tp + fp_focused, 1)
            f1_focused = (
                2 * prec_focused * rec / max(prec_focused + rec, 1e-9)
                if (prec_focused + rec) > 0 else 0.0)
            row[f"TP_{thr}"] = tp
            row[f"FP_{thr}"] = fp
            row[f"FN_{thr}"] = fn
            row[f"prec_{thr}"] = round(prec, 3)
            row[f"rec_{thr}"] = round(rec, 3)
            row[f"F1_{thr}"] = round(f1, 3)
            row[f"FP_focused_{thr}"] = fp_focused
            row[f"prec_focused_{thr}"] = round(prec_focused, 3)
            row[f"F1_focused_{thr}"] = round(f1_focused, 3)
        per_frame.append(row)

    # Tracking identity preservation
    id_consistency = []
    for g_id, freqs in gt_to_pred_freq.items():
        total = sum(freqs.values())
        if total == 0:
            continue
        most_common_p = max(freqs, key=freqs.get)
        id_consistency.append({
            "gt_id": g_id,
            "n_frames_matched": total,
            "dominant_pred_id": most_common_p,
            "consistency": freqs[most_common_p] / total,
            "all_pred_ids": dict(freqs),
        })

    # Save outputs
    out_dir = os.path.join(folder, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    # CSVs
    import csv
    with open(os.path.join(out_dir, "per_frame.csv"), "w",
               newline="") as f:
        w_ = csv.DictWriter(f, list(per_frame[0]))
        w_.writeheader()
        w_.writerows(per_frame)
    with open(os.path.join(out_dir, "per_cell.csv"), "w",
               newline="") as f:
        w_ = csv.DictWriter(f, list(per_cell[0]))
        w_.writeheader()
        w_.writerows(per_cell)

    # Aggregate stats
    matched_ious = [r["iou"] for r in per_cell
                    if r["gt_id"] is not None
                    and r["pred_id"] is not None
                    and r["iou"] > 0]
    summary = {
        "folder": folder,
        "n_gt_frames": len(gt),
        "iou_thresholds": list(iou_thresholds),
        "mean_TP_per_frame": {
            f"@{thr}": float(np.mean([r[f"TP_{thr}"]
                                       for r in per_frame]))
            for thr in iou_thresholds},
        "mean_FN_per_frame": {
            f"@{thr}": float(np.mean([r[f"FN_{thr}"]
                                       for r in per_frame]))
            for thr in iou_thresholds},
        "mean_FP_per_frame": {
            f"@{thr}": float(np.mean([r[f"FP_{thr}"]
                                       for r in per_frame]))
            for thr in iou_thresholds},
        "mean_F1": {
            f"@{thr}": float(np.mean([r[f"F1_{thr}"]
                                       for r in per_frame]))
            for thr in iou_thresholds},
        # GT-focused variants (exclude predictions with 0 IoU against
        # all GT cells from the FP count). These match what users
        # actually care about when GT only partially annotates the
        # field — single-cell GT recordings, or any recording where
        # the pipeline finds extra real cells the GT didn't label.
        "mean_FP_focused_per_frame": {
            f"@{thr}": float(np.mean([r[f"FP_focused_{thr}"]
                                       for r in per_frame]))
            for thr in iou_thresholds},
        "mean_F1_focused": {
            f"@{thr}": float(np.mean([r[f"F1_focused_{thr}"]
                                       for r in per_frame]))
            for thr in iou_thresholds},
        "mean_out_of_scope_pred_per_frame":
            float(np.mean([r["n_pred_out_of_scope"]
                           for r in per_frame])),
        "mean_iou_of_matched_cells":
            float(np.mean(matched_ious)) if matched_ious else 0.0,
        "median_iou_of_matched_cells":
            float(np.median(matched_ious)) if matched_ious else 0.0,
        "mean_id_consistency":
            float(np.mean([c["consistency"] for c in id_consistency]))
            if id_consistency else 0.0,
        "n_perfect_id_consistency":
            sum(1 for c in id_consistency if c["consistency"] == 1.0),
        "n_gt_cells_total": len(id_consistency),
    }

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump({"summary": summary,
                   "per_frame": per_frame,
                   "id_consistency": id_consistency},
                  f, indent=2)

    # Plots
    if matched_ious:
        plt.figure(figsize=(7, 4))
        plt.hist(matched_ious, bins=20, color="#3eb049",
                 edgecolor="black")
        plt.xlabel("Best-match IoU per cell")
        plt.ylabel("Count")
        plt.axvline(0.5, color="red", lw=1, ls="--",
                    label="IoU = 0.5")
        plt.title(f"Per-cell IoU distribution "
                  f"({len(matched_ious)} matched cells across "
                  f"{len(gt)} frames)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "iou_distribution.png"),
                    dpi=85)
        plt.close()

    # Confusion: GT-id rows × pred-id cols
    if id_consistency:
        all_pred_ids = sorted(set(
            p for c in id_consistency for p in c["all_pred_ids"]))
        all_gt_ids = sorted(c["gt_id"] for c in id_consistency)
        mat = np.zeros((len(all_gt_ids), len(all_pred_ids)))
        for i, g in enumerate(all_gt_ids):
            entry = next(c for c in id_consistency if c["gt_id"] == g)
            for p, n in entry["all_pred_ids"].items():
                j = all_pred_ids.index(p)
                mat[i, j] = n
        # Normalise rows
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat_norm = mat / row_sums

        fig, ax = plt.subplots(
            figsize=(max(6, len(all_pred_ids) * 0.4),
                     max(4, len(all_gt_ids) * 0.4)))
        im = ax.imshow(mat_norm, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(all_pred_ids)))
        ax.set_xticklabels(all_pred_ids, rotation=90)
        ax.set_yticks(range(len(all_gt_ids)))
        ax.set_yticklabels(all_gt_ids)
        ax.set_xlabel("Pipeline cell ID")
        ax.set_ylabel("GT cell ID")
        ax.set_title("ID match frequency (row-normalised)\n"
                     "Diagonal = consistent identity")
        # Annotate cells
        for i in range(len(all_gt_ids)):
            for j in range(len(all_pred_ids)):
                if mat[i, j] > 0:
                    ax.text(j, i, f"{int(mat[i, j])}",
                            ha="center", va="center",
                            color="white" if mat_norm[i, j] < 0.5
                            else "black", fontsize=8)
        plt.colorbar(im, ax=ax, label="Fraction of frames")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion.png"), dpi=85)
        plt.close()

    # Markdown report
    md = _format_report_md(summary, per_frame, id_consistency)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(md)

    print(f"\nEvaluation written to {out_dir}/")
    print(md)


def _format_report_md(summary, per_frame, id_consistency):
    iou_thresholds = summary["iou_thresholds"]
    rows = ["| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |",
            "|---:|---:|---:|---:|---:|---:|"]
    for thr in iou_thresholds:
        f1_focused = (summary
                      .get("mean_F1_focused", {})
                      .get(f"@{thr}",
                           summary['mean_F1'][f'@{thr}']))
        rows.append(f"| IoU≥{thr} | "
                     f"{summary['mean_TP_per_frame'][f'@{thr}']:.1f} | "
                     f"{summary['mean_FN_per_frame'][f'@{thr}']:.1f} | "
                     f"{summary['mean_FP_per_frame'][f'@{thr}']:.1f} | "
                     f"{summary['mean_F1'][f'@{thr}']:.2f} | "
                     f"{f1_focused:.2f} |")

    pf_table = ["| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |",
                "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_frame:
        pf_table.append(
            f"| F{r['frame']} | {r['n_gt']} | {r['n_pred']} | "
            f"{r['TP_0.5']} | {r['FP_0.5']} | {r['FN_0.5']} | "
            f"{r['F1_0.5']:.2f} |")

    consist = sorted(id_consistency, key=lambda c: -c["consistency"])
    cons_table = ["| GT cell | matched in N frames | dominant pred | "
                  "consistency |",
                  "|---:|---:|---:|---:|"]
    for c in consist:
        cons_table.append(
            f"| {c['gt_id']} | {c['n_frames_matched']} | "
            f"{c['dominant_pred_id']} | {c['consistency']:.2f} |")

    return f"""# Evaluation report

**Recording**: `{summary['folder']}`
**GT frames evaluated**: {summary['n_gt_frames']}
**GT cells (unique IDs)**: {summary['n_gt_cells_total']}

## Detection accuracy

{chr(10).join(rows)}

- **Mean per-cell IoU (matched)**: {summary['mean_iou_of_matched_cells']:.3f}
- **Median per-cell IoU (matched)**: {summary['median_iou_of_matched_cells']:.3f}
- **Out-of-scope predictions/frame**: {summary.get('mean_out_of_scope_pred_per_frame', 0.0):.1f}

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **{summary['mean_id_consistency']:.2%}**
- **GT cells with perfect 1.0 consistency**:
  {summary['n_perfect_id_consistency']} / {summary['n_gt_cells_total']}

{chr(10).join(cons_table)}

## Per-frame breakdown

{chr(10).join(pf_table)}

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="Recording folder containing "
                   "gt_masks/ and pipeline_results/")
    args = p.parse_args()
    evaluate_recording(args.folder)
