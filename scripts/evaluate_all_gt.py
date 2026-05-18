"""Evaluate pipeline against GT on every folder that has both
gt_masks/ and pipeline_results/. Produces individual reports +
an aggregate summary at data/gt_evaluation_summary.md.

Usage:
  python scripts/evaluate_all_gt.py

Scans data/ic295_gt_full/* and data/legacy_gt/* for folders with
labelled gt_masks/ and pipeline_results/masks.npz, runs the existing
evaluate_against_gt.evaluate_recording on each, then aggregates the
report.json files into a single markdown table.
"""
import os
import sys
import json
import glob
import logging
import numpy as np

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("eval_all")

ROOTS = ["data/ic295_gt_full", "data/legacy_gt"]
OUT_FILE = "data/gt_evaluation_summary.md"


def find_evaluable_folders():
    out = []
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for sub in sorted(os.listdir(root)):
            folder = os.path.join(root, sub)
            if not os.path.isdir(folder):
                continue
            gt = os.path.join(folder, "gt_masks")
            pipe = os.path.join(folder, "pipeline_results", "masks.npz")
            if not os.path.isdir(gt):
                continue
            n_masks = sum(1 for f in os.listdir(gt)
                          if f.endswith(".png"))
            if n_masks == 0:
                continue
            if not os.path.exists(pipe):
                log.info("Skipping %s — no pipeline_results/masks.npz",
                         folder)
                continue
            out.append(folder)
    return out


def main():
    from scripts.evaluate_against_gt import evaluate_recording

    folders = find_evaluable_folders()
    if not folders:
        log.info("No evaluable folders found")
        return

    log.info("Found %d folders to evaluate:", len(folders))
    for f in folders:
        log.info("  %s", f)

    summaries = []
    for folder in folders:
        log.info("\n=== Evaluating %s ===", folder)
        try:
            evaluate_recording(folder)
        except Exception as e:
            log.error("FAILED %s: %s", folder, e)
            continue
        rep_path = os.path.join(folder, "evaluation", "report.json")
        if os.path.exists(rep_path):
            with open(rep_path) as f:
                rep = json.load(f)
            summaries.append((folder, rep["summary"]))

    # Aggregate markdown
    rows = []
    rows.append("# GT evaluation aggregate")
    rows.append("")
    rows.append(f"_{len(summaries)} recordings, "
                f"generated {os.popen('date').read().strip()}_")
    rows.append("")
    rows.append(
        "| Recording | GT frames | Mean IoU | F1@.5 | "
        "Mean TP/frame | Mean FN | Mean FP | ID consistency |"
        " Perfect tracks |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for folder, s in summaries:
        name = os.path.basename(folder)
        rows.append(
            f"| {name} | {s['n_gt_frames']} | "
            f"{s['mean_iou_of_matched_cells']:.3f} | "
            f"{s['mean_F1']['@0.5']:.2f} | "
            f"{s['mean_TP_per_frame']['@0.5']:.1f} | "
            f"{s['mean_FN_per_frame']['@0.5']:.1f} | "
            f"{s['mean_FP_per_frame']['@0.5']:.1f} | "
            f"{s['mean_id_consistency']:.2%} | "
            f"{s['n_perfect_id_consistency']}/"
            f"{s['n_gt_cells_total']} |")

    # Per-recording links
    rows.append("")
    rows.append("## Per-recording reports")
    rows.append("")
    for folder, s in summaries:
        name = os.path.basename(folder)
        rel = os.path.relpath(folder, ".")
        rows.append(f"- **{name}** — `{rel}/evaluation/report.md`")

    # Aggregates
    if summaries:
        all_iou = [s["mean_iou_of_matched_cells"] for _, s in summaries]
        all_f1 = [s["mean_F1"]["@0.5"] for _, s in summaries]
        all_consist = [s["mean_id_consistency"] for _, s in summaries]
        rows.append("")
        rows.append("## Aggregate (across recordings)")
        rows.append("")
        rows.append(f"- Mean per-cell IoU: **"
                    f"{np.mean(all_iou):.3f}** "
                    f"(across {sum(s['n_gt_frames'] for _, s in summaries)} "
                    f"annotated frames)")
        rows.append(f"- Mean F1 @ IoU≥0.5: **"
                    f"{np.mean(all_f1):.2f}**")
        rows.append(f"- Mean ID consistency: **"
                    f"{np.mean(all_consist):.2%}**")

    md = "\n".join(rows) + "\n"
    with open(OUT_FILE, "w") as f:
        f.write(md)
    log.info("\nWrote %s", OUT_FILE)
    print()
    print(md)


if __name__ == "__main__":
    main()
