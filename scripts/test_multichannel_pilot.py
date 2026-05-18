"""Pilot run of the multichannel pipeline on real IC295 frames.

Runs cpsam(DIC) → Cy5 filter on a small set of frames. Compares
DIC-only baseline vs DIC + Cy5 AND-fusion. Renders side-by-side
overlays so the user can judge whether the filter dropped only
debris or also lost real cells.

Wait until the IC293 full run has finished before running this — it
uses the same cellpose4 GPU (cpsam).

Output: results/ic295_pilot/
  pilot_results.csv     per-frame, per-config row (cell counts, scores)
  overlays/<rec>_f<N>_<config>.png   DIC + masks per frame/config
  comparison_grid.png   one row per frame: DIC | Cy5 | DIC-only | filtered
  RUN_METADATA.md       reproducibility info
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

OUT_DIR = "results/ic295_pilot"
OVERLAY_DIR = os.path.join(OUT_DIR, "overlays")

DEFAULT_RECORDINGS = [
    "Pos14-KO",   # multiple cells with varied intensities
    "Pos26-GOF",  # GOF morphology test
    "Pos0-WT",    # WT baseline (1-2 cells)
]
DEFAULT_FRAMES = [5, 48, 91]


def overlay_with_masks(ax, image_uint8, mask_int32, scores=None,
                       title=""):
    ax.imshow(image_uint8, cmap="gray")
    if mask_int32 is not None and mask_int32.max() > 0:
        ncol = max(int(mask_int32.max()), 1) + 1
        cmap = plt.cm.tab20(np.linspace(0, 1, ncol))
        cmap[0] = (0, 0, 0, 0)
        ax.imshow(mask_int32, cmap=ListedColormap(cmap), alpha=0.45)
        # Label each cell with its ID and Cy5 score
        for lab in range(1, int(mask_int32.max()) + 1):
            yy, xx = np.where(mask_int32 == lab)
            if len(yy) == 0:
                continue
            cy, cx = yy.mean(), xx.mean()
            txt = f"{lab}"
            if scores and lab in scores:
                txt += f"\n{scores[lab]:.2f}"
            ax.text(cx, cy, txt, color="yellow", ha="center", va="center",
                    fontsize=7,
                    bbox=dict(facecolor="black", alpha=0.5,
                              edgecolor="none", pad=1))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/Volumes/GeorgeDrive/ignasi/IC295")
    ap.add_argument("--recordings", nargs="+",
                    default=DEFAULT_RECORDINGS)
    ap.add_argument("--frames", nargs="+", type=int,
                    default=DEFAULT_FRAMES)
    ap.add_argument("--min-score", type=float, default=0.3,
                    help="min Cy5 presence score to keep a mask")
    ap.add_argument("--cy5-recovery", action="store_true",
                    help="enable Cy5-flagged false-negative recovery: "
                         "for each Cy5+ region without a DIC mask, "
                         "crop and re-run cpsam to recover the cell")
    ap.add_argument("--recovery-tta", action="store_true",
                    default=True,
                    help="(if --cy5-recovery) use TTA on the small "
                         "recovery crop (cheap; default on)")
    args = ap.parse_args()

    os.makedirs(OVERLAY_DIR, exist_ok=True)

    from cellpose import models
    from core.multichannel import (
        to_uint8_dic, to_uint8_fluorescence,
        score_all_labels, filter_dic_labels_by_cy5,
    )
    from core.cy5_fallbacks import recover_missed_cells_via_dic_crop

    print(f"Loading cpsam model (cellpose 4)…")
    cpsam = models.CellposeModel(gpu=True)

    rows = []
    grid_panels = []  # list of (title_row, dic, cy5, dic_masks_overlay, filt_overlay, scores)
    t_start = time.time()
    for rec in args.recordings:
        tif = next((p for p in glob.glob(
                       os.path.join(args.src, "*.ome.tif"))
                    if rec in os.path.basename(p)), None)
        if not tif:
            print(f"  [skip] no .ome.tif matching {rec}")
            continue
        with tifffile.TiffFile(tif) as tf:
            for fi in args.frames:
                cy5_raw = tf.pages[fi * 3 + 0].asarray()
                dic_raw = tf.pages[fi * 3 + 1].asarray()
                dic_u8 = to_uint8_dic(dic_raw)
                cy5_u8 = to_uint8_fluorescence(cy5_raw)
                t0 = time.time()
                masks_raw, _, _ = cpsam.eval(dic_u8)
                elapsed = time.time() - t0
                masks_raw = masks_raw.astype(np.int32)
                # Optional recovery before scoring/filtering
                n_recovered = 0
                if args.cy5_recovery:
                    t_rec = time.time()
                    masks_raw, n_recovered, _ = (
                        recover_missed_cells_via_dic_crop(
                            dic_u8, masks_raw, cy5_u8, cpsam,
                            use_tta=args.recovery_tta))
                    elapsed += (time.time() - t_rec)
                scores = score_all_labels(masks_raw, cy5_u8)
                filtered, _, kept = filter_dic_labels_by_cy5(
                    masks_raw, cy5_u8, min_score=args.min_score)
                n_raw = int(masks_raw.max())
                n_drop = n_raw - kept
                print(f"  {rec} f{fi}: cpsam {n_raw} cells "
                      f"(+{n_recovered} recovered) in {elapsed:.1f}s, "
                      f"Cy5 filter kept {kept}, dropped {n_drop}")
                rows.append({
                    "recording": rec,
                    "frame": fi,
                    "n_dic": n_raw,
                    "n_kept": kept,
                    "n_dropped": n_drop,
                    "n_cy5_recovered": n_recovered,
                    "min_score_threshold": args.min_score,
                    "scores": scores,
                    "elapsed_s": round(elapsed, 1),
                })
                grid_panels.append((rec, fi, dic_u8, cy5_u8,
                                     masks_raw, filtered, scores))
    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {elapsed_total/60:.1f} min")

    # Comparison grid: 1 row per frame, 4 cols
    if grid_panels:
        n = len(grid_panels)
        fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
        for i, (rec, fi, dic_u8, cy5_u8, raw, filt, sc) in enumerate(grid_panels):
            axes[i, 0].imshow(dic_u8, cmap="gray")
            axes[i, 0].set_title(f"{rec} f{fi} — DIC", fontsize=10)
            axes[i, 0].axis("off")
            axes[i, 1].imshow(cy5_u8, cmap="gray")
            axes[i, 1].set_title("Cy5 (SiR-actin)", fontsize=10)
            axes[i, 1].axis("off")
            overlay_with_masks(axes[i, 2], dic_u8, raw, sc,
                                f"DIC-only ({int(raw.max())} cells)")
            overlay_with_masks(axes[i, 3], dic_u8, filt, sc,
                                f"AND-filter ({int(filt.max())} kept)")
        fig.suptitle(
            f"IC295 multichannel pilot — score threshold = "
            f"{args.min_score}, dropped count = "
            f"{sum(r['n_dropped'] for r in rows)}/"
            f"{sum(r['n_dic'] for r in rows)} DIC detections",
            fontsize=12)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, "comparison_grid.png")
        fig.savefig(out, dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")

    # Per-frame CSV
    csv_path = os.path.join(OUT_DIR, "pilot_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording", "frame", "n_dic", "n_kept",
                    "n_dropped", "min_score_threshold",
                    "score_distribution", "elapsed_s"])
        for r in rows:
            score_str = ";".join(f"{lab}:{s:.2f}"
                                 for lab, s in sorted(r["scores"].items()))
            w.writerow([r["recording"], r["frame"], r["n_dic"],
                        r["n_kept"], r["n_dropped"],
                        r["min_score_threshold"], score_str,
                        r["elapsed_s"]])
    print(f"Wrote {csv_path}")

    # Run metadata
    from output.run_metadata import write_run_metadata
    write_run_metadata(
        out_path=os.path.join(OUT_DIR, "RUN_METADATA.md"),
        title="IC295 multichannel pilot — cpsam(DIC) + Cy5 filter",
        sections={
            "Source": (
                f"`{args.src}` — IC295 dataset (3-channel: Cy5 SiR-actin,\n"
                f"DIC 10x, None). 19 recordings × 97 frames × 3 channels\n"
                f"× 2048×2048 uint16. Pixel size 0.6523 µm/px,\n"
                f"interval 10 min."),
            "Recordings tested":
                "\n".join(f"* {r}" for r in args.recordings),
            "Frames tested":
                "\n".join(f"* frame {fi}" for fi in args.frames),
            "Pipeline": (
                "1. Load DIC (ch 1) → flat-field σ=80 → uint8 p1/p99\n"
                "2. Load Cy5 (ch 0) → uint8 p1/p99.5 (no flat-field)\n"
                "3. cpsam (cellpose 4 base, no fine-tune) on DIC\n"
                "4. For each DIC mask: compute Cy5 presence score\n"
                "   (z-score of inside p75 vs local 30-px annulus\n"
                "   median, robust MAD denominator, mapped to 0-1)\n"
                f"5. Drop masks with score < {args.min_score}"),
            "Outputs": (
                "* `pilot_results.csv` — per-frame counts + score distributions\n"
                "* `comparison_grid.png` — 4-panel per frame: DIC, Cy5,\n"
                "  raw masks, filtered masks (with score labels)\n"
                "* `overlays/` — individual overlay PNGs\n"
                "* `RUN_METADATA.md` — this file"),
        },
        rerun_cli=(
            f"conda run -n cellpose4 python "
            f"scripts/test_multichannel_pilot.py \\\n"
            f"    --src {args.src} \\\n"
            f"    --recordings {' '.join(args.recordings)} \\\n"
            f"    --frames {' '.join(str(f) for f in args.frames)} \\\n"
            f"    --min-score {args.min_score}"),
        timing_seconds={
            "total_pilot_run": elapsed_total,
        },
        extra_notes=(
            "If the AND-filter drops cells that the user (looking at\n"
            "comparison_grid.png) confirms are real cells, lower\n"
            "`--min-score` (try 0.15) and re-run. If debris is still\n"
            "kept, raise it (try 0.5)."),
    )
    print(f"Wrote {OUT_DIR}/RUN_METADATA.md")


if __name__ == "__main__":
    main()
