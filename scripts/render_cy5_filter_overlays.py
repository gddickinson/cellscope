"""Render side-by-side overlays comparing Cy5 filter strategies.

For each recording in results/ic295_full/, samples 6 frames spread
across the timeline and renders a multi-page PDF with 4 panels per
frame:
  1. DIC + Cy5 (Cy5 in red overlay)
  2. Raw — all tracks colored by ID with score labels
  3. Conservative filter (kept solid, dropped dashed gray)
  4. Adaptive filter (kept solid, dropped dashed gray)

Plus an aggregate single-page grid showing 1 sample frame per
recording (19 rows × 3 cols: DIC | DIC+Cy5 | adaptive-filtered).

Output: results/ic295_filter_overlays/
  <pos>_<cond>_review.pdf      per-recording multi-page review
  aggregate_grid.png           one sample frame per recording
  filtered_summary.csv         kept/dropped per filter mode
  RUN_METADATA.md
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports
setup_imports()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

CACHE_DIR = "results/ic295_full"
OUT_DIR = "results/ic295_filter_overlays"
SAMPLE_FRAMES = [5, 24, 48, 72, 91]


def load_recording_data(npz_path):
    """Load DIC + Cy5 + per-track stacks + per-track Cy5 features."""
    z = np.load(npz_path, allow_pickle=False)
    frames = z["frames"]
    cy5_frames = z["cy5_frames"] if "cy5_frames" in z.files else None
    tracks = []
    for tid in range(int(z["tracks_n"]) * 3 + 5):
        if f"track_{tid}_stack" not in z.files:
            continue
        score_arr = z.get(f"track_{tid}_cy5_score") if (
            f"track_{tid}_cy5_score" in z.files) else None
        if score_arr is None:
            mean_score = 0.0
            p95_score = 0.0
        else:
            valid = ~np.isnan(score_arr)
            mean_score = float(np.nanmean(score_arr)) if valid.any() else 0.0
            p95_score = (float(np.nanpercentile(score_arr, 95))
                         if valid.any() else 0.0)
        tracks.append({
            "id": tid + 1,
            "stack": z[f"track_{tid}_stack"],
            "cy5_score": score_arr,
            "cy5_mean_score": mean_score,
            "cy5_p95_score": p95_score,
        })
    return frames, cy5_frames, tracks


def apply_filters(tracks):
    """Run conservative + adaptive filters; return dict of results."""
    from core.cy5_filter import (
        conservative_filter, adaptive_filter)
    cons_kept, cons_drop, cons_info = conservative_filter(tracks)
    adap_kept, adap_drop, adap_info = adaptive_filter(tracks)
    return {
        "conservative": (cons_kept, cons_drop, cons_info),
        "adaptive": (adap_kept, adap_drop, adap_info),
    }


def overlay_dic_cy5(ax, dic_u8, cy5_u8, title=""):
    """DIC grayscale with Cy5 in red overlay."""
    rgb = np.stack([dic_u8, dic_u8, dic_u8], axis=-1).astype(np.float32)
    rgb[..., 0] = np.maximum(rgb[..., 0], cy5_u8.astype(np.float32))
    ax.imshow(rgb.clip(0, 255).astype(np.uint8))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def overlay_tracks(ax, dic_u8, tracks, frame_idx, dropped=None,
                   title="", show_scores=True):
    """Show DIC + colored track contours on this frame.
    `dropped` (if given) tracks are drawn as dashed gray outlines."""
    ax.imshow(dic_u8, cmap="gray")
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(tracks), 1)))
    for ti, t in enumerate(tracks):
        m = t["stack"][frame_idx].astype(bool)
        if not m.any():
            continue
        ax.contour(m, levels=[0.5], colors=[cmap[ti % len(cmap)]],
                    linewidths=1.0, alpha=0.9)
        if show_scores:
            yy, xx = np.where(m)
            cy, cx = yy.mean(), xx.mean()
            ax.text(cx, cy, f"{t['cy5_mean_score']:.2f}",
                    color="yellow", ha="center", va="center",
                    fontsize=6,
                    bbox=dict(facecolor="black", alpha=0.5,
                              edgecolor="none", pad=0.5))
    if dropped:
        for t in dropped:
            m = t["stack"][frame_idx].astype(bool)
            if not m.any():
                continue
            ax.contour(m, levels=[0.5], colors=["gray"],
                        linewidths=0.7, alpha=0.5, linestyles="--")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def render_recording_pdf(name, frames, cy5_frames, tracks,
                          filters, out_pdf):
    """Per-recording multi-page PDF: 6 frames × 4 panels."""
    cons_kept, cons_drop, cons_info = filters["conservative"]
    adap_kept, adap_drop, adap_info = filters["adaptive"]
    sample_frames = [f for f in SAMPLE_FRAMES if f < len(frames)]
    with PdfPages(out_pdf) as pdf:
        for fi in sample_frames:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
            overlay_dic_cy5(axes[0], frames[fi], cy5_frames[fi],
                             f"DIC + Cy5 — {name} f{fi}")
            overlay_tracks(axes[1], frames[fi], tracks, fi,
                            title=f"Raw ({len(tracks)} tracks)")
            overlay_tracks(axes[2], frames[fi], cons_kept, fi,
                            dropped=cons_drop,
                            title=f"Conservative — kept "
                                  f"{len(cons_kept)}, dropped "
                                  f"{len(cons_drop)}")
            adap_thresh = adap_info.get("threshold", "?")
            adap_thresh_str = (f"{adap_thresh:.2f}"
                                if isinstance(adap_thresh, (int, float))
                                else str(adap_thresh))
            overlay_tracks(axes[3], frames[fi], adap_kept, fi,
                            dropped=adap_drop,
                            title=f"Adaptive (cut {adap_thresh_str}, "
                                  f"{adap_info.get('result', '?')}) — "
                                  f"kept {len(adap_kept)}, dropped "
                                  f"{len(adap_drop)}")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight", dpi=80)
            plt.close(fig)


def render_aggregate_grid(per_rec, out_png):
    """One sample frame per recording, 3 cols (DIC | Cy5 | adaptive)."""
    n = len(per_rec)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, (name, payload) in enumerate(sorted(per_rec.items())):
        frames = payload["frames"]
        cy5 = payload["cy5"]
        tracks = payload["tracks"]
        adap_kept, adap_drop, adap_info = payload["adaptive"]
        fi = min(48, len(frames) - 1)  # mid-timeline
        axes[i, 0].imshow(frames[fi], cmap="gray")
        axes[i, 0].set_title(f"{name} f{fi}: DIC", fontsize=8)
        axes[i, 0].axis("off")
        overlay_dic_cy5(axes[i, 1], frames[fi], cy5[fi],
                          f"{name}: DIC + Cy5")
        adap_thresh = adap_info.get("threshold", "?")
        adap_thresh_str = (f"{adap_thresh:.2f}"
                            if isinstance(adap_thresh, (int, float))
                            else str(adap_thresh))
        overlay_tracks(axes[i, 2], frames[fi], adap_kept, fi,
                        dropped=adap_drop,
                        title=f"{name}: adaptive cut {adap_thresh_str} "
                              f"(kept {len(adap_kept)} of "
                              f"{len(tracks)})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=70, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    npzs = sorted(glob.glob(os.path.join(args.cache_dir, "Pos*.npz")))
    print(f"Rendering overlays for {len(npzs)} recordings.\n")

    summary_rows = []
    per_rec = {}
    for npz in npzs:
        name = os.path.basename(npz).replace(".npz", "")
        print(f"  {name}…", flush=True)
        frames, cy5_frames, tracks = load_recording_data(npz)
        if cy5_frames is None or not tracks:
            print(f"    skip ({len(tracks)} tracks, "
                  f"cy5={'yes' if cy5_frames is not None else 'no'})")
            continue
        filters = apply_filters(tracks)
        cons_kept, cons_drop, cons_info = filters["conservative"]
        adap_kept, adap_drop, adap_info = filters["adaptive"]

        out_pdf = os.path.join(args.out_dir, f"{name}_review.pdf")
        render_recording_pdf(name, frames, cy5_frames, tracks,
                              filters, out_pdf)

        summary_rows.append({
            "recording": name,
            "n_tracks": len(tracks),
            "conservative_kept": len(cons_kept),
            "conservative_dropped": len(cons_drop),
            "adaptive_kept": len(adap_kept),
            "adaptive_dropped": len(adap_drop),
            "adaptive_threshold": round(
                adap_info.get("threshold", 0.0), 3),
            "adaptive_classification": adap_info.get("result", "?"),
        })
        per_rec[name] = {
            "frames": frames, "cy5": cy5_frames, "tracks": tracks,
            "conservative": filters["conservative"],
            "adaptive": filters["adaptive"],
        }

    # Aggregate grid (free up memory by closing per_rec frames after)
    print("\nRendering aggregate grid…")
    grid_png = os.path.join(args.out_dir, "aggregate_grid.png")
    render_aggregate_grid(per_rec, grid_png)
    print(f"Wrote {grid_png}")

    # Summary CSV
    csv_path = os.path.join(args.out_dir, "filtered_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # RUN_METADATA
    from output.run_metadata import write_run_metadata
    write_run_metadata(
        out_path=os.path.join(args.out_dir, "RUN_METADATA.md"),
        title="IC295 Cy5 filter overlays + comparison",
        sections={
            "Input": (
                f"`{args.cache_dir}/*.npz` (per-recording NPZ caches\n"
                f"from `scripts/run_ignasi_ic295_full.py`).\n"
                f"Each NPZ has DIC frames, Cy5 frames, per-track stacks\n"
                f"and per-frame Cy5 features (mean/p75/p95/score)."),
            "Filter strategies compared": (
                "* **Conservative**: drop tracks with NO Cy5 signal\n"
                "  (mean_score < 0.05 AND p95_score < 0.10).\n"
                "* **Adaptive**: per-recording bimodal-aware cut.\n"
                "  If sorted-score gaps show a clear separation\n"
                "  (max gap > 2× median, > 0.10 absolute), cut at\n"
                "  the gap midpoint. Otherwise fall back to the\n"
                "  conservative hard floor."),
            "Sampling": (
                f"Per-recording PDF: 5 sample frames {SAMPLE_FRAMES} "
                f"(skipped if recording shorter)."),
            "Outputs": (
                "* `<pos>_<cond>_review.pdf` — per-recording multi-page\n"
                "  side-by-side comparison (DIC+Cy5, raw, conservative,\n"
                "  adaptive).\n"
                "* `aggregate_grid.png` — single-page summary, 1 sample\n"
                "  frame per recording.\n"
                "* `filtered_summary.csv` — kept/dropped counts per mode.\n"
                "* `RUN_METADATA.md` — this file."),
            "How to read the overlays": (
                "* Yellow score labels = track-mean Cy5 score.\n"
                "* Solid colored contours = kept by that filter.\n"
                "* Dashed gray contours = dropped by that filter.\n"
                "* Visual review goal: decide if adaptive's drops\n"
                "  are real cells (use conservative) or debris\n"
                "  (use adaptive)."),
        },
        rerun_cli=(
            f"conda run -n cellpose python "
            f"scripts/render_cy5_filter_overlays.py \\\n"
            f"    --cache-dir {args.cache_dir} \\\n"
            f"    --out-dir {args.out_dir}"),
    )


if __name__ == "__main__":
    main()
