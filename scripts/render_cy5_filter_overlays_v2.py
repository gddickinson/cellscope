"""Compare 5 Cy5 filter strategies side-by-side per recording.

Reads from results/ic295_full_v2/ (NPZs enriched with io_ratio,
inside_cv, fraction_positive metrics from `recompute_cy5_metrics.py`).

Per recording, renders a multi-page PDF with 6 panels per frame:
  1. DIC + Cy5
  2. Raw (all tracks)
  3. conservative_strict
  4. adaptive_loose
  5. multi_metric
  6. temporal_stability

Plus aggregate grid + summary CSV showing kept/dropped per mode.
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

CACHE_DIR = "results/ic295_full_v2"
OUT_DIR = "results/ic295_filter_overlays_v2"
SAMPLE_FRAMES = [5, 24, 48, 72, 91]
MODES = ["conservative_strict", "adaptive_loose",
         "multi_metric", "composite_score"]


def load_recording_data(npz_path):
    """Load DIC + Cy5 + per-track stacks + ALL Cy5 features
    (including the v2 metrics: io_ratio, inside_cv, fraction_positive)."""
    z = np.load(npz_path, allow_pickle=False)
    frames = z["frames"]
    cy5_frames = z["cy5_frames"] if "cy5_frames" in z.files else None
    tracks = []
    n_tracks_max = int(z["tracks_n"]) if "tracks_n" in z.files else 0
    for tid in range(n_tracks_max * 3 + 5):
        if f"track_{tid}_stack" not in z.files:
            continue
        score_arr = z[f"track_{tid}_cy5_score"] if (
            f"track_{tid}_cy5_score" in z.files) else None
        if score_arr is None:
            mean_score = 0.0
        else:
            valid = ~np.isnan(score_arr)
            mean_score = float(np.nanmean(score_arr)) if valid.any() else 0.0
        t = {
            "id": tid + 1,
            "stack": z[f"track_{tid}_stack"],
            "cy5_score": score_arr,
            "cy5_mean_score": mean_score,
        }
        for extra in ("cy5_io_ratio", "cy5_inside_cv",
                       "cy5_fraction_positive"):
            key = f"track_{tid}_{extra}"
            if key in z.files:
                t[extra] = z[key]
        tracks.append(t)
    return frames, cy5_frames, tracks


def apply_all_filters(tracks):
    from core.cy5_filter import apply_cy5_filter
    out = {}
    for mode in MODES:
        kept, dropped, info = apply_cy5_filter(tracks, mode=mode)
        out[mode] = (kept, dropped, info)
    return out


def overlay_dic_cy5(ax, dic_u8, cy5_u8, title=""):
    rgb = np.stack([dic_u8, dic_u8, dic_u8], axis=-1).astype(np.float32)
    rgb[..., 0] = np.maximum(rgb[..., 0], cy5_u8.astype(np.float32))
    ax.imshow(rgb.clip(0, 255).astype(np.uint8))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def overlay_tracks(ax, dic_u8, tracks, frame_idx, dropped=None,
                   title="", show_scores=True):
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
            ax.text(cx, cy, f"{t.get('cy5_mean_score', 0):.2f}",
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
                          filter_results, out_pdf):
    """Per-recording PDF: 5 sample frames × 6 panels each."""
    sample_frames = [f for f in SAMPLE_FRAMES if f < len(frames)]
    with PdfPages(out_pdf) as pdf:
        for fi in sample_frames:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            overlay_dic_cy5(axes[0, 0], frames[fi], cy5_frames[fi],
                             f"DIC + Cy5 — {name} f{fi}")
            overlay_tracks(axes[0, 1], frames[fi], tracks, fi,
                            title=f"Raw ({len(tracks)} tracks)")
            # Top-right: render conservative_strict (most-similar to raw)
            mode = "conservative_strict"
            kept, dropped, info = filter_results[mode]
            overlay_tracks(axes[0, 2], frames[fi], kept, fi,
                            dropped=dropped,
                            title=f"{mode} — kept {len(kept)} "
                                  f"dropped {len(dropped)}")
            # Bottom row: 3 stricter filters
            for col_idx, mode in enumerate(
                    ["adaptive_loose", "multi_metric",
                     "composite_score"]):
                kept, dropped, info = filter_results[mode]
                thresh = info.get("threshold")
                thresh_str = (f" cut={thresh:.2f}"
                              if isinstance(thresh, (int, float))
                              and thresh != 0 else "")
                overlay_tracks(axes[1, col_idx], frames[fi], kept, fi,
                                dropped=dropped,
                                title=f"{mode}{thresh_str} — kept "
                                      f"{len(kept)} dropped "
                                      f"{len(dropped)}")
            fig.suptitle(
                f"{name} f{fi} — 4 Cy5 filter strategies "
                "(solid=kept, dashed=dropped, yellow=score)",
                fontsize=10, y=1.02)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight", dpi=80)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    npzs = sorted(glob.glob(os.path.join(args.cache_dir, "Pos*.npz")))
    print(f"Rendering v2 overlays for {len(npzs)} recordings.\n",
          flush=True)

    summary_rows = []
    for npz in npzs:
        name = os.path.basename(npz).replace(".npz", "")
        print(f"  {name}…", flush=True)
        frames, cy5_frames, tracks = load_recording_data(npz)
        if cy5_frames is None or not tracks:
            continue
        filter_results = apply_all_filters(tracks)
        out_pdf = os.path.join(args.out_dir, f"{name}_review_v2.pdf")
        render_recording_pdf(name, frames, cy5_frames, tracks,
                              filter_results, out_pdf)
        row = {"recording": name, "n_tracks": len(tracks)}
        for mode in MODES:
            kept, dropped, info = filter_results[mode]
            row[f"{mode}_kept"] = len(kept)
            row[f"{mode}_dropped"] = len(dropped)
        summary_rows.append(row)

    csv_path = os.path.join(args.out_dir, "filtered_summary_v2.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    from output.run_metadata import write_run_metadata
    write_run_metadata(
        out_path=os.path.join(args.out_dir, "RUN_METADATA.md"),
        title="IC295 Cy5 filter v2 — 4-strategy comparison",
        sections={
            "Input": (
                f"`{args.cache_dir}/*.npz` (NPZ caches enriched\n"
                f"with extra Cy5 metrics by\n"
                f"`scripts/recompute_cy5_metrics.py`)."),
            "Strategies": (
                "* `conservative_strict`: tighter cut on existing\n"
                "  z-score (mean<0.10 AND p95<0.20).\n"
                "* `adaptive_loose`: bimodal-aware, requires gap\n"
                "  > 4× median AND > 0.20 absolute.\n"
                "* `multi_metric`: composite — track is REAL if\n"
                "  ≥2 of 4 cellularity criteria pass: cy5_score,\n"
                "  io_ratio, inside_cv, fraction_positive.\n"
                "* `temporal_stability`: drop tracks whose Cy5\n"
                "  signal looks like noise (lag-1 autocorr < 0.20)."),
            "Outputs": (
                "* `<pos>_<cond>_review_v2.pdf` — per-recording\n"
                "  comparison: 5 sample frames × 6 panels each.\n"
                "* `filtered_summary_v2.csv` — kept/dropped per mode.\n"
                "* `RUN_METADATA.md` — this file."),
        },
        rerun_cli=(
            f"conda run -n cellpose python "
            f"scripts/recompute_cy5_metrics.py && "
            f"conda run -n cellpose python "
            f"scripts/render_cy5_filter_overlays_v2.py"),
    )


if __name__ == "__main__":
    main()
