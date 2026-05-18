"""Render clean overlays of FILTERED results (kept cells only).

Reads from results/ic295_filtered/ (output of
`apply_cy5_filter_to_results.py`). For each filtered NPZ, renders:
  * a multi-page PDF of 6 sample frames showing ONLY the kept tracks
    (colored contours + numeric IDs over the DIC + Cy5 composite)
  * a track-trajectory page showing all kept track centroids over time

Plus an aggregate summary PNG: 1 sample frame per recording with
kept tracks overlaid, organized by condition (WT/KO/GOF/OT/Y1/DMSO).

Output: results/ic295_filtered/overlays/
"""
import argparse
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

CACHE_DIR = "results/ic295_filtered"
OUT_DIR = "results/ic295_filtered/overlays"
SAMPLE_FRAMES = [5, 24, 48, 72, 91]


def load_filtered_npz(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    frames = z["frames"]
    cy5 = z["cy5_frames"]
    n_kept = int(z["filter_n_kept"]) if "filter_n_kept" in z.files else 0
    n_raw = int(z["filter_n_raw"]) if "filter_n_raw" in z.files else 0
    tracks = []
    for tid in range(n_kept * 3 + 5):
        if f"track_{tid}_stack" not in z.files:
            continue
        score = z.get(f"track_{tid}_cy5_score")
        valid = ~np.isnan(score) if score is not None else None
        ms = (float(np.nanmean(score))
               if score is not None and valid.any() else 0.0)
        tracks.append({
            "id": tid + 1,
            "stack": z[f"track_{tid}_stack"],
            "centroids": z.get(f"track_{tid}_centroids"),
            "cy5_mean_score": ms,
            "io_ratio": (float(np.nanmean(z[f"track_{tid}_cy5_io_ratio"]))
                          if f"track_{tid}_cy5_io_ratio" in z.files
                          else 0.0),
            "frac_pos": (float(np.nanmean(
                z[f"track_{tid}_cy5_fraction_positive"]))
                if f"track_{tid}_cy5_fraction_positive" in z.files
                else 0.0),
        })
    return frames, cy5, tracks, n_raw


def overlay_frame(ax, dic_u8, cy5_u8, tracks, frame_idx, title=""):
    rgb = np.stack([dic_u8] * 3, axis=-1).astype(np.float32)
    rgb[..., 0] = np.maximum(rgb[..., 0], cy5_u8.astype(np.float32))
    ax.imshow(rgb.clip(0, 255).astype(np.uint8))
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(tracks), 1)))
    for ti, t in enumerate(tracks):
        m = t["stack"][frame_idx].astype(bool)
        if not m.any():
            continue
        color = cmap[ti % len(cmap)]
        ax.contour(m, levels=[0.5], colors=[color], linewidths=1.2)
        yy, xx = np.where(m)
        cy, cx = yy.mean(), xx.mean()
        ax.text(cx, cy, f"{t['id']}",
                color="white", ha="center", va="center", fontsize=7,
                bbox=dict(facecolor=color, alpha=0.7,
                          edgecolor="none", pad=1))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def render_recording_pdf(name, frames, cy5_frames, tracks, n_raw,
                          out_pdf):
    sample_frames = [f for f in SAMPLE_FRAMES if f < len(frames)]
    with PdfPages(out_pdf) as pdf:
        # Per-frame pages
        for fi in sample_frames:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            axes[0].imshow(frames[fi], cmap="gray")
            axes[0].set_title(f"DIC — {name} f{fi}", fontsize=9)
            axes[0].axis("off")
            overlay_frame(
                axes[1], frames[fi], cy5_frames[fi], tracks, fi,
                f"Filtered (Cy5 multi_metric): kept "
                f"{len(tracks)}/{n_raw} tracks — frame {fi}")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight", dpi=80)
            plt.close(fig)
        # Trajectories page
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(frames[len(frames) // 2], cmap="gray", alpha=0.4)
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(tracks), 1)))
        for ti, t in enumerate(tracks):
            cents = t.get("centroids")
            if cents is None:
                continue
            valid = ~np.isnan(cents[:, 0])
            if valid.sum() < 2:
                continue
            ys = cents[valid, 0]
            xs = cents[valid, 1]
            color = cmap[ti % len(cmap)]
            ax.plot(xs, ys, "-", color=color, alpha=0.8, linewidth=1.5,
                     label=f"#{t['id']} score={t['cy5_mean_score']:.2f}")
            ax.plot(xs[0], ys[0], "o", color=color, markersize=8)
            ax.plot(xs[-1], ys[-1], "s", color=color, markersize=8)
        ax.set_title(
            f"{name}: kept-cell trajectories (○ start, □ end)\n"
            f"{len(tracks)} of {n_raw} raw tracks kept "
            f"(filter: multi_metric)",
            fontsize=10)
        ax.axis("off")
        if len(tracks) <= 20:
            ax.legend(loc="upper right", fontsize=7,
                      bbox_to_anchor=(1.3, 1.0))
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight", dpi=80)
        plt.close(fig)


def render_aggregate(per_rec, out_png):
    """Group recordings by condition, show 1 sample frame each."""
    by_cond = {}
    for name, payload in per_rec.items():
        cond = name.split("_")[-1]
        by_cond.setdefault(cond, []).append((name, payload))

    cond_order = ["WT", "KO", "GOF", "OT", "Y1", "DMSO"]
    cond_order = [c for c in cond_order if c in by_cond]
    n_rows = max(len(by_cond[c]) for c in cond_order)
    n_cols = len(cond_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols,
                                                          4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    for col_idx, cond in enumerate(cond_order):
        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            if row_idx < len(by_cond[cond]):
                name, (frames, cy5, tracks, n_raw) = by_cond[cond][row_idx]
                fi = min(48, len(frames) - 1)
                overlay_frame(ax, frames[fi], cy5[fi], tracks, fi,
                                f"{cond}: {name.replace('_'+cond,'')} "
                                f"({len(tracks)}/{n_raw})")
            else:
                ax.axis("off")
    fig.suptitle("IC295 filtered cells by condition (multi_metric, "
                 "1 sample frame per recording)", fontsize=12, y=1.0)
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
    print(f"Rendering filtered overlays for {len(npzs)} recordings.\n",
          flush=True)

    per_rec = {}
    for npz in npzs:
        name = os.path.basename(npz).replace(".npz", "")
        print(f"  {name}…", flush=True)
        frames, cy5, tracks, n_raw = load_filtered_npz(npz)
        if not tracks:
            print(f"    skip (0 tracks after filter)", flush=True)
            continue
        out_pdf = os.path.join(args.out_dir, f"{name}_filtered.pdf")
        render_recording_pdf(name, frames, cy5, tracks, n_raw, out_pdf)
        per_rec[name] = (frames, cy5, tracks, n_raw)

    if per_rec:
        agg_path = os.path.join(args.out_dir,
                                  "by_condition_aggregate.png")
        print(f"\nRendering aggregate by condition…", flush=True)
        render_aggregate(per_rec, agg_path)
        print(f"Wrote {agg_path}", flush=True)

    from output.run_metadata import write_run_metadata
    write_run_metadata(
        out_path=os.path.join(args.out_dir, "RUN_METADATA.md"),
        title="IC295 filtered overlays (kept cells only)",
        sections={
            "Source": (
                f"`{args.cache_dir}/*.npz` — filtered NPZs from\n"
                f"`apply_cy5_filter_to_results.py --filter-mode "
                f"multi_metric`."),
            "Per-recording PDFs": (
                "5 sample frames + 1 trajectories page each.\n"
                "Each frame shows DIC + Cy5 composite (Cy5 in red)\n"
                "with kept tracks colored + numbered."),
            "Aggregate": (
                "`by_condition_aggregate.png` — 1 sample frame per\n"
                "recording, grouped by condition (WT/KO/GOF/OT/Y1/DMSO).\n"
                "Useful for at-a-glance comparison of cell counts and\n"
                "morphologies across conditions."),
        },
        rerun_cli=(
            f"conda run -n cellpose python "
            f"scripts/render_filtered_overlays.py \\\n"
            f"    --cache-dir {args.cache_dir} \\\n"
            f"    --out-dir {args.out_dir}"),
    )


if __name__ == "__main__":
    main()
