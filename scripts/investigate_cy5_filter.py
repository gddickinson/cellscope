"""Per-recording Cy5 score distribution analysis to design the filter.

For each .npz cache in results/ic295_full/, extracts:
  - per-track Cy5 mean / p95 score
  - track lifetime
  - bimodality test on the score distribution

Outputs:
  results/ic295_filter_investigation/
    <pos>_<cond>_dist.png   per-recording histogram + scatter
    aggregate.png           all 19 distributions on one grid
    recommended.csv         per-recording filter recommendation
    investigation.md        human-readable summary
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

CACHE_DIR = "results/ic295_full"
OUT_DIR = "results/ic295_filter_investigation"
HARD_FLOOR_MEAN = 0.05
HARD_FLOOR_P95 = 0.10
GAP_RATIO = 2.0
MIN_GAP = 0.10


def extract_track_scores(npz_path):
    """Load NPZ, return list of (track_id, mean_score, p95_score, lifetime)."""
    z = np.load(npz_path, allow_pickle=False)
    rows = []
    n_tracks = int(z["tracks_n"]) if "tracks_n" in z.files else 0
    for tid in range(n_tracks * 3 + 5):  # over-loop, breaks on missing
        score_key = f"track_{tid}_cy5_score"
        stack_key = f"track_{tid}_stack"
        if score_key not in z.files:
            continue
        score = z[score_key]
        valid = ~np.isnan(score)
        if not valid.any():
            continue
        mean = float(np.nanmean(score))
        p95 = float(np.nanpercentile(score, 95))
        lifetime = int(valid.sum())
        rows.append((tid, mean, p95, lifetime))
    return rows


def recommend_threshold(scores):
    """Bimodal-aware threshold detection.
    Returns (threshold, classification_string)."""
    if len(scores) < 5:
        return HARD_FLOOR_MEAN, "too few tracks"
    sorted_s = np.sort(np.array(scores))
    gaps = np.diff(sorted_s)
    if len(gaps) == 0:
        return HARD_FLOOR_MEAN, "single track"
    median_gap = float(np.median(gaps))
    max_gap_idx = int(np.argmax(gaps))
    max_gap = float(gaps[max_gap_idx])
    if max_gap > GAP_RATIO * max(median_gap, 1e-6) and max_gap > MIN_GAP:
        thresh = (sorted_s[max_gap_idx] + sorted_s[max_gap_idx + 1]) / 2
        return float(thresh), f"bimodal (gap {max_gap:.3f})"
    return HARD_FLOOR_MEAN, "unimodal"


def plot_per_recording(rows, name, out_path):
    if not rows:
        return
    means = np.array([r[1] for r in rows])
    p95s = np.array([r[2] for r in rows])
    lifetimes = np.array([r[3] for r in rows])
    threshold, classification = recommend_threshold(means)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(means, bins=20, color="steelblue", edgecolor="white")
    axes[0].axvline(threshold, color="red", linestyle="--",
                    label=f"recommended cut={threshold:.3f}")
    axes[0].axvline(HARD_FLOOR_MEAN, color="orange", linestyle=":",
                    label=f"hard floor={HARD_FLOOR_MEAN}")
    axes[0].set_xlabel("Track mean Cy5 score")
    axes[0].set_ylabel("# tracks")
    axes[0].set_title(f"{name} — score distribution ({classification})")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_xlim(0, max(means.max() + 0.05, 0.6))

    sc = axes[1].scatter(means, p95s, c=lifetimes, cmap="viridis",
                          s=lifetimes * 2, alpha=0.6)
    axes[1].axvline(threshold, color="red", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Mean score")
    axes[1].set_ylabel("p95 score")
    axes[1].set_title(f"{name} — mean vs p95 (size = lifetime)")
    axes[1].set_xlim(0, max(means.max() + 0.05, 0.6))
    axes[1].set_ylim(0, max(p95s.max() + 0.05, 0.7))
    plt.colorbar(sc, ax=axes[1], label="lifetime (frames)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return threshold, classification


def plot_aggregate(all_data, out_path):
    """Overlay all recording histograms on one normalized plot."""
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(all_data)))
    for i, (name, means) in enumerate(sorted(all_data.items())):
        if len(means) < 2:
            continue
        ax.hist(means, bins=20, alpha=0.4, label=name,
                color=cmap[i], histtype="stepfilled", density=True)
    ax.axvline(HARD_FLOOR_MEAN, color="black", linestyle="--",
                label=f"hard floor {HARD_FLOOR_MEAN}")
    ax.set_xlabel("Track mean Cy5 score")
    ax.set_ylabel("Density")
    ax.set_title("Per-track Cy5 mean score, all 19 IC295 recordings")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_xlim(0, 0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    npzs = sorted(glob.glob(os.path.join(args.cache_dir, "Pos*.npz")))
    print(f"Found {len(npzs)} recordings.\n")

    recommendations = []
    aggregate_means = {}

    for npz in npzs:
        name = os.path.basename(npz).replace(".npz", "")
        rows = extract_track_scores(npz)
        if not rows:
            print(f"  {name}: no tracks"); continue
        means = [r[1] for r in rows]
        out_png = os.path.join(args.out_dir, f"{name}_dist.png")
        threshold, classification = plot_per_recording(rows, name, out_png)
        n_drop_hard = sum(1 for m, p, _, _ in
                           ((r[1], r[2], r[3], r[0]) for r in rows)
                           if m < HARD_FLOOR_MEAN)
        n_drop_thresh = sum(1 for m in means if m < threshold)
        n_total = len(rows)
        aggregate_means[name] = means
        recommendations.append({
            "recording": name,
            "n_tracks": n_total,
            "score_min": round(min(means), 3),
            "score_median": round(float(np.median(means)), 3),
            "score_max": round(max(means), 3),
            "classification": classification,
            "recommended_threshold": round(threshold, 3),
            "n_drop_conservative": n_drop_hard,
            "n_drop_recommended": n_drop_thresh,
        })
        print(f"  {name}: {n_total:3d} tracks, "
              f"score median={np.median(means):.3f}, "
              f"threshold={threshold:.3f} ({classification}), "
              f"would drop conservative={n_drop_hard} / recommended={n_drop_thresh}")

    # Aggregate plot
    plot_aggregate(aggregate_means,
                    os.path.join(args.out_dir, "aggregate.png"))

    # CSV
    csv_path = os.path.join(args.out_dir, "recommended.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recommendations[0].keys()))
        w.writeheader()
        for r in recommendations:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # Markdown summary
    md_path = os.path.join(args.out_dir, "investigation.md")
    with open(md_path, "w") as f:
        f.write("# IC295 Cy5 filter investigation\n\n")
        f.write(f"19 recordings, hard floor={HARD_FLOOR_MEAN}, "
                f"bimodal gap-ratio={GAP_RATIO}, min gap={MIN_GAP}.\n\n")
        f.write("## Per-recording recommendations\n\n")
        f.write("| Recording | Tracks | Score median | Classification | "
                "Threshold | Drop (cons) | Drop (rec) |\n")
        f.write("|---|---:|---:|---|---:|---:|---:|\n")
        n_bimodal = 0
        for r in recommendations:
            if "bimodal" in r["classification"]:
                n_bimodal += 1
            f.write(f"| {r['recording']} | {r['n_tracks']} | "
                    f"{r['score_median']} | {r['classification']} | "
                    f"{r['recommended_threshold']} | "
                    f"{r['n_drop_conservative']} | "
                    f"{r['n_drop_recommended']} |\n")
        total_tracks = sum(r["n_tracks"] for r in recommendations)
        total_cons = sum(r["n_drop_conservative"] for r in recommendations)
        total_rec = sum(r["n_drop_recommended"] for r in recommendations)
        f.write(f"\n**Totals**: {total_tracks} tracks across "
                f"{len(recommendations)} recordings. "
                f"Conservative filter would drop {total_cons} "
                f"({100*total_cons/total_tracks:.1f}%); "
                f"per-recording adaptive would drop {total_rec} "
                f"({100*total_rec/total_tracks:.1f}%).\n\n")
        f.write(f"**Bimodal recordings**: {n_bimodal}/{len(recommendations)} "
                "show clear bimodal Cy5-score distribution (debris vs "
                "real cells).\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
