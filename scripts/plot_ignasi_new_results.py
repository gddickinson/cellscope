"""Post-processor for run_ignasi_new_full.py output.

Generates two figures aggregating across all .npz caches in
results/ignasi_new_full/:

  comparison.png     box+strip plots of speed/persistence/lifetime/
                     n_tracks per condition, with statistical tests
                     (Mann-Whitney for 2-group, Kruskal-Wallis for 3+)
  overlay_grid.png   3-frame overlay strip per recording showing
                     tracked cells coloured by track ID

Both can be re-run any time without recomputing the pipeline.
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PIXEL_SIZE_UM = 0.6523
INTERVAL_MIN = 10.0
SPEED_CAP = 15.0
ROOT_DEFAULT = "results/ignasi_new_full"


def per_track_metrics(stack):
    """Return dict of (mean_speed, persistence, lifetime, mean_area)
    or None if track too short."""
    from core.tracking import extract_centroids
    cents = extract_centroids(stack)
    valid = ~np.isnan(cents[:, 0])
    if valid.sum() < 3:
        return None
    cv = cents[valid] * PIXEL_SIZE_UM
    d = np.diff(cv, axis=0)
    speeds = np.linalg.norm(d, axis=1) / INTERVAL_MIN
    speeds = speeds[speeds <= SPEED_CAP]
    if len(speeds) == 0:
        return None
    total_path = float(np.sum(np.linalg.norm(d, axis=1)))
    net_disp = float(np.linalg.norm(cv[-1] - cv[0]))
    pers = net_disp / total_path if total_path > 0 else 0.0
    areas = stack[valid].sum(axis=(1, 2)).astype(float) * (PIXEL_SIZE_UM ** 2)
    return {
        "mean_speed": float(np.mean(speeds)),
        "persistence": float(pers),
        "lifetime_frames": int(valid.sum()),
        "mean_area_um2": float(np.mean(areas)),
    }


def collect_all(root):
    rows = []
    npzs = sorted(glob.glob(os.path.join(root, "*.npz")))
    for npz in npzs:
        stem = os.path.basename(npz).replace(".npz", "")
        try:
            pos, cond = stem.split("_", 1)
        except ValueError:
            continue
        z = np.load(npz, allow_pickle=False)
        for i in range(500):
            key = f"track_{i}_stack"
            if key not in z.files:
                continue
            stack = z[key]
            m = per_track_metrics(stack)
            if m is None:
                continue
            m.update({"position": pos, "condition": cond,
                      "track_id": i})
            rows.append(m)
    return rows


def stats_for_groups(groups):
    """Return dict {comparison_str: (p, test_name)}.
    For 2 groups: Mann-Whitney U. For 3+: Kruskal-Wallis (omnibus only).
    """
    from scipy import stats as ss
    out = {}
    items = sorted(groups.items())
    if len(items) >= 3:
        try:
            stat, p = ss.kruskal(*[v for _, v in items])
            out["overall"] = (float(p), "Kruskal-Wallis")
        except Exception:
            pass
    if len(items) == 2:
        try:
            stat, p = ss.mannwhitneyu(items[0][1], items[1][1],
                                       alternative="two-sided")
            out[f"{items[0][0]} vs {items[1][0]}"] = (
                float(p), "Mann-Whitney U")
        except Exception:
            pass
    return out


def plot_comparison(rows, out_path):
    if not rows:
        print("[plot_comparison] no rows")
        return
    metrics = [
        ("mean_speed", "Mean speed (µm/min)"),
        ("persistence", "Persistence ratio"),
        ("lifetime_frames", "Lifetime (frames)"),
        ("mean_area_um2", "Mean area (µm²)"),
    ]
    by_cond = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, _ in metrics:
            by_cond[k][r["condition"]].append(r[k])
    n_metric = len(metrics)
    fig, axes = plt.subplots(1, n_metric, figsize=(4 * n_metric, 5))
    for ax, (key, label) in zip(axes, metrics):
        groups = by_cond[key]
        cond_order = sorted(groups.keys())
        data = [groups[c] for c in cond_order]
        bp = ax.boxplot(data, labels=cond_order, patch_artist=True,
                        widths=0.55, showfliers=False)
        cmap = plt.cm.Set2(np.linspace(0, 1, len(cond_order)))
        for patch, color in zip(bp["boxes"], cmap):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for i, vals in enumerate(data):
            x = np.full(len(vals), i + 1) + np.random.uniform(
                -0.1, 0.1, len(vals))
            ax.scatter(x, vals, s=12, color="black", alpha=0.5)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.2, axis="y")
        # Statistics annotation
        s = stats_for_groups(groups)
        if s:
            txt_lines = []
            for name, (p, test) in s.items():
                star = ("***" if p < 0.001
                        else "**" if p < 0.01
                        else "*" if p < 0.05
                        else "n.s.")
                txt_lines.append(f"{name}: p={p:.3f} {star}")
            ax.text(0.5, 0.97, "\n".join(txt_lines),
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="none", pad=2))
    fig.suptitle(
        f"Ignasi recordings — per-cell metrics by condition "
        f"(n={len(rows)} cells across {len({r['position'] for r in rows})} recordings)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_overlay_grid(root, out_path, max_recordings=None):
    """One row per recording: 3 sample frames with coloured track contours."""
    npzs = sorted(glob.glob(os.path.join(root, "*.npz")))
    if max_recordings:
        npzs = npzs[:max_recordings]
    if not npzs:
        return
    nrows = len(npzs)
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 3.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for i, npz in enumerate(npzs):
        stem = os.path.basename(npz).replace(".npz", "")
        z = np.load(npz, allow_pickle=False)
        frames = z["frames"]
        labels = z["labels"]
        n = len(frames)
        picks = [n // 10, n // 2, n - 1 - n // 10]
        for j, fi in enumerate(picks):
            ax = axes[i, j]
            ax.imshow(frames[fi], cmap="gray")
            label_frame = labels[fi]
            if label_frame.max() > 0:
                from matplotlib.colors import ListedColormap
                ncol = max(int(label_frame.max()), 1) + 1
                cmap = plt.cm.tab20(np.linspace(0, 1, ncol))
                cmap[0] = (0, 0, 0, 0)
                ax.imshow(label_frame, cmap=ListedColormap(cmap),
                          alpha=0.45)
            ax.set_title(f"{stem}  f{fi}  ({int(label_frame.max())} cells)",
                         fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--max-recordings", type=int, default=None,
                    help="cap recordings rendered into overlay_grid")
    args = ap.parse_args()

    rows = collect_all(args.root)
    if not rows:
        print(f"No track data in {args.root}/. Run "
              f"scripts/run_ignasi_new_full.py first.")
        return
    print(f"[plot] {len(rows)} cells "
          f"across {len({r['position'] for r in rows})} recordings")
    plot_comparison(rows, os.path.join(args.root, "comparison.png"))
    plot_overlay_grid(args.root,
                       os.path.join(args.root, "overlay_grid.png"),
                       max_recordings=args.max_recordings)


if __name__ == "__main__":
    main()
