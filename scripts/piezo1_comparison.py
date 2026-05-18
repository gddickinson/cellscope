"""Piezo1 WT vs KO comparison across all recordings.

Aggregates per-cell metrics from results/full_dataset/*.npz, computed
over all kept tracks (those present ≥ 50% of the recording window).
For each genotype × modality pair, reports:

  • Per-cell migration speed (rolling-mean smoothed, 15 µm/min cap)
  • Persistence ratio (net / path)
  • MSD diffusion exponent (alpha) — log-log slope of MSD vs lag
  • Mean cell area (µm²)

Statistical tests on each metric:
  • Two-sided Mann–Whitney U (non-parametric — small samples, no
    normality assumption)
  • Cohen's d effect size

Outputs:
  results/piezo1_comparison/comparison_DIC.png
  results/piezo1_comparison/comparison_phase.png
  results/piezo1_comparison/comparison.csv     — per-cell metrics
  results/piezo1_comparison/results.md         — Markdown summary

Usage:
  conda run -n cellpose python scripts/piezo1_comparison.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

CACHE_DIR = "results/full_dataset"
OUT = "results/piezo1_comparison"
os.makedirs(OUT, exist_ok=True)

UM_PER_PX = 0.65
DT_MIN = 5.0
SPEED_CAP = 15.0
TRACK_KEEP_FRACTION = 0.5


# ──────────────────────────────────────────────────────────────────────
# Per-cell metric extraction
# ──────────────────────────────────────────────────────────────────────
def smoothed_speeds(centroids, um_per_px, dt_min, cap=SPEED_CAP, win=3):
    valid = ~np.isnan(centroids[:, 0])
    smooth = centroids.copy()
    half = win // 2
    for i in range(len(centroids)):
        if valid[i]:
            chunks = []
            for k in range(-half, half + 1):
                j = i + k
                if 0 <= j < len(centroids) and valid[j]:
                    chunks.append(centroids[j])
            if chunks:
                smooth[i] = np.mean(chunks, axis=0)
    speeds = np.full(len(centroids), np.nan)
    for i in range(1, len(centroids)):
        if valid[i] and valid[i - 1]:
            d = (smooth[i] - smooth[i - 1]) * um_per_px
            v = float(np.linalg.norm(d) / dt_min)
            if v <= cap:
                speeds[i] = v
    return speeds


def msd_alpha(centroids, um_per_px, dt_min, max_lag=None):
    """Diffusion exponent: slope of log(MSD) vs log(lag)."""
    valid = ~np.isnan(centroids[:, 0])
    if valid.sum() < 5:
        return np.nan
    cs = centroids[valid] * um_per_px
    n_lag = max_lag or min(20, len(cs) // 3)
    if n_lag < 3:
        return np.nan
    msd = np.zeros(n_lag)
    for k in range(1, n_lag + 1):
        disp = cs[k:] - cs[:-k]
        msd[k - 1] = np.mean(np.sum(disp ** 2, axis=1))
    lags = np.arange(1, n_lag + 1) * dt_min
    if (msd > 0).sum() < 3:
        return np.nan
    ok = msd > 0
    return float(np.polyfit(np.log(lags[ok]), np.log(msd[ok]), 1)[0])


def per_cell_metrics(stack, centroids, um_per_px, dt_min):
    """Return dict of per-cell metrics for one track."""
    valid = ~np.isnan(centroids[:, 0])
    if valid.sum() < 3:
        return None
    cs_um = centroids[valid] * um_per_px
    diffs = np.diff(cs_um, axis=0)
    path = float(np.sum(np.linalg.norm(diffs, axis=1)))
    net = float(np.linalg.norm(cs_um[-1] - cs_um[0]))
    persistence = net / path if path > 0 else 0.0

    speeds = smoothed_speeds(centroids, um_per_px, dt_min)
    speeds = speeds[~np.isnan(speeds)]

    a = stack.astype(bool).sum(axis=(1, 2)) * (um_per_px ** 2)
    a = a[a > 0]
    return {
        "n_frames_present": int(valid.sum()),
        "mean_speed": float(np.mean(speeds)) if len(speeds) else 0.0,
        "median_speed": float(np.median(speeds)) if len(speeds) else 0.0,
        "persistence": float(persistence),
        "msd_alpha": msd_alpha(centroids, um_per_px, dt_min),
        "mean_area_um2": float(np.mean(a)) if len(a) else 0.0,
        "path_length_um": path,
        "net_displacement_um": net,
    }


def collect_metrics():
    """Walk all .npz cache files, extract per-cell metrics."""
    rows = []
    for fn in sorted(os.listdir(CACHE_DIR)):
        if not fn.endswith(".npz"):
            continue
        path = os.path.join(CACHE_DIR, fn)
        stem = fn[:-4]
        if "_" not in stem:
            continue
        modality, *_ = stem.split("_", 1)
        # naming: dic_pos0_wt / phase_pos17_ko
        # genotype is "wt" or "ko" suffix
        geno = "WT" if "_wt" in stem else (
            "KO" if "_ko" in stem else "?")
        if geno == "?":
            continue
        z = np.load(path, allow_pickle=False)
        if "tracks_n" not in z.files:
            continue
        n_frames = z["frames"].shape[0]
        for i in range(50):
            if f"track_{i}_stack" not in z.files:
                continue
            s = z[f"track_{i}_stack"]
            cents = z[f"track_{i}_centroids"]
            n_present = int(s.any(axis=(1, 2)).sum())
            if n_present < TRACK_KEEP_FRACTION * n_frames:
                continue
            m = per_cell_metrics(s, cents, UM_PER_PX, DT_MIN)
            if m is None:
                continue
            m.update({
                "recording": stem,
                "modality": modality,
                "genotype": geno,
                "track_id": i,
            })
            rows.append(m)
    return rows


# ──────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────
def cohen_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) +
                      (len(b) - 1) * np.var(b, ddof=1))
                     / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def compare_metric(rows, modality, metric, label):
    """Return dict with {wt_n, ko_n, wt_mean, ko_mean, p_value, d}."""
    wt = [r[metric] for r in rows
          if r["modality"] == modality and r["genotype"] == "WT"
          and not np.isnan(r[metric])]
    ko = [r[metric] for r in rows
          if r["modality"] == modality and r["genotype"] == "KO"
          and not np.isnan(r[metric])]
    if len(wt) < 2 or len(ko) < 2:
        return None
    u, p = stats.mannwhitneyu(wt, ko, alternative="two-sided")
    d = cohen_d(wt, ko)
    return {
        "metric": metric, "label": label,
        "wt_n": len(wt), "ko_n": len(ko),
        "wt_mean": float(np.mean(wt)), "wt_std": float(np.std(wt, ddof=1)),
        "wt_median": float(np.median(wt)),
        "ko_mean": float(np.mean(ko)), "ko_std": float(np.std(ko, ddof=1)),
        "ko_median": float(np.median(ko)),
        "p_value": float(p),
        "cohen_d": d,
        "wt": wt, "ko": ko,
    }


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────
def plot_comparison(modality, results, out_path):
    """4-panel: speed / persistence / msd_alpha / area."""
    panels = ["mean_speed", "persistence", "msd_alpha", "mean_area_um2"]
    titles = {
        "mean_speed": "Mean speed (µm/min)",
        "persistence": "Persistence ratio (net / path)",
        "msd_alpha": "MSD diffusion exponent α",
        "mean_area_um2": "Mean cell area (µm²)",
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, key in zip(axes.flat, panels):
        r = next((r for r in results if r["metric"] == key), None)
        if r is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(titles[key])
            ax.set_xticks([]); ax.set_yticks([])
            continue
        bp = ax.boxplot([r["wt"], r["ko"]],
                        labels=[f"WT (n={r['wt_n']})",
                                f"KO (n={r['ko_n']})"],
                        patch_artist=True, widths=0.55,
                        showfliers=False)
        for patch, c in zip(bp["boxes"], ["#5e9ed6", "#d65e5e"]):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        rng = np.random.default_rng(0)
        for x, vals in zip([1, 2], [r["wt"], r["ko"]]):
            xs = rng.normal(x, 0.05, len(vals))
            ax.scatter(xs, vals, color="black", s=18, alpha=0.55,
                       zorder=5)

        all_v = list(r["wt"]) + list(r["ko"])
        if all_v:
            ymax = max(all_v) * 1.15
            ax.plot([1, 1, 2, 2],
                    [ymax, ymax * 1.04, ymax * 1.04, ymax],
                    "k-", linewidth=1)
            stars = ("***" if r["p_value"] < 0.001
                     else "**" if r["p_value"] < 0.01
                     else "*" if r["p_value"] < 0.05
                     else "n.s.")
            ax.text(1.5, ymax * 1.06,
                    f"{stars} (p={r['p_value']:.3f}, d={r['cohen_d']:+.2f})",
                    ha="center", fontsize=9)

        ax.set_title(titles[key], fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Piezo1 WT vs KO — {modality.upper()} cohort",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    rows = collect_metrics()
    if not rows:
        print("No cached results found. Run scripts/run_full_dataset.py "
              "first.")
        sys.exit(1)

    print(f"Collected metrics from {len(rows)} kept tracks across "
          f"{len({r['recording'] for r in rows})} recordings:")
    for mod in sorted({r["modality"] for r in rows}):
        for geno in ("WT", "KO"):
            sub = [r for r in rows if r["modality"] == mod
                   and r["genotype"] == geno]
            print(f"  {mod} {geno}: {len(sub)} cells from "
                  f"{len({r['recording'] for r in sub})} recordings")

    # Write per-cell CSV
    fields = ["recording", "modality", "genotype", "track_id",
              "n_frames_present", "mean_speed", "median_speed",
              "persistence", "msd_alpha", "mean_area_um2",
              "path_length_um", "net_displacement_um"]
    with open(f"{OUT}/comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {OUT}/comparison.csv ({len(rows)} cells)")

    # Plot + stats per modality
    md_lines = ["# Piezo1 WT vs KO — comparison\n",
                f"*{len(rows)} kept cells across "
                f"{len({r['recording'] for r in rows})} recordings*\n"]
    metrics = [("mean_speed", "Mean speed"),
               ("persistence", "Persistence"),
               ("msd_alpha", "MSD α"),
               ("mean_area_um2", "Mean area")]

    for mod in sorted({r["modality"] for r in rows}):
        print(f"\n=== {mod} cohort ===")
        results = []
        for key, label in metrics:
            r = compare_metric(rows, mod, key, label)
            if r:
                results.append(r)
                stars = ("***" if r["p_value"] < 0.001
                         else "**" if r["p_value"] < 0.01
                         else "*" if r["p_value"] < 0.05
                         else "")
                print(f"  {label:18s} WT={r['wt_mean']:7.3f} ± "
                      f"{r['wt_std']:.3f}  "
                      f"KO={r['ko_mean']:7.3f} ± {r['ko_std']:.3f}  "
                      f"p={r['p_value']:.3f}  d={r['cohen_d']:+.2f}  "
                      f"{stars}")

        out_png = f"{OUT}/comparison_{mod}.png"
        plot_comparison(mod, results, out_png)
        print(f"  saved {out_png}")

        md_lines.append(f"\n## {mod.upper()} cohort\n")
        md_lines.append(f"![](comparison_{mod}.png)\n")
        md_lines.append("| Metric | WT (mean ± std) | KO (mean ± std) | p | Cohen's d |")
        md_lines.append("|---|---:|---:|---:|---:|")
        for r in results:
            md_lines.append(
                f"| {r['label']} | "
                f"{r['wt_mean']:.3f} ± {r['wt_std']:.3f} (n={r['wt_n']}) | "
                f"{r['ko_mean']:.3f} ± {r['ko_std']:.3f} (n={r['ko_n']}) | "
                f"{r['p_value']:.3f} | {r['cohen_d']:+.2f} |")

    md_path = f"{OUT}/results.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
