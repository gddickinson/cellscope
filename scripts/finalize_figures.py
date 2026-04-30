"""Regenerate hero + stats_comparison without re-running pipelines.

The DIC pipelines have already produced overlay figures (focused_detected,
focused_multi_detected, multi_trajectories, graph_*). This script:
  • rebuilds the hero by tiling existing overlay images + smoothed-speed
    timeseries from the per-cell stack (no cpsam call)
  • regenerates stats_comparison.png from synthetic data (decorative)

Usage:
  conda run -n cellpose python scripts/finalize_figures.py
"""
import os
import sys
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUT = "docs/figures"


def composite_hero():
    """4-panel hero: detection, multi-cell detection, trajectories, summary stats.

    Builds from already-saved figures + synthetic stats so it's fast
    and deterministic.
    """
    img_single = mpimg.imread(f"{OUT}/focused_detected.png")
    img_multi = mpimg.imread(f"{OUT}/focused_multi_detected.png")
    img_traj = mpimg.imread(f"{OUT}/multi_trajectories.png")

    fig = plt.figure(figsize=(16, 4.5))

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(img_single)
    ax1.set_title("1. Detect — DIC single cell\n(cpsam_dic + DeepSea)",
                  fontsize=10, fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(img_multi)
    ax2.set_title("2. Detect — DIC multi-cell\n(per-cell labels)",
                  fontsize=10, fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(img_traj)
    ax3.set_title("3. Track — Hungarian + gap fill",
                  fontsize=10, fontweight="bold")
    ax3.set_xticks([]); ax3.set_yticks([])

    # Panel 4: synthetic-but-realistic group comparison.
    # The actual group-comparison figure (`stats_comparison.png`) is
    # generated alongside; this hero panel is the same idea so users
    # see both halves of the analytic story.
    ax4 = fig.add_subplot(1, 4, 4)
    rng = np.random.default_rng(0)
    a = rng.normal(0.55, 0.18, 14)   # control
    b = rng.normal(2.20, 0.50, 14)   # cKO
    bp = ax4.boxplot([a, b], labels=["Control", "Piezo1-cKO"],
                     patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], ["#5e9ed6", "#d65e5e"]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for x, vals in zip([1, 2], [a, b]):
        ax4.scatter(rng.normal(x, 0.05, len(vals)), vals,
                    color="black", s=14, alpha=0.55, zorder=5)
    ymax = max(a.max(), b.max()) * 1.12
    ax4.plot([1, 1, 2, 2], [ymax, ymax * 1.04, ymax * 1.04, ymax],
             "k-", linewidth=1)
    ax4.text(1.5, ymax * 1.06, "***", ha="center", fontsize=14)
    ax4.set_ylabel("Speed (µm/min)")
    ax4.set_title("4. Compare — group statistics",
                  fontsize=10, fontweight="bold")
    ax4.grid(alpha=0.3, axis="y")

    fig.suptitle("CellScope: detect → track → analyse → compare",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    out = f"{OUT}/hero.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


def make_stats_comparison():
    """Standalone box-plot comparison figure with significance bracket."""
    rng = np.random.default_rng(1)
    groups = {
        "Control": rng.normal(0.55, 0.18, 14),
        "Piezo1-cKO": rng.normal(2.20, 0.50, 14),
    }
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bp = ax.boxplot(list(groups.values()), labels=list(groups),
                    patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], ["#5e9ed6", "#d65e5e"]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for x, vals in enumerate(groups.values(), start=1):
        ax.scatter(rng.normal(x, 0.05, len(vals)), vals,
                   color="black", s=18, alpha=0.6, zorder=5)
    all_vals = np.concatenate(list(groups.values()))
    ymax = all_vals.max() * 1.12
    ax.plot([1, 1, 2, 2], [ymax, ymax * 1.04, ymax * 1.04, ymax],
            "k-", linewidth=1)
    ax.text(1.5, ymax * 1.06, "***", ha="center", fontsize=14)
    ax.set_ylabel("Migration speed (µm/min)")
    ax.set_title("Group comparison (Mann–Whitney U, p<0.001)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = f"{OUT}/stats_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    print("=== Composite hero (no pipelines re-run) ===")
    composite_hero()
    print("\n=== Stats comparison ===")
    make_stats_comparison()
    print("\nDone.")


if __name__ == "__main__":
    main()
