"""VAMPIRE shape-mode plot functions for the analysis tab.

Split out of analysis_plots.py to keep that file under the
500-line limit. Imported back into the GRAPH_REGISTRY there.
Each function follows the same `(fig, result, **_)` contract as
the other plot functions; the unused `**_` swallows kwargs the
shared dispatcher passes (e.g. ``gap_interp_max``).
"""
import numpy as np
from matplotlib.figure import Figure


_VAMPIRE_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63",
                   "#9C27B0", "#00BCD4", "#FF5722", "#607D8B"]


def plot_vampire_modes(fig: Figure, result: dict, **_):
    """VAMPIRE shape mode scatter (PC1 vs PC2, colored by mode)."""
    fig.clear()
    ax = fig.add_subplot(111)
    vamp = result.get("vampire")
    if vamp is None:
        ax.text(0.5, 0.5, "Run VAMPIRE analysis first\n"
                "(enable in Analysis params)",
                ha="center", va="center", transform=ax.transAxes)
        return
    pc = vamp["modes"]["principal_components"]
    ids = vamp["modes"]["cluster_ids"]
    n_cl = vamp["n_clusters"]
    for k in range(n_cl):
        mask = ids == k
        if mask.any():
            ax.scatter(pc[mask, 0], pc[mask, 1], s=15, alpha=0.6,
                       label=f"Mode {k+1} ({mask.sum()})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"VAMPIRE Shape Modes ({n_cl} clusters, "
                 f"{len(ids)} contours)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()


def plot_vampire_distribution(fig: Figure, result: dict, **_):
    """VAMPIRE shape mode frequency histogram."""
    fig.clear()
    ax = fig.add_subplot(111)
    vamp = result.get("vampire")
    if vamp is None:
        ax.text(0.5, 0.5, "No VAMPIRE data", ha="center",
                va="center", transform=ax.transAxes)
        return
    h = vamp["heterogeneity"]
    fracs = h["mode_fractions"]
    n = len(fracs)
    ax.bar(range(1, n+1), fracs,
           color=[_VAMPIRE_COLORS[i % len(_VAMPIRE_COLORS)]
                  for i in range(n)])
    ax.set_xlabel("Shape Mode")
    ax.set_ylabel("Fraction of Frames")
    ax.set_title(f"Shape Mode Distribution "
                 f"(H={h['entropy']:.2f} / {h['max_entropy']:.2f})")
    ax.set_xticks(range(1, n+1))
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()


def plot_vampire_timeseries(fig: Figure, result: dict, **_):
    """VAMPIRE shape mode assignment over time."""
    fig.clear()
    ax = fig.add_subplot(111)
    vamp = result.get("vampire")
    if vamp is None:
        ax.text(0.5, 0.5, "No VAMPIRE data", ha="center",
                va="center", transform=ax.transAxes)
        return
    fm = vamp["frame_modes"]
    valid = ~np.isnan(fm)
    for k in range(vamp["n_clusters"]):
        mask = valid & (fm == k)
        if mask.any():
            ax.scatter(np.where(mask)[0], fm[mask] + 1, s=20,
                       color=_VAMPIRE_COLORS[k % len(_VAMPIRE_COLORS)],
                       label=f"Mode {k+1}", zorder=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Shape Mode")
    ax.set_title("Shape Mode Over Time")
    ax.set_yticks(range(1, vamp["n_clusters"] + 1))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    fig.tight_layout()


def plot_vampire_eigenshapes(fig: Figure, result: dict, **_):
    """VAMPIRE mean shape + top eigenshape variations."""
    fig.clear()
    vamp = result.get("vampire")
    if vamp is None:
        return
    mean_c = vamp["modes"]["mean_contour"]
    n_pts = len(mean_c) // 2
    mx, my = mean_c[:n_pts], mean_c[n_pts:]
    pd = vamp["modes"]["principal_directions"]
    ev = vamp["modes"]["explained_variance"]

    n_show = min(4, len(ev))
    for i in range(n_show):
        ax = fig.add_subplot(1, n_show, i + 1)
        dx, dy = pd[:n_pts, i], pd[n_pts:, i]
        scale = 0.3
        ax.fill(mx, my, alpha=0.15, color="gray")
        ax.plot(mx, my, "k-", lw=0.5)
        ax.plot(mx + scale*dx, my + scale*dy, "r-", lw=1, label="+")
        ax.plot(mx - scale*dx, my - scale*dy, "b-", lw=1, label="-")
        ax.set_title(f"PC{i+1} ({ev[i]*100:.0f}%)", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Eigenshape Variations", fontsize=11)
    fig.tight_layout()
