"""Statistical comparison plots: box/violin with significance brackets."""
import numpy as np
from matplotlib.figure import Figure


def _significance_y(ax, data_max, bracket_idx):
    """Compute y position for significance bracket."""
    return data_max * (1.05 + 0.08 * bracket_idx)


def add_significance_brackets(ax, pairs, y_base):
    """Draw significance brackets between group pairs.

    Args:
        ax: matplotlib axes
        pairs: list of (x1, x2, sig_marker) tuples
        y_base: starting y for brackets
    """
    for i, (x1, x2, sig) in enumerate(pairs):
        if sig == "ns":
            continue
        y = y_base * (1.05 + 0.08 * i)
        ax.plot([x1, x1, x2, x2], [y * 0.98, y, y, y * 0.98],
                "k-", lw=0.8)
        ax.text((x1 + x2) / 2, y, sig, ha="center", va="bottom",
                fontsize=9, fontweight="bold")


def plot_group_boxplot(fig, groups, metric_name, stats_result=None):
    """Box plot with individual data points and significance brackets.

    Args:
        fig: matplotlib Figure
        groups: dict[group_name → list of values]
        metric_name: y-axis label
        stats_result: output from core.statistics.group_comparison()
    """
    fig.clear()
    ax = fig.add_subplot(111)
    names = list(groups.keys())
    data = [np.asarray(v) for v in groups.values()]
    data = [d[~np.isnan(d)] for d in data]

    bp = ax.boxplot(data, labels=names, patch_artist=True,
                    widths=0.5, showfliers=False)
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63",
              "#9C27B0", "#00BCD4"]
    for i, patch in enumerate(bp["boxes"]):
        c = colors[i % len(colors)]
        patch.set_facecolor(c + "40")
        patch.set_edgecolor(c)

    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.06, len(d))
        ax.scatter(x, d, alpha=0.6, s=20,
                   color=colors[i % len(colors)], zorder=3,
                   edgecolors="none")

    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by Group")
    ax.grid(alpha=0.2, axis="y")

    if stats_result and stats_result.get("pairwise"):
        all_vals = np.concatenate([d for d in data if len(d)])
        y_max = all_vals.max() if len(all_vals) else 1
        pairs = []
        for pw in stats_result["pairwise"]:
            if pw["group_a"] in names and pw["group_b"] in names:
                x1 = names.index(pw["group_a"]) + 1
                x2 = names.index(pw["group_b"]) + 1
                pairs.append((x1, x2, pw["sig"]))
        if pairs:
            add_significance_brackets(ax, pairs, y_max)

    fig.tight_layout()


def plot_group_violin(fig, groups, metric_name, stats_result=None):
    """Violin plot variant of group comparison."""
    fig.clear()
    ax = fig.add_subplot(111)
    names = list(groups.keys())
    data = [np.asarray(v) for v in groups.values()]
    data = [d[~np.isnan(d)] for d in data]
    valid = [(n, d) for n, d in zip(names, data) if len(d) >= 2]
    if not valid:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                transform=ax.transAxes)
        return

    vnames = [n for n, _ in valid]
    vdata = [d for _, d in valid]
    parts = ax.violinplot(vdata, showmeans=True, showmedians=True)
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.4)
    ax.set_xticks(range(1, len(vnames) + 1))
    ax.set_xticklabels(vnames)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by Group")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()


def plot_metric_summary(fig, group_metrics, metric_names):
    """Bar chart comparing means ± SEM across groups for multiple metrics.

    Args:
        fig: matplotlib Figure
        group_metrics: dict[group_name → dict[metric_name → value]]
        metric_names: list of metric names to plot
    """
    fig.clear()
    groups = list(group_metrics.keys())
    n_metrics = len(metric_names)
    n_groups = len(groups)
    if not n_metrics or not n_groups:
        return
    ax = fig.add_subplot(111)
    x = np.arange(n_metrics)
    width = 0.8 / n_groups
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
    for i, grp in enumerate(groups):
        vals = [group_metrics[grp].get(m, 0) for m in metric_names]
        offset = (i - n_groups / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=grp, color=colors[i % len(colors)], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("Group Comparison")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
