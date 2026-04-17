"""Statistical comparison engine for inter-group analysis.

Compares metrics across treatment groups (identified by folder name).
Auto-selects appropriate tests based on group count:
  2 groups:  t-test + Mann-Whitney U + Cohen's d
  3+ groups: one-way ANOVA + Kruskal-Wallis + pairwise Bonferroni
"""
import numpy as np
from scipy import stats


def significance_marker(p):
    """Return significance marker for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def cohens_d(a, b):
    """Cohen's d effect size between two groups."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def _group_stats(values):
    """Compute per-group summary statistics."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0, "n": 0,
                "median": 0.0, "values": []}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "sem": float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "n": n,
        "median": float(np.median(arr)),
        "values": arr.tolist(),
    }


def group_comparison(groups, metric_name="metric"):
    """Compare a metric across treatment groups.

    Args:
        groups: dict mapping group_name → list of float values
                (one value per recording in that group)
        metric_name: human-readable name for the metric

    Returns:
        dict with keys:
            metric_name: str
            group_stats: dict[group_name → {mean, std, sem, n, median}]
            tests: list of {test_name, statistic, p_value, sig}
            pairwise: list of {group_a, group_b, test, p_value, sig,
                               effect_size}
            n_groups: int
    """
    names = list(groups.keys())
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    arrays = [a[~np.isnan(a)] for a in arrays]
    n_groups = len(names)

    result = {
        "metric_name": metric_name,
        "group_stats": {n: _group_stats(a) for n, a in zip(names, arrays)},
        "tests": [],
        "pairwise": [],
        "n_groups": n_groups,
    }

    valid = [(n, a) for n, a in zip(names, arrays) if len(a) >= 2]
    if len(valid) < 2:
        return result

    if n_groups == 2:
        a, b = valid[0][1], valid[1][1]
        t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
        result["tests"].append({
            "test_name": "Welch's t-test",
            "statistic": float(t_stat),
            "p_value": float(t_p),
            "sig": significance_marker(t_p),
        })
        try:
            u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
            result["tests"].append({
                "test_name": "Mann-Whitney U",
                "statistic": float(u_stat),
                "p_value": float(u_p),
                "sig": significance_marker(u_p),
            })
        except ValueError:
            pass
        d = cohens_d(a, b)
        result["pairwise"].append({
            "group_a": valid[0][0],
            "group_b": valid[1][0],
            "test": "Welch's t-test",
            "p_value": float(t_p),
            "sig": significance_marker(t_p),
            "effect_size": float(d),
        })
    else:
        valid_arrays = [a for _, a in valid]
        valid_names = [n for n, _ in valid]
        f_stat, f_p = stats.f_oneway(*valid_arrays)
        result["tests"].append({
            "test_name": "One-way ANOVA",
            "statistic": float(f_stat),
            "p_value": float(f_p),
            "sig": significance_marker(f_p),
        })
        try:
            h_stat, h_p = stats.kruskal(*valid_arrays)
            result["tests"].append({
                "test_name": "Kruskal-Wallis",
                "statistic": float(h_stat),
                "p_value": float(h_p),
                "sig": significance_marker(h_p),
            })
        except ValueError:
            pass
        n_pairs = len(valid_names) * (len(valid_names) - 1) // 2
        for i in range(len(valid_names)):
            for j in range(i + 1, len(valid_names)):
                a, b = valid_arrays[i], valid_arrays[j]
                _, p_raw = stats.ttest_ind(a, b, equal_var=False)
                p_corr = min(1.0, p_raw * n_pairs)
                d = cohens_d(a, b)
                result["pairwise"].append({
                    "group_a": valid_names[i],
                    "group_b": valid_names[j],
                    "test": "t-test (Bonferroni)",
                    "p_value": float(p_corr),
                    "p_raw": float(p_raw),
                    "sig": significance_marker(p_corr),
                    "effect_size": float(d),
                })

    return result


def format_comparison_text(result):
    """Format a group_comparison result as human-readable text."""
    lines = [f"Metric: {result['metric_name']}",
             f"Groups: {result['n_groups']}", ""]
    for name, gs in result["group_stats"].items():
        lines.append(f"  {name}: {gs['mean']:.4f} +/- {gs['sem']:.4f} "
                     f"(n={gs['n']})")
    lines.append("")
    for t in result["tests"]:
        lines.append(f"  {t['test_name']}: "
                     f"stat={t['statistic']:.4f}, "
                     f"p={t['p_value']:.4g} {t['sig']}")
    if result["pairwise"]:
        lines.append("\nPairwise:")
        for pw in result["pairwise"]:
            lines.append(f"  {pw['group_a']} vs {pw['group_b']}: "
                         f"p={pw['p_value']:.4g} {pw['sig']} "
                         f"(d={pw['effect_size']:.3f})")
    return "\n".join(lines)
