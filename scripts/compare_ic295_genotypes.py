"""Per-metric genotype comparison from the aggregated IC295 corpus.

Reads `corpus_per_cell.csv` (output of
`scripts/aggregate_ic295_batch.py`), groups by genotype, and produces:

  - A violin plot per metric in <out>/<metric>_violin.png
  - A statistics table per metric in <out>/<metric>_stats.{md,json}
    (ANOVA + Kruskal-Wallis + Bonferroni-corrected pairwise tests
    via core/statistics.py::group_comparison)
  - <out>/REPORT.md — single-page summary with all violins inline +
    significant-pair callouts

Usage:
  python scripts/compare_ic295_genotypes.py \\
      --csv ~/iCloud/.../aggregated/corpus_per_cell.csv \\
      --out ~/iCloud/.../aggregated/figures
"""
import os, sys, csv, json, argparse
from collections import defaultdict


METRICS = [
    ("mean_speed_um_per_min", "Migration speed (µm/min)"),
    ("diffusion_D",           "MSD diffusion coefficient"),
    ("persistence",           "Directional persistence"),
    ("mean_area_um2",         "Cell area (µm²)"),
    ("mean_circularity",      "Circularity"),
    ("mean_protrusion_rate",  "Protrusion rate (µm/min)"),
]
GENOTYPE_ORDER = ["WT", "KO", "GOF", "OT", "Y1", "DMSO", "unknown"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cells", type=int, default=3,
                    help="Skip genotypes with fewer than N cells")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load
    by_genotype = defaultdict(lambda: defaultdict(list))
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            g = r["genotype"]
            for k, _ in METRICS:
                try:
                    v = float(r.get(k, "nan"))
                    if v == v:  # not NaN
                        by_genotype[g][k].append(v)
                except (ValueError, TypeError):
                    pass

    # Filter sparse genotypes
    present = [g for g in GENOTYPE_ORDER
                if g in by_genotype and len(
                    by_genotype[g].get(METRICS[0][0], [])) >= args.min_cells]
    if len(present) < 2:
        print(f"Need ≥2 genotypes with ≥{args.min_cells} cells; "
              f"have {present}")
        sys.exit(2)

    # Stats + plots
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plots")
        sys.exit(2)
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from core.statistics import group_comparison

    report_lines = ["# IC295 corpus — genotype comparison\n",
                    f"Genotypes (≥{args.min_cells} cells each): "
                    f"{', '.join(present)}\n"]

    for metric, pretty in METRICS:
        groups = {g: by_genotype[g][metric] for g in present
                   if len(by_genotype[g][metric]) >= args.min_cells}
        if len(groups) < 2:
            continue
        # Stats
        try:
            stats = group_comparison(groups, metric_name=metric)
        except Exception as e:
            stats = {"error": str(e)}
        stats_json = os.path.join(args.out, f"{metric}_stats.json")
        with open(stats_json, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        # Violin
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(groups)), 5))
        data = [groups[g] for g in present if g in groups]
        labels = [g for g in present if g in groups]
        parts = ax.violinplot(data, showmedians=True, widths=0.7)
        for body in parts["bodies"]:
            body.set_alpha(0.6)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(pretty)
        ax.set_title(f"{pretty} — IC295 corpus\n"
                      f"N cells: {dict(zip(labels, [len(d) for d in data]))}")
        ax.grid(alpha=0.3)
        # P-value annotation if available
        pv_main = stats.get("p_anova") or stats.get("p_kruskal")
        if pv_main is not None:
            ax.text(0.02, 0.95,
                    f"Omnibus p = {pv_main:.3g}",
                    transform=ax.transAxes, fontsize=9, va="top")
        fig.tight_layout()
        out_png = os.path.join(args.out, f"{metric}_violin.png")
        fig.savefig(out_png, dpi=120); plt.close(fig)

        # Append to report
        report_lines.append(f"\n## {pretty}\n")
        report_lines.append(f"![{metric}]({metric}_violin.png)\n")
        report_lines.append("```json")
        report_lines.append(json.dumps(stats, indent=2, default=str))
        report_lines.append("```\n")

    report_path = os.path.join(args.out, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote {len(METRICS)} violins + stats tables to {args.out}/")
    print(f"Wrote summary: {report_path}")


if __name__ == "__main__":
    main()
