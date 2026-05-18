"""Compare state-stratified motility across three datasets:
  * IC295 filtered    (multi_metric Cy5 filter applied) — 19 recordings
  * IC295 unfiltered  (no Cy5 filter, all DIC detections)— 19 recordings
  * IC293             (older, no fluorescence)          — 16 recordings

For each (condition × state × metric), tabulate mean ± SE across
recordings from each dataset. Generates side-by-side bar charts and
a master comparison report.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports
setup_imports()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DATASETS = {
    "IC295 b1+2 filtered (65)":   "results/ic295_combined_state_analysis",
    "IC295 b1+2 unfiltered (65)": "results/ic295_combined_state_analysis_unfiltered",
    "IC295 batch1 only (19)":     "results/ic295_state_analysis",
    "IC293 (no fluo, 16)":        "results/ignasi_state_analysis_old",
}
OUT_DIR = "results/state_comparison"
CONDITION_ORDER = ["WT", "KO", "GOF", "OT", "Y1", "DMSO"]


def load_means_se(path):
    """Read condition_means_se.csv → dict {condition: dict of {metric: (mean, se)}}."""
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row["condition"]
            metrics = {}
            for k, v in row.items():
                if k in ("condition", "n_recordings"):
                    continue
                try:
                    metrics[k] = float(v) if v not in ("", "nan") else float("nan")
                except ValueError:
                    pass
            metrics["n_recordings"] = int(row["n_recordings"])
            out[cond] = metrics
    return out


def figure_metric_three_datasets(datasets, metric_ball, metric_atta,
                                   ylabel, title, out_png):
    """3 datasets × 2 states × 6 conditions grouped bars with SE."""
    n_ds = len(datasets)
    n_cond = len(CONDITION_ORDER)
    fig, (ax_ball, ax_atta) = plt.subplots(1, 2, figsize=(15, 5),
                                              sharey=True)

    width = 0.20
    ds_colors = {"IC295 b1+2 filtered (65)": "#CC4400",
                  "IC295 b1+2 unfiltered (65)": "#FFC080",
                  "IC295 batch1 only (19)": "#FF8800",
                  "IC293 (no fluo, 16)": "#888888"}

    for ax, metric, state in [(ax_ball, metric_ball, "balled"),
                                (ax_atta, metric_atta, "attached")]:
        x = np.arange(n_cond)
        for i, (ds_name, ds_data) in enumerate(datasets.items()):
            means, ses = [], []
            for cond in CONDITION_ORDER:
                d = ds_data.get(cond, {})
                means.append(d.get(f"{metric}_mean", float("nan")))
                ses.append(d.get(f"{metric}_se", 0.0))
            offset = (i - (n_ds - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=ses, capsize=3,
                    color=ds_colors.get(ds_name, "gray"),
                    edgecolor="black", label=ds_name)
        ax.set_xticks(x); ax.set_xticklabels(CONDITION_ORDER)
        ax.set_title(f"{state} cells")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=7)

    fig.suptitle(f"{title}\n(error bars: SE between recordings)",
                  y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def figure_state_composition_three(datasets, out_png):
    """Stacked bar: balled fraction per condition for each dataset."""
    n_ds = len(datasets)
    n_cond = len(CONDITION_ORDER)
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.20
    x = np.arange(n_cond)
    ds_colors = {"IC295 b1+2 filtered (65)": "#CC4400",
                  "IC295 b1+2 unfiltered (65)": "#FFC080",
                  "IC295 batch1 only (19)": "#FF8800",
                  "IC293 (no fluo, 16)": "#888888"}
    for i, (ds_name, ds_data) in enumerate(datasets.items()):
        means, ses = [], []
        for cond in CONDITION_ORDER:
            d = ds_data.get(cond, {})
            means.append(d.get("frac_balled_mean", float("nan")))
            ses.append(d.get("frac_balled_se", 0.0))
        offset = (i - (n_ds - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=ses, capsize=3,
                color=ds_colors.get(ds_name, "gray"),
                edgecolor="black", label=ds_name)
    ax.set_xticks(x); ax.set_xticklabels(CONDITION_ORDER)
    ax.set_ylabel("Mean fraction of cell-frames in BALLED state")
    ax.set_title("Cell state composition: % balled cells per condition\n"
                  "across the three datasets")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def write_report(datasets, out_md):
    def fmt(d, key):
        m = d.get(f"{key}_mean")
        s = d.get(f"{key}_se")
        n = d.get("n_recordings", 0)
        if m is None or np.isnan(m):
            return "—"
        return f"{m:.3f} ± {s:.3f} (n={n})"

    with open(out_md, "w") as f:
        f.write("# State-stratified motility — cross-dataset comparison\n\n")
        f.write("Three datasets analyzed with the **identical** "
                 "state-stratified pipeline:\n\n")
        f.write("| Dataset | n recordings | Cells | Notes |\n")
        f.write("|---|---:|---:|---|\n")
        for ds_name, ds_data in datasets.items():
            n_recs = sum(d["n_recordings"] for d in ds_data.values())
            f.write(f"| {ds_name} | {n_recs} | — | |\n")
        f.write("\nAll metrics: **mean ± SE between recordings**.\n\n")

        for metric_pair in [
                ("frac_balled", "frac_attached", "State composition (fractions)"),
                ("balled_speed", "attached_speed",
                  "Speed (µm/min)"),
                ("balled_persistence", "attached_persistence",
                  "Directional persistence (lag-1 cosine)"),
                ("balled_msd_50min", "attached_msd_50min",
                  "MSD at lag 50 min (µm²)"),
                ("balled_total_displacement", "attached_total_displacement",
                  "Total displacement per segment (µm)"),
                ("balled_straightness", "attached_straightness",
                  "Trajectory straightness (0-1)"),
                ("balled_circularity", "attached_circularity",
                  "Circularity (shape)"),
                ("balled_area_px", "attached_area_px",
                  "Cell area (px)"),
                ]:
            ball_key, atta_key, title = metric_pair
            f.write(f"\n## {title}\n\n")
            f.write("| Condition |")
            for ds_name in datasets:
                f.write(f" {ds_name} balled | {ds_name} attached |")
            f.write("\n|---|")
            for _ in datasets:
                f.write("---:|---:|")
            f.write("\n")
            for cond in CONDITION_ORDER:
                f.write(f"| {cond} |")
                for ds_name, ds_data in datasets.items():
                    d = ds_data.get(cond, {})
                    f.write(f" {fmt(d, ball_key)} | {fmt(d, atta_key)} |")
                f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    datasets = {}
    for ds_name, ds_path in DEFAULT_DATASETS.items():
        csv_path = os.path.join(ds_path, "condition_means_se.csv")
        if not os.path.exists(csv_path):
            print(f"  [skip] {csv_path} missing")
            continue
        datasets[ds_name] = load_means_se(csv_path)
        print(f"  loaded {ds_name}: {len(datasets[ds_name])} conditions")

    if not datasets:
        print("No datasets loaded."); return

    # Figures
    figure_state_composition_three(
        datasets, os.path.join(fig_dir, "state_composition.png"))
    figure_metric_three_datasets(
        datasets, "balled_speed", "attached_speed",
        "Mean speed (µm/min)", "Speed by state",
        os.path.join(fig_dir, "speed.png"))
    figure_metric_three_datasets(
        datasets, "balled_persistence", "attached_persistence",
        "Lag-1 cosine", "Directional persistence by state",
        os.path.join(fig_dir, "persistence.png"))
    figure_metric_three_datasets(
        datasets, "balled_msd_50min", "attached_msd_50min",
        "MSD@50min (µm²)", "MSD at 50-min lag by state",
        os.path.join(fig_dir, "msd.png"))
    figure_metric_three_datasets(
        datasets, "balled_total_displacement", "attached_total_displacement",
        "Total displacement (µm)", "Total displacement by state",
        os.path.join(fig_dir, "displacement.png"))
    figure_metric_three_datasets(
        datasets, "balled_straightness", "attached_straightness",
        "Straightness (0-1)", "Path straightness by state",
        os.path.join(fig_dir, "straightness.png"))
    figure_metric_three_datasets(
        datasets, "balled_circularity", "attached_circularity",
        "Mean circularity", "Cell circularity by state",
        os.path.join(fig_dir, "circularity.png"))
    figure_metric_three_datasets(
        datasets, "balled_area_px", "attached_area_px",
        "Cell area (px)", "Cell area by state",
        os.path.join(fig_dir, "area.png"))
    print(f"Wrote 8 figures to {fig_dir}")

    write_report(datasets, os.path.join(args.out_dir, "report.md"))
    print(f"Wrote report.md")

    from output.run_metadata import write_run_metadata
    write_run_metadata(
        out_path=os.path.join(args.out_dir, "RUN_METADATA.md"),
        title="State-stratified motility — 3-dataset comparison",
        sections={
            "Datasets": "\n".join(
                f"* `{ds_name}` ← `{path}/condition_means_se.csv`"
                for ds_name, path in DEFAULT_DATASETS.items()),
            "Outputs": (
                "* `report.md` — cross-dataset side-by-side tables.\n"
                "* `figures/` — 8 grouped bar charts (balled vs\n"
                "  attached × condition × dataset) with SE error\n"
                "  bars between recordings."),
        },
        rerun_cli=(
            "conda run -n cellpose python "
            "scripts/compare_state_datasets.py"),
    )


if __name__ == "__main__":
    main()
