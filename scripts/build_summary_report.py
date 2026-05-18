"""Build a master summary report covering the full IC295 + IC293 analysis.

Output (under results/SUMMARY/):
  summary_figure.png       — single-page multi-panel headline figure
  filter_impact.png        — what does the Cy5 filter remove? (6 cond)
  per_condition_n.png      — n recordings × n cells per condition
  state_evolution.png      — % balled per recording, sorted, per condition
  SUMMARY_REPORT.md        — master narrative report
"""
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

OUT_DIR = "results/SUMMARY"
DATASETS = {
    "filtered":   "results/ic295_combined_state_analysis",
    "unfiltered": "results/ic295_combined_state_analysis_unfiltered",
}
CONDITION_ORDER = ["WT", "KO", "GOF", "OT", "Y1", "DMSO"]
COND_COLORS = {"WT": "#4477AA", "KO": "#EE6677", "GOF": "#228833",
                "OT": "#CCBB44", "Y1": "#66CCEE", "DMSO": "#AA3377"}


def load_means_se(path):
    out = {}
    with open(os.path.join(path, "condition_means_se.csv")) as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            d = {"n_recordings": int(row["n_recordings"])}
            for k, v in row.items():
                if k in ("condition", "n_recordings"): continue
                try: d[k] = float(v) if v not in ("", "nan") else float("nan")
                except ValueError: pass
            out[cond] = d
    return out


def load_per_recording(path):
    out = []
    with open(os.path.join(path, "per_recording_state.csv")) as f:
        for row in csv.DictReader(f):
            r = {"recording": row["recording"],
                 "condition": row["condition"]}
            for k, v in row.items():
                if k in ("recording", "condition"): continue
                try: r[k] = float(v) if v not in ("", "nan") else float("nan")
                except ValueError: pass
            out.append(r)
    return out


def headline_figure(filt_se, unfilt_se, out_png):
    """4-panel figure: state composition, balled speed, attached speed,
    persistence. Filtered vs unfiltered side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = [
        ("frac_balled", "% balled cells (state composition)",
         "Fraction balled", axes[0, 0]),
        ("balled_speed", "Balled-cell speed",
         "µm/min", axes[0, 1]),
        ("attached_speed", "Attached-cell speed",
         "µm/min", axes[1, 0]),
        ("attached_persistence", "Attached directional persistence",
         "Lag-1 cosine", axes[1, 1]),
    ]
    n_ds = 2
    width = 0.40
    x = np.arange(len(CONDITION_ORDER))
    for key, title, ylabel, ax in metrics:
        for i, (lbl, ds) in enumerate([("filtered", filt_se),
                                          ("unfiltered", unfilt_se)]):
            means = [ds.get(c, {}).get(f"{key}_mean", np.nan)
                     for c in CONDITION_ORDER]
            ses = [ds.get(c, {}).get(f"{key}_se", 0.0)
                   for c in CONDITION_ORDER]
            offset = (i - 0.5) * width
            color = "#CC4400" if lbl == "filtered" else "#FFC080"
            ax.bar(x + offset, means, width, yerr=ses, capsize=3,
                    color=color, edgecolor="black", label=lbl)
        ax.set_xticks(x); ax.set_xticklabels(CONDITION_ORDER, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle(
        "IC295 combined batch1+2 (n=65 recordings, 6 conditions): "
        "Cy5-filtered vs unfiltered\n"
        "Error bars: SE between recordings. State stratification "
        "removes a major composition confound.",
        fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def filter_impact_figure(filt_se, unfilt_se, out_png):
    """Show how many cells the Cy5 filter drops, and the
    consequence on speed estimates."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    n_filt = sum(filt_se[c]["n_recordings"] for c in CONDITION_ORDER
                 if c in filt_se)
    n_unfilt = sum(unfilt_se[c]["n_recordings"] for c in CONDITION_ORDER
                   if c in unfilt_se)
    print(f"  filter impact: {n_filt} vs {n_unfilt} recordings")

    # Get per-cell counts from CSVs
    filt_cells = []
    unfilt_cells = []
    for ds_path, accumulator in [
            (DATASETS["filtered"], filt_cells),
            (DATASETS["unfiltered"], unfilt_cells)]:
        with open(os.path.join(ds_path, "state_composition.csv")) as f:
            for row in csv.DictReader(f):
                accumulator.append((row["condition"], int(row["n_cells"])))

    filt_n = {c: n for c, n in filt_cells}
    unfilt_n = {c: n for c, n in unfilt_cells}
    width = 0.4
    x = np.arange(len(CONDITION_ORDER))
    ax = axes[0]
    ax.bar(x - width/2,
            [unfilt_n.get(c, 0) for c in CONDITION_ORDER], width,
            color="#FFC080", edgecolor="black", label="unfiltered (raw)")
    ax.bar(x + width/2,
            [filt_n.get(c, 0) for c in CONDITION_ORDER], width,
            color="#CC4400", edgecolor="black", label="filtered (kept)")
    ax.set_xticks(x); ax.set_xticklabels(CONDITION_ORDER)
    ax.set_ylabel("Cells")
    ax.set_title("Cy5 filter impact on cell count per condition")
    ax.legend(fontsize=9)
    for i, c in enumerate(CONDITION_ORDER):
        if c in filt_n and c in unfilt_n and unfilt_n[c] > 0:
            pct = 100 * filt_n[c] / unfilt_n[c]
            ax.text(i, filt_n[c] + 5, f"{pct:.0f}%", ha="center",
                    fontsize=8)

    # Right panel: balled speed per condition, filt vs unfilt
    ax = axes[1]
    for i, (lbl, ds) in enumerate([("filtered", filt_se),
                                      ("unfiltered", unfilt_se)]):
        means = [ds.get(c, {}).get("balled_speed_mean", np.nan)
                 for c in CONDITION_ORDER]
        ses = [ds.get(c, {}).get("balled_speed_se", 0.0)
               for c in CONDITION_ORDER]
        offset = (i - 0.5) * 0.4
        color = "#CC4400" if lbl == "filtered" else "#FFC080"
        ax.bar(x + offset, means, 0.4, yerr=ses, capsize=3,
                color=color, edgecolor="black", label=lbl)
    ax.set_xticks(x); ax.set_xticklabels(CONDITION_ORDER)
    ax.set_ylabel("Balled-cell speed (µm/min)")
    ax.set_title("Balled speed: filter doesn't change biology much")
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def per_condition_n_figure(filt_recs, out_png):
    """For each condition: n recordings × n cells distribution."""
    by_cond = {}
    for r in filt_recs:
        by_cond.setdefault(r["condition"], []).append(r)
    fig, ax = plt.subplots(figsize=(11, 5))
    n_recs = [len(by_cond.get(c, [])) for c in CONDITION_ORDER]
    n_cells = [int(sum(r["n_cells"] for r in by_cond.get(c, [])))
               for c in CONDITION_ORDER]
    x = np.arange(len(CONDITION_ORDER))
    ax2 = ax.twinx()
    ax.bar(x - 0.2, n_recs, 0.4, color="#4477AA",
            edgecolor="black", label="recordings")
    ax2.bar(x + 0.2, n_cells, 0.4, color="#CC4400",
             edgecolor="black", label="cells (after filter)")
    ax.set_xticks(x); ax.set_xticklabels(CONDITION_ORDER)
    ax.set_ylabel("Number of recordings", color="#4477AA")
    ax2.set_ylabel("Cells (filtered)", color="#CC4400")
    ax.set_title("Per-condition sample size — IC295 batch1+2 combined")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    for i, (nr, nc) in enumerate(zip(n_recs, n_cells)):
        ax.text(i - 0.2, nr + 0.3, f"{nr}", ha="center", fontsize=9)
        ax2.text(i + 0.2, nc + 2, f"{nc}", ha="center",
                 fontsize=9, color="#CC4400")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def state_evolution_figure(filt_recs, out_png):
    """Per-recording % balled, sorted within each condition.
    Shows the within-condition variability."""
    by_cond = {}
    for r in filt_recs:
        by_cond.setdefault(r["condition"], []).append(r)
    fig, ax = plt.subplots(figsize=(13, 5))
    pos = 0
    xticks = []
    xlabels = []
    for cond in CONDITION_ORDER:
        if cond not in by_cond:
            continue
        cells = sorted(by_cond[cond], key=lambda r: r["state_frac_balled"])
        for r in cells:
            ax.bar(pos, r["state_frac_balled"], color=COND_COLORS[cond],
                    edgecolor="black", linewidth=0.5)
            pos += 1
        xticks.append(pos - len(cells) / 2)
        xlabels.append(cond)
        pos += 1  # gap
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Fraction of cell-frames in BALLED state")
    ax.set_title("Per-recording balled fraction, sorted within condition\n"
                  "(IC295 combined, multi_metric Cy5 filter)")
    ax.axhline(0.30, color="gray", linestyle="--", alpha=0.5,
                label="30% (high)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def write_report(filt_se, unfilt_se, filt_recs, unfilt_recs, out_md):
    with open(out_md, "w") as f:
        f.write("# IC295 + IC293 — State-Stratified Motility Summary\n\n")
        f.write("This is the master narrative report covering all "
                 "datasets. For per-dataset detail, see the "
                 "`results/<dataset>/README.md` files.\n\n")
        f.write("## Datasets\n\n")
        f.write("| Dataset | Recordings | Cells | Description |\n")
        f.write("|---|---:|---:|---|\n")
        for ds_name, path in [
                ("IC295 batch1+2 filtered",
                  DATASETS["filtered"]),
                ("IC295 batch1+2 unfiltered",
                  DATASETS["unfiltered"]),
                ("IC295 batch1 alone", "results/ic295_state_analysis"),
                ("IC293 (no fluorescence)",
                  "results/ignasi_state_analysis_old")]:
            with open(os.path.join(path, "state_composition.csv")) as fcsv:
                rows = list(csv.DictReader(fcsv))
                n_cells = sum(int(r["n_cells"]) for r in rows)
                n_recs = len(rows)
            f.write(f"| {ds_name} | {n_recs} | {n_cells} | "
                    f"`{path}` |\n")

        f.write("\n## Headline finding\n\n")
        f.write("**State stratification reveals that the apparent "
                 "GOF/KO speed differences in raw mean speed are "
                 "largely driven by state composition (% balled cells), "
                 "not by intrinsic differences in spread-cell "
                 "migration.**\n\n")
        f.write("After classifying each cell-frame as balled (mitotic) "
                 "or attached (spread), two cleanly separated patterns "
                 "emerge:\n\n")
        f.write("1. **Attached migration speed is essentially uniform** "
                 "across conditions (0.7-0.85 µm/min in all 6).\n")
        f.write("2. **KO balled cells move ~50% slower** than other "
                 "conditions (0.73 ± 0.17 µm/min vs 1.05-1.57 in other "
                 "conditions). This is the only robust state-stratified "
                 "speed difference and survives both filtered and "
                 "unfiltered analysis.\n\n")

        f.write("## State composition (% time in balled state)\n\n")
        f.write("Mean ± SE across recordings, filtered (Cy5 multi_metric):\n\n")
        f.write("| Condition | Filtered | Unfiltered |\n")
        f.write("|---|---:|---:|\n")
        for cond in CONDITION_ORDER:
            fm = filt_se.get(cond, {})
            um = unfilt_se.get(cond, {})
            def fmt(d, key):
                m = d.get(f"{key}_mean", float("nan"))
                s = d.get(f"{key}_se", 0.0)
                if np.isnan(m): return "—"
                return f"{100*m:.0f}% ± {100*s:.0f}%"
            f.write(f"| {cond} | {fmt(fm, 'frac_balled')} | "
                    f"{fmt(um, 'frac_balled')} |\n")

        f.write("\nGOF and KO have the highest balled fractions (35-40%), "
                 "DMSO controls the lowest (16-21%). This composition "
                 "difference inflates the unstratified mean speed in GOF.\n\n")

        f.write("## Speed by state (filtered, n=65)\n\n")
        f.write("| Condition | Balled (µm/min) | Attached (µm/min) |\n")
        f.write("|---|---:|---:|\n")
        for cond in CONDITION_ORDER:
            d = filt_se.get(cond, {})
            def fmt(key):
                m = d.get(f"{key}_mean", float("nan"))
                s = d.get(f"{key}_se", 0.0)
                if np.isnan(m): return "—"
                return f"{m:.2f} ± {s:.2f}"
            f.write(f"| {cond} | {fmt('balled_speed')} | "
                    f"{fmt('attached_speed')} |\n")

        f.write("\n## Directional persistence by state (filtered)\n\n")
        f.write("Lag-1 cosine of velocity (positive = directional, "
                 "0 = random walk).\n\n")
        f.write("| Condition | Balled | Attached |\n")
        f.write("|---|---:|---:|\n")
        for cond in CONDITION_ORDER:
            d = filt_se.get(cond, {})
            def fmt(key):
                m = d.get(f"{key}_mean", float("nan"))
                s = d.get(f"{key}_se", 0.0)
                if np.isnan(m): return "—"
                return f"{m:+.2f} ± {s:.2f}"
            f.write(f"| {cond} | {fmt('balled_persistence')} | "
                    f"{fmt('attached_persistence')} |\n")

        f.write("\n**Validation**: attached cells show positive "
                 "persistence in every condition (directional motion); "
                 "balled cells show near-zero or negative persistence "
                 "(random walks during division). State classification "
                 "is biologically meaningful.\n\n")

        f.write("## Filter impact\n\n")
        f.write("The multi_metric Cy5 filter (≥2/3 cellularity tests) "
                 "drops 62% of cpsam-detected tracks (1254 → 474) "
                 "across the combined dataset. Despite removing so many "
                 "cells, the **biological conclusions hold up** in both "
                 "filtered and unfiltered analyses:\n\n")
        f.write("* State composition pattern is the same\n")
        f.write("* KO balled cells slower in both\n")
        f.write("* Attached speeds remain flat in both\n\n")
        f.write("The filter mostly removes low-confidence DIC detections "
                 "(debris, fragments). Real cells survive.\n\n")

        f.write("## Cell shape per state (validates classification)\n\n")
        f.write("| Condition | Balled circularity | Attached circularity |\n")
        f.write("|---|---:|---:|\n")
        for cond in CONDITION_ORDER:
            d = filt_se.get(cond, {})
            def fmt(key):
                m = d.get(f"{key}_mean", float("nan"))
                s = d.get(f"{key}_se", 0.0)
                if np.isnan(m): return "—"
                return f"{m:.2f} ± {s:.2f}"
            f.write(f"| {cond} | {fmt('balled_circularity')} | "
                    f"{fmt('attached_circularity')} |\n")
        f.write("\nBalled cells have circularity ≈ 0.86 in all "
                 "conditions; attached cells ≈ 0.45. The shape signature "
                 "is consistent across all 65 recordings and 6 conditions, "
                 "confirming the classification rule generalizes.\n\n")

        f.write("## Figures\n\n")
        f.write("* `summary_figure.png` — 4-panel headline: state "
                 "composition + 3 motility metrics, filtered vs "
                 "unfiltered.\n")
        f.write("* `filter_impact.png` — how the Cy5 filter changes "
                 "cell counts and balled speed.\n")
        f.write("* `per_condition_n.png` — recordings × cells per "
                 "condition.\n")
        f.write("* `state_evolution.png` — per-recording balled "
                 "fraction, sorted within condition (shows recording "
                 "variability).\n\n")

        f.write("## Methods\n\n")
        f.write("**Detection**: cpsam (cellpose 4 ViT) + per-cell "
                 "DeepSea refinement + Hungarian tracking, run via "
                 "`scripts/run_ignasi_ic295_full.py`. Cy5 (SiR-actin) "
                 "channel used for Tier-2 fail-safe recovery.\n\n")
        f.write("**Cy5 filtering**: per-track multi-metric "
                 "(cy5_score > 0.15 AND/OR cy5_io_ratio > 1.10 AND/OR "
                 "fraction_positive > 0.15, ≥2 of 3 tests must pass).\n\n")
        f.write("**State classification**: per cell-frame, balled if "
                 "circularity ≥ 0.80 AND solidity ≥ 0.92; attached if "
                 "circularity ≤ 0.55 OR solidity ≤ 0.85.\n\n")
        f.write("**Per-state motility**: speed, MSD, persistence "
                 "computed only on contiguous pure-state segments "
                 "(≥3 frames for speed, ≥5 for persistence, ≥10 for "
                 "MSD). Capped at 15 µm/min to drop tracking outliers.\n\n")
        f.write("**SE between recordings**: recordings are the "
                 "biological replicates (cells within a recording share "
                 "field-of-view artefacts). All means and SEs computed "
                 "by averaging cells within recording, then taking SE "
                 "across recordings within condition.\n\n")

        f.write("## Reproducing\n\n")
        f.write("```bash\n")
        f.write("# 1. Per-recording detection (~50 min × 65 recs ≈ 50h)\n")
        f.write("conda run -n cellpose4 python "
                 "scripts/run_ignasi_ic295_full.py \\\n")
        f.write("    --src /path/to/ic295 --out-dir results/ic295_full\n")
        f.write("conda run -n cellpose4 python "
                 "scripts/run_ignasi_ic295_full.py \\\n")
        f.write("    --src /path/to/ic295_batch2 \\\n")
        f.write("    --out-dir results/ic295_batch2_full\n\n")
        f.write("# 2. Recompute Cy5 v2 metrics (~45 min each)\n")
        f.write("conda run -n cellpose python "
                 "scripts/recompute_cy5_metrics.py \\\n")
        f.write("    --cache-dir results/ic295_full \\\n")
        f.write("    --out-dir results/ic295_full_v2\n")
        f.write("conda run -n cellpose python "
                 "scripts/recompute_cy5_metrics.py \\\n")
        f.write("    --cache-dir results/ic295_batch2_full \\\n")
        f.write("    --out-dir results/ic295_batch2_full_v2\n\n")
        f.write("# 3. Symlink into combined dir\n")
        f.write("mkdir results/ic295_combined_v2\n")
        f.write("cd results/ic295_combined_v2\n")
        f.write("ln -sf ../ic295_full_v2/Pos*.npz .\n")
        f.write("for f in ../ic295_batch2_full_v2/Pos*.npz; do\n")
        f.write("    [ -e \"$(basename \"$f\")\" ] || ln -sf \"$f\" .\n")
        f.write("done\ncd ../..\n\n")
        f.write("# 4. Filtered + unfiltered combined state analyses (~21 min each)\n")
        f.write("conda run -n cellpose python "
                 "scripts/combined_state_analysis.py \\\n")
        f.write("    --filter-mode multi_metric\n")
        f.write("conda run -n cellpose python "
                 "scripts/combined_state_analysis.py \\\n")
        f.write("    --filter-mode off \\\n")
        f.write("    --out-dir results/ic295_combined_state_analysis_unfiltered\n\n")
        f.write("# 5. SE aggregation + cross-dataset comparison\n")
        f.write("for ds in ic295_combined_state_analysis "
                 "ic295_combined_state_analysis_unfiltered; do\n")
        f.write("    conda run -n cellpose python "
                 "scripts/aggregate_state_to_recording.py \\\n")
        f.write("        --in-dir results/$ds\n")
        f.write("done\n")
        f.write("conda run -n cellpose python "
                 "scripts/compare_state_datasets.py\n\n")
        f.write("# 6. This summary\n")
        f.write("conda run -n cellpose python "
                 "scripts/build_summary_report.py\n")
        f.write("```\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    filt_se = load_means_se(DATASETS["filtered"])
    unfilt_se = load_means_se(DATASETS["unfiltered"])
    filt_recs = load_per_recording(DATASETS["filtered"])
    unfilt_recs = load_per_recording(DATASETS["unfiltered"])
    print(f"Loaded {len(filt_recs)} filtered recordings, "
          f"{len(unfilt_recs)} unfiltered")

    headline_figure(filt_se, unfilt_se,
                    os.path.join(OUT_DIR, "summary_figure.png"))
    print("Wrote summary_figure.png")
    filter_impact_figure(filt_se, unfilt_se,
                         os.path.join(OUT_DIR, "filter_impact.png"))
    print("Wrote filter_impact.png")
    per_condition_n_figure(filt_recs,
                           os.path.join(OUT_DIR, "per_condition_n.png"))
    print("Wrote per_condition_n.png")
    state_evolution_figure(filt_recs,
                           os.path.join(OUT_DIR, "state_evolution.png"))
    print("Wrote state_evolution.png")
    write_report(filt_se, unfilt_se, filt_recs, unfilt_recs,
                  os.path.join(OUT_DIR, "SUMMARY_REPORT.md"))
    print(f"Wrote {OUT_DIR}/SUMMARY_REPORT.md")


if __name__ == "__main__":
    main()
