"""Recording-level aggregation of per-cell state-stratified metrics.

Recordings (not cells) are the independent biological replicates:
cells within a recording share field-of-view, focal plane, dish
position, and any local artefacts. Computing condition-level SE
across CELLS would massively over-state precision. The right unit
of replication is the recording.

This script reads `per_cell_state.csv` (one row per cell) and
produces:

  * `per_recording_state.csv` — averages all cells within each
    recording into one row per recording.
  * `condition_means_se.csv` — mean ± SE *across recordings* per
    (condition × state) for speed, persistence, MSD-at-lag-5,
    total displacement, straightness, area, circularity, solidity.
  * `figures_se/`:
      - speed_with_se.png            (bar + SE error bars per state)
      - persistence_with_se.png
      - msd_curves_with_se.png       (MSD vs lag, shaded SE band)
      - shape_with_se.png            (area + circularity per state)
      - state_composition_with_se.png
  * `report_with_se.md` — recording-aggregated tables with SE.

Run:
    conda run -n cellpose python scripts/aggregate_state_to_recording.py \\
        --in-dir results/ic295_state_analysis_unfiltered
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

INTERVAL_MIN = 10.0
CONDITION_ORDER = ["WT", "KO", "GOF", "OT", "Y1", "DMSO"]
COND_COLORS = {"WT": "#4477AA", "KO": "#EE6677", "GOF": "#228833",
                "OT": "#CCBB44", "Y1": "#66CCEE", "DMSO": "#AA3377"}


def parse_condition(name):
    import re
    m = re.search(r"(Pos\d+)_?-?(WT|KO|GOF|OT|Y1|DMSO)", name, re.I)
    return (m.group(1), m.group(2).upper()) if m else ("?", "?")


def load_per_cell(in_dir):
    """Load per_cell_state.csv → list of dicts (numeric fields cast)."""
    csv_path = os.path.join(in_dir, "per_cell_state.csv")
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            d = {}
            for k, v in r.items():
                if v in ("", "nan"):
                    d[k] = float("nan")
                else:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
            rows.append(d)
    return rows


def aggregate_to_recordings(rows):
    """Average all cells within each recording → one row per recording."""
    by_rec = {}
    for r in rows:
        by_rec.setdefault(r["recording"], []).append(r)
    out = []
    for rec_name, cells in by_rec.items():
        pos, cond = parse_condition(rec_name)
        agg = {"recording": rec_name, "condition": cond,
               "n_cells": len(cells)}
        # Average all numeric fields
        keys = [k for k, v in cells[0].items()
                if isinstance(v, float) and k != "track_id"]
        for k in keys:
            vals = [c[k] for c in cells if not np.isnan(c[k])]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        out.append(agg)
    return out


def write_per_recording_csv(rec_rows, out_csv):
    if not rec_rows:
        return
    fields = ["recording", "condition", "n_cells"] + [
        k for k in rec_rows[0].keys()
        if k not in ("recording", "condition", "n_cells")]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rec_rows:
            w.writerow(r)


def condition_mean_se(rec_rows, key, cond):
    """For one (condition, metric), return (mean, SE, n) across
    recordings. Skips recordings with NaN."""
    vals = [r[key] for r in rec_rows
            if r["condition"] == cond and not np.isnan(r.get(key, np.nan))]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, se, n


def write_condition_means_se(rec_rows, out_csv):
    metrics = [
        ("ball_mean_speed_um_per_min", "balled_speed"),
        ("atta_mean_speed_um_per_min", "attached_speed"),
        ("ball_persistence_lag1", "balled_persistence"),
        ("atta_persistence_lag1", "attached_persistence"),
        ("ball_msd_lag5_um2", "balled_msd_50min"),
        ("atta_msd_lag5_um2", "attached_msd_50min"),
        ("ball_total_displacement_um", "balled_total_displacement"),
        ("atta_total_displacement_um", "attached_total_displacement"),
        ("ball_straightness", "balled_straightness"),
        ("atta_straightness", "attached_straightness"),
        ("ball_mean_circularity", "balled_circularity"),
        ("atta_mean_circularity", "attached_circularity"),
        ("ball_mean_area_px", "balled_area_px"),
        ("atta_mean_area_px", "attached_area_px"),
        ("state_frac_balled", "frac_balled"),
        ("state_frac_attached", "frac_attached"),
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["condition", "n_recordings"]
        for _, label in metrics:
            header += [f"{label}_mean", f"{label}_se"]
        w.writerow(header)
        for cond in CONDITION_ORDER:
            row = [cond, sum(1 for r in rec_rows if r["condition"] == cond)]
            for key, _ in metrics:
                m, se, _n = condition_mean_se(rec_rows, key, cond)
                row += [round(m, 4) if not np.isnan(m) else "",
                        round(se, 4) if not np.isnan(se) else ""]
            w.writerow(row)


# ────────────────────────────────────────────────────────────────────
# Figures
# ────────────────────────────────────────────────────────────────────

def _bar_metric_by_state(rec_rows, ball_key, atta_key, ylabel,
                          title, out_png):
    """Side-by-side bar chart per condition: balled and attached
    means with SE error bars."""
    conds = CONDITION_ORDER
    ball_means, ball_ses, atta_means, atta_ses = [], [], [], []
    for cond in conds:
        m1, s1, _ = condition_mean_se(rec_rows, ball_key, cond)
        m2, s2, _ = condition_mean_se(rec_rows, atta_key, cond)
        ball_means.append(m1)
        ball_ses.append(s1)
        atta_means.append(m2)
        atta_ses.append(s2)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conds))
    width = 0.35
    ax.bar(x - width / 2, ball_means, width, yerr=ball_ses, capsize=4,
            color="orange", edgecolor="black", label="balled")
    ax.bar(x + width / 2, atta_means, width, yerr=atta_ses, capsize=4,
            color="steelblue", edgecolor="black", label="attached")
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n(error bars: SE between recordings)")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def figure_state_composition_se(rec_rows, out_png):
    conds = CONDITION_ORDER
    bm, bs, am, as_, tm, ts = [], [], [], [], [], []
    for cond in conds:
        m1, s1, _ = condition_mean_se(rec_rows, "state_frac_balled", cond)
        m2, s2, _ = condition_mean_se(rec_rows, "state_frac_attached", cond)
        m3, s3, _ = condition_mean_se(
            rec_rows, "state_frac_transitional", cond)
        bm.append(m1); bs.append(s1)
        am.append(m2); as_.append(s2)
        tm.append(m3); ts.append(s3)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conds))
    width = 0.27
    ax.bar(x - width, am, width, yerr=as_, capsize=4,
            color="steelblue", edgecolor="black", label="attached")
    ax.bar(x, bm, width, yerr=bs, capsize=4,
            color="orange", edgecolor="black", label="balled")
    ax.bar(x + width, tm, width, yerr=ts, capsize=4,
            color="lightgray", edgecolor="black", label="transitional")
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("Mean fraction of cell-frames")
    ax.set_title("Cell state composition by condition\n"
                  "(error bars: SE between recordings)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def figure_msd_curves_se(per_cell_rows_with_curves, out_png,
                          full_data_dir):
    """Re-extract MSD curves from per_cell_state... actually we don't
    have curves in CSV. Skip this figure here — see msd_by_state.png
    in the parent directory for cell-pooled curves.
    Generate a placeholder note instead."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5,
             "Per-recording MSD curves require re-loading NPZs.\n"
             "See `figures/msd_by_state.png` in parent dir for\n"
             "cell-pooled MSD vs lag (no SE band).\n\n"
             "If recording-level MSD bands are needed, run\n"
             "`scripts/msd_with_recording_se.py` (not yet built).",
             ha="center", va="center", fontsize=11,
             family="monospace",
             transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def write_report(rec_rows, out_md):
    def fmt(rec_rows, key, cond):
        m, se, n = condition_mean_se(rec_rows, key, cond)
        if np.isnan(m):
            return "—"
        return f"{m:.3f} ± {se:.3f} (n={n})"

    with open(out_md, "w") as f:
        f.write("# IC295 state-stratified motility — recording-level\n\n")
        f.write("All metrics are reported as **mean ± SE between "
                 "recordings** (recordings, not cells, are the "
                 "independent biological replicates).\n\n")
        f.write("## Recordings per condition\n\n")
        f.write("| Condition | n recordings | n cells (sum) |\n")
        f.write("|---|---:|---:|\n")
        for cond in CONDITION_ORDER:
            recs = [r for r in rec_rows if r["condition"] == cond]
            f.write(f"| {cond} | {len(recs)} | "
                    f"{int(sum(r['n_cells'] for r in recs))} |\n")

        f.write("\n## State composition (mean ± SE)\n\n")
        f.write("| Condition | % balled | % attached | % transitional |\n")
        f.write("|---|---:|---:|---:|\n")
        for cond in CONDITION_ORDER:
            f.write(f"| {cond} | "
                    f"{fmt(rec_rows, 'state_frac_balled', cond)} | "
                    f"{fmt(rec_rows, 'state_frac_attached', cond)} | "
                    f"{fmt(rec_rows, 'state_frac_transitional', cond)} |\n")

        f.write("\n## Speed (µm/min, mean ± SE)\n\n")
        f.write("| Condition | Balled speed | Attached speed |\n")
        f.write("|---|---:|---:|\n")
        for cond in CONDITION_ORDER:
            f.write(f"| {cond} | "
                    f"{fmt(rec_rows, 'ball_mean_speed_um_per_min', cond)} | "
                    f"{fmt(rec_rows, 'atta_mean_speed_um_per_min', cond)} |\n")

        f.write("\n## Directional persistence (lag-1 cosine, mean ± SE)\n\n")
        f.write("| Condition | Balled persistence | Attached persistence |\n")
        f.write("|---|---:|---:|\n")
        for cond in CONDITION_ORDER:
            f.write(f"| {cond} | "
                    f"{fmt(rec_rows, 'ball_persistence_lag1', cond)} | "
                    f"{fmt(rec_rows, 'atta_persistence_lag1', cond)} |\n")

        f.write("\n## MSD at lag = 50 min (µm², mean ± SE)\n\n")
        f.write("| Condition | Balled MSD@50min | Attached MSD@50min |\n")
        f.write("|---|---:|---:|\n")
        for cond in CONDITION_ORDER:
            f.write(f"| {cond} | "
                    f"{fmt(rec_rows, 'ball_msd_lag5_um2', cond)} | "
                    f"{fmt(rec_rows, 'atta_msd_lag5_um2', cond)} |\n")

        f.write("\n## Total displacement (µm, mean ± SE)\n\n")
        f.write("| Condition | Balled | Attached |\n")
        f.write("|---|---:|---:|\n")
        for cond in CONDITION_ORDER:
            f.write(f"| {cond} | "
                    f"{fmt(rec_rows, 'ball_total_displacement_um', cond)} | "
                    f"{fmt(rec_rows, 'atta_total_displacement_um', cond)} |\n")

        f.write("\n## Straightness (mean ± SE)\n\n")
        f.write("| Condition | Balled | Attached |\n")
        f.write("|---|---:|---:|\n")
        for cond in CONDITION_ORDER:
            f.write(f"| {cond} | "
                    f"{fmt(rec_rows, 'ball_straightness', cond)} | "
                    f"{fmt(rec_rows, 'atta_straightness', cond)} |\n")

        f.write("\n## Shape: cell area (px, mean ± SE)\n\n")
        f.write("| Condition | Balled area | Attached area |\n")
        f.write("|---|---:|---:|\n")
        for cond in CONDITION_ORDER:
            f.write(f"| {cond} | "
                    f"{fmt(rec_rows, 'ball_mean_area_px', cond)} | "
                    f"{fmt(rec_rows, 'atta_mean_area_px', cond)} |\n")

        f.write("\n## Shape: circularity (mean ± SE)\n\n")
        f.write("| Condition | Balled circularity | Attached circularity |\n")
        f.write("|---|---:|---:|\n")
        for cond in CONDITION_ORDER:
            f.write(f"| {cond} | "
                    f"{fmt(rec_rows, 'ball_mean_circularity', cond)} | "
                    f"{fmt(rec_rows, 'atta_mean_circularity', cond)} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True,
                    help="state-analysis output dir (containing "
                         "per_cell_state.csv)")
    args = ap.parse_args()
    out_dir = args.in_dir  # write next to existing CSVs
    fig_dir = os.path.join(out_dir, "figures_se")
    os.makedirs(fig_dir, exist_ok=True)

    rows = load_per_cell(args.in_dir)
    print(f"Loaded {len(rows)} per-cell rows")
    rec_rows = aggregate_to_recordings(rows)
    print(f"Aggregated into {len(rec_rows)} recordings")

    rec_csv = os.path.join(out_dir, "per_recording_state.csv")
    write_per_recording_csv(rec_rows, rec_csv)
    print(f"Wrote {rec_csv}")

    se_csv = os.path.join(out_dir, "condition_means_se.csv")
    write_condition_means_se(rec_rows, se_csv)
    print(f"Wrote {se_csv}")

    figure_state_composition_se(
        rec_rows, os.path.join(fig_dir, "state_composition_with_se.png"))
    _bar_metric_by_state(
        rec_rows, "ball_mean_speed_um_per_min",
        "atta_mean_speed_um_per_min",
        "Mean speed (µm/min)",
        "Per-recording mean speed by state and condition",
        os.path.join(fig_dir, "speed_with_se.png"))
    _bar_metric_by_state(
        rec_rows, "ball_persistence_lag1",
        "atta_persistence_lag1",
        "Lag-1 persistence (cosine)",
        "Directional persistence by state and condition",
        os.path.join(fig_dir, "persistence_with_se.png"))
    _bar_metric_by_state(
        rec_rows, "ball_msd_lag5_um2",
        "atta_msd_lag5_um2",
        "MSD at lag = 50 min (µm²)",
        "MSD at 50-min lag by state and condition",
        os.path.join(fig_dir, "msd_50min_with_se.png"))
    _bar_metric_by_state(
        rec_rows, "ball_total_displacement_um",
        "atta_total_displacement_um",
        "Total displacement (µm)",
        "Total per-segment displacement by state and condition",
        os.path.join(fig_dir, "displacement_with_se.png"))
    _bar_metric_by_state(
        rec_rows, "ball_straightness",
        "atta_straightness",
        "Path straightness (0-1)",
        "Trajectory straightness by state and condition",
        os.path.join(fig_dir, "straightness_with_se.png"))
    _bar_metric_by_state(
        rec_rows, "ball_mean_area_px",
        "atta_mean_area_px",
        "Mean cell area (px)",
        "Cell area by state and condition",
        os.path.join(fig_dir, "area_with_se.png"))
    _bar_metric_by_state(
        rec_rows, "ball_mean_circularity",
        "atta_mean_circularity",
        "Mean circularity",
        "Circularity by state and condition",
        os.path.join(fig_dir, "circularity_with_se.png"))
    print(f"Wrote figures to {fig_dir}")

    write_report(rec_rows, os.path.join(out_dir, "report_with_se.md"))
    print(f"Wrote {os.path.join(out_dir, 'report_with_se.md')}")


if __name__ == "__main__":
    main()
