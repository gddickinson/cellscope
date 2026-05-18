"""State-stratified motility + shape analysis on filtered IC295 results.

For each cell in each filtered NPZ:
  1. Per frame: shape metrics + classify state (balled/attached/transitional)
  2. Per state: speed distribution, MSD curve, directional persistence,
     total displacement, straightness, mean shape
  3. Per cell summary: time fraction in each state

Then aggregate per condition (WT/KO/GOF/OT/Y1/DMSO) and write:
  * `per_cell_state.csv` — one row per cell, with state-stratified
    metrics
  * `state_composition.csv` — per condition: % balled, % attached,
    average per-cell time in each state
  * `motility_by_state_condition.csv` — per (condition × state):
    mean speed, mean persistence, mean MSD slope, n cells
  * `report.md` — interpretive summary, including state-corrected
    speed comparison
  * `figures/`:
      - state_composition_by_condition.png  (stacked bar)
      - speed_by_state_condition.png        (box plot, balled vs attached
        per condition)
      - msd_by_state_condition.png          (log-log curves per condition,
        2 lines each — balled + attached)
      - persistence_by_state_condition.png  (box plot)
      - shape_by_state.png                  (histograms of circularity
        per state, all conditions pooled)

Output: results/ic295_state_analysis/
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports
setup_imports()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "results/ic295_filtered"
OUT_DIR = "results/ic295_state_analysis"
PIXEL_SIZE_UM = 0.6523
INTERVAL_MIN = 10.0
SPEED_CAP_UM_PER_MIN = 15.0
CONDITION_ORDER = ["WT", "KO", "GOF", "OT", "Y1", "DMSO"]
STATES = ["balled", "attached", "transitional"]


def parse_condition(name):
    import re
    m = re.search(r"(Pos\d+)_?-?(WT|KO|GOF|OT|Y1|DMSO)", name, re.I)
    return (m.group(1), m.group(2).upper()) if m else ("?", "?")


def load_filtered_tracks(npz_path):
    """Load filtered NPZ → list of track dicts with stack + centroids."""
    z = np.load(npz_path, allow_pickle=False)
    tracks = []
    n = int(z["filter_n_kept"]) if "filter_n_kept" in z.files else (
        int(z["tracks_n"]) if "tracks_n" in z.files else 0)
    for tid in range(n * 3 + 5):
        if f"track_{tid}_stack" not in z.files:
            continue
        tracks.append({
            "id": tid + 1,
            "stack": z[f"track_{tid}_stack"],
            "centroids": z.get(f"track_{tid}_centroids"),
        })
    return tracks


def analyze_one_track(track):
    """Run state classification + per-state motility for one track."""
    from core.cell_state import (
        classify_track_states, state_fraction, per_state_means,
        STATE_BALLED, STATE_ATTACHED, STATE_TRANSITIONAL)
    from core.motility_state import (
        state_speeds, state_msd, state_persistence,
        state_total_displacement)

    state_data = classify_track_states(track["stack"])
    states = state_data["states"]
    metrics = state_data["metrics"]
    cents = track["centroids"]
    if cents is None:
        return None

    out = {"track_id": track["id"]}
    out["lifetime_frames"] = int(sum(1 for s in states
                                       if s != "unknown"))
    out["state_frac_balled"] = state_fraction(states, STATE_BALLED)
    out["state_frac_attached"] = state_fraction(states, STATE_ATTACHED)
    out["state_frac_transitional"] = state_fraction(
        states, STATE_TRANSITIONAL)
    out["mean_circularity"] = float(
        np.nanmean(metrics["circularity"]))
    out["mean_area"] = float(np.nanmean(metrics["area"]))

    # Per-state motility + shape
    for state in (STATE_BALLED, STATE_ATTACHED):
        sp = state_speeds(cents, states, state, PIXEL_SIZE_UM,
                            INTERVAL_MIN, speed_cap=SPEED_CAP_UM_PER_MIN)
        msd = state_msd(cents, states, state, PIXEL_SIZE_UM,
                          max_lag=20)
        per = state_persistence(cents, states, state, PIXEL_SIZE_UM)
        td = state_total_displacement(cents, states, state,
                                        PIXEL_SIZE_UM)
        sh = per_state_means(metrics, states, state)
        prefix = state[:4]
        out[f"{prefix}_n_speed_samples"] = int(len(sp))
        out[f"{prefix}_mean_speed_um_per_min"] = (
            float(np.mean(sp)) if len(sp) else float("nan"))
        out[f"{prefix}_median_speed_um_per_min"] = (
            float(np.median(sp)) if len(sp) else float("nan"))
        out[f"{prefix}_persistence_lag1"] = per["lag1"]
        out[f"{prefix}_n_msd_segments"] = msd["n_segments"]
        out[f"{prefix}_msd_lag1_um2"] = (
            float(msd["msd"][0]) if not np.isnan(msd["msd"][0])
            else float("nan"))
        out[f"{prefix}_msd_lag5_um2"] = (
            float(msd["msd"][4]) if not np.isnan(msd["msd"][4])
            else float("nan"))
        out[f"{prefix}_total_displacement_um"] = td["total_displacement_um"]
        out[f"{prefix}_straightness"] = td["straightness"]
        out[f"{prefix}_mean_circularity"] = sh.get("circularity",
                                                    float("nan"))
        out[f"{prefix}_mean_area_px"] = sh.get("area", float("nan"))
        out[f"{prefix}_mean_solidity"] = sh.get("solidity",
                                                 float("nan"))

        # MSD curve (full length)
        out[f"_{prefix}_msd_curve"] = msd["msd"]

    return out


def aggregate_by_condition(all_cells, condition_lookup):
    """Bucket per-cell rows by condition for downstream stats."""
    by_cond = {}
    for row in all_cells:
        cond = condition_lookup.get(row["recording"], "?")
        by_cond.setdefault(cond, []).append(row)
    return by_cond


def write_per_cell_csv(rows, out_csv):
    fields = [k for k in rows[0].keys() if not k.startswith("_")]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_state_composition(by_cond, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_cells",
                    "mean_frac_balled", "mean_frac_attached",
                    "mean_frac_transitional",
                    "median_frac_balled", "median_frac_attached"])
        for cond in CONDITION_ORDER:
            if cond not in by_cond:
                continue
            rows = by_cond[cond]
            balled = np.array([r["state_frac_balled"] for r in rows])
            attached = np.array([r["state_frac_attached"] for r in rows])
            trans = np.array([r["state_frac_transitional"] for r in rows])
            w.writerow([cond, len(rows),
                        round(float(np.mean(balled)), 3),
                        round(float(np.mean(attached)), 3),
                        round(float(np.mean(trans)), 3),
                        round(float(np.median(balled)), 3),
                        round(float(np.median(attached)), 3)])


def write_motility_by_state_condition(by_cond, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "state", "n_cells_with_state",
                    "mean_speed_um_per_min", "median_speed_um_per_min",
                    "mean_persistence_lag1",
                    "mean_msd_lag5_um2",
                    "mean_total_displacement_um",
                    "mean_straightness",
                    "mean_circularity", "mean_area_px"])
        for cond in CONDITION_ORDER:
            if cond not in by_cond:
                continue
            for state in ("ball", "atta"):
                rows = [r for r in by_cond[cond]
                        if r[f"{state}_n_speed_samples"] > 0]
                if not rows:
                    w.writerow([cond, state, 0,
                                "", "", "", "", "", "", "", ""])
                    continue
                w.writerow([
                    cond, state, len(rows),
                    round(_safe_mean(rows, f"{state}_mean_speed_um_per_min"), 4),
                    round(_safe_mean(rows, f"{state}_median_speed_um_per_min"), 4),
                    round(_safe_mean(rows, f"{state}_persistence_lag1"), 3),
                    round(_safe_mean(rows, f"{state}_msd_lag5_um2"), 2),
                    round(_safe_mean(rows, f"{state}_total_displacement_um"), 1),
                    round(_safe_mean(rows, f"{state}_straightness"), 3),
                    round(_safe_mean(rows, f"{state}_mean_circularity"), 3),
                    round(_safe_mean(rows, f"{state}_mean_area_px"), 0),
                ])


def _safe_mean(rows, key):
    vals = [r[key] for r in rows
            if r.get(key) is not None and not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else 0.0


def figure_state_composition(by_cond, out_png):
    conds = [c for c in CONDITION_ORDER if c in by_cond]
    balled = [np.mean([r["state_frac_balled"]
                        for r in by_cond[c]]) for c in conds]
    attached = [np.mean([r["state_frac_attached"]
                          for r in by_cond[c]]) for c in conds]
    trans = [np.mean([r["state_frac_transitional"]
                        for r in by_cond[c]]) for c in conds]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conds))
    ax.bar(x, attached, color="steelblue", label="attached (spread)")
    ax.bar(x, balled, bottom=attached,
            color="orange", label="balled (rounded)")
    ax.bar(x, trans, bottom=np.array(attached) + np.array(balled),
            color="lightgray", label="transitional")
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("Mean fraction of frames")
    ax.set_title("Cell state composition by condition")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def figure_speed_by_state(by_cond, out_png):
    conds = [c for c in CONDITION_ORDER if c in by_cond]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.35
    x = np.arange(len(conds))
    balled_data = [[r["ball_mean_speed_um_per_min"] for r in by_cond[c]
                     if not np.isnan(r["ball_mean_speed_um_per_min"])]
                    for c in conds]
    attached_data = [[r["atta_mean_speed_um_per_min"] for r in by_cond[c]
                       if not np.isnan(r["atta_mean_speed_um_per_min"])]
                      for c in conds]
    bp1 = ax.boxplot(balled_data, positions=x - width / 2,
                      widths=width, patch_artist=True,
                      boxprops=dict(facecolor="orange"),
                      medianprops=dict(color="black"))
    bp2 = ax.boxplot(attached_data, positions=x + width / 2,
                      widths=width, patch_artist=True,
                      boxprops=dict(facecolor="steelblue"),
                      medianprops=dict(color="black"))
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("Mean speed (µm/min)")
    ax.set_title("Per-cell mean speed by state and condition")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
               ["balled", "attached"], loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def figure_msd_by_state(by_cond, out_png, all_cells_full):
    """Log-log MSD curves, two lines per condition (balled, attached)."""
    conds = [c for c in CONDITION_ORDER if c in by_cond]
    n_cols = 3
    n_rows = (len(conds) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols,
                                                        3.5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    full_by_cond = {c: [] for c in conds}
    for cell, full in all_cells_full:
        # _condition lives on the `full` dict (the row was stripped
        # of underscore-prefixed keys before being CSV-serialized).
        cond = full.get("_condition")
        if cond in full_by_cond:
            full_by_cond[cond].append(full)

    for c, cond in enumerate(conds):
        ax = axes[c]
        for state, color in (("ball", "orange"),
                              ("atta", "steelblue")):
            curves = []
            for full in full_by_cond[cond]:
                msd_curve = full.get(f"_{state}_msd_curve")
                if msd_curve is None:
                    continue
                if not np.all(np.isnan(msd_curve)):
                    curves.append(msd_curve)
            if not curves:
                continue
            arr = np.array(curves)
            mean_msd = np.nanmean(arr, axis=0)
            lag = np.arange(1, len(mean_msd) + 1) * INTERVAL_MIN
            ax.loglog(lag, mean_msd, "-", color=color, linewidth=2,
                       label=state)
        ax.set_xlabel("Lag (min)")
        ax.set_ylabel("MSD (µm²)")
        ax.set_title(cond)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    for c in range(len(conds), len(axes)):
        axes[c].axis("off")
    fig.suptitle("MSD by state and condition", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def figure_persistence_by_state(by_cond, out_png):
    conds = [c for c in CONDITION_ORDER if c in by_cond]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.35
    x = np.arange(len(conds))
    balled_p = [[r["ball_persistence_lag1"] for r in by_cond[c]
                  if not np.isnan(r["ball_persistence_lag1"])]
                 for c in conds]
    attached_p = [[r["atta_persistence_lag1"] for r in by_cond[c]
                    if not np.isnan(r["atta_persistence_lag1"])]
                   for c in conds]
    bp1 = ax.boxplot(balled_p, positions=x - width / 2,
                      widths=width, patch_artist=True,
                      boxprops=dict(facecolor="orange"),
                      medianprops=dict(color="black"))
    bp2 = ax.boxplot(attached_p, positions=x + width / 2,
                      widths=width, patch_artist=True,
                      boxprops=dict(facecolor="steelblue"),
                      medianprops=dict(color="black"))
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("Lag-1 directional persistence (cosine)")
    ax.set_title("Directional persistence by state and condition\n"
                  "(0 = random walk, +1 = perfectly persistent)")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
               ["balled", "attached"], loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def figure_shape_by_state(all_cells_full, out_png):
    """Histogram of mean circularity per state (pooled across conditions)."""
    balled_circ = [r["ball_mean_circularity"] for r, _ in all_cells_full
                    if not np.isnan(r["ball_mean_circularity"])]
    attached_circ = [r["atta_mean_circularity"] for r, _ in all_cells_full
                      if not np.isnan(r["atta_mean_circularity"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(balled_circ, bins=20, alpha=0.6, color="orange",
             label=f"balled (n={len(balled_circ)})", density=True)
    ax.hist(attached_circ, bins=20, alpha=0.6, color="steelblue",
             label=f"attached (n={len(attached_circ)})", density=True)
    ax.axvline(0.80, color="orange", linestyle="--",
                label="balled threshold (0.80)", alpha=0.7)
    ax.axvline(0.55, color="steelblue", linestyle="--",
                label="attached threshold (0.55)", alpha=0.7)
    ax.set_xlabel("Mean circularity in state"); ax.set_ylabel("Density")
    ax.set_title("Per-cell mean circularity by classified state\n"
                  "(validates the classification)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def write_report(by_cond, out_md):
    with open(out_md, "w") as f:
        f.write("# IC295 state-stratified motility analysis\n\n")
        f.write("Multi-metric Cy5-filtered tracks, classified per "
                 "frame as **balled** (mitotic / pre-mitotic, rounded), "
                 "**attached** (spread, adherent), or "
                 "**transitional**.\n\n")
        f.write("Classification rule: balled if circularity ≥ 0.80 "
                 "AND solidity ≥ 0.92; attached if circularity ≤ 0.55 "
                 "OR solidity ≤ 0.85; transitional otherwise.\n\n")
        f.write("## State composition by condition\n\n")
        f.write("| Condition | n cells | Mean % balled | Mean % attached | Mean % transitional |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for cond in CONDITION_ORDER:
            if cond not in by_cond:
                continue
            rows = by_cond[cond]
            f.write(f"| {cond} | {len(rows)} | "
                    f"{100*np.mean([r['state_frac_balled'] for r in rows]):.0f}% | "
                    f"{100*np.mean([r['state_frac_attached'] for r in rows]):.0f}% | "
                    f"{100*np.mean([r['state_frac_transitional'] for r in rows]):.0f}% |\n")
        f.write("\n## Motility by state and condition\n\n")
        f.write("| Condition | State | n cells | Mean speed (µm/min) | "
                 "Median speed | Persistence (lag1) | MSD@50min (µm²) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for cond in CONDITION_ORDER:
            if cond not in by_cond:
                continue
            for state, label in [("ball", "balled"), ("atta", "attached")]:
                rows = [r for r in by_cond[cond]
                        if r[f"{state}_n_speed_samples"] > 0]
                if not rows:
                    f.write(f"| {cond} | {label} | 0 | — | — | — | — |\n")
                    continue
                f.write(f"| {cond} | {label} | {len(rows)} | "
                        f"{_safe_mean(rows, f'{state}_mean_speed_um_per_min'):.3f} | "
                        f"{_safe_mean(rows, f'{state}_median_speed_um_per_min'):.3f} | "
                        f"{_safe_mean(rows, f'{state}_persistence_lag1'):.2f} | "
                        f"{_safe_mean(rows, f'{state}_msd_lag5_um2'):.1f} |\n")
        f.write("\n## State-corrected speed comparison\n\n")
        f.write("Raw mean speed (the mistake) blends both states. Real "
                "biology lives in the per-state values above.\n\n")
        for cond in CONDITION_ORDER:
            if cond not in by_cond:
                continue
            rows = by_cond[cond]
            balled_rows = [r for r in rows
                            if r["ball_n_speed_samples"] > 0]
            attached_rows = [r for r in rows
                              if r["atta_n_speed_samples"] > 0]
            ball_speed = (_safe_mean(balled_rows, "ball_mean_speed_um_per_min")
                          if balled_rows else float("nan"))
            atta_speed = (_safe_mean(attached_rows, "atta_mean_speed_um_per_min")
                          if attached_rows else float("nan"))
            ball_frac = float(np.mean([r["state_frac_balled"]
                                         for r in rows]))
            atta_frac = float(np.mean([r["state_frac_attached"]
                                         for r in rows]))
            f.write(f"* **{cond}**: balled {ball_speed:.2f} µm/min "
                    f"({100*ball_frac:.0f}% of frames), "
                    f"attached {atta_speed:.2f} µm/min "
                    f"({100*atta_frac:.0f}% of frames)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    npzs = sorted(glob.glob(os.path.join(args.cache_dir, "Pos*.npz")))
    print(f"Analyzing {len(npzs)} filtered recordings.\n", flush=True)

    all_cells = []   # CSV-friendly rows (no _ underscore keys)
    all_cells_full = []  # (cell_row, full_data_with_msd_curves)
    cond_lookup = {}
    for npz in npzs:
        name = os.path.basename(npz).replace(".npz", "")
        pos, cond = parse_condition(name)
        cond_lookup[name] = cond
        tracks = load_filtered_tracks(npz)
        if not tracks:
            print(f"  {name}: no tracks"); continue
        for t in tracks:
            full = analyze_one_track(t)
            if full is None:
                continue
            full["recording"] = name
            full["_condition"] = cond
            row = {k: v for k, v in full.items() if not k.startswith("_")}
            all_cells.append(row)
            all_cells_full.append((row, full))
        print(f"  {name} ({cond}): {len(tracks)} cells analyzed",
              flush=True)

    # Write per-cell CSV (no MSD curves)
    csv_path = os.path.join(args.out_dir, "per_cell_state.csv")
    write_per_cell_csv(all_cells, csv_path)
    print(f"\nWrote {csv_path} ({len(all_cells)} cells)")

    # Aggregate by condition
    by_cond = aggregate_by_condition(all_cells, cond_lookup)

    write_state_composition(by_cond,
                             os.path.join(args.out_dir,
                                            "state_composition.csv"))
    write_motility_by_state_condition(
        by_cond, os.path.join(args.out_dir,
                                "motility_by_state_condition.csv"))
    print("Wrote state composition + motility-by-state CSVs")

    # Figures
    figure_state_composition(by_cond,
                              os.path.join(fig_dir,
                                            "state_composition.png"))
    figure_speed_by_state(by_cond,
                           os.path.join(fig_dir,
                                          "speed_by_state.png"))
    figure_msd_by_state(by_cond,
                         os.path.join(fig_dir,
                                        "msd_by_state.png"),
                         all_cells_full)
    figure_persistence_by_state(by_cond,
                                 os.path.join(fig_dir,
                                                "persistence_by_state.png"))
    figure_shape_by_state(all_cells_full,
                           os.path.join(fig_dir,
                                          "shape_by_state.png"))
    print(f"Wrote 5 figures to {fig_dir}")

    # Markdown report
    write_report(by_cond, os.path.join(args.out_dir, "report.md"))
    print(f"Wrote report.md")

    from output.run_metadata import write_run_metadata
    write_run_metadata(
        out_path=os.path.join(args.out_dir, "RUN_METADATA.md"),
        title="IC295 state-stratified motility (balled vs attached)",
        sections={
            "Source": (
                f"`{args.cache_dir}/Pos*.npz` — multi_metric Cy5-filtered\n"
                f"tracks from `apply_cy5_filter_to_results.py`."),
            "Classification": (
                "Per cell-frame: shape metrics (circularity, solidity)\n"
                "from skimage.regionprops. Rules:\n"
                "  balled       if circ ≥ 0.80 AND solid ≥ 0.92\n"
                "  attached     if circ ≤ 0.55 OR  solid ≤ 0.85\n"
                "  transitional otherwise"),
            "Motility metrics (per state)": (
                "* speed (µm/min): per-step within state segments\n"
                "  ≥ 3 frames; capped at 15 µm/min.\n"
                "* MSD vs lag: per-segment ≥ 10 frames, max lag 20.\n"
                "* persistence: lag-1 cosine of velocity unit vectors,\n"
                "  per-segment ≥ 5 frames.\n"
                "* total displacement / straightness per segment."),
            "Outputs": (
                "* `per_cell_state.csv` — one row per cell with all\n"
                "  state-stratified metrics.\n"
                "* `state_composition.csv` — per condition: % time\n"
                "  in each state.\n"
                "* `motility_by_state_condition.csv` — per (cond × state):\n"
                "  speed, persistence, MSD, displacement, shape.\n"
                "* `report.md` — interpretive summary.\n"
                "* `figures/` — composition, speed, MSD, persistence,\n"
                "  shape-by-state plots."),
        },
        rerun_cli=(
            f"conda run -n cellpose python "
            f"scripts/analyze_state_motility.py \\\n"
            f"    --cache-dir {args.cache_dir} \\\n"
            f"    --out-dir {args.out_dir}"),
    )


if __name__ == "__main__":
    main()
