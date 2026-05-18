"""Combined IC295 batch1 + batch2 — filter + state-analyze inline.

Avoids writing 45 GB of filtered NPZs by:
  1. Loading each enriched NPZ (with v2 Cy5 metrics)
  2. Applying multi_metric Cy5 filter in memory
  3. Running state classification + per-state motility on kept tracks
  4. Streaming results straight to per_cell_state.csv

Output (under results/ic295_combined_state_analysis/):
  per_cell_state.csv          one row per filtered cell
  state_composition.csv       per-condition fractions
  motility_by_state_condition.csv   per (cond × state) means
  RUN_METADATA.md
"""
import argparse
import csv
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports
setup_imports()

import numpy as np

CACHE_DIR = "results/ic295_combined_v2"
OUT_DIR = "results/ic295_combined_state_analysis"
PIXEL_SIZE_UM = 0.6523
INTERVAL_MIN = 10.0
SPEED_CAP_UM_PER_MIN = 15.0
CONDITION_ORDER = ["WT", "KO", "GOF", "OT", "Y1", "DMSO"]


def parse_condition(name):
    import re
    m = re.search(r"(Pos\d+)_?-?(WT|KO|GOF|OT|Y1|DMSO)", name, re.I)
    return (m.group(1), m.group(2).upper()) if m else ("?", "?")


def load_tracks(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    tracks = []
    n_max = int(z["tracks_n"]) if "tracks_n" in z.files else 0
    for tid in range(n_max * 3 + 5):
        if f"track_{tid}_stack" not in z.files:
            continue
        score = z.get(f"track_{tid}_cy5_score")
        valid = ~np.isnan(score) if score is not None else None
        ms = (float(np.nanmean(score))
               if score is not None and valid.any() else 0.0)
        t = {
            "id": tid + 1,
            "stack": z[f"track_{tid}_stack"],
            "centroids": z.get(f"track_{tid}_centroids"),
            "cy5_score": score,
            "cy5_mean_score": ms,
        }
        for extra in ("cy5_io_ratio", "cy5_inside_cv",
                       "cy5_fraction_positive"):
            key = f"track_{tid}_{extra}"
            if key in z.files:
                t[extra] = z[key]
        tracks.append(t)
    return tracks


def analyze_track(track, recording, condition):
    """Same per-track analysis as analyze_state_motility.py."""
    from core.cell_state import (
        classify_track_states, state_fraction, per_state_means,
        STATE_BALLED, STATE_ATTACHED, STATE_TRANSITIONAL)
    from core.motility_state import (
        state_speeds, state_msd, state_persistence,
        state_total_displacement)

    sd = classify_track_states(track["stack"])
    states = sd["states"]
    metrics = sd["metrics"]
    cents = track["centroids"]
    if cents is None:
        return None
    out = {"recording": recording, "condition": condition,
           "track_id": track["id"]}
    out["lifetime_frames"] = int(sum(1 for s in states
                                       if s != "unknown"))
    out["state_frac_balled"] = state_fraction(states, STATE_BALLED)
    out["state_frac_attached"] = state_fraction(states, STATE_ATTACHED)
    out["state_frac_transitional"] = state_fraction(
        states, STATE_TRANSITIONAL)
    out["mean_circularity"] = float(np.nanmean(metrics["circularity"]))
    out["mean_area"] = float(np.nanmean(metrics["area"]))
    for state, prefix in ((STATE_BALLED, "ball"),
                           (STATE_ATTACHED, "atta")):
        sp = state_speeds(cents, states, state, PIXEL_SIZE_UM,
                            INTERVAL_MIN, speed_cap=SPEED_CAP_UM_PER_MIN)
        msd = state_msd(cents, states, state, PIXEL_SIZE_UM, max_lag=20)
        per = state_persistence(cents, states, state, PIXEL_SIZE_UM)
        td = state_total_displacement(cents, states, state, PIXEL_SIZE_UM)
        sh = per_state_means(metrics, states, state)
        out[f"{prefix}_n_speed_samples"] = int(len(sp))
        out[f"{prefix}_mean_speed_um_per_min"] = (
            float(np.mean(sp)) if len(sp) else float("nan"))
        out[f"{prefix}_median_speed_um_per_min"] = (
            float(np.median(sp)) if len(sp) else float("nan"))
        out[f"{prefix}_persistence_lag1"] = per["lag1"]
        out[f"{prefix}_msd_lag5_um2"] = (
            float(msd["msd"][4])
            if len(msd["msd"]) > 4 and not np.isnan(msd["msd"][4])
            else float("nan"))
        out[f"{prefix}_total_displacement_um"] = td["total_displacement_um"]
        out[f"{prefix}_straightness"] = td["straightness"]
        out[f"{prefix}_mean_circularity"] = sh.get("circularity",
                                                    float("nan"))
        out[f"{prefix}_mean_area_px"] = sh.get("area", float("nan"))
        out[f"{prefix}_mean_solidity"] = sh.get("solidity",
                                                 float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--filter-mode", default="multi_metric",
                    help="Cy5 filter mode (default: multi_metric)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "per_cell_state.csv")

    from core.cy5_filter import apply_cy5_filter

    npzs = sorted(glob.glob(os.path.join(args.cache_dir, "Pos*.npz")))
    print(f"Combined analysis: {len(npzs)} recordings → {args.out_dir}",
          flush=True)

    rows = []
    n_raw_total = 0
    n_kept_total = 0
    t0 = time.time()
    for i, npz_path in enumerate(npzs, 1):
        name = os.path.basename(npz_path).replace(".npz", "")
        pos, cond = parse_condition(name)
        all_tracks = load_tracks(npz_path)
        kept, dropped, info = apply_cy5_filter(
            all_tracks, mode=args.filter_mode)
        n_raw_total += len(all_tracks)
        n_kept_total += len(kept)
        for t in kept:
            row = analyze_track(t, name, cond)
            if row:
                rows.append(row)
        print(f"  [{i}/{len(npzs)}] {name} ({cond}): "
              f"kept {len(kept)}/{len(all_tracks)}", flush=True)

    print(f"\nTotal: {n_kept_total}/{n_raw_total} cells kept after "
          f"{args.filter_mode} filter ({time.time()-t0:.0f}s)",
          flush=True)

    # Write per-cell CSV
    if rows:
        fields = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {csv_path}")

    # Per-condition aggregates
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)
    sc_path = os.path.join(args.out_dir, "state_composition.csv")
    with open(sc_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_cells",
                    "mean_frac_balled", "mean_frac_attached",
                    "mean_frac_transitional"])
        for cond in CONDITION_ORDER:
            if cond not in by_cond:
                continue
            cs = by_cond[cond]
            w.writerow([cond, len(cs),
                         round(np.mean([r["state_frac_balled"]
                                          for r in cs]), 3),
                         round(np.mean([r["state_frac_attached"]
                                          for r in cs]), 3),
                         round(np.mean([r["state_frac_transitional"]
                                          for r in cs]), 3)])
    print(f"Wrote {sc_path}")

    mb_path = os.path.join(args.out_dir,
                            "motility_by_state_condition.csv")
    with open(mb_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "state", "n_cells",
                    "mean_speed_um_per_min", "median_speed_um_per_min",
                    "mean_persistence_lag1", "mean_msd_lag5_um2",
                    "mean_total_displacement_um", "mean_straightness",
                    "mean_circularity", "mean_area_px"])
        for cond in CONDITION_ORDER:
            if cond not in by_cond:
                continue
            for state in ("ball", "atta"):
                cs = [r for r in by_cond[cond]
                      if r[f"{state}_n_speed_samples"] > 0]
                if not cs:
                    w.writerow([cond, state, 0, "", "", "",
                                "", "", "", "", ""])
                    continue
                def _m(k):
                    vals = [r[k] for r in cs
                            if not np.isnan(r[k])]
                    return float(np.mean(vals)) if vals else 0.0
                w.writerow([cond, state, len(cs),
                            round(_m(f"{state}_mean_speed_um_per_min"), 4),
                            round(_m(f"{state}_median_speed_um_per_min"), 4),
                            round(_m(f"{state}_persistence_lag1"), 3),
                            round(_m(f"{state}_msd_lag5_um2"), 2),
                            round(_m(f"{state}_total_displacement_um"), 1),
                            round(_m(f"{state}_straightness"), 3),
                            round(_m(f"{state}_mean_circularity"), 3),
                            round(_m(f"{state}_mean_area_px"), 0)])
    print(f"Wrote {mb_path}")

    from output.run_metadata import write_run_metadata
    write_run_metadata(
        out_path=os.path.join(args.out_dir, "RUN_METADATA.md"),
        title=f"IC295 combined batch1+2 state analysis "
               f"(filter={args.filter_mode})",
        sections={
            "Source": (
                f"`{args.cache_dir}/Pos*.npz` — symlinks to enriched\n"
                f"NPZs from `recompute_cy5_metrics.py` for both\n"
                f"batches. Pos51-Y1 deduplicated (kept batch1 version)."),
            "Pipeline": (
                f"In-memory: load NPZ → apply {args.filter_mode}\n"
                f"Cy5 filter → state-classify each kept track\n"
                f"→ per-state speed/MSD/persistence/displacement\n"
                f"→ stream to per_cell_state.csv. No intermediate\n"
                f"NPZ writes (avoids ~45 GB disk usage)."),
            "Outputs": (
                "* `per_cell_state.csv` — one row per kept cell\n"
                "* `state_composition.csv` — per condition\n"
                "* `motility_by_state_condition.csv`\n"
                "* `RUN_METADATA.md`"),
        },
        rerun_cli=(
            f"conda run -n cellpose python "
            f"scripts/combined_state_analysis.py \\\n"
            f"    --cache-dir {args.cache_dir} \\\n"
            f"    --filter-mode {args.filter_mode}"),
    )
    print(f"Wrote RUN_METADATA.md")


if __name__ == "__main__":
    main()
