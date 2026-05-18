"""Apply a Cy5 filter to all enriched NPZs and write filtered outputs.

Reads from `results/ic295_full_v2/` (NPZs with the v2 metrics).
For each recording:
  * applies the chosen filter (default: multi_metric)
  * writes a filtered NPZ keeping only kept tracks (re-numbered 1..N)
  * rebuilds the (N, H, W) int32 label stack so downstream readers
    only see kept cells
  * computes per-recording analytics on filtered tracks

Outputs (under results/ic295_filtered/):
  <pos>_<cond>.npz                filtered NPZ
  <pos>_<cond>_overlay.tif        Fiji-ready DIC + filtered labels
  summary.csv                     per-recording stats with filter info
  by_condition.csv                aggregate by genotype/treatment
  filter_drop_log.csv             per-track drop reasons
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

CACHE_DIR = "results/ic295_full_v2"
OUT_DIR = "results/ic295_filtered"
PIXEL_SIZE_UM = 0.6523
INTERVAL_MIN = 10.0
SPEED_CAP_UM_PER_MIN = 15.0


def parse_condition(name):
    import re
    m = re.search(r"(Pos\d+)_?-?(WT|KO|GOF|OT|Y1|DMSO)", name, re.I)
    return (m.group(1), m.group(2).upper()) if m else ("?", "?")


def load_tracks_from_npz(z):
    """Reconstruct track list from NPZ with all per-track metrics."""
    tracks = []
    n_max = int(z["tracks_n"]) if "tracks_n" in z.files else 0
    for tid in range(n_max * 3 + 5):
        if f"track_{tid}_stack" not in z.files:
            continue
        score = z.get(f"track_{tid}_cy5_score")
        valid = ~np.isnan(score) if score is not None else None
        mean_score = (float(np.nanmean(score))
                       if score is not None and valid.any() else 0.0)
        t = {
            "id": tid + 1,
            "stack": z[f"track_{tid}_stack"],
            "centroids": z.get(f"track_{tid}_centroids"),
            "first": (int(z[f"track_{tid}_first"])
                       if f"track_{tid}_first" in z.files else 0),
            "last": (int(z[f"track_{tid}_last"])
                      if f"track_{tid}_last" in z.files else 0),
            "cy5_score": score,
            "cy5_mean_score": mean_score,
        }
        for extra in ("cy5_mean", "cy5_p75", "cy5_p95", "cy5_io_ratio",
                       "cy5_inside_cv", "cy5_fraction_positive"):
            key = f"track_{tid}_{extra}"
            if key in z.files:
                t[extra] = z[key]
        tracks.append(t)
    return tracks


def rebuild_label_stack(tracks, shape):
    """Rebuild (N, H, W) int32 label stack from a kept-tracks list.
    New IDs are 1..len(tracks)."""
    out = np.zeros(shape, dtype=np.int32)
    for new_id, t in enumerate(tracks, start=1):
        s = t["stack"]
        for i in range(shape[0]):
            out[i][s[i] & (out[i] == 0)] = new_id
    return out


def per_recording_stats(kept, n_frames):
    """Compute summary stats from kept tracks (same as IC295 main run)."""
    speeds, lifetimes, areas = [], [], []
    cy5_scores = []
    for t in kept:
        cents = t.get("centroids")
        if cents is None:
            continue
        valid = ~np.isnan(cents[:, 0])
        if valid.sum() < 2:
            continue
        lifetimes.append(int(valid.sum()))
        d = np.diff(cents[valid], axis=0) * PIXEL_SIZE_UM
        sp = np.linalg.norm(d, axis=1) / INTERVAL_MIN
        sp = sp[sp <= SPEED_CAP_UM_PER_MIN]
        if len(sp):
            speeds.append(float(sp.mean()))
        s = t["stack"]
        per_frame_areas = s.reshape(s.shape[0], -1).sum(axis=1)
        per_frame_areas = per_frame_areas[per_frame_areas > 0]
        if len(per_frame_areas):
            areas.append(float(per_frame_areas.mean()))
        cy5_scores.append(t.get("cy5_mean_score", 0.0))
    return {
        "n_kept": len(kept),
        "mean_lifetime_frames": (round(float(np.mean(lifetimes)), 1)
                                  if lifetimes else 0),
        "mean_speed_um_per_min": (round(float(np.mean(speeds)), 4)
                                   if speeds else 0.0),
        "median_speed_um_per_min": (round(float(np.median(speeds)), 4)
                                     if speeds else 0.0),
        "mean_area_px": (round(float(np.mean(areas)), 0)
                          if areas else 0),
        "mean_cy5_score": (round(float(np.mean(cy5_scores)), 3)
                            if cy5_scores else 0.0),
    }


def export_overlay_tiff(npz_path, out_dir):
    try:
        from scripts.cellscope_export_fiji import export_one
        export_one(npz_path, out_dir)
    except Exception as e:
        print(f"    [overlay export skipped: {e}]", flush=True)


def filter_one(npz_in, out_dir, filter_mode):
    """Apply filter to one recording, write filtered NPZ + overlay."""
    from core.cy5_filter import apply_cy5_filter
    z = np.load(npz_in, allow_pickle=False)
    name = os.path.basename(npz_in).replace(".npz", "")
    pos, cond = parse_condition(name)
    frames = z["frames"]
    cy5_frames = z["cy5_frames"]
    tracks = load_tracks_from_npz(z)
    raw_n = len(tracks)
    kept, dropped, info = apply_cy5_filter(tracks, mode=filter_mode)

    # Build filtered NPZ
    new_labels = rebuild_label_stack(kept, frames.shape)
    save = {
        "frames": frames,
        "cy5_frames": cy5_frames,
        "labels": new_labels,
        "tracks_n": np.array(len(kept)),
        "filter_mode": np.array(filter_mode),
        "filter_n_raw": np.array(raw_n),
        "filter_n_kept": np.array(len(kept)),
        "filter_n_dropped": np.array(len(dropped)),
    }
    for new_id, t in enumerate(kept):
        save[f"track_{new_id}_stack"] = t["stack"]
        for k in ("centroids", "cy5_score", "cy5_mean", "cy5_p75",
                   "cy5_p95", "cy5_io_ratio", "cy5_inside_cv",
                   "cy5_fraction_positive"):
            v = t.get(k)
            if v is not None:
                save[f"track_{new_id}_{k}"] = v
        save[f"track_{new_id}_first"] = np.array(t["first"])
        save[f"track_{new_id}_last"] = np.array(t["last"])
    npz_out = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(npz_out, **save)

    # Drop-reasons log rows
    drop_rows = []
    for t in dropped:
        drop_rows.append({
            "recording": name,
            "old_track_id": t["id"],
            "first_frame": t.get("first", 0),
            "last_frame": t.get("last", 0),
            "cy5_mean_score": round(t.get("cy5_mean_score", 0.0), 3),
            "drop_reason": t.get("drop_reason", "?"),
        })

    # Stats
    stats = per_recording_stats(kept, len(frames))
    summary_row = {
        "name": name,
        "condition": cond,
        "n_frames": len(frames),
        "n_raw_tracks": raw_n,
        "n_kept_tracks": stats["n_kept"],
        "n_dropped_tracks": len(dropped),
        "filter_mode": filter_mode,
        "mean_lifetime_frames": stats["mean_lifetime_frames"],
        "mean_speed_um_per_min": stats["mean_speed_um_per_min"],
        "median_speed_um_per_min": stats["median_speed_um_per_min"],
        "mean_area_px": stats["mean_area_px"],
        "mean_cy5_score_kept": stats["mean_cy5_score"],
    }
    return summary_row, drop_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--filter-mode", default="multi_metric",
                    choices=["off", "conservative", "conservative_strict",
                             "adaptive", "adaptive_loose",
                             "multi_metric", "composite_score",
                             "consensus", "temporal_stability"])
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip Fiji overlay TIFF export (faster)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    npzs = sorted(glob.glob(os.path.join(args.cache_dir, "Pos*.npz")))
    print(f"Filtering {len(npzs)} recordings with mode={args.filter_mode}\n",
          flush=True)

    summary_rows = []
    drop_rows = []
    t_start = time.time()
    for npz in npzs:
        name = os.path.basename(npz).replace(".npz", "")
        print(f"  {name}…", flush=True)
        try:
            row, drops = filter_one(npz, args.out_dir, args.filter_mode)
            summary_rows.append(row)
            drop_rows.extend(drops)
            print(f"    kept {row['n_kept_tracks']}/{row['n_raw_tracks']}, "
                  f"mean speed {row['mean_speed_um_per_min']} µm/min",
                  flush=True)
            if not args.no_overlay:
                export_overlay_tiff(
                    os.path.join(args.out_dir, f"{name}.npz"),
                    args.out_dir)
        except Exception as e:
            print(f"    [FAIL] {e}", flush=True)
            continue
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min", flush=True)

    # Summary CSV
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"Wrote {csv_path}", flush=True)

    # By-condition aggregate
    by_cond = {}
    for r in summary_rows:
        by_cond.setdefault(r["condition"], []).append(r)
    bc_path = os.path.join(args.out_dir, "by_condition.csv")
    with open(bc_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_recordings", "total_kept_tracks",
                    "mean_kept_per_rec", "mean_lifetime",
                    "mean_speed_um_per_min", "median_speed_um_per_min",
                    "mean_area_px", "mean_cy5_score"])
        for cond, rs in sorted(by_cond.items()):
            w.writerow([
                cond, len(rs),
                sum(r["n_kept_tracks"] for r in rs),
                round(np.mean([r["n_kept_tracks"] for r in rs]), 1),
                round(np.mean([r["mean_lifetime_frames"] for r in rs]), 1),
                round(np.mean([r["mean_speed_um_per_min"] for r in rs]), 4),
                round(np.mean([r["median_speed_um_per_min"] for r in rs]), 4),
                round(np.mean([r["mean_area_px"] for r in rs]), 0),
                round(np.mean([r["mean_cy5_score_kept"] for r in rs]), 3),
            ])
    print(f"Wrote {bc_path}", flush=True)

    # Drop log
    drop_path = os.path.join(args.out_dir, "filter_drop_log.csv")
    if drop_rows:
        with open(drop_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(drop_rows[0].keys()))
            w.writeheader()
            for r in drop_rows:
                w.writerow(r)
        print(f"Wrote {drop_path}", flush=True)

    # Metadata
    from output.run_metadata import write_run_metadata
    write_run_metadata(
        out_path=os.path.join(args.out_dir, "RUN_METADATA.md"),
        title=f"IC295 filtered results — Cy5 mode={args.filter_mode}",
        sections={
            "Source": (
                f"`{args.cache_dir}/*.npz` — IC295 enriched NPZs from\n"
                f"`scripts/recompute_cy5_metrics.py` (which adds the\n"
                f"io_ratio, inside_cv, fraction_positive metrics on\n"
                f"top of the original cpsam(DIC) + Cy5-recovery run)."),
            "Filter": (
                f"`{args.filter_mode}` from `core.cy5_filter`.\n"
                f"See `core/cy5_filter.py` for definition."),
            "Outputs": (
                "* `<pos>_<cond>.npz` — filtered NPZ (kept tracks only,\n"
                "  re-labelled 1..N).\n"
                "* `<pos>_<cond>_overlay.tif` — Fiji-ready DIC + filtered\n"
                "  labels (skipped if --no-overlay).\n"
                "* `summary.csv` — per-recording stats with filter info.\n"
                "* `by_condition.csv` — aggregate by WT/KO/GOF/OT/Y1/DMSO.\n"
                "* `filter_drop_log.csv` — per-dropped-track reason.\n"
                "* `RUN_METADATA.md` — this file."),
            "Metrics": (
                f"Pixel size {PIXEL_SIZE_UM} µm/px, interval "
                f"{INTERVAL_MIN} min, speed cap "
                f"{SPEED_CAP_UM_PER_MIN} µm/min."),
        },
        rerun_cli=(
            f"conda run -n cellpose python "
            f"scripts/apply_cy5_filter_to_results.py \\\n"
            f"    --cache-dir {args.cache_dir} \\\n"
            f"    --out-dir {args.out_dir} \\\n"
            f"    --filter-mode {args.filter_mode}"),
        timing_seconds={"total_run": time.time() - t_start},
    )


if __name__ == "__main__":
    main()
