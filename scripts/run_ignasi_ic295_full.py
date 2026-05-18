"""Full multichannel analysis on IC295 recordings (DIC + SiR-actin Cy5).

Pipeline:
  1. Load 3-channel TIFF (Cy5 ch0, DIC ch1, None ch2) → flat-field both
     channels (DIC σ=80, Cy5 σ=200), p1/p99 → uint8.
  2. Run hybrid_cpsam_multi on the DIC channel only — same proven
     config as IC293 full run (cpsam_base, no TTA, no gap-fill, deepsea).
  3. After tracking, compute per-track per-frame Cy5 stats
     (mean, p75, p95, presence score). Cy5 is **annotation, not filter**
     — the pilot showed some recordings (Pos14-KO) have weak Cy5 across
     the board so a hard filter would lose real cells.
  4. Save per-recording .npz with Cy5 features alongside masks.
  5. Summary CSV adds: cy5_mean_score, cy5_p75_mean, n_cy5_positive_tracks
     (track-mean score ≥ 0.15), so users can post-hoc filter low-Cy5
     tracks if needed.

Outputs (under results/ic295_full/):
  <pos>_<cond>.npz           pipeline + Cy5 feature cache
  <pos>_<cond>.html          quality dashboard
  <pos>_<cond>_overlay.tif   Fiji-ready DIC + label overlay
  summary.csv                per-recording stats (incl. Cy5 columns)
  by_condition.csv           aggregate by WT/KO/GOF/OT/Y1/DMSO
  report.md                  human-readable summary
  RUN_METADATA.md            reproducibility info

Run from cellpose4 env (hybrid_cpsam_multi requires cpsam):

  conda run -n cellpose4 python scripts/run_ignasi_ic295_full.py \\
      --src /Volumes/GeorgeDrive/ignasi/IC295
"""
import argparse
import csv
import glob
import os
import re
import sys
import time
import warnings
import logging

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

import numpy as np
import tifffile

OUT_DIR = "results/ic295_full"
PIXEL_SIZE_UM = 0.6523
INTERVAL_MIN = 10.0
SPEED_CAP_UM_PER_MIN = 15.0
CY5_POSITIVE_THRESHOLD = 0.15  # track-mean score above this = "Cy5+"
DIC_CHANNEL = 1
CY5_CHANNEL = 0

_COND_RE = re.compile(r"_(Pos\d+)-(WT|KO|GOF|OT|Y1|DMSO)", re.I)


def parse_condition(filename):
    m = _COND_RE.search(filename)
    return (m.group(1), m.group(2).upper()) if m else ("?", "?")


def _print_progress(msg, pct):
    print(f"    [{pct:>3}%] {msg}", flush=True)


def load_dic_and_cy5(tif_path, max_frames=None):
    """Load multichannel IC295 TIFF → (dic_u8, cy5_u8) NCYX uint8 stacks.

    Both channels flat-fielded + percentile-rescaled via
    core.multichannel utilities.
    """
    from core.multichannel import load_recording_multi
    return load_recording_multi(tif_path, dic_ch=DIC_CHANNEL,
                                 fluo_ch=CY5_CHANNEL,
                                 max_frames=max_frames)


def run_pipeline_for_frame(dic_frames, cy5_frames=None,
                            use_tta=False,
                            cpsam_model_path=None,
                            use_gap_fill=False,
                            use_fallback=False,
                            use_deepsea=True,
                            recover_with_cy5=False):
    """Same defaults as IC293 full run — winning config.

    recover_with_cy5: if True (and cy5_frames given), use Cy5 to find
        cells that cpsam(DIC) missed; crop+rerun cpsam(TTA) on each
        Cy5+ region not covered by a DIC mask. Defensive layer for
        false-negatives. Cheap when no recoveries fire.
    """
    if cpsam_model_path:
        os.environ["CPSAM_PRETRAINED"] = cpsam_model_path
    else:
        os.environ.pop("CPSAM_PRETRAINED", None)
    from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi
    return detect_hybrid_cpsam_multi(
        dic_frames,
        progress_fn=_print_progress,
        min_area_px=200,
        use_fallback=use_fallback,
        use_deepsea=use_deepsea,
        use_gap_fill=use_gap_fill,
        use_tta=use_tta,
        cy5_frames=cy5_frames,
        recover_with_cy5=recover_with_cy5)


def compute_track_cy5_features(track_stack, cy5_frames):
    """Per-frame Cy5 stats for one track stack.

    Returns dict {
        'mean':   (T,) float — per-frame mean Cy5 inside mask
        'p75':    (T,)
        'p95':    (T,)
        'score':  (T,) — Cy5 presence score (z vs local annulus, 0-1)
    }
    NaN where the track is absent in that frame.
    """
    from core.multichannel import (
        per_cell_cy5_features, cy5_presence_score)
    T = len(cy5_frames)
    mean = np.full(T, np.nan, dtype=np.float32)
    p75 = np.full(T, np.nan, dtype=np.float32)
    p95 = np.full(T, np.nan, dtype=np.float32)
    score = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        m = track_stack[t].astype(bool)
        if not m.any():
            continue
        feats = per_cell_cy5_features(m, cy5_frames[t])
        mean[t] = feats["mean"]
        p75[t] = feats["p75"]
        p95[t] = feats["p95"]
        score[t] = cy5_presence_score(m, cy5_frames[t])
    return {"mean": mean, "p75": p75, "p95": p95, "score": score}


def cache_and_report(out_stem, dic_frames, cy5_frames, name,
                      run_pipeline_fn, condition):
    """Run pipeline + compute Cy5 features + save .npz + dashboard."""
    npz_path = f"{OUT_DIR}/{out_stem}.npz"
    html_path = f"{OUT_DIR}/{out_stem}.html"

    if os.path.exists(npz_path):
        print(f"  [cache hit] {npz_path}")
    else:
        print(f"  running pipeline on DIC channel…")
        t0 = time.time()
        result = run_pipeline_fn(dic_frames, cy5_frames)
        print(f"    DIC pipeline done in {time.time() - t0:.0f}s "
              f"(Cy5-recovered cells: {result.get('n_cy5_recovered', 0)})")

        from core.tracking import extract_centroids
        save = {
            "frames": dic_frames,
            "cy5_frames": cy5_frames,
            "labels": result["labels"],
            "tracks_n": np.array(len(result["tracks"])),
            "n_cy5_recovered": np.array(
                int(result.get("n_cy5_recovered", 0))),
        }
        print(f"  scoring Cy5 per track per frame…")
        t1 = time.time()
        for i, t in enumerate(result["tracks"]):
            s = t.get("stack")
            if s is None or not s.any():
                continue
            save[f"track_{i}_stack"] = s
            save[f"track_{i}_centroids"] = extract_centroids(s)
            present = np.where(s.any(axis=(1, 2)))[0]
            save[f"track_{i}_first"] = np.array(int(present.min()))
            save[f"track_{i}_last"] = np.array(int(present.max()))
            cy5_feats = compute_track_cy5_features(s, cy5_frames)
            save[f"track_{i}_cy5_mean"] = cy5_feats["mean"]
            save[f"track_{i}_cy5_p75"] = cy5_feats["p75"]
            save[f"track_{i}_cy5_p95"] = cy5_feats["p95"]
            save[f"track_{i}_cy5_score"] = cy5_feats["score"]
        print(f"    Cy5 scoring done in {time.time() - t1:.0f}s")
        np.savez_compressed(npz_path, **save)
        print(f"  saved {npz_path}")

    z = np.load(npz_path, allow_pickle=False)
    tracks = []
    track_cy5_means = []
    track_cy5_scores = []
    for i in range(200):
        if f"track_{i}_stack" not in z.files:
            continue
        tracks.append({"id": i, "stack": z[f"track_{i}_stack"]})
        if f"track_{i}_cy5_mean" in z.files:
            tm = z[f"track_{i}_cy5_mean"]
            ts = z[f"track_{i}_cy5_score"]
            valid = ~np.isnan(tm)
            if valid.any():
                track_cy5_means.append(float(np.nanmean(tm)))
                track_cy5_scores.append(float(np.nanmean(ts)))

    from output.quality_report import write_quality_report
    write_quality_report(html_path,
                         frames=z["frames"],
                         labels=z["labels"],
                         tracks=tracks,
                         recording_name=name,
                         um_per_px=PIXEL_SIZE_UM, dt_min=INTERVAL_MIN)
    print(f"  saved {html_path}")

    from core.tracking import extract_centroids
    n_frames = int(z["frames"].shape[0])
    detected = int(sum(1 for L in z["labels"] if L.any()))
    track_speeds = []
    track_lifetimes = []
    for t in tracks:
        s = t["stack"]
        if not s.any():
            continue
        cents = extract_centroids(s)
        valid = ~np.isnan(cents[:, 0])
        if valid.sum() < 2:
            continue
        track_lifetimes.append(int(valid.sum()))
        d = np.diff(cents[valid], axis=0) * PIXEL_SIZE_UM
        speeds = np.linalg.norm(d, axis=1) / INTERVAL_MIN
        speeds = speeds[speeds <= SPEED_CAP_UM_PER_MIN]
        if len(speeds):
            track_speeds.append(float(speeds.mean()))

    n_cy5_pos = int(sum(1 for s in track_cy5_scores
                        if s >= CY5_POSITIVE_THRESHOLD))
    n_cy5_recovered = int(z["n_cy5_recovered"]) if (
        "n_cy5_recovered" in z.files) else 0
    return {
        "name": out_stem,
        "condition": condition,
        "n_frames": n_frames,
        "n_frames_detected": detected,
        "detection_rate": round(detected / max(n_frames, 1), 3),
        "n_tracks": int(z["tracks_n"]) if "tracks_n" in z.files else 0,
        "n_kept_tracks": len(track_speeds),
        "n_cy5_recovered_cells": n_cy5_recovered,
        "n_cy5_positive_tracks": n_cy5_pos,
        "cy5_mean_score": (round(float(np.mean(track_cy5_scores)), 3)
                            if track_cy5_scores else 0.0),
        "cy5_p75_mean": (round(float(np.mean(track_cy5_means)), 1)
                          if track_cy5_means else 0.0),
        "mean_track_lifetime_frames": (
            round(float(np.mean(track_lifetimes)), 1)
            if track_lifetimes else 0),
        "mean_speed_um_per_min": (
            round(float(np.mean(track_speeds)), 4)
            if track_speeds else 0.0),
        "median_speed_um_per_min": (
            round(float(np.median(track_speeds)), 4)
            if track_speeds else 0.0),
    }


def export_overlay_tiff(npz_path, out_dir):
    from scripts.cellscope_export_fiji import export_one
    export_one(npz_path, out_dir)


def write_summary(rows):
    out_csv = f"{OUT_DIR}/summary.csv"
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSummary: {out_csv}  ({len(rows)} recordings)")


def write_by_condition(rows):
    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r)
    out_csv = f"{OUT_DIR}/by_condition.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_recordings",
                    "mean_n_tracks", "mean_n_cy5_positive_tracks",
                    "mean_cy5_score", "mean_lifetime_frames",
                    "mean_speed_um_per_min", "median_speed_um_per_min",
                    "mean_detection_rate"])
        for cond, rs in sorted(by.items()):
            w.writerow([
                cond, len(rs),
                round(float(np.mean([r["n_kept_tracks"] for r in rs])), 1),
                round(float(np.mean(
                    [r["n_cy5_positive_tracks"] for r in rs])), 1),
                round(float(np.mean(
                    [r["cy5_mean_score"] for r in rs])), 3),
                round(float(np.mean(
                    [r["mean_track_lifetime_frames"] for r in rs])), 1),
                round(float(np.mean(
                    [r["mean_speed_um_per_min"] for r in rs])), 4),
                round(float(np.mean(
                    [r["median_speed_um_per_min"] for r in rs])), 4),
                round(float(np.mean([r["detection_rate"] for r in rs])), 3),
            ])
    print(f"By-condition: {out_csv}")


def write_markdown(rows):
    md_path = f"{OUT_DIR}/report.md"
    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r)
    with open(md_path, "w") as f:
        f.write("# IC295 multichannel — full analysis\n\n")
        f.write("DIC detection (cpsam_base, no TTA) + Cy5 SiR-actin "
                "annotation. Cy5 is annotation, not filter; per-track "
                "score (`cy5_mean_score`) lets users post-hoc filter "
                f"low-Cy5 tracks (suggested ≥ {CY5_POSITIVE_THRESHOLD}).\n\n")
        f.write("## Per-recording\n\n")
        f.write("| Position | Cond | Frames | Det | Tracks | Cy5+ | "
                "Cy5 score | Lifetime | Speed (µm/min) | Median |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['name'].split('_')[0]} | {r['condition']} | "
                    f"{r['n_frames']} | {r['n_frames_detected']} | "
                    f"{r['n_kept_tracks']} | {r['n_cy5_positive_tracks']} | "
                    f"{r['cy5_mean_score']} | "
                    f"{r['mean_track_lifetime_frames']} | "
                    f"{r['mean_speed_um_per_min']} | "
                    f"{r['median_speed_um_per_min']} |\n")
        f.write("\n## By condition\n\n")
        f.write("| Cond | n | Mean tracks | Mean Cy5+ | Mean Cy5 score | "
                "Mean lifetime | Mean speed | Median speed |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for cond, rs in sorted(by.items()):
            f.write(f"| {cond} | {len(rs)} | "
                    f"{np.mean([r['n_kept_tracks'] for r in rs]):.1f} | "
                    f"{np.mean([r['n_cy5_positive_tracks'] for r in rs]):.1f} | "
                    f"{np.mean([r['cy5_mean_score'] for r in rs]):.3f} | "
                    f"{np.mean([r['mean_track_lifetime_frames'] for r in rs]):.1f} | "
                    f"{np.mean([r['mean_speed_um_per_min'] for r in rs]):.3f} | "
                    f"{np.mean([r['median_speed_um_per_min'] for r in rs]):.3f} |\n")
    print(f"Report:    {md_path}")


def write_metadata(args, t_start, n_recordings):
    from output.run_metadata import write_run_metadata
    write_run_metadata(
        out_path=os.path.join(OUT_DIR, "RUN_METADATA.md"),
        title="IC295 full multichannel analysis (DIC + Cy5 SiR-actin)",
        sections={
            "Source": (
                f"`{args.src}` — IC295 dataset, 19 recordings × 97 frames\n"
                f"× 3 channels (Cy5 ch0, DIC 10x ch1, empty ch2)\n"
                f"× 2048×2048 uint16. Pixel {PIXEL_SIZE_UM} µm/px,\n"
                f"interval {INTERVAL_MIN} min."),
            "Conditions": "WT (4) · KO (3) · GOF (3) · OT (3) · Y1 (3) · DMSO (3)",
            "Preprocessing": (
                "* DIC: flat-field σ=80, p1/p99 → uint8\n"
                "* Cy5: flat-field σ=200, p1/p99.5 → uint8\n"
                "  (σ broader for Cy5 — fluorescent clusters span\n"
                "   larger spatial scales than DIC features)"),
            "Detection pipeline": (
                "1. cpsam_base (cellpose 4 ViT, no fine-tune, no TTA)\n"
                "2. **Cy5 recovery (Tier 2)**: for each frame, find\n"
                "   bright Cy5 regions not covered by any cpsam mask;\n"
                "   crop the DIC + rerun cpsam(TTA) on the crop only;\n"
                "   add any new detections. Defensive layer for cpsam\n"
                "   false-negatives. Disable with --no-cy5-recovery.\n"
                "3. per-cell DeepSea boundary refinement\n"
                "4. Hungarian multi-cell tracking with division detection\n"
                "5. Cy5 stats per track per frame: mean, p75, p95,\n"
                "   presence score (z vs local 30-px annulus)\n"
                "6. **No Cy5-based filtering** — Cy5 is annotation only.\n"
                "   Pilot showed some recordings (Pos14-KO) have weak\n"
                "   Cy5 across the board so a hard filter would lose\n"
                "   real cells. Per-track `cy5_mean_score` lets users\n"
                f"   post-hoc filter (suggested ≥{CY5_POSITIVE_THRESHOLD})."),
            "Why this config": (
                "Same proven defaults as IC293 full run — model "
                "comparison showed cpsam_base wins (11.7 cells/frame "
                "vs cpsam_dic 4.6); TTA didn't help; gap fill on "
                "2048² with TTA cost ~4h/recording for marginal "
                "improvement (cpsam_base hits 100% per-frame "
                "detection)."),
            "Outputs": (
                "* `<pos>_<cond>.npz` — frames + labels + per-track\n"
                "  stacks + per-track Cy5 features (mean/p75/p95/score)\n"
                "* `<pos>_<cond>.html` — quality dashboard\n"
                "* `<pos>_<cond>_overlay.tif` — Fiji-ready DIC + labels\n"
                "* `summary.csv` — per-recording with Cy5 columns\n"
                "* `by_condition.csv` — aggregate by genotype/treatment\n"
                "* `report.md` — human-readable\n"
                "* `RUN_METADATA.md` — this file"),
        },
        rerun_cli=(
            f"conda run -n cellpose4 python "
            f"scripts/run_ignasi_ic295_full.py \\\n"
            f"    --src {args.src}"
            + (f" \\\n    --tta" if args.tta else "")
            + (f" \\\n    --gap-fill" if args.gap_fill else "")
            + (f" \\\n    --cpsam-model {args.cpsam_model}"
               if args.cpsam_model else "")),
        timing_seconds={
            "total_run": time.time() - t_start,
            "n_recordings_processed": n_recordings,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/Volumes/GeorgeDrive/ignasi/IC295")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--positions", nargs="+",
                    help="restrict to specific positions (Pos14 Pos26 ...)")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--cpsam-model", default=None)
    ap.add_argument("--gap-fill", action="store_true")
    ap.add_argument("--no-cy5-recovery", action="store_true",
                    help="disable Cy5-flagged false-negative recovery "
                         "(default ON: per frame, search Cy5+ regions "
                         "not in any DIC mask, crop+rerun cpsam(TTA))")
    ap.add_argument("--out-dir", default=None,
                    help="override output directory "
                         "(default: results/ic295_full)")
    args = ap.parse_args()
    use_cy5_recovery = not args.no_cy5_recovery

    if args.out_dir:
        global OUT_DIR
        OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    tifs = sorted(glob.glob(os.path.join(args.src, "*.ome.tif")))
    if args.positions:
        keep = {p.lower() for p in args.positions}
        tifs = [t for t in tifs
                if any(k in os.path.basename(t).lower() for k in keep)]
    if not tifs:
        print("No recordings selected.")
        sys.exit(1)
    print(f"Processing {len(tifs)} recordings → {OUT_DIR}/")

    t_start = time.time()
    rows = []
    for tif in tifs:
        base = os.path.basename(tif).replace(".ome.tif", "")
        pos, cond = parse_condition(base)
        stem = f"{pos}_{cond}"
        print(f"\n=== {stem} ===")
        npz_path = f"{OUT_DIR}/{stem}.npz"
        if os.path.exists(npz_path):
            print(f"  [skip detection — cache exists]")
            dic_frames = cy5_frames = None
        else:
            print(f"  loading {os.path.basename(tif)} (DIC + Cy5)")
            t0 = time.time()
            dic_frames, cy5_frames = load_dic_and_cy5(
                tif, max_frames=args.max_frames)
            print(f"    DIC {dic_frames.shape}, Cy5 {cy5_frames.shape} "
                  f"in {time.time() - t0:.0f}s")
        try:
            pipe = lambda f, c: run_pipeline_for_frame(
                f, cy5_frames=c, use_tta=args.tta,
                cpsam_model_path=args.cpsam_model,
                use_gap_fill=args.gap_fill,
                recover_with_cy5=use_cy5_recovery)
            row = cache_and_report(stem, dic_frames, cy5_frames,
                                    stem, pipe, cond)
            rows.append(row)
        except Exception as e:
            print(f"  [FAIL] {stem}: {e}")
            continue
        try:
            export_overlay_tiff(npz_path, OUT_DIR)
        except Exception as e:
            print(f"  [overlay export skipped: {e}]")

    if rows:
        write_summary(rows)
        write_by_condition(rows)
        write_markdown(rows)
        write_metadata(args, t_start, len(rows))


if __name__ == "__main__":
    main()
