"""Full analysis on the new Ignasi recordings.

Best-effort detection options for these 2048×2048 phase recordings:
  * Channel 0 only (channel 1 is empty)
  * Flat-field correction (subtract Gaussian-blurred background, σ=80)
    — essential, the raw frames have severe vignette
  * hybrid_cpsam_multi pipeline with TTA on by default (cpsam at
    augment=True averages 4 rotations; ~2.5× slower per frame but
    recovers the cells the base model misses on these sparse-field
    recordings — the no-TTA smoke test found 0 cells in Pos27-GOF
    despite cells being visible)
      - cpsam (cellpose 4 ViT) for instance segmentation
      - per-cell DeepSea refinement
      - Hungarian multi-cell tracking
      - track gap fill via cpsam(augment=True) + CP3 fallback

Output (under results/ignasi_new_full/):
  <pos>_<cond>.npz       pipeline cache (frames, labels, per-track stacks)
  <pos>_<cond>.html      quality dashboard
  <pos>_<cond>_overlay.tif   Fiji-ready overlay TIFF
  summary.csv            per-recording stats
  by_condition.csv       aggregate by WT/KO/GOF/Y1/DMSO
  report.md              human-readable summary

Run from cellpose4 env (hybrid_cpsam_multi requires cpsam):

  conda run -n cellpose4 python scripts/run_ignasi_new_full.py \\
      --src /Users/george/Desktop/ignasi_cellscope_test_data
"""
import argparse
import csv
import glob
import json
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

OUT_DIR = "results/ignasi_new_full"
PIXEL_SIZE_UM = 0.6523        # from metadata sidecar
INTERVAL_MIN = 10.0           # 600000 ms / 60000
SPEED_CAP_UM_PER_MIN = 15.0


_COND_RE = re.compile(r"_(Pos\d+)-(WT|KO|GOF|Y1|DMSO)", re.I)


def parse_condition(filename):
    m = _COND_RE.search(filename)
    return (m.group(1), m.group(2).upper()) if m else ("?", "?")


def _read_metadata(tif_path):
    sidecar = tif_path.replace(".ome.tif", "_metadata.txt")
    out = {"n_channels": 1, "n_frames": None}
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            txt = f.read()
        if (m := re.search(r'"Channels"\s*:\s*(\d+)', txt)):
            out["n_channels"] = int(m.group(1))
        if (m := re.search(r'"Frames"\s*:\s*(\d+)', txt)):
            out["n_frames"] = int(m.group(1))
    return out


def flat_field(frame_uint16, sigma=80):
    from scipy.ndimage import gaussian_filter
    f = frame_uint16.astype(np.float32)
    bg = gaussian_filter(f, sigma=sigma)
    bg = np.where(bg < 1, 1, bg)
    return f / bg - 1.0


def load_recording_uint8(tif_path, channel=0, max_frames=None):
    """Load a recording, take channel 0, apply flat-field, return uint8."""
    meta = _read_metadata(tif_path)
    nch = meta["n_channels"]
    n = meta["n_frames"]
    with tifffile.TiffFile(tif_path) as tf:
        if n is None:
            n = max(1, len(tf.pages) // nch)
        if max_frames is not None:
            n = min(n, max_frames)
        out = np.empty((n, *tf.pages[0].shape), dtype=np.uint8)
        for i in range(n):
            page = i * nch + channel
            raw = tf.pages[page].asarray()
            flat = flat_field(raw)
            p1, p99 = np.percentile(flat, [1, 99])
            out[i] = np.clip((flat - p1) / max(p99 - p1, 1e-6) * 255,
                             0, 255).astype(np.uint8)
    return out


def cache_and_report(out_stem, frames, name, run_pipeline_fn, condition):
    """Run pipeline + save .npz + quality dashboard. Returns summary row."""
    npz_path = f"{OUT_DIR}/{out_stem}.npz"
    html_path = f"{OUT_DIR}/{out_stem}.html"

    if os.path.exists(npz_path):
        print(f"  [cache hit] {npz_path}")
    else:
        print(f"  running pipeline…")
        t0 = time.time()
        result = run_pipeline_fn(frames)
        print(f"    done in {time.time() - t0:.0f}s")
        from core.tracking import extract_centroids
        save = {
            "frames": frames,
            "labels": result["labels"],
            "tracks_n": np.array(len(result["tracks"])),
        }
        for i, t in enumerate(result["tracks"]):
            s = t.get("stack")
            if s is None or not s.any():
                continue
            save[f"track_{i}_stack"] = s
            save[f"track_{i}_centroids"] = extract_centroids(s)
            present = np.where(s.any(axis=(1, 2)))[0]
            save[f"track_{i}_first"] = np.array(int(present.min()))
            save[f"track_{i}_last"] = np.array(int(present.max()))
        np.savez_compressed(npz_path, **save)
        print(f"  saved {npz_path}")

    z = np.load(npz_path, allow_pickle=False)
    tracks = []
    for i in range(200):
        if f"track_{i}_stack" not in z.files:
            continue
        tracks.append({"id": i, "stack": z[f"track_{i}_stack"]})

    from output.quality_report import write_quality_report
    write_quality_report(html_path,
                         frames=z["frames"],
                         labels=z["labels"],
                         tracks=tracks,
                         recording_name=name,
                         um_per_px=PIXEL_SIZE_UM, dt_min=INTERVAL_MIN)
    print(f"  saved {html_path}")

    # Per-recording stats
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
    return {
        "name": out_stem,
        "condition": condition,
        "n_frames": n_frames,
        "n_frames_detected": detected,
        "detection_rate": round(detected / max(n_frames, 1), 3),
        "n_tracks": int(z["tracks_n"]) if "tracks_n" in z.files else 0,
        "n_kept_tracks": len(track_speeds),
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


def _print_progress(msg, pct):
    """Live progress callback — prints to stdout with flush."""
    print(f"    [{pct:>3}%] {msg}", flush=True)


def run_pipeline_for_frame(frames, use_tta=False,
                           cpsam_model_path=None,
                           use_gap_fill=False,
                           use_fallback=False,
                           use_deepsea=True):
    """The 'best detection' pipeline for these recordings.

    Defaults chosen from the model comparison + the Pos19_KO timing
    breakdown:
      * cpsam_base (no model_path) — 11.7 cells/frame vs cpsam_dic 4.6
      * use_tta=False — TTA actually hurt slightly on these recordings
        in the comparison, and triples per-frame time (27s → 77s)
      * use_gap_fill=False — gap fill on 2048² with TTA spent ~3.8 h
        per recording on Pos19_KO (176 gaps × 77 s). cpsam_base hits
        100% per-frame detection so gap fill rarely helps anyway;
        Hungarian tracker bridges brief gaps via interpolation.
      * use_fallback=False — same reasoning; no empty frames expected
      * use_deepsea=True — refines cell boundaries, runs ONCE per
        frame regardless of cell count, so cheap.

    cpsam_model_path: optional path to a fine-tuned cpsam model
    (e.g. data/models/cpsam_dic). None = default cpsam (the winner).
    """
    if cpsam_model_path:
        os.environ["CPSAM_PRETRAINED"] = cpsam_model_path
    else:
        os.environ.pop("CPSAM_PRETRAINED", None)
    from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi
    return detect_hybrid_cpsam_multi(
        frames,
        progress_fn=_print_progress,
        min_area_px=200,
        use_fallback=use_fallback,
        use_deepsea=use_deepsea,
        use_gap_fill=use_gap_fill,
        use_tta=use_tta)


def export_overlay_tiff(npz_path, out_dir):
    """Write Fiji-ready (image, labels) TIFF pair for each cache."""
    from scripts.cellscope_export_fiji import export_one  # noqa
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
                    "mean_n_tracks", "mean_lifetime_frames",
                    "mean_speed_um_per_min", "median_speed_um_per_min",
                    "mean_detection_rate"])
        for cond, rs in sorted(by.items()):
            w.writerow([
                cond, len(rs),
                round(float(np.mean([r["n_kept_tracks"] for r in rs])), 1),
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
        f.write("# Ignasi recordings — full analysis\n\n")
        f.write("## Per-recording\n\n")
        f.write("| Position | Condition | Frames | Detected | Tracks | "
                "Kept | Mean lifetime | Mean speed (µm/min) | Median |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['name'].split('_')[0]} | {r['condition']} | "
                    f"{r['n_frames']} | {r['n_frames_detected']} | "
                    f"{r['n_tracks']} | {r['n_kept_tracks']} | "
                    f"{r['mean_track_lifetime_frames']} | "
                    f"{r['mean_speed_um_per_min']} | "
                    f"{r['median_speed_um_per_min']} |\n")
        f.write("\n## Aggregate by condition\n\n")
        f.write("| Condition | n | Mean tracks | Mean lifetime | "
                "Mean speed | Median speed |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for cond, rs in sorted(by.items()):
            f.write(f"| {cond} | {len(rs)} | "
                    f"{np.mean([r['n_kept_tracks'] for r in rs]):.1f} | "
                    f"{np.mean([r['mean_track_lifetime_frames'] for r in rs]):.1f} | "
                    f"{np.mean([r['mean_speed_um_per_min'] for r in rs]):.3f} | "
                    f"{np.mean([r['median_speed_um_per_min'] for r in rs]):.3f} |\n")
    print(f"Report:    {md_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames per recording (default = full)")
    ap.add_argument("--positions", nargs="+",
                    help="restrict to specific positions (Pos5 Pos19 ...)")
    ap.add_argument("--tta", action="store_true",
                    help="enable cpsam test-time augmentation "
                         "(3× slower; on these recordings it didn't "
                         "help in the comparison)")
    ap.add_argument("--cpsam-model", default=None,
                    help="path to cpsam fine-tune (default = base cpsam). "
                         "e.g. data/models/cpsam_dic")
    ap.add_argument("--gap-fill", action="store_true",
                    help="enable track gap fill (slow! ~4 h per "
                         "recording on these 2048×2048 frames)")
    args = ap.parse_args()
    use_tta = bool(args.tta)
    use_gap_fill = bool(args.gap_fill)

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

    rows = []
    for tif in tifs:
        base = os.path.basename(tif).replace(".ome.tif", "")
        pos, cond = parse_condition(base)
        stem = f"{pos}_{cond}"
        print(f"\n=== {stem} ===")
        npz_path = f"{OUT_DIR}/{stem}.npz"
        if os.path.exists(npz_path):
            print(f"  [skip detection — cache exists]")
            frames = None  # don't load if not needed
        else:
            print(f"  loading {os.path.basename(tif)} (flat-field corrected)")
            t0 = time.time()
            frames = load_recording_uint8(tif, channel=0,
                                           max_frames=args.max_frames)
            print(f"    loaded {frames.shape} {frames.dtype} "
                  f"in {time.time() - t0:.0f}s")
        try:
            pipe = lambda f: run_pipeline_for_frame(
                f, use_tta=use_tta,
                cpsam_model_path=args.cpsam_model,
                use_gap_fill=use_gap_fill)
            row = cache_and_report(stem, frames, stem, pipe, cond)
            rows.append(row)
        except Exception as e:
            print(f"  [FAIL] {stem}: {e}")
            continue
        # Overlay TIFF for Fiji
        try:
            export_overlay_tiff(npz_path, OUT_DIR)
        except Exception as e:
            print(f"  [overlay export skipped: {e}]")

    if rows:
        write_summary(rows)
        write_by_condition(rows)
        write_markdown(rows)


if __name__ == "__main__":
    main()
