"""End-to-end pipeline run on every available recording.

Two cohorts:
  • DIC (Jesse): 60-frame slices through hybrid_dic_multi
  • Phase-contrast (Ignasi): full recording through hybrid_cpsam_multi

Outputs go to:
  results/full_dataset/<recording_name>.npz       — pipeline cache
  results/full_dataset/<recording_name>.html      — quality dashboard
  results/full_dataset/summary.csv                — per-recording stats

Run DIC (default, in cellpose env):
  conda run -n cellpose python scripts/run_full_dataset.py --modality dic

Run phase (cellpose4 env):
  conda run -n cellpose4 python scripts/run_full_dataset.py --modality phase

Or both in series:
  bash -c 'conda run -n cellpose python scripts/run_full_dataset.py --modality dic && conda run -n cellpose4 python scripts/run_full_dataset.py --modality phase'
"""
import argparse
import csv
import os
import sys
import time
import warnings
import logging

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports, benchmark_data_root  # noqa
setup_imports()

import numpy as np
import tifffile

OUT = "results/full_dataset"
os.makedirs(OUT, exist_ok=True)

UM_PER_PX = 0.65
DT_MIN = 5.0


def safe_read_tiff(path):
    try:
        return tifffile.imread(path)
    except AttributeError as e:
        if "newbyteorder" not in str(e):
            raise
    pages = []
    with tifffile.TiffFile(path) as tf:
        with open(path, "rb") as fh:
            for pg in tf.pages:
                bps = pg.bitspersample; sf = pg.sampleformat
                bo = pg.parent.byteorder
                offsets = pg.dataoffsets; bcs = pg.databytecounts
                chunks = []
                for off, n in zip(offsets, bcs):
                    fh.seek(off); chunks.append(fh.read(n))
                raw = b"".join(chunks)
                kind = {1: "u", 2: "i", 3: "f"}.get(sf, "u")
                dt = np.dtype(f"{bo}{kind}{bps // 8}")
                arr = np.frombuffer(raw, dtype=dt).reshape(
                    pg.imagelength, pg.imagewidth)
                if arr.dtype.byteorder not in ("=", "|"):
                    arr = arr.astype(arr.dtype.newbyteorder("=")).copy()
                pages.append(arr)
    return np.stack(pages) if len(pages) > 1 else pages[0]


def load_uint8(path, max_frames=None):
    s = safe_read_tiff(path)
    if s.dtype != np.uint8:
        p1, p99 = np.percentile(s, [1, 99])
        s = np.clip((s.astype(np.float32) - p1) /
                    max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
    return s if max_frames is None else s[:max_frames]


def cache_and_report(out_stem, frames, name, run_pipeline_fn, genotype):
    """Run pipeline + save cache + write quality report."""
    npz_path = f"{OUT}/{out_stem}.npz"
    html_path = f"{OUT}/{out_stem}.html"

    if os.path.exists(npz_path):
        print(f"  [cache hit] {npz_path}")
        z = np.load(npz_path, allow_pickle=False)
        labels = z.get("labels")
        n_tracks = int(z["tracks_n"]) if "tracks_n" in z.files else 0
    else:
        print(f"  running pipeline…")
        t0 = time.time()
        result = run_pipeline_fn(frames)
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.0f}s")

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
        labels = result["labels"]
        n_tracks = len(result["tracks"])
        print(f"  saved {npz_path}")

    # Quality report
    z = np.load(npz_path, allow_pickle=False)
    tracks = []
    for i in range(50):
        if f"track_{i}_stack" not in z.files:
            continue
        tracks.append({
            "id": i,
            "stack": z[f"track_{i}_stack"],
        })
    from output.quality_report import write_quality_report
    write_quality_report(html_path,
                         frames=z["frames"],
                         labels=z["labels"],
                         tracks=tracks,
                         recording_name=name,
                         um_per_px=UM_PER_PX, dt_min=DT_MIN)
    print(f"  saved {html_path}")

    # Compute summary stats for the CSV
    from core.tracking import extract_centroids
    n_frames = z["frames"].shape[0]
    detected = sum(1 for L in z["labels"] if L.any())
    track_speeds = []
    for t in tracks:
        s = t["stack"]
        if not s.any():
            continue
        cents = extract_centroids(s)
        valid = ~np.isnan(cents[:, 0])
        if valid.sum() < 2:
            continue
        d = np.diff(cents[valid], axis=0) * UM_PER_PX
        speeds = np.linalg.norm(d, axis=1) / DT_MIN
        speeds = speeds[speeds <= 15.0]   # cap at biologically plausible
        if len(speeds):
            track_speeds.append(float(speeds.mean()))
    return {
        "name": out_stem, "genotype": genotype,
        "n_frames": n_frames,
        "detection_rate": detected / max(n_frames, 1),
        "n_tracks": n_tracks,
        "n_kept_tracks": len(track_speeds),
        "mean_speed_um_per_min": (
            float(np.mean(track_speeds)) if track_speeds else 0.0),
        "median_speed_um_per_min": (
            float(np.median(track_speeds)) if track_speeds else 0.0),
    }


def run_dic_recordings(slice_len=60):
    """Jesse DIC recordings via hybrid_dic_multi."""
    from core.hybrid_dic import detect_hybrid_dic_multi

    def _pipe(frames):
        return detect_hybrid_dic_multi(
            frames, model_path="data/models/cpsam_dic",
            min_area_px=500,
            use_preprocess=False, use_deepsea=True, use_retry=False,
            use_gap_fill=True, use_tta=False)

    bd = benchmark_data_root()
    cohort = [
        ("dic_pos0_wt", "WT",
         bd / "data/examples/jesse_wt/pos0_wt.ome.tif"),
        ("dic_pos17_wt", "WT",
         bd / "data/examples/jesse_wt/pos17_wt.ome.tif"),
        ("dic_pos59_ko", "KO",
         bd / "data/examples/jesse_ko/pos59_ko.ome.tif"),
        ("dic_pos65_ko", "KO",
         bd / "data/examples/jesse_ko/pos65_ko.ome.tif"),
    ]
    rows = []
    for stem, geno, path in cohort:
        if not path.exists():
            print(f"\n=== {stem}: SKIP (missing) ===")
            continue
        print(f"\n=== {stem} ({geno}) ===")
        frames = load_uint8(str(path), max_frames=slice_len)
        rows.append(cache_and_report(stem, frames, stem, _pipe, geno))
    return rows


def run_phase_recordings(max_frames=None):
    """Ignasi phase-contrast recordings via hybrid_cpsam_multi."""
    from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi

    def _pipe(frames):
        return detect_hybrid_cpsam_multi(
            frames, min_area_px=500,
            use_fallback=True, use_deepsea=True, use_gap_fill=True)

    bd = benchmark_data_root()
    cohort = [
        ("phase_pos0_wt", "WT",
         bd / "data/ignasi/C1-IC293__1_MMStack_Pos0-WT.ome-1cropped.tif"),
        ("phase_pos2_wt", "WT",
         bd / "data/ignasi/IC293__1_MMStack_Pos2-WT.ome-cropped.tif"),
        ("phase_pos3_wt", "WT",
         bd / "data/ignasi/IC293__1_MMStack_Pos3-WT.ome-cropped.tif"),
        ("phase_pos17_ko", "KO",
         bd / "data/ignasi/IC293__1_MMStack_Pos17-KO.ome-cropped.tif"),
        ("phase_pos19_ko", "KO",
         bd / "data/ignasi/IC293__1_MMStack_Pos19-KO.ome-cropped.tif"),
    ]
    rows = []
    for stem, geno, path in cohort:
        if not path.exists():
            print(f"\n=== {stem}: SKIP (missing) ===")
            continue
        print(f"\n=== {stem} ({geno}) ===")
        frames = load_uint8(str(path), max_frames=max_frames)
        rows.append(cache_and_report(stem, frames, stem, _pipe, geno))
    return rows


def append_summary(rows):
    out_csv = f"{OUT}/summary.csv"
    existing = []
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            existing = list(csv.DictReader(f))
    by_name = {r["name"]: r for r in existing}
    for r in rows:
        # cast numeric values to strings (csv compatibility)
        rec = {k: (f"{v:.4f}" if isinstance(v, float) else str(v))
               for k, v in r.items()}
        by_name[r["name"]] = rec
    if not by_name:
        return
    fields = list(next(iter(by_name.values())).keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in by_name.values():
            w.writerow(r)
    print(f"\nSummary written to {out_csv} ({len(by_name)} recordings)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["dic", "phase", "both"],
                    default="both")
    ap.add_argument("--slice-len", type=int, default=60,
                    help="DIC frame budget per recording (default 60)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="phase frame budget per recording (default = full)")
    args = ap.parse_args()

    rows = []
    if args.modality in ("dic", "both"):
        try:
            rows.extend(run_dic_recordings(args.slice_len))
        except RuntimeError as e:
            print(f"  DIC pipeline aborted: {e}")
    if args.modality in ("phase", "both"):
        try:
            rows.extend(run_phase_recordings(args.max_frames))
        except RuntimeError as e:
            print(f"  Phase pipeline aborted: {e}")

    if rows:
        append_summary(rows)


if __name__ == "__main__":
    main()
