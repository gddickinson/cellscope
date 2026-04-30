"""Run pipelines once and cache outputs for figure generation.

Run from `cellpose` env. DIC pipelines subprocess once into `cellpose4`
for cpsam_dic detection — strictly one-way, no env round-trips, so gap
fill and DeepSea complete reliably.

Outputs (results/figure_pipelines/):
  pos0_wt_60f.npz       — single-cell DIC (frames + mask stack)
  pos17_wt_60f.npz      — multi-cell DIC (frames + labels + tracks)

Each .npz contains:
  frames        (N, H, W) uint8
  masks         (N, H, W) bool        — single-cell only
  labels        (N, H, W) int32       — multi-cell only (tracker IDs)
  tracks_n      int                   — multi-cell only
  track_<i>_stack    (N, H, W) bool   — per-cell mask stack
  track_<i>_centroids (N, 2) float    — per-cell centroids (NaN where missing)
  track_<i>_first    int              — first frame of track
  track_<i>_last     int              — last frame of track

Usage:
  conda run -n cellpose python scripts/cache_pipelines.py
  conda run -n cellpose python scripts/cache_pipelines.py --force
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports, benchmark_data_root  # noqa
setup_imports()

import numpy as np
import tifffile

OUT = "results/figure_pipelines"
os.makedirs(OUT, exist_ok=True)


def safe_read_tiff(path):
    """Read TIFF stack tolerating big-endian float32 (numpy 2.0)."""
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


def load_uint8(path, start, end):
    s = safe_read_tiff(path)
    if s.dtype != np.uint8:
        p1, p99 = np.percentile(s, [1, 99])
        s = np.clip((s.astype(np.float32) - p1) /
                    max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
    return s[start:end]


def cache_single_cell(out_path, frames):
    """Run hybrid_dic + DeepSea on a single-cell stack."""
    from core.hybrid_dic import detect_hybrid_dic
    print(f"  hybrid_dic on {len(frames)} frames (DeepSea, no TTA)…")
    t0 = time.time()
    masks, missed = detect_hybrid_dic(
        frames,
        model_path="data/models/cpsam_dic",
        use_preprocess=False,
        use_deepsea=True,
        use_retry=False,
        use_tta=False)
    print(f"  done in {time.time() - t0:.0f}s, missed: {len(missed)}")
    np.savez_compressed(out_path,
                        frames=frames,
                        masks=masks)
    print(f"  saved {out_path}")


def cache_multi_cell(out_path, frames):
    """Run hybrid_dic_multi with TTA + DeepSea + gap fill."""
    from core.hybrid_dic import detect_hybrid_dic_multi
    print(f"  hybrid_dic_multi on {len(frames)} frames "
          f"(DeepSea + gap fill, no TTA)…")
    t0 = time.time()

    def cb(msg, pct):
        print(f"    [{pct:3d}%] {msg}")

    result = detect_hybrid_dic_multi(
        frames,
        progress_fn=cb,
        model_path="data/models/cpsam_dic",
        min_area_px=500,
        use_preprocess=False,
        use_deepsea=True,
        use_retry=False,
        use_gap_fill=True,
        use_tta=False)
    elapsed = time.time() - t0
    n_tracks = len(result["tracks"])
    print(f"  done in {elapsed:.0f}s; {n_tracks} tracks")

    # Pack tracks. Compute per-track centroids on the fly.
    from core.tracking import extract_centroids
    save = {
        "frames": frames,
        "labels": result["labels"],
        "tracks_n": np.array(n_tracks),
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
    np.savez_compressed(out_path, **save)
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-run even if cache exists")
    args = ap.parse_args()

    bd = benchmark_data_root()

    # Recording 1: single-cell DIC, very clean
    out_a = f"{OUT}/pos0_wt_60f.npz"
    if args.force or not os.path.exists(out_a):
        print(f"\n=== Recording A: pos0_wt 60-frame slice (single-cell) ===")
        path = bd / "data" / "examples" / "jesse_wt" / "pos0_wt.ome.tif"
        # Frames 20-79 — skip the first 20 in case the cell isn't yet
        # well-positioned, then 60 frames of stable migration
        frames = load_uint8(str(path), 20, 80)
        cache_single_cell(out_a, frames)
    else:
        print(f"  [skip] {out_a} already cached (use --force to redo)")

    # Recording 2: multi-cell DIC, 4 cells
    out_b = f"{OUT}/pos17_wt_60f.npz"
    if args.force or not os.path.exists(out_b):
        print(f"\n=== Recording B: pos17_wt 60-frame slice (multi-cell) ===")
        path = bd / "data" / "examples" / "jesse_wt" / "pos17_wt.ome.tif"
        # Frames 30-89 — middle of the recording where most cells are stable
        frames = load_uint8(str(path), 30, 90)
        cache_multi_cell(out_b, frames)
    else:
        print(f"  [skip] {out_b} already cached (use --force to redo)")

    print(f"\nAll cached to {OUT}/")


if __name__ == "__main__":
    main()
