"""Cache a phase-contrast multi-cell pipeline run.

Phase-contrast recordings (Ignasi-style) show clearer cell migration
and division events than the slow keratinocyte DIC recordings used
elsewhere in the figure set. Cache once, render figures from cache.

Run from cellpose4 env (hybrid_cpsam_multi requires cellpose 4):
  conda run -n cellpose4 python scripts/cache_phase.py
"""
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

OUT = "results/figure_pipelines"
os.makedirs(OUT, exist_ok=True)


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


def main():
    bd = benchmark_data_root()
    path = (bd / "data" / "ignasi" /
            "IC293__1_MMStack_Pos3-WT.ome-cropped.tif")
    if not path.exists():
        print(f"MISSING: {path}")
        sys.exit(1)

    print(f"Loading {path.name} (97 frames)…")
    s = safe_read_tiff(str(path))
    if s.dtype != np.uint8:
        p1, p99 = np.percentile(s, [1, 99])
        s = np.clip((s.astype(np.float32) - p1) /
                    max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
    # Use full 97 frames for richer motion signal
    frames = s

    print(f"  shape: {frames.shape}, dtype: {frames.dtype}")
    print(f"\n=== hybrid_cpsam_multi (DeepSea, no gap fill, no TTA) ===")
    from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi
    t0 = time.time()

    def cb(msg, pct):
        print(f"  [{pct:3d}%] {msg}")

    result = detect_hybrid_cpsam_multi(
        frames, progress_fn=cb,
        min_area_px=500,
        use_fallback=True,
        use_deepsea=True,
        use_gap_fill=False)   # cellpose round-trip is slow
    elapsed = time.time() - t0
    print(f"\n  done in {elapsed:.0f}s; {len(result['tracks'])} tracks")

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
    out = f"{OUT}/pos3_wt_phase.npz"
    np.savez_compressed(out, **save)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
