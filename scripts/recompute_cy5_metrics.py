"""Recompute extended Cy5 cellularity metrics from existing NPZs.

For each .npz in results/ic295_full/, loads the cy5_frames + per-track
stacks and computes the new per-frame metrics needed by the
multi-metric and temporal-stability filters:
    cy5_io_ratio          (inside/outside ring ratio)
    cy5_inside_cv         (texture proxy)
    cy5_fraction_positive (spatial coverage)

These complement the existing cy5_score / cy5_mean / cy5_p75 / cy5_p95.

Output: enriched .npz files saved to results/ic295_full_v2/ alongside
the original. Originals are not modified.

No re-detection — uses cached labels + cy5_frames. ~2-5 min/recording.
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports
setup_imports()

import numpy as np

CACHE_DIR = "results/ic295_full"
OUT_DIR = "results/ic295_full_v2"


def recompute_one(npz_in, npz_out, log_fn=print):
    z = np.load(npz_in, allow_pickle=False)
    cy5_frames = z["cy5_frames"]
    n = len(cy5_frames)
    save = {k: z[k] for k in z.files}

    from core.multichannel import (
        cy5_inside_outside_ratio, cy5_inside_cv,
        cy5_fraction_positive)
    n_tracks = int(z["tracks_n"]) if "tracks_n" in z.files else 0
    t_start = time.time()
    n_done = 0
    for tid in range(n_tracks * 3 + 5):
        stack_key = f"track_{tid}_stack"
        if stack_key not in z.files:
            continue
        stack = z[stack_key]
        io_ratio = np.full(n, np.nan, dtype=np.float32)
        inside_cv = np.full(n, np.nan, dtype=np.float32)
        frac_pos = np.full(n, np.nan, dtype=np.float32)
        for i in range(n):
            m = stack[i].astype(bool)
            if not m.any():
                continue
            io_ratio[i] = cy5_inside_outside_ratio(m, cy5_frames[i])
            inside_cv[i] = cy5_inside_cv(m, cy5_frames[i])
            frac_pos[i] = cy5_fraction_positive(m, cy5_frames[i])
        save[f"track_{tid}_cy5_io_ratio"] = io_ratio
        save[f"track_{tid}_cy5_inside_cv"] = inside_cv
        save[f"track_{tid}_cy5_fraction_positive"] = frac_pos
        n_done += 1

    np.savez_compressed(npz_out, **save)
    log_fn(f"  recomputed {n_done} tracks × {n} frames in "
           f"{time.time() - t_start:.1f}s → {npz_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    npzs = sorted(glob.glob(os.path.join(args.cache_dir, "Pos*.npz")))
    print(f"Recomputing extended Cy5 metrics for {len(npzs)} recordings\n",
          flush=True)
    for npz in npzs:
        name = os.path.basename(npz)
        out = os.path.join(args.out_dir, name)
        if os.path.exists(out):
            print(f"  [skip — exists] {name}", flush=True)
            continue
        print(f"  {name}", flush=True)
        recompute_one(npz, out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
