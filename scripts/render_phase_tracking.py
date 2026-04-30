"""Render the tracking-derived figures (multi-cell detection, trajectories,
analysis graphs) from the phase-contrast cache, where cells visibly migrate.

Keeps focused_detected.png from the existing DIC render (still good).

Usage:
  conda run -n cellpose python scripts/render_phase_tracking.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

import numpy as np

import render_v2_figures as r

CACHE = "results/figure_pipelines/pos3_wt_phase.npz"

UM_PER_PX = 0.65
DT_MIN = 5.0


def main():
    if not os.path.exists(CACHE):
        print(f"MISSING: {CACHE}")
        sys.exit(1)

    frames, labels, tracks = r.load_multi(CACHE)
    n = len(frames)
    print(f"=== {n}-frame phase-contrast cache: {len(tracks)} raw tracks ===")
    kept = r.filter_tracks(tracks, n, r.TRACK_KEEP_FRACTION)
    print(f"After ≥{int(r.TRACK_KEEP_FRACTION*100)}% filter: {len(kept)} tracks")
    for t in kept:
        cents = t["centroids"]
        ok = ~np.isnan(cents[:, 0])
        if ok.sum() < 2:
            continue
        cs = cents[ok]
        path_px = float(np.sum(np.linalg.norm(np.diff(cs, axis=0), axis=1)))
        net_px = float(np.linalg.norm(cs[-1] - cs[0]))
        print(f"  Cell {t['id']}: {ok.sum()}/{n} frames, "
              f"path {path_px:.0f}px, net {net_px:.0f}px")

    if not kept:
        print("ERROR: no tracks after filter")
        return

    print()
    print("Multi-cell detection overlay:")
    r.fig_focused_multi_detected(frames, labels, kept, fi=n // 3)

    print("Trajectories:")
    r.fig_trajectories(frames, kept, n)

    # For analysis graphs, use the cell with the longest path (most signal).
    longest = None
    longest_path = -1
    for t in kept:
        cents = t["centroids"]
        ok = ~np.isnan(cents[:, 0])
        if ok.sum() < 2:
            continue
        cs = cents[ok]
        p = float(np.sum(np.linalg.norm(np.diff(cs, axis=0), axis=1)))
        if p > longest_path:
            longest_path = p
            longest = t
    print(f"\nMost-mobile cell: track {longest['id']} (path {longest_path:.0f}px)")

    # Crop frames + stack to that cell's bbox
    fr_c, masks_c = r.crop_to_track(frames, longest, pad=30)
    print(f"  cropped to {fr_c.shape[1:]}")

    # focused_detected from this cell — phase contrast single-cell
    r.fig_focused_detected(fr_c, masks_c, fi=n // 3)
    # The default title says "Single-cell DIC" — patch label for accuracy
    # (we'll fix this in the function via a custom call)

    from core.tracking import extract_centroids
    cents_c = extract_centroids(masks_c)
    single_track = [(masks_c, cents_c)]
    r.fig_speed(single_track, UM_PER_PX, DT_MIN, label_prefix="Cell")
    r.fig_trajectory_xy(single_track, UM_PER_PX)
    r.fig_msd(single_track, UM_PER_PX, DT_MIN)
    r.fig_area(single_track, UM_PER_PX, DT_MIN)
    r.fig_kymograph_and_shape(masks_c, UM_PER_PX, DT_MIN)

    print("\nHero composite:")
    r.fig_hero(fr_c, masks_c, frames, kept)

    print("\nDone.")


if __name__ == "__main__":
    main()
