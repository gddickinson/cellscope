"""Capture the Tracking GUI populated with real data.

Loads the cached pos17_wt multi-cell pipeline output into TrackingWindow
programmatically, runs the tracker (it re-computes from labels), waits
for the table to populate, then snapshots the window.

Usage:
  conda run -n cellpose python scripts/screenshot_tracking_gui.py
"""
import os
import sys
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

import numpy as np
from PyQt5.QtWidgets import QApplication

CACHE = "results/figure_pipelines/pos17_wt_60f.npz"
OUT = "docs/figures/gui_tracking.png"


def main():
    if not os.path.exists(CACHE):
        print(f"ERROR: missing cache file {CACHE}")
        print("  Run scripts/cache_pipelines.py first.")
        sys.exit(1)

    print(f"Loading cached pipeline output from {CACHE}…")
    z = np.load(CACHE, allow_pickle=False)
    frames = z["frames"]
    labels = z["labels"]
    print(f"  frames: {frames.shape}, labels: {labels.shape}")

    app = QApplication.instance() or QApplication(sys.argv)

    from gui_tracking.tracking_window import TrackingWindow
    w = TrackingWindow()
    w.resize(1500, 900)
    w.show()
    app.processEvents()

    sv = w.single_view
    # Inject recording + masks as if user had clicked Load + Load Masks.
    sv.recording = {
        "frames": frames,
        "name": "pos17_wt",
        "um_per_px": 0.65,
        "time_interval_min": 5.0,
    }
    sv.viewer.set_data(frames)
    sv.masks = labels.astype(np.int32)
    sv.btn_track.setEnabled(True)
    sv.status_label.setText(
        f"Loaded — {len(frames)} frames, label range "
        f"0..{int(labels.max())}")
    app.processEvents()

    # Run tracking
    print("Running tracker…")
    sv._on_track()
    app.processEvents()

    n_tracks = len(sv.tracks) if sv.tracks else 0
    print(f"  → {n_tracks} tracks populated in the table")

    # Select the longest track so the viewer shows a nice contour.
    if sv.tracks:
        longest_idx = max(
            range(len(sv.tracks)),
            key=lambda i: int(sv.tracks[i]["stack"].any(axis=(1, 2)).sum()))
        sv.track_table.setCurrentCell(longest_idx, 0)
        app.processEvents()
        print(f"  selected track {longest_idx} (longest)")

    # Run analysis so the graph dropdown is meaningful
    if sv.tracks:
        try:
            sv._on_analyze()
            app.processEvents()
            print("  ran analysis")
        except Exception as e:
            print(f"  analyze step failed: {e}")

    # Step the frame slider to the middle so we see migrating cells
    try:
        from PyQt5.QtCore import Qt
        sv.viewer.set_frame(len(frames) // 3)
        app.processEvents()
    except Exception:
        pass

    pix = w.grab()
    pix.save(OUT)
    print(f"\nSaved {OUT}")
    w.close()


if __name__ == "__main__":
    main()
