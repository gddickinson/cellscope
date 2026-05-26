"""Open the Pos7_WT recording + its pipeline results in the focused
GUI.

Loads:
  - The full multichannel TIFF (DIC + Cy5, both channels)
  - The pipeline_results/masks.npz (labels + fusion source stack)
  - Auto-reconstructs the tracks list from labels so multi-cell
    analysis can run without re-detecting

Then opens the GUI with everything pre-populated. Pipeline stage
shows "detect ✓" so you can click Analyze immediately, or just scrub
through the frames and toggle Cell IDs / Tracks / Source view.

Run:
    cd /Users/george/claude_test/cellscope
    conda run -n cellpose4 python scripts/open_pos7_in_gui.py
"""
import os
import sys
import numpy as np

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

REC_DIR = os.path.join(
    CELLSCOPE_ROOT, "data/ic295_gt_full/Pos7_WT")
TIF = os.path.join(REC_DIR, "IC295__1_MMStack_Pos7-WT.ome.tif")
RESULTS_NPZ = os.path.join(REC_DIR, "pipeline_results/masks.npz")


def tracks_from_labels(labels, source_stack=None):
    """Rebuild a list of per-cell track dicts from an (N, H, W) int32
    label stack. Optionally annotates fusion_source per track."""
    tracks = []
    cell_ids = sorted(int(c) for c in np.unique(labels) if c != 0)
    for cid in cell_ids:
        stack = (labels == cid)
        per_frame = stack.any(axis=(1, 2))
        if not per_frame.any():
            continue
        first = int(np.argmax(per_frame))
        t = {"stack": stack, "first_frame": first, "parent_id": None}
        if source_stack is not None:
            ns = {1: 0, 2: 0, 3: 0}   # dic_only / cy5_only / both
            for fi in range(len(stack)):
                m = stack[fi]
                if not m.any():
                    continue
                src = source_stack[fi][m]
                for code in ns:
                    ns[code] += int((src == code).sum())
            if ns[3] >= max(ns[1], ns[2]):
                t["fusion_source"] = "both"
            elif ns[2] > ns[1]:
                t["fusion_source"] = "cy5_only"
            else:
                t["fusion_source"] = "dic_only"
        tracks.append(t)
    return tracks


def main():
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer

    app = QApplication(sys.argv)
    from gui_focused.main_window import FocusedMainWindow

    w = FocusedMainWindow()
    w.resize(1600, 1000)
    w.show()
    w.raise_()
    w.activateWindow()

    # 1. Load the recording with both channels
    print(f"Loading {TIF} …")
    from core.io import load_recording
    w.recording = load_recording(TIF, dic_channel=1, fluo_channel=0)
    n = len(w.recording["frames"])
    has_cy5 = w.recording.get("cy5_frames") is not None
    cy5_note = " [+ Cy5 channel]" if has_cy5 else ""
    w.logger.log("info",
                  f"Loaded {os.path.basename(TIF)}: "
                  f"{n} frames{cy5_note}")
    w.viewer.set_data(w.recording["frames"],
                       fluo_frames=w.recording.get("cy5_frames"))
    w.params.set_from_recording(w.recording)
    if hasattr(w.params, "set_cy5_available"):
        w.params.set_cy5_available(has_cy5)

    # 2. Load pipeline labels + reconstruct tracks
    if os.path.exists(RESULTS_NPZ):
        print(f"Loading pipeline results from {RESULTS_NPZ} …")
        data = np.load(RESULTS_NPZ)
        labels = data["labels"]
        masks = data["masks"]
        source = (data["fusion_source_stack"]
                  if "fusion_source_stack" in data.files else None)
        tracks = tracks_from_labels(labels, source)
        w.detect_result = {
            "masks": masks,
            "labels": labels,
            "tracks": tracks,
        }
        if source is not None:
            w.detect_result["fusion_source_stack"] = source
        # Force multi-cell mode so the tracks are usable
        w.mode = "multi"
        w.pipeline.set_mode("multi")
        # Push labels + (optional) source stack to the viewer
        w.viewer.update_masks(labels)
        if hasattr(w.viewer, "set_source_stack") and source is not None:
            w.viewer.set_source_stack(source)
        w.viewer.nav_bar.set_status(masks)
        w.pipeline.set_stage_status("detect", "done")
        w.pipeline.enable_stage("edit", True)
        w.pipeline.enable_stage("analyze", True)
        w.logger.log(
            "info",
            f"Loaded pipeline labels: {len(tracks)} tracks, "
            f"{int(labels.max())} max ID")
    else:
        print(f"WARNING: no pipeline_results found at {RESULTS_NPZ}")

    w.status.showMessage(
        f"Pos7_WT — {n} frames, {len(w.detect_result['tracks'])} "
        f"tracked cells. Toggle Cell IDs / Tracks in the viewer, or "
        f"click Analyze.")

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
