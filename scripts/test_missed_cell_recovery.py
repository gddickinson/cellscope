"""Two diagnostic figures probing missed-cell recovery on Ignasi.

Run in cellpose4 env:
  conda run -n cellpose4 python scripts/test_missed_cell_recovery.py

Outputs:
  results/best_detections/05_tta_recovery.png
    For 6 multi-cell-capable frames, default cpsam vs cpsam(augment=True).
    Title shows cell counts at each — a jump means TTA recovered a cell.

  results/best_detections/06_pipeline_gap_fill.png
    For Pos2 WT, runs detect_hybrid_cpsam_multi (full production pipeline
    including gap fill) on a 25-frame window covering frame 10. Shows the
    tracked label stack vs default cpsam on the same frames.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports, benchmark_data_root  # noqa
setup_imports()

from scripts.make_overlay_figures import (
    overlay_panel, safe_read_tiff, OUT_DIR,
)


# ──────────────────────────────────────────────────────────────────────
# Figure 5: TTA recovery — default cpsam vs cpsam(augment=True)
# ──────────────────────────────────────────────────────────────────────
def figure_tta_recovery(model):
    print("[fig 5] TTA recovery — default vs augment=True")
    base = str(benchmark_data_root() / "data" / "ignasi")
    # 6 frames mostly from Pos2 (the missed-cell case) + cross-recording
    # samples. Each row is one frame; left = default, right = +TTA.
    cases = [
        ("IC293__1_MMStack_Pos2-WT.ome-cropped.tif",   "Pos2 WT", 10),
        ("IC293__1_MMStack_Pos2-WT.ome-cropped.tif",   "Pos2 WT", 30),
        ("IC293__1_MMStack_Pos2-WT.ome-cropped.tif",   "Pos2 WT", 60),
        ("IC293__1_MMStack_Pos3-WT.ome-cropped.tif",   "Pos3 WT", 10),
        ("IC293__1_MMStack_Pos17-KO.ome-cropped.tif",  "Pos17 KO", 10),
        ("IC293__1_MMStack_Pos17-KO.ome-cropped.tif",  "Pos17 KO", 60),
    ]

    # Cache stacks so we don't re-read big files
    cache = {}
    def stack_for(rel):
        if rel not in cache:
            s = safe_read_tiff(f"{base}/{rel}")
            if s.dtype != np.uint8:
                p1, p99 = np.percentile(s, [1, 99])
                s = np.clip((s.astype(np.float32) - p1) /
                            max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
            cache[rel] = s
        return cache[rel]

    fig, axes = plt.subplots(len(cases), 2, figsize=(10, 3.2 * len(cases)))
    fig.suptitle("Ignasi: default cpsam vs cpsam(augment=True / TTA)",
                 fontsize=12, y=0.995)

    for row, (rel, label, fi) in enumerate(cases):
        img = stack_for(rel)[fi]
        # Default
        lbl_default = model.eval(img, augment=False)[0].astype(np.int32)
        # TTA
        lbl_tta = model.eval(img, augment=True)[0].astype(np.int32)
        # Filter < 500 px to mirror production
        for arr in (lbl_default, lbl_tta):
            for cid in list(np.unique(arr)):
                if cid == 0:
                    continue
                if int((arr == cid).sum()) < 500:
                    arr[arr == cid] = 0
        # Count distinct surviving cells (label IDs are not contiguous
        # after min_area filter, so .max() lies — use unique count).
        n_default = len(set(np.unique(lbl_default).tolist()) - {0})
        n_tta = len(set(np.unique(lbl_tta).tolist()) - {0})

        delta = n_tta - n_default
        marker = " ★ TTA recovered" if delta > 0 else \
                 " (no change)" if delta == 0 else \
                 " (TTA dropped)"

        overlay_panel(axes[row, 0], img, gt=None, pred=lbl_default,
                      title=f"{label} f{fi}  default  n={n_default}")
        overlay_panel(axes[row, 1], img, gt=None, pred=lbl_tta,
                      title=f"{label} f{fi}  TTA  n={n_tta}{marker}")

    fig.text(0.5, 0.01,
             "left = default cpsam   right = cpsam(augment=True)   "
             "min_area=500 applied to both",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    out = f"{OUT_DIR}/05_tta_recovery.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Figure 6: Multi-cell pipeline + gap fill on Pos2 WT window
# ──────────────────────────────────────────────────────────────────────
def figure_multicell_pipeline(model):
    print("[fig 6] Multi-cell pipeline + gap fill on Pos2 WT window")
    from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi

    base = str(benchmark_data_root() / "data" / "ignasi")
    rel = "IC293__1_MMStack_Pos2-WT.ome-cropped.tif"
    stack = safe_read_tiff(f"{base}/{rel}")
    if stack.dtype != np.uint8:
        p1, p99 = np.percentile(stack, [1, 99])
        stack = np.clip((stack.astype(np.float32) - p1) /
                        max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)

    # 25-frame window centred on frame 10 — long enough for tracking +
    # gap fill to work, short enough to render in a few minutes.
    start = 0
    end = 25
    window = stack[start:end]
    print(f"  Running production pipeline on {len(window)} frames "
          f"({rel.split('.')[0][-12:]} {start}–{end-1})…")

    # Light progress reporter
    last_pct = [-1]
    def cb(msg, pct):
        if pct >= last_pct[0] + 10 or pct == 100:
            print(f"    [{pct:3d}%] {msg}")
            last_pct[0] = pct

    result = detect_hybrid_cpsam_multi(
        window, progress_fn=cb,
        min_area_px=500,
        use_fallback=True,
        use_deepsea=True,
        use_gap_fill=True,
    )
    tracked = result["labels"]   # (N,H,W) int32 with consistent IDs
    n_tracks = len(result["tracks"])
    missed = result.get("missed_frames", [])
    print(f"  → {n_tracks} tracks, {len(missed)} cpsam-missed frames "
          f"(fallback fired)")

    # Default cpsam (per-frame, no pipeline) for comparison
    default_labels = np.zeros(window.shape, dtype=np.int32)
    for i in range(len(window)):
        lbl = model.eval(window[i], augment=False)[0].astype(np.int32)
        # Apply min_area filter
        for cid in list(np.unique(lbl)):
            if cid == 0:
                continue
            if int((lbl == cid).sum()) < 500:
                lbl[lbl == cid] = 0
        default_labels[i] = lbl

    # Pick 4 frames to show (ensure frame 10 is in there)
    show_frames = [5, 10, 15, 20]   # local indices into window
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"Pos2 WT — production pipeline (top: default cpsam, "
        f"bottom: hybrid_cpsam_multi + gap fill)\n"
        f"{n_tracks} tracks found, {len(missed)} fallback frames",
        fontsize=11, y=0.99)

    def cell_count(arr):
        return len(set(np.unique(arr).tolist()) - {0})

    for col, fi in enumerate(show_frames):
        global_fi = start + fi
        n_def = cell_count(default_labels[fi])
        overlay_panel(axes[0, col], window[fi], gt=None,
                      pred=default_labels[fi],
                      title=f"frame {global_fi}  default  n={n_def}")
        n_pipe = cell_count(tracked[fi])
        overlay_panel(axes[1, col], window[fi], gt=None,
                      pred=tracked[fi],
                      title=f"frame {global_fi}  pipeline  n={n_pipe}")

    fig.text(0.5, 0.01,
             "Pipeline = cpsam → min_area filter → DeepSea → tracking → "
             "gap fill (recovers missed cells in tracked frames)",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = f"{OUT_DIR}/06_pipeline_gap_fill.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    from cellpose import models
    print("Loading default cpsam…")
    model = models.CellposeModel(gpu=True)
    figure_tta_recovery(model)
    figure_multicell_pipeline(model)
    print(f"\nFigures saved under {OUT_DIR}/")


if __name__ == "__main__":
    main()
