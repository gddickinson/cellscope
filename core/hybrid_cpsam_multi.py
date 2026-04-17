"""Multi-cell hybrid cpsam detection and tracking.

Pipeline (run from cellpose4 env):
  1. cpsam at defaults → keep ALL instance labels (not just largest)
  2. Filter debris (area < min_area_px)
  3. Frames with 0 significant cells → cellpose+MedSAM+DeepSea
     fallback (subprocess in cellpose env)
  4. Per-cell DeepSea refinement (preserves multi-cell identity)
  5. Hungarian tracker (cross-frame identity assignment)
  6. Returns tracked label stack + track list

Requires:
  - cellpose4 env active (cellpose >= 4, cpsam)
  - cellpose env available (cellpose 3.x, for CP3 model fallback)
"""
import os
import subprocess
import tempfile
import logging
import numpy as np

log = logging.getLogger(__name__)

AREA_THRESHOLD = 500
CELLPOSE_ENV = "cellpose"

_FALLBACK_LABELS_SCRIPT = '''
import sys, warnings, logging, numpy as np
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)
sys.path.insert(0, "{project_root}")

from core.detection import detect_cellpose_labels

data = np.load("{input_path}", allow_pickle=True)
frames = data["frames"]
labels = detect_cellpose_labels(
    frames, gpu=True,
    model_path="data/models/cellpose_combined_robust",
    flow_threshold=0.0, cellprob_threshold=-2.0,
    min_area_px=100)
np.savez_compressed("{output_path}", labels=labels)
print("FALLBACK_OK")
'''


def _run_cp3_fallback_labels(frames, project_root):
    """Run cellpose detection in the CP3 env via subprocess.
    Returns int32 label stack."""
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "input.npz")
        outp = os.path.join(tmp, "output.npz")
        np.savez_compressed(inp, frames=frames)
        script = _FALLBACK_LABELS_SCRIPT.format(
            project_root=project_root,
            input_path=inp,
            output_path=outp,
        )
        result = subprocess.run(
            ["conda", "run", "-n", CELLPOSE_ENV, "python", "-c", script],
            capture_output=True, text=True, timeout=600,
            cwd=project_root,
        )
        if "FALLBACK_OK" not in result.stdout:
            log.error("CP3 fallback failed:\n%s\n%s",
                      result.stdout, result.stderr)
            raise RuntimeError(
                f"CP3 fallback subprocess failed: {result.stderr[-500:]}")
        return np.load(outp)["labels"]


def _filter_labels(label_frame, min_area_px):
    """Remove small components and re-compact label IDs."""
    if label_frame.max() == 0:
        return label_frame, 0
    out = np.zeros_like(label_frame)
    new_id = 0
    for lab in range(1, int(label_frame.max()) + 1):
        mask = label_frame == lab
        if mask.sum() >= min_area_px:
            new_id += 1
            out[mask] = new_id
    return out, new_id


def _build_tracked_labels(tracks, shape):
    """Build (N, H, W) int32 label stack from track list,
    where pixel value = 1-indexed track ID."""
    out = np.zeros(shape, dtype=np.int32)
    for tid, t in enumerate(tracks, start=1):
        stack = t["stack"]
        for i in range(shape[0]):
            out[i][stack[i] & (out[i] == 0)] = tid
    return out


def detect_hybrid_cpsam_multi(frames, progress_fn=None,
                               min_area_px=500,
                               use_fallback=True,
                               use_deepsea=True,
                               use_gap_fill=True):
    """Multi-cell hybrid cpsam detection + tracking.

    Args:
        use_fallback: run cellpose fallback on cpsam-missed frames
        use_deepsea: run per-cell DeepSea refinement
        use_gap_fill: fill internal track gaps with augmented detection

    Returns dict with keys:
        masks:         (N, H, W) bool — union of all tracked cells
        labels:        (N, H, W) int32 — tracked cell IDs (consistent
                       across frames; 0=background)
        tracks:        list of track dicts from track_all_cells
        missed_frames: list of frame indices rescued by fallback
        cell_count:    max cells detected in any frame
    """
    import cellpose
    if not cellpose.version.startswith("4"):
        raise RuntimeError(
            f"detect_hybrid_cpsam_multi needs cellpose >=4, got "
            f"{cellpose.version}. Run from cellpose4 env.")

    from cellpose import models
    from core.deepsea_multicell import refine_labels_with_deepsea
    from core.multi_cell import track_all_cells

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    n = len(frames)

    # Step 1: cpsam at defaults — keep instance labels
    m = models.CellposeModel(gpu=True)
    raw_labels = np.zeros(frames.shape, dtype=np.int32)
    for i in range(n):
        if progress_fn and (i % 10 == 0 or i == n - 1):
            progress_fn(f"cpsam {i+1}/{n}",
                        int(30 * i / max(n - 1, 1)))
        masks_i, _, _ = m.eval(frames[i])
        raw_labels[i] = masks_i.astype(np.int32)

    # Step 2: debris filter
    cell_counts = np.zeros(n, dtype=int)
    for i in range(n):
        raw_labels[i], cell_counts[i] = _filter_labels(
            raw_labels[i], min_area_px)
    max_cells = int(cell_counts.max())
    log.info("cpsam cell counts: min=%d max=%d mean=%.1f",
             cell_counts.min(), max_cells, cell_counts.mean())

    # Step 3: fallback for empty frames
    missed = list(np.where(cell_counts == 0)[0])
    if missed and use_fallback:
        log.info("cpsam missed %d frames, running CP3 fallback", len(missed))
        if progress_fn:
            progress_fn(
                f"Fallback: cellpose labels on {len(missed)} frames", 35)
        fallback = _run_cp3_fallback_labels(frames[missed], project_root)
        for j, fi in enumerate(missed):
            raw_labels[fi], cell_counts[fi] = _filter_labels(
                fallback[j], min_area_px)

    # Step 4: per-cell DeepSea refinement
    if use_deepsea:
        if progress_fn:
            progress_fn("Per-cell DeepSea refinement", 40)
        refined = refine_labels_with_deepsea(
            frames, raw_labels, expand_px=20,
            progress_fn=lambda msg, pct: progress_fn(
                msg, int(40 + 30 * pct / 100)) if progress_fn else None)
    else:
        refined = raw_labels

    # Step 5: track across frames
    if progress_fn:
        progress_fn("Tracking cells across frames", 75)
    tracks = track_all_cells(
        refined,
        min_area_px=min_area_px,
        max_hop_px=150,
        spawn_new_tracks=True,
        min_track_length=3,
    )
    log.info("Found %d tracks (max %d cells/frame)", len(tracks), max_cells)

    # Step 5b: fill internal track gaps (re-detect missing cells)
    if use_gap_fill:
        from core.track_gap_fill import fill_track_gaps
        if progress_fn:
            progress_fn("Filling track gaps", 80)
        n_filled = fill_track_gaps(
            tracks, frames, min_area=min_area_px,
            search_radius=150,
            project_root=project_root,
            progress_fn=lambda msg, pct: progress_fn(
                msg, int(80 + 15 * pct / 100)) if progress_fn else None)
        if n_filled:
            log.info("Filled %d track gaps via cpsam(augment=True)",
                     n_filled)

    # Step 6: build tracked label stack
    tracked = _build_tracked_labels(tracks, frames.shape)
    if progress_fn:
        progress_fn("Done", 100)

    return {
        "masks": tracked > 0,
        "labels": tracked,
        "tracks": tracks,
        "missed_frames": missed,
        "cell_count": max_cells,
    }


def scan_cell_count(frames, n_sample=10, min_area_px=500):
    """Quick scan to estimate max cells per frame.
    Samples n_sample frames, runs cpsam, returns max cell count."""
    import cellpose
    if not cellpose.version.startswith("4"):
        raise RuntimeError("scan_cell_count needs cellpose >=4")
    from cellpose import models
    m = models.CellposeModel(gpu=True)
    indices = np.linspace(0, len(frames) - 1, min(n_sample, len(frames)),
                          dtype=int)
    max_count = 0
    for i in indices:
        masks_i, _, _ = m.eval(frames[i])
        count = 0
        for lab in range(1, int(masks_i.max()) + 1):
            if (masks_i == lab).sum() >= min_area_px:
                count += 1
        max_count = max(max_count, count)
    return max_count
