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
                               use_gap_fill=True,
                               use_tta=False,
                               use_mirror_pad="auto",
                               expected_cells=0,
                               cy5_frames=None,
                               recover_with_cy5=False,
                               use_cy5_fusion=False):
    """Multi-cell hybrid cpsam detection + tracking.

    Args:
        use_fallback: run cellpose fallback on cpsam-missed frames
        use_deepsea: run per-cell DeepSea refinement
        use_gap_fill: fill internal track gaps with augmented detection
        use_tta: pass augment=True to cpsam (4-rotation TTA, ~2.5×
            slower per frame but recovers cells the base model misses;
            recommended for low-density / OOD recordings)
        cy5_frames: optional (N, H, W) uint8 fluorescence stack
        recover_with_cy5: if True (and cy5_frames provided), search
            for Cy5+ regions not covered by any cpsam mask, then crop
            and re-run cpsam(TTA) on that crop only to try to recover
            the missed cell. Cheap defensive layer for false-negatives.

    Returns dict with keys:
        masks:         (N, H, W) bool — union of all tracked cells
        labels:        (N, H, W) int32 — tracked cell IDs (consistent
                       across frames; 0=background)
        tracks:        list of track dicts from track_all_cells
        missed_frames: list of frame indices rescued by fallback
        n_cy5_recovered: count of cells added via Cy5 recovery
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

    # Step 1: cpsam at defaults — keep instance labels.
    # Honour CPSAM_PRETRAINED if set so callers can swap the model.
    cpsam_path = os.environ.get("CPSAM_PRETRAINED")
    if cpsam_path:
        log.info("loading cpsam from %s", cpsam_path)
        m = models.CellposeModel(gpu=True, pretrained_model=cpsam_path)
    else:
        m = models.CellposeModel(gpu=True)
    raw_labels = np.zeros(frames.shape, dtype=np.int32)
    eval_kwargs = {"augment": True} if use_tta else {}
    # Resolve mirror-padding policy: enabled for large detection
    # frames where the reflection border is a small fraction of the
    # image. Validated across 9 GT recordings (2026-05-21).
    from core.pipeline_defaults import (
        should_use_mirror_pad, MIRROR_PAD_PX)
    pad_enabled = should_use_mirror_pad(use_mirror_pad, frames.shape[1:])
    pad_px = MIRROR_PAD_PX if pad_enabled else 0
    if pad_enabled:
        log.info("Mirror-padding cpsam input by %d px "
                 "(detection-min-dim=%d ≥ threshold)",
                 pad_px, min(frames.shape[1:]))
    for i in range(n):
        if progress_fn and (i % 10 == 0 or i == n - 1):
            label = "cpsam"
            if use_tta:
                label += "+TTA"
            if pad_enabled:
                label += "+pad"
            progress_fn(f"{label} {i+1}/{n}",
                        int(30 * i / max(n - 1, 1)))
        if pad_enabled:
            padded = np.pad(
                frames[i], ((pad_px, pad_px), (pad_px, pad_px)),
                mode="reflect")
            masks_i, _, _ = m.eval(padded, **eval_kwargs)
            H_, W_ = frames.shape[1:]
            masks_i = masks_i[pad_px:pad_px + H_, pad_px:pad_px + W_]
        else:
            masks_i, _, _ = m.eval(frames[i], **eval_kwargs)
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

    # Step 3b: Cy5-based recovery of cells missed by cpsam.
    # For each frame, find bright Cy5 regions not covered by any
    # cpsam label, crop around them, re-run cpsam(TTA). Newly
    # detected cells are appended as new label IDs.
    # k_mad=5.0 (vs default 4.0) chosen via diagnostic on Pos14_KO:
    # at k=4 every frame triggers 2-3 candidate cpsam crops that all
    # reject (~60s/frame waste); at k=5 candidates drop to 0 when no
    # genuine missed cells exist. Pilot confirmed 0 recoveries at
    # k=4 so this sacrifices nothing while restoring throughput.
    # Step 3a: Cy5 fusion — run cpsam on Cy5 stack, merge new cells
    # in before refinement + tracking. Separate from recover_with_cy5
    # which is a slower fallback that re-runs cpsam on DIC crops.
    n_cy5_fusion_added = 0
    fusion_source_stack = None
    if use_cy5_fusion and cy5_frames is not None:
        from core.dic_cy5_fusion import detect_dic_cy5_fusion
        if cy5_frames.shape != frames.shape:
            log.warning(
                "Cy5 fusion skipped: shape mismatch dic=%s cy5=%s",
                frames.shape, cy5_frames.shape)
        else:
            if progress_fn:
                progress_fn("Cy5 fusion: cpsam on fluorescence", 37)
            fusion = detect_dic_cy5_fusion(
                raw_labels, cy5_frames, project_root,
                min_area_px=min_area_px,
                augment_cpsam=False)
            raw_labels = fusion["labels"]
            fusion_source_stack = fusion["source_stack"]
            n_cy5_fusion_added = fusion["n_added_total"]
            cell_counts = np.array(
                [int(raw_labels[i].max()) for i in range(n)])
            max_cells = max(max_cells, int(cell_counts.max()))
            log.info("After Cy5 fusion: max %d cells/frame "
                     "(+%d cells added)",
                     max_cells, n_cy5_fusion_added)

    n_cy5_recovered = 0
    if recover_with_cy5 and cy5_frames is not None:
        from core.cy5_fallbacks import recover_missed_cells_via_dic_crop
        if progress_fn:
            progress_fn("Cy5 recovery scan", 38)
        for i in range(n):
            new_labels, n_added, _ = recover_missed_cells_via_dic_crop(
                frames[i], raw_labels[i], cy5_frames[i], m,
                use_tta=True, k_mad=5.0)
            if n_added > 0:
                raw_labels[i] = new_labels
                cell_counts[i] = int(new_labels.max())
                n_cy5_recovered += n_added
        if n_cy5_recovered:
            log.info("Cy5 recovery added %d cells across %d frames",
                     n_cy5_recovered, n)
            max_cells = max(max_cells, int(cell_counts.max()))

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

    # Step 4b: detection post-processing (close splits, merge
    # fragments, enforce expected cell count)
    from core.detection_postprocess import preprocess_detection
    if progress_fn:
        progress_fn("Detection post-processing", 72)
    refined = preprocess_detection(
        refined, expected_cells=expected_cells, min_area=min_area_px)

    # Step 5: track across frames
    if progress_fn:
        progress_fn("Tracking cells across frames", 75)
    tracks = track_all_cells(
        refined,
        min_area_px=min_area_px,
        max_hop_px=150,
        spawn_new_tracks=True,
        min_track_length=3,
        frames=frames,
    )
    log.info("Found %d tracks (max %d cells/frame)", len(tracks), max_cells)

    # Step 5b: fill internal track gaps (re-detect missing cells)
    if use_gap_fill:
        from core.track_gap_fill import fill_track_gaps
        if progress_fn:
            progress_fn("Filling track gaps", 80)
        from core.pipeline_defaults import DEFAULTS as _PD
        n_filled = fill_track_gaps(
            tracks, frames, min_area=min_area_px,
            search_radius=150,
            project_root=project_root,
            use_sam2_video=_PD.use_sam2_video_gap_fill,
            progress_fn=lambda msg, pct: progress_fn(
                msg, int(80 + 15 * pct / 100)) if progress_fn else None)
        if n_filled:
            log.info("Filled %d track gaps via gap-fill cascade",
                     n_filled)

    # Step 5c: post-process tracks (merge splits, fix ID switches)
    from core.track_postprocess import postprocess_tracks
    if progress_fn:
        progress_fn("Post-processing tracks", 95)
    pp = postprocess_tracks(tracks, frames=frames)
    if (pp.get("fps_rejected") or pp.get("tracks_removed")
            or pp.get("edge_slivers_dropped")
            or pp.get("edge_sliver_frames_zeroed")):
        log.info("Post-process: %d FPs rejected, %d tracks removed, "
                 "%d edge-sliver tracks dropped, "
                 "%d sliver frames zeroed, %d tracks remaining",
                 pp.get("fps_rejected", 0), pp.get("tracks_removed", 0),
                 pp.get("edge_slivers_dropped", 0),
                 pp.get("edge_sliver_frames_zeroed", 0),
                 pp.get("tracks_remaining", len(tracks)))

    # Step 6: build tracked label stack
    tracked = _build_tracked_labels(tracks, frames.shape)
    if progress_fn:
        progress_fn("Done", 100)

    if fusion_source_stack is not None:
        from core.fusion_diagnostics import (
            annotate_tracks_with_source, per_track_source_summary)
        annotate_tracks_with_source(tracks, fusion_source_stack)
        src_summary = per_track_source_summary(tracks)
        log.info(
            "Track sources: %d dic_only, %d both, %d cy5_only",
            src_summary["dic_only"], src_summary["both"],
            src_summary["cy5_only"])

    # Step 7: annotate cell-division lineage (sets parent_id on
    # daughter tracks). Adds a `divisions` list to the result for
    # sidecar output. Cheap (<1s on 97-frame stacks).
    divisions = []
    try:
        from core.division_annotator import annotate_track_lineage
        divisions, n_set = annotate_track_lineage(tracks, tracked)
        if n_set:
            log.info("Division annotator: %d lineage link(s) set",
                     n_set)
    except Exception as e:
        log.warning("Division annotator failed: %s", e)

    return {
        "masks": tracked > 0,
        "labels": tracked,
        "tracks": tracks,
        "missed_frames": missed,
        "n_cy5_recovered": n_cy5_recovered,
        "n_cy5_fusion_added": n_cy5_fusion_added,
        "fusion_source_stack": fusion_source_stack,
        "cell_count": max_cells,
        "divisions": divisions,
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
