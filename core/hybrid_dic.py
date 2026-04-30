"""DIC-specific hybrid detection pipeline.

For DIC (Differential Interference Contrast) recordings where cpsam
picks up background texture as false positives. Uses cellpose_dic
(CP3) or cpsam_dic (CP4 ViT fine-tune) with optional preprocessing
and DeepSea refinement.

Pipeline (CP3 fine-tunes — cellpose_dic, cellpose_dic_v2/v3):
  1. (Optional) Temporal median background subtraction + spatial HP
  2. cellpose_dic detection (CP3 model, ft=0.0, ct=0.0)
  3. Retry missed frames at ct=-2.0
  4. (Optional) DeepSea union for boundary refinement

Pipeline (cpsam_dic — runs cellpose step in cellpose4 env via subprocess):
  1. cpsam_dic detection at defaults (no preprocessing — model trained
     on raw VAMPIRE crops; preprocessing tends to hurt)
  2. Retry skipped (det rate ~100% on benchmarks)
  3. (Optional) DeepSea union for boundary refinement
"""
import os
import json
import subprocess
import tempfile
import logging
import numpy as np

log = logging.getLogger(__name__)

AREA_THRESHOLD = 200
CELLPOSE4_ENV = "cellpose4"

# Inline script run via `conda run -n cellpose4 python -c ...`.
# Loads cpsam_dic in the CP4 env, runs detection per frame, dumps result.
_CPSAM_DIC_SCRIPT = '''
import sys, warnings, logging, numpy as np
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)
sys.path.insert(0, "{project_root}")

from cellpose import models
data = np.load("{input_path}", allow_pickle=True)
frames = data["frames"]
n = len(frames)
m = models.CellposeModel(gpu=True, pretrained_model="{model_path}")
result = np.zeros(frames.shape, dtype=bool)
for i in range(n):
    masks_i = m.eval(frames[i])[0]
    result[i] = masks_i > 0
np.savez_compressed("{output_path}", masks=result)
print("CPSAM_DIC_OK")
'''


def _is_cpsam_model(model_path):
    """True if the model lives in cellpose 4.x territory (cpsam fine-tunes)."""
    return model_path is not None and "cpsam" in os.path.basename(
        model_path).lower()


_CPSAM_DIC_LABELS_SCRIPT = '''
import sys, warnings, logging, numpy as np
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)
sys.path.insert(0, "{project_root}")

from cellpose import models
data = np.load("{input_path}", allow_pickle=True)
frames = data["frames"]
n = len(frames)
m = models.CellposeModel(gpu=True, pretrained_model="{model_path}")
out = np.zeros(frames.shape, dtype=np.int32)
for i in range(n):
    out[i] = m.eval(frames[i])[0].astype(np.int32)
np.savez_compressed("{output_path}", labels=out)
print("CPSAM_DIC_LABELS_OK")
'''


def _run_cpsam_dic_labels_subprocess(frames, model_path, project_root,
                                     progress_fn=None):
    """Run cpsam_dic in cellpose4; return int32 (N,H,W) label stack."""
    if progress_fn:
        progress_fn(f"cpsam_dic detection ({CELLPOSE4_ENV} env, multi-cell)",
                    10)
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "input.npz")
        outp = os.path.join(tmp, "output.npz")
        np.savez_compressed(inp, frames=frames)
        script = _CPSAM_DIC_LABELS_SCRIPT.format(
            project_root=project_root,
            input_path=inp,
            output_path=outp,
            model_path=model_path,
        )
        proc = subprocess.run(
            ["conda", "run", "-n", CELLPOSE4_ENV, "python", "-c", script],
            capture_output=True, text=True, timeout=1800,
            cwd=project_root,
        )
        if "CPSAM_DIC_LABELS_OK" not in proc.stdout:
            log.error("cpsam_dic labels subprocess failed:\n"
                      "STDOUT:\n%s\nSTDERR:\n%s",
                      proc.stdout, proc.stderr)
            raise RuntimeError(
                f"cpsam_dic labels subprocess failed: {proc.stderr[-500:]}")
        if progress_fn:
            progress_fn("cpsam_dic detection done", 40)
        return np.load(outp)["labels"]


def _run_cpsam_dic_subprocess(frames, model_path, project_root,
                              progress_fn=None):
    """Run cpsam_dic detection in cellpose4 env; return bool mask array."""
    if progress_fn:
        progress_fn(f"cpsam_dic detection ({CELLPOSE4_ENV} env)", 10)
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "input.npz")
        outp = os.path.join(tmp, "output.npz")
        np.savez_compressed(inp, frames=frames)
        script = _CPSAM_DIC_SCRIPT.format(
            project_root=project_root,
            input_path=inp,
            output_path=outp,
            model_path=model_path,
        )
        proc = subprocess.run(
            ["conda", "run", "-n", CELLPOSE4_ENV, "python", "-c", script],
            capture_output=True, text=True, timeout=1800,
            cwd=project_root,
        )
        if "CPSAM_DIC_OK" not in proc.stdout:
            log.error("cpsam_dic subprocess failed:\nSTDOUT:\n%s\nSTDERR:\n%s",
                      proc.stdout, proc.stderr)
            raise RuntimeError(
                f"cpsam_dic subprocess failed: {proc.stderr[-500:]}")
        if progress_fn:
            progress_fn("cpsam_dic detection done", 50)
        return np.load(outp)["masks"]


def detect_hybrid_dic(frames, progress_fn=None, area_threshold=None,
                      use_preprocess=True, use_deepsea=True,
                      use_retry=True, model_path=None):
    """DIC-optimized single-cell detection.

    Args:
        use_preprocess: apply temporal bg subtraction + spatial HP
        use_deepsea: run DeepSea union on all frames
        use_retry: retry missed frames with lower threshold

    Returns (N, H, W) bool mask array, list of missed frame indices.
    """
    from core.detection import detect_cellpose
    from core.preprocess import preprocess_sequence

    if area_threshold is None:
        area_threshold = AREA_THRESHOLD
    if model_path is None:
        # Prefer cpsam_dic > cellpose_dic_v3 > v2 > v1.
        cpsam_dic = "data/models/cpsam_dic"
        v3 = "data/models/cellpose_dic_v3"
        v2 = "data/models/cellpose_dic_v2"
        v1 = "data/models/cellpose_dic"
        if os.path.exists(cpsam_dic):
            model_path = cpsam_dic
        elif os.path.exists(v3):
            model_path = v3
        elif os.path.exists(v2):
            model_path = v2
        else:
            model_path = v1

    n = len(frames)
    is_cpsam = _is_cpsam_model(model_path)

    # cpsam_dic was trained on raw VAMPIRE crops — preprocessing tends
    # to hurt. Skip it regardless of caller's use_preprocess flag.
    if is_cpsam and use_preprocess:
        log.info("cpsam_dic detected; skipping preprocessing "
                 "(model was trained on raw crops)")
        use_preprocess = False

    # Step 1: preprocessing (CP3 path only)
    if use_preprocess:
        if progress_fn:
            progress_fn("Preprocessing (bg subtraction + HP)", 5)
        processed = preprocess_sequence(
            frames,
            temporal_method="median",
            spatial_highpass_sigma=40.0,
            debris_diameter_um=0,
            pixel_size_um=None,
        )
    else:
        processed = frames

    # Step 2: detection
    if is_cpsam:
        # cpsam_dic via cellpose4 subprocess
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        masks = _run_cpsam_dic_subprocess(
            processed, model_path, project_root, progress_fn=progress_fn)
        # cpsam_dic gets ~100% det rate on benchmarks; skip retry
        missed = list(np.where(
            np.array([m.sum() for m in masks]) < area_threshold)[0])
    else:
        # CP3 cellpose_dic path
        if progress_fn:
            progress_fn("cellpose_dic detection", 10)
        masks = detect_cellpose(
            processed, gpu=True,
            model_path=model_path,
            flow_threshold=0.0,
            cellprob_threshold=0.0,
            progress_fn=lambda i: progress_fn(
                f"cellpose_dic {i+1}/{n}",
                int(10 + 40 * i / max(n-1, 1))) if progress_fn else None,
        )

        # Step 3: retry missed frames (CP3 only)
        missed = list(np.where(
            np.array([m.sum() for m in masks]) < area_threshold)[0])
        if missed and use_retry:
            log.info("cellpose_dic missed %d frames, retrying at ct=-2.0",
                     len(missed))
            if progress_fn:
                progress_fn(f"Retry {len(missed)} missed frames", 55)
            retry_masks = detect_cellpose(
                processed[missed], gpu=True,
                model_path=model_path,
                flow_threshold=0.0,
                cellprob_threshold=-2.0,
            )
            for j, fi in enumerate(missed):
                if retry_masks[j].sum() >= area_threshold:
                    masks[fi] = retry_masks[j]
            missed = list(np.where(
                np.array([m.sum() for m in masks]) < area_threshold)[0])

    # Step 4: DeepSea union
    if use_deepsea:
        from core.medsam_deepsea_union import union_with_deepsea
        result = np.zeros(frames.shape, dtype=bool)
        for i in range(n):
            if progress_fn and (i % 10 == 0 or i == n-1):
                progress_fn(f"DeepSea union {i+1}/{n}",
                            int(60 + 40 * i / max(n-1, 1)))
            result[i] = union_with_deepsea(frames[i], masks[i])
    else:
        result = masks

    log.info("DIC detection: %d/%d detected, %d missed",
             n - len(missed), n, len(missed))
    return result, missed


def detect_hybrid_dic_multi(frames, progress_fn=None,
                            min_area_px=200,
                            use_preprocess=True,
                            use_deepsea=True,
                            use_retry=True,
                            use_gap_fill=True,
                            expected_cells=0,
                            model_path=None):
    """DIC-optimized multi-cell detection + tracking.

    Same structure as hybrid_cpsam_multi but uses cellpose_dic
    instead of cpsam for initial detection.

    Returns dict with keys: masks, labels, tracks, missed_frames,
    cell_count.
    """
    from core.detection import detect_cellpose_labels
    from core.preprocess import preprocess_sequence
    from core.multi_cell import track_all_cells

    if model_path is None:
        cpsam_dic = "data/models/cpsam_dic"
        v3 = "data/models/cellpose_dic_v3"
        v2 = "data/models/cellpose_dic_v2"
        v1 = "data/models/cellpose_dic"
        if os.path.exists(cpsam_dic):
            model_path = cpsam_dic
        elif os.path.exists(v3):
            model_path = v3
        elif os.path.exists(v2):
            model_path = v2
        else:
            model_path = v1

    n = len(frames)
    is_cpsam = _is_cpsam_model(model_path)

    if is_cpsam and use_preprocess:
        log.info("cpsam_dic detected; skipping preprocessing "
                 "(model trained on raw crops)")
        use_preprocess = False

    # Step 1: preprocess (CP3 only)
    if use_preprocess:
        if progress_fn:
            progress_fn("Preprocessing", 5)
        processed = preprocess_sequence(
            frames,
            temporal_method="median",
            spatial_highpass_sigma=40.0,
            debris_diameter_um=0,
            pixel_size_um=None,
        )
    else:
        processed = frames

    # Step 2: detection
    if is_cpsam:
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        raw_labels = _run_cpsam_dic_labels_subprocess(
            processed, model_path, project_root, progress_fn=progress_fn)
        # Filter by min area
        if min_area_px > 0:
            for i in range(n):
                lbl = raw_labels[i]
                for cid in np.unique(lbl):
                    if cid == 0:
                        continue
                    if int((lbl == cid).sum()) < min_area_px:
                        lbl[lbl == cid] = 0
                raw_labels[i] = lbl
        cell_counts = np.array([int(raw_labels[i].max()) for i in range(n)])
        missed = list(np.where(cell_counts == 0)[0])
        # Skip retry — cpsam_dic ~100% det rate
    else:
        if progress_fn:
            progress_fn("cellpose_dic detection", 10)
        raw_labels = detect_cellpose_labels(
            processed, gpu=True,
            model_path=model_path,
            flow_threshold=0.0,
            cellprob_threshold=0.0,
            min_area_px=min_area_px,
        )

        # Step 3: retry empty frames (CP3 only)
        cell_counts = np.array([int(raw_labels[i].max()) for i in range(n)])
        missed = list(np.where(cell_counts == 0)[0])
        if missed and use_retry:
            log.info("DIC missed %d frames, retrying at ct=-2.0", len(missed))
            if progress_fn:
                progress_fn(f"Retry {len(missed)} frames", 40)
            retry_labels = detect_cellpose_labels(
                processed[missed], gpu=True,
                model_path=model_path,
                flow_threshold=0.0,
                cellprob_threshold=-2.0,
                min_area_px=min_area_px,
            )
            for j, fi in enumerate(missed):
                raw_labels[fi] = retry_labels[j]
                cell_counts[fi] = int(raw_labels[fi].max())

    max_cells = int(cell_counts.max())
    log.info("DIC cell counts: min=%d max=%d mean=%.1f",
             cell_counts.min(), max_cells, cell_counts.mean())

    # Step 4: per-cell DeepSea refinement
    if use_deepsea:
        from core.deepsea_multicell import refine_labels_with_deepsea
        if progress_fn:
            progress_fn("Per-cell DeepSea refinement", 50)
        refined = refine_labels_with_deepsea(
            frames, raw_labels, expand_px=20,
            progress_fn=lambda msg, pct: progress_fn(
                msg, int(50 + 15 * pct / 100)) if progress_fn else None)
    else:
        refined = raw_labels

    # Step 4b: detection post-processing
    from core.detection_postprocess import preprocess_detection
    if progress_fn:
        progress_fn("Detection post-processing", 70)
    refined = preprocess_detection(
        refined, expected_cells=expected_cells, min_area=min_area_px)

    # Step 5: track across frames
    if progress_fn:
        progress_fn("Tracking cells", 75)
    tracks = track_all_cells(
        refined,
        min_area_px=min_area_px,
        max_hop_px=150,
        spawn_new_tracks=True,
        min_track_length=3,
        frames=frames,
    )
    log.info("Found %d tracks (max %d cells/frame)", len(tracks), max_cells)

    # Step 5b: gap fill
    if use_gap_fill and tracks:
        from core.track_gap_fill import fill_track_gaps
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        if progress_fn:
            progress_fn("Filling track gaps", 85)
        n_filled = fill_track_gaps(
            tracks, frames, min_area=min_area_px,
            search_radius=150,
            project_root=project_root,
            progress_fn=lambda msg, pct: progress_fn(
                msg, int(85 + 10 * pct / 100)) if progress_fn else None)
        if n_filled:
            log.info("Filled %d track gaps", n_filled)

    # Step 5c: post-process tracks
    from core.track_postprocess import postprocess_tracks
    if progress_fn:
        progress_fn("Post-processing tracks", 95)
    postprocess_tracks(tracks, frames=frames)

    # Step 6: build label stack
    from core.hybrid_cpsam_multi import _build_tracked_labels
    tracked = _build_tracked_labels(tracks, frames.shape)
    missed = list(np.where(cell_counts == 0)[0])
    if progress_fn:
        progress_fn("Done", 100)

    return {
        "masks": tracked > 0,
        "labels": tracked,
        "tracks": tracks,
        "missed_frames": missed,
        "cell_count": max_cells,
    }
