"""Hybrid cpsam + cellpose fallback detection.

Pipeline (run from cellpose4 env):
  1. cpsam at defaults (ViT backbone)
  2. For frames where cpsam area < AREA_THRESHOLD:
     → fall back to cellpose_combined_robust + MedSAM + DeepSea
       (run via subprocess in cellpose env for CP3 model compat)
  3. DeepSea union on all frames (fills under-segmentation, drops debris)

Requires:
  - cellpose4 env active (cellpose >= 4)
  - cellpose env available (cellpose 3.x with CP3 models)
"""
import os
import json
import subprocess
import tempfile
import logging
import numpy as np

log = logging.getLogger(__name__)

AREA_THRESHOLD = 500
CELLPOSE_ENV = "cellpose"

# Inline script run by subprocess in the cellpose (CP3) env
_FALLBACK_SCRIPT = '''
import sys, json, warnings, logging, numpy as np
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)
sys.path.insert(0, "{project_root}")

from core.io import load_recording
from core.detection import detect_cellpose
from core.medsam_refine import refine_all_with_medsam
from core.medsam_deepsea_union import union_with_deepsea

data = np.load("{input_path}", allow_pickle=True)
frames = data["frames"]
n = len(frames)
seeds = detect_cellpose(
    frames, gpu=True,
    model_path="data/models/cellpose_combined_robust",
    flow_threshold=0.0, cellprob_threshold=-2.0, diameter=50) > 0
refined = refine_all_with_medsam(frames, seeds)
result = np.array([union_with_deepsea(frames[i], refined[i])
                    for i in range(n)])
np.savez_compressed("{output_path}", masks=result)
print("FALLBACK_OK")
'''


def _run_cp3_fallback(frames, project_root):
    """Run cellpose+MedSAM+DeepSea on frames via subprocess in the
    cellpose (CP3) env. Returns bool mask array."""
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "input.npz")
        outp = os.path.join(tmp, "output.npz")
        np.savez_compressed(inp, frames=frames)
        script = _FALLBACK_SCRIPT.format(
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
        return np.load(outp)["masks"]


def detect_hybrid_cpsam(frames, progress_fn=None, area_threshold=None,
                        use_fallback=True, use_deepsea=True):
    """Full hybrid detection: cpsam → CP3 fallback → DeepSea union.

    Args:
        use_fallback: run cellpose+MedSAM+DeepSea on cpsam-missed frames
        use_deepsea: run DeepSea union on all frames

    Returns (N, H, W) bool mask array.
    """
    import cellpose
    if not cellpose.version.startswith("4"):
        raise RuntimeError(
            f"detect_hybrid_cpsam needs cellpose >=4, got "
            f"{cellpose.version}. Run from cellpose4 env.")

    from cellpose import models
    from core.medsam_deepsea_union import union_with_deepsea

    if area_threshold is None:
        area_threshold = AREA_THRESHOLD
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n = len(frames)

    # Step 1: cpsam at defaults
    m = models.CellposeModel(gpu=True)
    cpsam_masks = np.zeros(frames.shape, dtype=bool)
    cpsam_areas = np.zeros(n, dtype=int)
    for i in range(n):
        if progress_fn:
            progress_fn(f"cpsam {i+1}/{n}",
                        int(50 * i / max(n-1, 1)))
        masks_i, _, _ = m.eval(frames[i])
        cpsam_masks[i] = masks_i > 0
        cpsam_areas[i] = int(cpsam_masks[i].sum())

    # Step 2: identify missed frames
    missed = list(np.where(cpsam_areas < area_threshold)[0])
    if missed and use_fallback:
        log.info("cpsam missed %d frames (area<%d): %s",
                 len(missed), area_threshold, missed)
        if progress_fn:
            progress_fn(
                f"Fallback: cellpose+MedSAM+DeepSea on {len(missed)} frames",
                55)
        fallback = _run_cp3_fallback(frames[missed], project_root)
        for j, fi in enumerate(missed):
            cpsam_masks[fi] = fallback[j]

    # Step 3: DeepSea union on all frames
    if use_deepsea:
        result = np.zeros(frames.shape, dtype=bool)
        for i in range(n):
            if progress_fn and (i % 10 == 0 or i == n-1):
                progress_fn(f"DeepSea union {i+1}/{n}",
                            int(60 + 40 * i / max(n-1, 1)))
            result[i] = union_with_deepsea(frames[i], cpsam_masks[i])
    else:
        result = cpsam_masks

    return result, missed
