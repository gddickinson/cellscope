"""DIC model resolution and the auto-select probe (runs raw cpsam in the cellpose4 env).

Split out of `core/pipeline_defaults.py` (2026-11-12) to keep that file under the
500-line budget; `pipeline_defaults` re-exports every name here, so import from
there as before. Contents: `DIC_MODEL_PREFERENCE`, `resolve_dic_model_path`, the
probe subprocess (`_PROBE_SCRIPT`, `_run_probe_subprocess`) and
`select_pipeline_for_recording` (the decision itself is
`core.pipeline_policy.decide_backbone`).
"""
from __future__ import annotations

import os
from typing import Optional

from core.conda_run import conda_exe
from core.pipeline_policy import N_SAMPLE_FRAMES_FOR_AUTO, decide_backbone

# ------------------------------------------------------------------
# Where to find the GUI-installed models. Resolution order is the same
# as `gui_focused/params_panel.py::_populate_dic_models`: cpsam_dic
# first (best, ViT fine-tune), then v3 > v2 > v1.
# ------------------------------------------------------------------
DIC_MODEL_PREFERENCE = (
    "cpsam_dic",
    "cellpose_dic_v3",
    "cellpose_dic_v2",
    "cellpose_dic",
)


def resolve_dic_model_path(project_root: Optional[str] = None) -> str:
    """Return the best available DIC model path.

    Mirrors the "Auto (best available)" choice from the GUI dropdown.
    If `project_root` is None, paths are returned relative to the
    current working directory (matching `core.detection`'s convention).
    """
    base = (os.path.join(project_root, "data/models")
            if project_root else "data/models")
    for name in DIC_MODEL_PREFERENCE:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    # Fall back to the oldest name even if it doesn't exist — caller
    # gets a clear FileNotFoundError instead of None.
    return os.path.join(base, DIC_MODEL_PREFERENCE[-1])


_PROBE_SCRIPT = '''
import sys, os, json, warnings, logging, numpy as np
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)

from cellpose import models
data = np.load("{input_path}", allow_pickle=True)
frames = data["frames"]
indices = data["indices"]
min_area_px = int({min_area_px})
model_path = "{model_path}"
if model_path:
    m = models.CellposeModel(gpu=True, pretrained_model=model_path)
else:
    # Raw cpsam (CP4 ViT default). Used by the auto-select probe so
    # the density estimate isn't biased by cpsam_dic's single-cell
    # fine-tune (which merges touching cells and undercounts).
    m = models.CellposeModel(gpu=True)
counts = []
for fi in indices:
    labels = m.eval(frames[int(fi)], augment=False)[0]
    n = sum(1 for c in range(1, int(labels.max()) + 1)
            if (labels == c).sum() >= min_area_px)
    counts.append(int(n))
print("PROBE_OK", json.dumps(counts))
'''


def _run_probe_subprocess(frames, indices, model_path, min_area_px,
                           env_name="cellpose4", project_root=None):
    """Run the cell-count probe in `env_name` via subprocess.

    `env_name` defaults to cellpose4 so the probe can use the CP4 ViT
    backbone (either a fine-tune like cpsam_dic, or raw cpsam when
    `model_path` is empty).
    """
    import subprocess
    import tempfile
    import numpy as np

    if project_root is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))

    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "input.npz")
        sub_frames = np.array(
            [frames[int(i)] for i in indices])
        np.savez_compressed(
            inp, frames=sub_frames,
            indices=np.arange(len(sub_frames)))
        script = _PROBE_SCRIPT.format(
            input_path=inp, model_path=model_path,
            min_area_px=min_area_px)
        proc = subprocess.run(
            [conda_exe(), "run", "-n", env_name, "python", "-c", script],
            capture_output=True, text=True, timeout=3600,
            cwd=project_root)
        for line in proc.stdout.splitlines():
            if line.startswith("PROBE_OK "):
                import json as _json
                return _json.loads(line[len("PROBE_OK "):])
        raise RuntimeError(
            f"probe subprocess failed:\nSTDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr[-500:]}")


def select_pipeline_for_recording(
        frames, min_area_px=None, sample_n=N_SAMPLE_FRAMES_FOR_AUTO,
        verbose=False):
    """Pick the right detection pipeline + model for a recording.

    Probes a few frames with **raw cpsam** (CP4 ViT default — not the
    cpsam_dic fine-tune), counts cells per frame, and returns either
    ('cpsam_dic', model_path) or ('cpsam', None) based on whether the
    recording looks multi-cell.

    Why probe with raw cpsam rather than cpsam_dic: cpsam_dic was
    fine-tuned on single-cell VAMPIRE crops and tends to MERGE
    touching cells in multi-cell scenes, **so using it to estimate
    density biases the count downward**. Pos68_DMSO showed this — 11
    probed frames at downsampled 1024² returned `[1,0,0,1,1,0,0,1,1,
    0,0]` from cpsam_dic while the recording actually has 9–14
    cells/frame (visible once Cy5 fusion adds 644 cells). Switching
    the probe model to raw cpsam removes that bias: raw cpsam doesn't
    merge touching cells and gives an honest density estimate.

    The downstream pipeline still chooses cpsam_dic (tighter
    boundaries on isolated cells) when raw cpsam reports a sparse
    field; raw cpsam wins when the field is multi-cell (cpsam_dic
    merges hurt more than the looser boundaries cost).

    The probe runs in the cellpose4 env via subprocess so CP4 ViT
    weights load correctly even when the calling process is in the
    cellpose (CP3) env.

    Args:
      frames: (N, H, W) uint8 stack
      min_area_px: drop detections smaller than this when counting
      sample_n: number of frames to sample evenly across the recording
      verbose: print decision rationale

    Returns:
      tuple ('cpsam_dic' | 'cpsam', model_path_or_None, info_dict)
    """
    if min_area_px is None:
        from core.pipeline_defaults import DEFAULTS
        min_area_px = DEFAULTS.min_area_px

    import numpy as np
    n_frames = len(frames)
    if n_frames == 0:
        return ("cpsam_dic", resolve_dic_model_path(),
                {"reason": "no frames"})

    indices = list(np.linspace(0, n_frames - 1, min(sample_n, n_frames),
                                dtype=int))
    # Empty model_path → probe script loads raw cpsam (CP4 ViT default).
    probe_model_path = ""
    if verbose:
        print(f"[auto-select] probing {len(indices)} frames "
              f"with raw cpsam (CP4 ViT default, via cellpose4 "
              f"subprocess) …")
    counts = _run_probe_subprocess(
        frames, indices, probe_model_path, min_area_px)

    info = decide_backbone(counts)
    info["probe_model"] = "cpsam"
    info["sampled_frames"] = [int(i) for i in indices]
    if verbose:
        print(f"[auto-select] {info['reason']}")
    if info["choice"] == "cpsam":
        return "cpsam", None, info
    return "cpsam_dic", resolve_dic_model_path(), info
