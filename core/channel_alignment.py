"""Measure + correct alignment offset between DIC and Cy5 channels.

Many multichannel scopes have a small but systematic shift between
channels (mechanical filter-wheel registration error, dichroic
mirror, time-multiplexed acquisition with slight stage drift). On
IC295 Pos7_WT we measured ~(-3.5, -10.5) px → about 7 µm of dx
shift, large enough to produce duplicate cell detections (one from
DIC, one from Cy5) when Cy5 fusion can't recognise them as the same
cell.

Two-part fix:

1. `measure_dic_cy5_offset(dic, cy5)` — runs cpsam on both channels
   in a few sampled frames, matches centroids, returns the median
   (dy, dx) shift in pixels (DIC minus Cy5). Cell-detection-based —
   robust to the very different signatures of DIC vs fluorescence
   that defeat phase correlation.

2. `apply_offset_to_stack(stack, dy, dx)` — bilinear shift of a
   uint8 (N, H, W) stack so Cy5 pixels land in DIC coordinates.

`pipeline_defaults` + `dic_cy5_fusion` are wired to auto-measure +
apply the shift before merging label stacks. Recordings where the
offset is < 1 px in both axes are left unshifted to save the cost.
"""
from __future__ import annotations

import os
import logging
import numpy as np

log = logging.getLogger(__name__)

# Cells further than this from any partner aren't considered the
# same cell when matching (px in DIC space). Tight enough to avoid
# wild mis-matches; loose enough that a 15-20 px alignment offset
# still finds plenty of pairs.
MAX_CENTROID_MATCH_PX = 50
DEFAULT_N_SAMPLE_FRAMES = 5
DEFAULT_MIN_AREA_PX = 200
ALIGNMENT_NEGLIGIBLE_PX = 1.0   # skip shift when offset < this in both axes


def _centroids_with_area(labels, min_area=DEFAULT_MIN_AREA_PX):
    out = []
    if labels is None or labels.max() == 0:
        return out
    for cid in range(1, int(labels.max()) + 1):
        m = labels == cid
        n = int(m.sum())
        if n < min_area:
            continue
        ys, xs = np.where(m)
        out.append((float(ys.mean()), float(xs.mean()), n))
    return out


def _match_centroids_hungarian(a, b, max_dist):
    """Return list of {dic_centroid, cy5_centroid, dy, dx, dist}."""
    if not a or not b:
        return []
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    ap = np.array([(y, x) for y, x, _ in a])
    bp = np.array([(y, x) for y, x, _ in b])
    cost = cdist(ap, bp)
    rows, cols = linear_sum_assignment(cost)
    out = []
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_dist:
            out.append({
                "dy": float(ap[r, 0] - bp[c, 0]),
                "dx": float(ap[r, 1] - bp[c, 1]),
                "dist": float(cost[r, c]),
            })
    return out


_DIC_CPSAM_PROBE = '''
import sys, os, warnings, logging, json, numpy as np
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)

from cellpose import models
data = np.load("{input_path}", allow_pickle=True)
dic_frames = data["dic"]
cy5_frames = data["cy5"]
n = len(dic_frames)
m = models.CellposeModel(gpu=True)
out = {{"dic": [], "cy5": []}}
for i in range(n):
    out["dic"].append(
        m.eval(dic_frames[i], augment=False)[0].astype(np.int32).tolist())
    out["cy5"].append(
        m.eval(cy5_frames[i], augment=False)[0].astype(np.int32).tolist())
np.savez_compressed("{output_path}",
                    dic=np.array(out["dic"], dtype=np.int32),
                    cy5=np.array(out["cy5"], dtype=np.int32))
print("PROBE_OK")
'''


def _run_cpsam_pair_subprocess(dic_frames, cy5_frames,
                                env_name="cellpose4",
                                project_root=None):
    """Run cpsam on both DIC + Cy5 sampled frames via cellpose4
    subprocess. Returns (dic_labels, cy5_labels) — each a stack."""
    import subprocess
    import tempfile
    if project_root is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "input.npz")
        outp = os.path.join(tmp, "output.npz")
        np.savez_compressed(inp, dic=dic_frames, cy5=cy5_frames)
        script = _DIC_CPSAM_PROBE.format(
            input_path=inp, output_path=outp)
        # Timeout=1800: 10 cpsam calls on 2048² frames take ~5 min on
        # MPS GPU, but the FIRST call in a fresh `cellpose4` env also
        # downloads ~360 MB of ViT weights AND JIT-compiles for MPS —
        # together easily blowing past 10 min. After warm-up, normal
        # alignment is well under 600s. 1800s gives cold-start
        # headroom on slower networks / older hardware.
        proc = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", script],
            capture_output=True, text=True, timeout=1800,
            cwd=project_root)
        if "PROBE_OK" not in proc.stdout:
            raise RuntimeError(
                f"cpsam pair probe failed:\nSTDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr[-500:]}")
        d = np.load(outp)
        return d["dic"], d["cy5"]


def measure_dic_cy5_offset(dic_frames, cy5_frames,
                            n_sample=DEFAULT_N_SAMPLE_FRAMES,
                            min_area_px=DEFAULT_MIN_AREA_PX,
                            verbose=False):
    """Return (dy, dx) median offset (DIC − Cy5) in pixels + info dict.

    Samples `n_sample` evenly-spaced frames, runs cpsam on each
    channel via cellpose4 subprocess, Hungarian-matches centroids,
    and reports the median shift across all matched pairs.

    A positive `dy` means DIC pixels appear LOWER (larger y) than the
    corresponding Cy5 pixels. To align Cy5 to DIC, shift Cy5 by
    `(+dy, +dx)`.
    """
    n_frames = len(dic_frames)
    idx = np.linspace(
        0, n_frames - 1, min(n_sample, n_frames), dtype=int)
    if verbose:
        log.info("measuring DIC↔Cy5 offset on %d sampled frames",
                 len(idx))

    sub_dic = np.array([dic_frames[int(i)] for i in idx])
    sub_cy5 = np.array([cy5_frames[int(i)] for i in idx])
    dic_labels, cy5_labels = _run_cpsam_pair_subprocess(sub_dic,
                                                          sub_cy5)

    all_matches = []
    for j in range(len(idx)):
        a = _centroids_with_area(dic_labels[j], min_area_px)
        b = _centroids_with_area(cy5_labels[j], min_area_px)
        all_matches.extend(_match_centroids_hungarian(
            a, b, MAX_CENTROID_MATCH_PX))

    if not all_matches:
        return (0.0, 0.0), {"n_pairs": 0, "reason": "no matches"}

    dys = np.array([m["dy"] for m in all_matches])
    dxs = np.array([m["dx"] for m in all_matches])
    info = {
        "n_pairs": len(all_matches),
        "n_sample_frames": int(len(idx)),
        "median_dy_px": float(np.median(dys)),
        "median_dx_px": float(np.median(dxs)),
        "iqr_dy_px": float(np.percentile(dys, 75)
                            - np.percentile(dys, 25)),
        "iqr_dx_px": float(np.percentile(dxs, 75)
                            - np.percentile(dxs, 25)),
    }
    return (info["median_dy_px"], info["median_dx_px"]), info


def apply_offset_to_stack(stack, dy, dx, order=1):
    """Translate every frame in a (N, H, W) uint8 stack by (dy, dx).

    Sub-pixel via bilinear interpolation (order=1). Returns a copy
    of the same dtype.
    """
    if abs(dy) < ALIGNMENT_NEGLIGIBLE_PX and \
            abs(dx) < ALIGNMENT_NEGLIGIBLE_PX:
        return stack
    from scipy.ndimage import shift as nd_shift
    out = np.empty_like(stack)
    for i in range(len(stack)):
        out[i] = nd_shift(stack[i].astype(np.float32),
                           shift=(dy, dx), order=order,
                           mode="constant", cval=0).astype(stack.dtype)
    return out


def align_cy5_to_dic(dic_frames, cy5_frames, n_sample=DEFAULT_N_SAMPLE_FRAMES,
                      min_area_px=DEFAULT_MIN_AREA_PX,
                      verbose=False):
    """One-shot helper: measure offset + apply shift.

    Returns (aligned_cy5, info_dict). When offset < 1 px in both axes
    the input is returned unmodified and `info["shift_applied"]` is
    False.
    """
    (dy, dx), info = measure_dic_cy5_offset(
        dic_frames, cy5_frames, n_sample=n_sample,
        min_area_px=min_area_px, verbose=verbose)
    info["shift_applied"] = (abs(dy) >= ALIGNMENT_NEGLIGIBLE_PX
                              or abs(dx) >= ALIGNMENT_NEGLIGIBLE_PX)
    if info["shift_applied"]:
        if verbose:
            log.info(
                "applying Cy5 shift (dy=%+.2f, dx=%+.2f) px "
                "(%d matched pairs, IQR %.1f/%.1f)",
                dy, dx, info["n_pairs"], info["iqr_dy_px"],
                info["iqr_dx_px"])
        aligned = apply_offset_to_stack(cy5_frames, dy, dx)
    else:
        aligned = cy5_frames
    return aligned, info
