"""Stage-1 DIC ∪ Cy5 detection fusion.

When a multichannel recording has a fluorescence channel (Cy5 / SiR-
actin), DIC alone may miss cells that are visibly stained but have
weak DIC contrast — debris-overlapping, faint, or sitting in
regions where the DIC interference produces low gradient.

Pattern:
  1. Detect on DIC with the configured model (cellpose_dic family,
     fine-tuned and conservative).
  2. Detect on Cy5 with raw cpsam — generic vit_h backbone, no
     domain bias. Bright, well-stained cells are obvious here.
  3. Merge label stacks: every cell in (1) kept verbatim; cells in
     (2) added only if they don't substantially overlap a (1) cell
     (Jaccard < `jaccard_thresh`) and their position isn't already
     covered. Each kept (2)-only mask gets a fresh label ID.

The cpsam call always runs in the `cellpose4` conda env via subprocess
(same pattern as `core/hybrid_dic.py::_run_cpsam_dic_labels_subprocess`),
so this module is callable from any env. Returns int32 (N, H, W).

Failure modes guarded against:
  - cpsam(Cy5) returns lots of small bright spots (dust, hot pixels):
    we filter by `min_area_px` before merging.
  - Cy5 cell partly overlaps a DIC cell (e.g. cytoplasm vs nucleus):
    Jaccard ≥ 0.3 with any existing DIC label → don't add (avoid
    label duplication).
  - Background fluorescence ring (vignette / mounting medium):
    cpsam typically rejects these (no cell shape); the
    downstream Cy5 multi-metric filter rejects them anyway.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import logging
import numpy as np

log = logging.getLogger(__name__)

CELLPOSE4_ENV = "cellpose4"

# Run raw cpsam (vit_h default) on every Cy5 frame; return int32 label
# stack. Honours CPSAM_PRETRAINED if a custom model path is wanted.
_CPSAM_CY5_LABELS_SCRIPT = '''
import sys, os, warnings, logging, numpy as np
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)
sys.path.insert(0, "{project_root}")

from cellpose import models
data = np.load("{input_path}", allow_pickle=True)
frames = data["frames"]
n = len(frames)
cpsam_path = os.environ.get("CPSAM_PRETRAINED")
if cpsam_path:
    m = models.CellposeModel(gpu=True, pretrained_model=cpsam_path)
else:
    m = models.CellposeModel(gpu=True)
augment = {augment}
out = np.zeros(frames.shape, dtype=np.int32)
for i in range(n):
    out[i] = m.eval(frames[i], augment=augment)[0].astype(np.int32)
np.savez_compressed("{output_path}", labels=out)
print("CPSAM_CY5_LABELS_OK")
'''


def detect_cpsam_on_cy5(cy5_frames, project_root, augment=False,
                         progress_fn=None):
    """Run cpsam on the Cy5 stack via cellpose4 subprocess.

    Returns int32 (N, H, W) label stack."""
    if progress_fn:
        suffix = " +TTA" if augment else ""
        progress_fn(f"cpsam(Cy5) detection ({CELLPOSE4_ENV} env{suffix})",
                    10)
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "input.npz")
        outp = os.path.join(tmp, "output.npz")
        np.savez_compressed(inp, frames=cy5_frames)
        script = _CPSAM_CY5_LABELS_SCRIPT.format(
            project_root=project_root,
            input_path=inp,
            output_path=outp,
            augment=str(augment),
        )
        proc = subprocess.run(
            ["conda", "run", "-n", CELLPOSE4_ENV, "python", "-c", script],
            capture_output=True, text=True, timeout=7200,
            cwd=project_root,
        )
        if "CPSAM_CY5_LABELS_OK" not in proc.stdout:
            log.error("cpsam(Cy5) labels subprocess failed:\n"
                      "STDOUT:\n%s\nSTDERR:\n%s",
                      proc.stdout, proc.stderr)
            raise RuntimeError(
                f"cpsam(Cy5) labels subprocess failed: "
                f"{proc.stderr[-500:]}")
        if progress_fn:
            progress_fn("cpsam(Cy5) detection done", 30)
        return np.load(outp)["labels"]


def _filter_small_labels(label_frame, min_area_px):
    """Drop components smaller than min_area_px; re-compact IDs to 1..N."""
    out = np.zeros_like(label_frame)
    new_id = 0
    if label_frame.max() == 0:
        return out, 0
    for lab in range(1, int(label_frame.max()) + 1):
        m = label_frame == lab
        if m.sum() >= min_area_px:
            new_id += 1
            out[m] = new_id
    return out, new_id


def _jaccard(a_mask, b_mask):
    inter = (a_mask & b_mask).sum()
    if inter == 0:
        return 0.0
    return inter / (a_mask | b_mask).sum()


# Source codes carried in the per-frame source maps.
SRC_BACKGROUND = 0
SRC_DIC_ONLY = 1
SRC_CY5_ONLY = 2
SRC_BOTH = 3


def merge_label_frames(dic_frame, cy5_frame, min_area_px=200,
                        jaccard_thresh=0.3, max_overlap_frac=0.5,
                        max_centroid_dist_px=None):
    """Merge one frame: keep all DIC labels, add Cy5 labels that
    don't overlap heavily with DIC.

    Args:
      dic_frame: (H, W) int32
      cy5_frame: (H, W) int32
      min_area_px: drop Cy5 components smaller than this
      jaccard_thresh: a Cy5 label is "the same cell as" a DIC label
        if their Jaccard ≥ jaccard_thresh → not added.
      max_overlap_frac: if a Cy5 mask is mostly covered by existing
        DIC labels (covers > max_overlap_frac of itself), don't add
        (avoids partial-overlap duplicates like cytoplasm-vs-nucleus).
      max_centroid_dist_px: BACKUP same-cell criterion. If a Cy5
        label's centroid is within this distance of any existing
        DIC label's centroid, treat as same cell even if Jaccard
        is below `jaccard_thresh`. Catches the common failure mode
        where residual channel misalignment puts matched cells just
        below the IoU cutoff. Default None = compute from
        sqrt(min_area_px) (≈ cell-radius worth of slack).

    Returns (merged_int32, n_added_from_cy5, source_frame_uint8).
      source_frame_uint8: (H, W) uint8 with values
        0 = background
        1 = DIC-only (DIC label with no Cy5 match)
        2 = Cy5-only (added by fusion, no DIC overlap)
        3 = both (DIC label with a Cy5 match ≥ jaccard_thresh)
    """
    if max_centroid_dist_px is None:
        max_centroid_dist_px = float(np.sqrt(min_area_px))

    out = dic_frame.astype(np.int32).copy()
    source = np.zeros(out.shape, dtype=np.uint8)
    # Start by marking every DIC label as dic_only; promote to "both"
    # below if a Cy5 label matches it.
    source[out > 0] = SRC_DIC_ONLY

    nxt = int(out.max()) + 1
    n_added = 0
    if cy5_frame.max() == 0:
        return out, 0, source

    # Precompute DIC centroids for the cheap centroid-distance check
    dic_centroids = {}
    for ex in range(1, int(out.max()) + 1):
        em = out == ex
        if not em.any():
            continue
        ys, xs = np.where(em)
        dic_centroids[ex] = (float(ys.mean()), float(xs.mean()))

    for lab in range(1, int(cy5_frame.max()) + 1):
        cm = cy5_frame == lab
        if cm.sum() < min_area_px:
            continue
        # Find the BEST-matching existing label (not just first ≥ thresh)
        best_j = 0.0
        best_ex = 0
        for ex in range(1, int(out.max()) + 1):
            em = out == ex
            if not em.any():
                continue
            j = _jaccard(cm, em)
            if j > best_j:
                best_j = j
                best_ex = ex
        if best_j >= jaccard_thresh:
            # Promote the matched DIC label to "both"
            source[out == best_ex] = SRC_BOTH
            continue
        # BACKUP same-cell check: centroid distance. Catches the
        # "misaligned cell" failure mode where Jaccard is just below
        # threshold because the two channels' masks are translated
        # by a few pixels.
        if dic_centroids:
            ys, xs = np.where(cm)
            cy5_cy, cy5_cx = float(ys.mean()), float(xs.mean())
            nearest_dic_id = None
            nearest_dist = float("inf")
            for ex, (dy, dx) in dic_centroids.items():
                d = ((cy5_cy - dy) ** 2 + (cy5_cx - dx) ** 2) ** 0.5
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_dic_id = ex
            if nearest_dist <= max_centroid_dist_px:
                source[out == nearest_dic_id] = SRC_BOTH
                continue
        cov = (cm & (out > 0)).sum() / max(cm.sum(), 1)
        if cov > max_overlap_frac:
            continue
        # New cell from Cy5 — write into background pixels only
        new_pixels = cm & (out == 0)
        out[new_pixels] = nxt
        source[new_pixels] = SRC_CY5_ONLY
        nxt += 1
        n_added += 1
    return out, n_added, source


def merge_label_stacks(dic_labels, cy5_labels, **kwargs):
    """Apply merge_label_frames frame-by-frame.

    Returns (merged_int32_stack, list_of_n_added_per_frame,
             source_stack_uint8).
    source_stack_uint8 carries SRC_BACKGROUND / DIC_ONLY / CY5_ONLY /
    BOTH per pixel per frame.
    """
    assert dic_labels.shape == cy5_labels.shape, \
        (f"shape mismatch: dic={dic_labels.shape} "
         f"cy5={cy5_labels.shape}")
    out = np.zeros_like(dic_labels)
    source = np.zeros(dic_labels.shape, dtype=np.uint8)
    n_added = []
    for i in range(len(dic_labels)):
        merged, n, src = merge_label_frames(
            dic_labels[i], cy5_labels[i], **kwargs)
        out[i] = merged
        source[i] = src
        n_added.append(n)
    return out, n_added, source


def detect_dic_cy5_fusion(dic_labels, cy5_frames, project_root,
                           min_area_px=200, augment_cpsam=False,
                           jaccard_thresh=0.3, max_overlap_frac=0.5,
                           progress_fn=None):
    """End-to-end fusion: take DIC labels (already produced by the
    DIC detector), run cpsam on Cy5, merge.

    Caller produces `dic_labels` so this works in both pipelines
    (hybrid_dic_multi where DIC runs in-process via cellpose env,
    or hybrid_cpsam_multi where DIC also went through cellpose4).

    Returns dict:
      labels:       (N, H, W) int32 merged stack
      cy5_labels:   (N, H, W) int32 raw cpsam(Cy5) (filtered for area)
      n_added_per_frame: list[int] new labels added per frame
      n_added_total: int — sum across frames
    """
    if progress_fn:
        progress_fn("cpsam(Cy5) fusion: starting", 5)
    cy5_labels_raw = detect_cpsam_on_cy5(
        cy5_frames, project_root,
        augment=augment_cpsam, progress_fn=progress_fn)

    # Re-compact + size-filter per frame
    cy5_filtered = np.zeros_like(cy5_labels_raw)
    for i in range(len(cy5_labels_raw)):
        cy5_filtered[i], _ = _filter_small_labels(
            cy5_labels_raw[i], min_area_px)

    if progress_fn:
        progress_fn("merging DIC + cpsam(Cy5) labels", 35)
    merged, n_added, source_stack = merge_label_stacks(
        dic_labels, cy5_filtered,
        min_area_px=min_area_px,
        jaccard_thresh=jaccard_thresh,
        max_overlap_frac=max_overlap_frac)

    n_added_total = int(sum(n_added))
    log.info("Cy5 fusion: +%d cells across %d frames",
             n_added_total, len(dic_labels))
    if progress_fn:
        progress_fn(
            f"Cy5 fusion done (+{n_added_total} cells)", 45)
    return {
        "labels": merged,
        "cy5_labels": cy5_filtered,
        "source_stack": source_stack,
        "dic_labels_pre_fusion": dic_labels.copy(),
        "n_added_per_frame": n_added,
        "n_added_total": n_added_total,
    }
