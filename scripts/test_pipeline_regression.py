"""Pipeline regression test.

Re-runs hybrid_dic on a fixed 15-frame slice of a canonical recording
and compares the output to a pinned reference. Catches silent
regressions when modules get refactored.

The reference is auto-generated on first run (or with --update).
Subsequent runs compare against it and fail if any of:
  - n_detected_frames drops
  - mean mask area changes by > 5%
  - per-frame area std changes by > 10%
  - mask centroids drift > 5 px on average
  - mask IoU vs reference < 0.95 (pinned masks)

Run:
  conda run -n cellpose python scripts/test_pipeline_regression.py
  conda run -n cellpose python scripts/test_pipeline_regression.py --update
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports, benchmark_data_root  # noqa
setup_imports()

import numpy as np
import tifffile

REFERENCE = "results/regression/reference.npz"
SLICE_LEN = 15
CANONICAL_REL = "data/examples/jesse_wt/pos0_wt.ome.tif"


def safe_read_tiff(path):
    try:
        return tifffile.imread(path)
    except AttributeError as e:
        if "newbyteorder" not in str(e):
            raise
    pages = []
    with tifffile.TiffFile(path) as tf:
        with open(path, "rb") as fh:
            for pg in tf.pages:
                bps = pg.bitspersample; sf = pg.sampleformat
                bo = pg.parent.byteorder
                offsets = pg.dataoffsets; bcs = pg.databytecounts
                chunks = []
                for off, n in zip(offsets, bcs):
                    fh.seek(off); chunks.append(fh.read(n))
                raw = b"".join(chunks)
                kind = {1: "u", 2: "i", 3: "f"}.get(sf, "u")
                dt = np.dtype(f"{bo}{kind}{bps // 8}")
                arr = np.frombuffer(raw, dtype=dt).reshape(
                    pg.imagelength, pg.imagewidth)
                if arr.dtype.byteorder not in ("=", "|"):
                    arr = arr.astype(arr.dtype.newbyteorder("=")).copy()
                pages.append(arr)
    return np.stack(pages) if len(pages) > 1 else pages[0]


def run_pipeline():
    """Run hybrid_dic on the canonical slice; return masks (N,H,W bool)."""
    bd = benchmark_data_root()
    path = bd / CANONICAL_REL
    if not path.exists():
        raise SystemExit(
            f"Canonical recording missing: {path}. Place a Jesse "
            f"pos0_wt.ome.tif there or update CANONICAL_REL.")
    print(f"Loading {path.name}…")
    s = safe_read_tiff(str(path))
    if s.dtype != np.uint8:
        p1, p99 = np.percentile(s, [1, 99])
        s = np.clip((s.astype(np.float32) - p1) /
                    max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
    frames = s[20:20 + SLICE_LEN]

    print(f"Running hybrid_dic on {len(frames)} frames…")
    from core.hybrid_dic import detect_hybrid_dic
    t0 = time.time()
    masks, missed = detect_hybrid_dic(
        frames, model_path="data/models/cpsam_dic",
        use_preprocess=False, use_deepsea=True, use_retry=False,
        use_tta=False)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s, missed={len(missed)}")
    return frames, masks


def summarise(masks):
    """Reduce a mask stack to scalar/vector statistics."""
    n = len(masks)
    detected = np.array([m.any() for m in masks])
    areas = np.array([int(m.sum()) for m in masks])
    centroids = []
    for m in masks:
        if m.any():
            ys, xs = np.where(m)
            centroids.append([float(ys.mean()), float(xs.mean())])
        else:
            centroids.append([np.nan, np.nan])
    return {
        "n_frames": np.array(n),
        "n_detected": np.array(int(detected.sum())),
        "areas": areas,
        "mean_area": np.array(float(areas[detected].mean())
                              if detected.any() else 0.0),
        "std_area": np.array(float(areas[detected].std())
                             if detected.any() else 0.0),
        "centroids": np.array(centroids),
    }


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 1.0


def compare(ref, cur, masks_ref, masks_cur, *, area_tol=0.05,
            std_tol=0.10, centroid_tol_px=5.0, iou_tol=0.95):
    failures = []

    if int(cur["n_detected"]) < int(ref["n_detected"]):
        failures.append(
            f"detection regressed: {int(cur['n_detected'])}/"
            f"{int(cur['n_frames'])} vs ref "
            f"{int(ref['n_detected'])}/{int(ref['n_frames'])}")

    rel_area = abs(float(cur["mean_area"]) - float(ref["mean_area"])) / max(
        float(ref["mean_area"]), 1)
    if rel_area > area_tol:
        failures.append(
            f"mean area changed {rel_area * 100:.1f}% "
            f"(tol {area_tol * 100:.0f}%): {float(ref['mean_area']):.0f} → "
            f"{float(cur['mean_area']):.0f}")

    rel_std = abs(float(cur["std_area"]) - float(ref["std_area"])) / max(
        float(ref["std_area"]), 1)
    if rel_std > std_tol:
        failures.append(
            f"area std changed {rel_std * 100:.1f}%: "
            f"{float(ref['std_area']):.0f} → "
            f"{float(cur['std_area']):.0f}")

    cs_ref = ref["centroids"]; cs_cur = cur["centroids"]
    valid = ~np.isnan(cs_ref[:, 0]) & ~np.isnan(cs_cur[:, 0])
    if valid.any():
        drift = np.linalg.norm(cs_ref[valid] - cs_cur[valid], axis=1).mean()
        if drift > centroid_tol_px:
            failures.append(
                f"mean centroid drift {drift:.1f} px (tol "
                f"{centroid_tol_px:.0f})")

    ious = [iou(masks_ref[i], masks_cur[i]) for i in range(len(masks_ref))
            if masks_ref[i].any() and masks_cur[i].any()]
    if ious:
        mean_iou = float(np.mean(ious))
        if mean_iou < iou_tol:
            failures.append(
                f"mean IoU vs reference: {mean_iou:.3f} (tol {iou_tol})")

    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="overwrite the reference with current output")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
    frames, masks = run_pipeline()
    summary = summarise(masks)

    if args.update or not os.path.exists(REFERENCE):
        np.savez_compressed(REFERENCE, masks=masks, **summary)
        print(f"\n✓ wrote reference {REFERENCE}")
        print(f"  detected {int(summary['n_detected'])}/"
              f"{int(summary['n_frames'])} frames, "
              f"mean area {float(summary['mean_area']):.0f} px")
        return 0

    ref = np.load(REFERENCE, allow_pickle=False)
    failures = compare(
        {k: ref[k] for k in summary},
        summary,
        ref["masks"],
        masks,
    )
    if failures:
        print("\n✗ REGRESSION FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\n✓ regression OK (vs {REFERENCE})")
    print(f"  detected {int(summary['n_detected'])}/"
          f"{int(summary['n_frames'])}, "
          f"mean area {float(summary['mean_area']):.0f} px, "
          f"matches reference within tolerances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
