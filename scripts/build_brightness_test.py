"""Generate brightness-perturbed copies of the v3 test set.

Phase 15 of the legacy piezo1_analysis project showed every detector
losing ~30% IoU on bright_dark / bright_bright perturbations. This
builds a labelled test set so we can measure that drop on cpsam_dic
and verify it shrinks after the brightness-augmented retrain (3.5).

Output (`data/training/dic_splits_v3/test_brightness/<perturbation>/`):
  clean              identity copy (sanity baseline)
  b_plus_30          add +30 to all pixels (clip)
  b_plus_60          add +60 to all pixels
  b_minus_30         subtract 30
  b_minus_60         subtract 60
  gamma_05           gamma 0.5 — heavy contrast lift
  gamma_18           gamma 1.8 — heavy contrast crush
  vignette           multiplicative 2-D Gaussian shading bias

Each subdirectory gets the same `<name>_img.tif` + `<name>_masks.tif`
pairs as the source, so `bench_cpsam_dic.py --test-dir <subdir>` works
unchanged.
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

SRC_DIR_DEFAULT = "data/training/dic_splits_v3/test"
OUT_ROOT_DEFAULT = "data/training/dic_splits_v3/test_brightness"


PERTURBATIONS = {
    "clean":      lambda img: img,
    "b_plus_30":  lambda img: np.clip(img.astype(np.int16) + 30,
                                       0, 255).astype(np.uint8),
    "b_plus_60":  lambda img: np.clip(img.astype(np.int16) + 60,
                                       0, 255).astype(np.uint8),
    "b_minus_30": lambda img: np.clip(img.astype(np.int16) - 30,
                                       0, 255).astype(np.uint8),
    "b_minus_60": lambda img: np.clip(img.astype(np.int16) - 60,
                                       0, 255).astype(np.uint8),
    "gamma_05":   lambda img: (np.clip(
        (img.astype(np.float32) / 255.0) ** 0.5, 0, 1) * 255
    ).astype(np.uint8),
    "gamma_18":   lambda img: (np.clip(
        (img.astype(np.float32) / 255.0) ** 1.8, 0, 1) * 255
    ).astype(np.uint8),
}


def _vignette(img):
    """Multiplicative 2-D Gaussian shading bias.

    Centred random offset, σ ≈ 0.4 of the smaller dim. The bias map
    ranges from ~0.55 at the corners to 1.0 in the centre — what an
    uneven illumination flat-field looks like in practice.
    """
    h, w = img.shape
    rng = np.random.default_rng(int(h * 1000 + w))
    cy = h * 0.5 + rng.uniform(-0.1, 0.1) * h
    cx = w * 0.5 + rng.uniform(-0.1, 0.1) * w
    sigma = 0.4 * min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    bias = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2)
                    / (2 * sigma ** 2)))
    bias = 0.55 + 0.45 * bias  # 0.55 at edges, 1.0 at peak
    return np.clip(img.astype(np.float32) * bias,
                   0, 255).astype(np.uint8)


PERTURBATIONS["vignette"] = _vignette


def build(src_dir, out_root, names=None):
    img_files = sorted(glob.glob(os.path.join(src_dir, "*_img.tif")))
    if names is not None:
        wanted = set(names)
        img_files = [f for f in img_files
                     if os.path.basename(f).replace("_img.tif", "")
                     in wanted]
    if not img_files:
        print(f"[build_brightness_test] no images in {src_dir}")
        return
    print(f"[build_brightness_test] source: {src_dir} "
          f"({len(img_files)} pairs)")
    for pert_name, fn in PERTURBATIONS.items():
        out_dir = os.path.join(out_root, pert_name)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        n_done = 0
        for img_path in img_files:
            mask_path = img_path.replace("_img.tif", "_masks.tif")
            if not os.path.exists(mask_path):
                continue
            img = tifffile.imread(img_path)
            if img.ndim == 3:
                img = img[0]
            if img.dtype != np.uint8:
                p1, p99 = np.percentile(img, [1, 99])
                img = np.clip((img.astype(np.float32) - p1)
                              / max(p99 - p1, 1e-6) * 255,
                              0, 255).astype(np.uint8)
            perturbed = fn(img)
            base = os.path.basename(img_path)
            tifffile.imwrite(os.path.join(out_dir, base), perturbed)
            shutil.copy(mask_path,
                        os.path.join(out_dir,
                                     os.path.basename(mask_path)))
            n_done += 1
        print(f"  ✓ {pert_name:<12} {n_done} pairs → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DIR_DEFAULT,
                    help=f"source test dir (default {SRC_DIR_DEFAULT})")
    ap.add_argument("--out", default=OUT_ROOT_DEFAULT,
                    help=f"output root dir (default {OUT_ROOT_DEFAULT})")
    args = ap.parse_args()
    build(args.src, args.out)
    print(f"\nNext: bench against each subdirectory, e.g.\n"
          f"  conda run -n cellpose4 python scripts/bench_cpsam_dic.py \\\n"
          f"      --model data/models/cpsam_dic \\\n"
          f"      --test-dir {args.out}/b_plus_60 \\\n"
          f"      --out results/brightness_eval/cpsam_dic_b_plus_60.json\n"
          f"Or use scripts/bench_brightness.py to do all 8 in one shot.")


if __name__ == "__main__":
    main()
