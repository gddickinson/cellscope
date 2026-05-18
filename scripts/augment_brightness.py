"""Generate brightness-augmented training pairs for cpsam_dic v4 retrain.

Reads `data/training/dic_splits_v3/train` (2,644 standardised 448²
DIC pairs) and writes:

  data/training/dic_splits_v4_brightness/train/
    <name>_img.tif         (original — copy not perturbed)
    <name>_masks.tif
    <name>_b30+_img.tif    +30 brightness
    <name>_b60+_img.tif    +60 brightness
    <name>_b30-_img.tif    -30
    <name>_b60-_img.tif    -60
    <name>_g05_img.tif     gamma 0.5
    <name>_g18_img.tif     gamma 1.8
    <name>_vig_img.tif     vignette shading

(Mask files are simply copied; geometry is unchanged.)

Result: training set ~8× the original, dominated by brightness
variants. Cellpose's internal augmentation handles flip/rotate;
this script's job is the photometric distribution we currently miss.

The val/test directories are NOT touched. Use
``scripts/build_brightness_test.py`` to make a held-out perturbed
test set for evaluation.
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

SRC_DEFAULT = "data/training/dic_splits_v3/train"
OUT_DEFAULT = "data/training/dic_splits_v4_brightness/train"


def add_brightness(img, delta):
    return np.clip(img.astype(np.int16) + delta,
                   0, 255).astype(np.uint8)


def gamma(img, g):
    f = img.astype(np.float32) / 255.0
    return (np.clip(f ** g, 0, 1) * 255).astype(np.uint8)


def vignette(img, seed):
    h, w = img.shape
    rng = np.random.default_rng(seed)
    cy = h * 0.5 + rng.uniform(-0.15, 0.15) * h
    cx = w * 0.5 + rng.uniform(-0.15, 0.15) * w
    sigma = rng.uniform(0.35, 0.5) * min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    bias = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2)
                    / (2 * sigma ** 2)))
    bias = 0.55 + 0.45 * bias
    return np.clip(img.astype(np.float32) * bias,
                   0, 255).astype(np.uint8)


VARIANTS = {
    "b30+": lambda img, _: add_brightness(img, +30),
    "b60+": lambda img, _: add_brightness(img, +60),
    "b30-": lambda img, _: add_brightness(img, -30),
    "b60-": lambda img, _: add_brightness(img, -60),
    "g05":  lambda img, _: gamma(img, 0.5),
    "g18":  lambda img, _: gamma(img, 1.8),
    "vig":  vignette,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT,
                    help=f"source train dir (default {SRC_DEFAULT})")
    ap.add_argument("--out", default=OUT_DEFAULT,
                    help=f"output train dir (default {OUT_DEFAULT})")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="cap on source pairs (for testing)")
    args = ap.parse_args()

    img_files = sorted(glob.glob(os.path.join(args.src, "*_img.tif")))
    if args.max_pairs:
        img_files = img_files[:args.max_pairs]
    if not img_files:
        print(f"[augment_brightness] no images in {args.src}")
        return
    if os.path.exists(args.out):
        print(f"[augment_brightness] clearing {args.out}")
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    print(f"[augment_brightness] {len(img_files)} source pairs → "
          f"{args.out} ({len(VARIANTS) + 1}× expansion expected)")

    counts = {"orig": 0}
    counts.update({k: 0 for k in VARIANTS})

    for i, img_path in enumerate(img_files):
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
        base = os.path.basename(img_path).replace("_img.tif", "")
        # Copy original (unperturbed)
        tifffile.imwrite(os.path.join(args.out, f"{base}_img.tif"), img)
        shutil.copy(mask_path,
                    os.path.join(args.out, f"{base}_masks.tif"))
        counts["orig"] += 1
        for variant, fn in VARIANTS.items():
            perturbed = fn(img, i)
            tifffile.imwrite(
                os.path.join(args.out, f"{base}_{variant}_img.tif"),
                perturbed)
            shutil.copy(
                mask_path,
                os.path.join(args.out, f"{base}_{variant}_masks.tif"))
            counts[variant] += 1
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(img_files)} sources processed")

    print("\n=== Done ===")
    for k, n in counts.items():
        print(f"  {k:<6} {n}")
    total = sum(counts.values())
    print(f"  total {total} pairs in {args.out}")


if __name__ == "__main__":
    main()
