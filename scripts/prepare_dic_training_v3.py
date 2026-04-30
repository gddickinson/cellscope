"""Prepare standardized DIC training data at a fixed tile size.

All training images are resized/cropped to TARGET_SIZE x TARGET_SIZE
to match tiled inference at the same scale.

Strategy per image:
  - If smaller than TARGET_SIZE: pad with edge reflection
  - If larger: center-crop to TARGET_SIZE
  - Same transform applied to both image and mask

This ensures the model sees cells at exactly the scale it will
encounter during tiled DIC inference.
"""
import os
import sys
import glob
import shutil
import random
import numpy as np
import cv2
from collections import defaultdict

random.seed(42)
np.random.seed(42)

VAMPIRE_DIR = str(benchmark_data_root() / "data" / "training" / "vampire")
TRAINING_DIR = str(benchmark_data_root() / "data" / "training")
OUT_DIR = "data/training/dic_splits_v3"
TARGET_SIZE = 448

# Same genotype map as v1
GENOTYPE_MAP = {
    "Pos10-WT": "control", "Pos16-WT": "control", "Pos3-WT": "control",
    "Pos6-WT_Cell1": "control", "Pos6-WT_Cell2": "control",
    "Pos8-WT": "control", "Pos70-WT": "control",
    "Pos1-WT": "control", "Pos14-WT": "control", "Pos22-WT": "control",
    "Pos15-wt": "control", "Pos40-con": "control", "Pos54-con": "control",
    "Pos63-KO": "cKO", "Pos58-KO": "cKO", "Pos62-KO": "cKO",
    "Pos64-KO": "cKO", "Pos27-KO": "cKO", "Pos51-KO": "cKO",
    "Pos54-GoF": "gof", "Pos73-GoF": "gof", "Pos75-GoF": "gof",
    "Pos0-gof": "gof", "Pos1-gof": "gof", "Pos23-gof": "gof",
    "Pos34-gof": "gof",
}


def get_genotype(seq_name):
    for key, geno in GENOTYPE_MAP.items():
        if key in seq_name:
            return geno
    return "unknown"


def standardize(img, mask, target=TARGET_SIZE):
    """Resize image+mask to target x target."""
    h, w = img.shape[:2]
    if h >= target and w >= target:
        # Center crop
        r0 = (h - target) // 2
        c0 = (w - target) // 2
        img_out = img[r0:r0+target, c0:c0+target]
        mask_out = mask[r0:r0+target, c0:c0+target]
    elif h < target and w < target:
        # Pad both dimensions with reflection
        pad_h = target - h
        pad_w = target - w
        img_out = cv2.copyMakeBorder(
            img, pad_h//2, pad_h - pad_h//2,
            pad_w//2, pad_w - pad_w//2,
            cv2.BORDER_REFLECT_101)
        mask_out = cv2.copyMakeBorder(
            mask, pad_h//2, pad_h - pad_h//2,
            pad_w//2, pad_w - pad_w//2,
            cv2.BORDER_CONSTANT, value=0)
    else:
        # Mixed: crop the large dimension, pad the small one
        if h >= target:
            r0 = (h - target) // 2
            img = img[r0:r0+target, :]
            mask = mask[r0:r0+target, :]
        else:
            pad_h = target - h
            img = cv2.copyMakeBorder(
                img, pad_h//2, pad_h - pad_h//2, 0, 0,
                cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(
                mask, pad_h//2, pad_h - pad_h//2, 0, 0,
                cv2.BORDER_CONSTANT, value=0)
        if w >= target:
            c0 = (w - target) // 2
            img_out = img[:, c0:c0+target]
            mask_out = mask[:, c0:c0+target]
        else:
            pad_w = target - w
            img_out = cv2.copyMakeBorder(
                img, 0, 0, pad_w//2, pad_w - pad_w//2,
                cv2.BORDER_REFLECT_101)
            mask_out = cv2.copyMakeBorder(
                mask, 0, 0, pad_w//2, pad_w - pad_w//2,
                cv2.BORDER_CONSTANT, value=0)

    assert img_out.shape == (target, target), \
        f"Expected ({target},{target}), got {img_out.shape}"
    return img_out, mask_out


def find_vampire_sequences():
    img_files = sorted(glob.glob(os.path.join(VAMPIRE_DIR, "vampire_*_????.tif")))
    img_files = [f for f in img_files if "_masks" not in f]
    sequences = defaultdict(list)
    for f in img_files:
        base = os.path.basename(f).replace(".tif", "")
        parts = base.split("_")
        seq = "_".join(parts[1:-1])
        mask_f = f.replace(".tif", "_masks.tif")
        if os.path.exists(mask_f):
            sequences[seq].append((f, mask_f))
    return dict(sequences)


def split_sequences(sequences):
    by_geno = defaultdict(list)
    for seq_name in sequences:
        geno = get_genotype(seq_name)
        by_geno[geno].append(seq_name)
    test_seqs, val_seqs, train_seqs = [], [], []
    for geno, seqs in sorted(by_geno.items()):
        random.shuffle(seqs)
        n = len(seqs)
        if n >= 4:
            test_seqs.extend(seqs[:2])
            val_seqs.extend(seqs[2:3])
            train_seqs.extend(seqs[3:])
        elif n >= 2:
            test_seqs.extend(seqs[:1])
            val_seqs.extend(seqs[1:2])
            train_seqs.extend(seqs[2:])
        else:
            train_seqs.extend(seqs)
    return train_seqs, val_seqs, test_seqs


def process_pairs(pairs, out_dir, prefix="", max_per_seq=120):
    """Standardize and save image+mask pairs."""
    import tifffile
    os.makedirs(out_dir, exist_ok=True)
    if len(pairs) > max_per_seq:
        pairs = random.sample(pairs, max_per_seq)
    count = 0
    for img_path, mask_path in pairs:
        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)
        if img.ndim == 3:
            img = img[0]
        if mask.ndim == 3:
            mask = mask[0]
        if mask.max() == 255:
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = (mask > 0).astype(np.uint8)
        img_s, mask_s = standardize(img, mask)
        name = prefix + os.path.basename(img_path).replace(".tif", "")
        tifffile.imwrite(os.path.join(out_dir, name + "_img.tif"), img_s)
        tifffile.imwrite(
            os.path.join(out_dir, name + "_masks.tif"),
            mask_s.astype(np.uint16))
        count += 1
    return count


def add_our_gt(out_dir):
    """Add our 526x526 GT (center-cropped to TARGET_SIZE)."""
    import tifffile
    count = 0
    for gtype in ["ctrl", "cko"]:
        pattern = os.path.join(TRAINING_DIR, "our_%s_gt_*.png" % gtype)
        img_files = sorted(glob.glob(pattern))
        img_files = [f for f in img_files if "_masks" not in f]
        for img_path in img_files:
            mask_path = img_path.replace(".png", "_masks.png")
            if not os.path.exists(mask_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = (mask > 0).astype(np.uint8)
            img_s, mask_s = standardize(img, mask)
            name = os.path.basename(img_path).replace(".png", "")
            tifffile.imwrite(
                os.path.join(out_dir, name + "_img.tif"), img_s)
            tifffile.imwrite(
                os.path.join(out_dir, name + "_masks.tif"),
                mask_s.astype(np.uint16))
            count += 1
    return count


def main():
    print("=== Preparing Standardized DIC Training Data (v3) ===")
    print("Target size: %dx%d\n" % (TARGET_SIZE, TARGET_SIZE))

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    sequences = find_vampire_sequences()
    print("Found %d VAMPIRE sequences" % len(sequences))

    train_seqs, val_seqs, test_seqs = split_sequences(sequences)
    print("Split: train=%d val=%d test=%d sequences\n" % (
        len(train_seqs), len(val_seqs), len(test_seqs)))

    train_dir = os.path.join(OUT_DIR, "train")
    val_dir = os.path.join(OUT_DIR, "val")
    test_dir = os.path.join(OUT_DIR, "test")

    n_train = sum(process_pairs(sequences[s], train_dir, "vamp_")
                  for s in train_seqs)
    print("VAMPIRE train: %d pairs" % n_train)

    n_val = sum(process_pairs(sequences[s], val_dir, "vamp_", max_per_seq=999)
                for s in val_seqs)
    print("VAMPIRE val: %d pairs" % n_val)

    n_test = sum(process_pairs(sequences[s], test_dir, "vamp_", max_per_seq=999)
                 for s in test_seqs)
    print("VAMPIRE test: %d pairs" % n_test)

    n_gt = add_our_gt(train_dir)
    print("Our GT added to train: %d pairs" % n_gt)

    # Verify all images are TARGET_SIZE x TARGET_SIZE
    import tifffile
    train_imgs = glob.glob(os.path.join(train_dir, "*_img.tif"))
    shapes = set()
    for f in train_imgs[:50]:
        shapes.add(tifffile.imread(f).shape)
    print("\nVerified shapes: %s" % shapes)

    total = len(glob.glob(os.path.join(train_dir, "*_img.tif")))
    print("\n=== Final: %d train, %d val, %d test ===" % (total, n_val, n_test))
    print("Output: %s" % OUT_DIR)


if __name__ == "__main__":
    main()
