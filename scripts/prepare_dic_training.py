"""Prepare DIC training data with proper sequence-based splits.

Builds train/val/test directories in cellpose format from:
  - 5,290 VAMPIRE pairs (29 sequences, split by sequence)
  - 244 our full-frame GT pairs (122 ctrl + 122 cKO)
  - 168 CTC DIC-HeLa pairs

Split strategy: hold out entire sequences to prevent data leakage.
Test set: 6 sequences (2 per genotype category).
Val set: 3 sequences (1 per category).
Train: remaining 20 sequences + our GT + CTC.

Output: data/training/dic_splits/{train,val,test}/ with
  <name>_img.tif + <name>_masks.tif pairs.
"""
import os
import sys
import glob
import shutil
import random
import numpy as np
from collections import defaultdict

random.seed(42)
np.random.seed(42)

VAMPIRE_DIR = str(benchmark_data_root() / "data" / "training" / "vampire")
TRAINING_DIR = str(benchmark_data_root() / "data" / "training")
OUT_DIR = "data/training/dic_splits"

# Genotype categories for stratified splitting
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


def find_vampire_sequences():
    """Group VAMPIRE files by cell sequence."""
    img_files = sorted(glob.glob(os.path.join(VAMPIRE_DIR, "vampire_*_????.tif")))
    img_files = [f for f in img_files if "_masks" not in f]

    sequences = defaultdict(list)
    for f in img_files:
        base = os.path.basename(f)
        # vampire_<seq>_<NNNN>.tif
        parts = base.replace(".tif", "").split("_")
        frame_idx = parts[-1]
        seq = "_".join(parts[1:-1])
        mask_f = f.replace(".tif", "_masks.tif")
        if os.path.exists(mask_f):
            sequences[seq].append((f, mask_f))

    return dict(sequences)


def split_sequences(sequences):
    """Split sequences into train/val/test by genotype."""
    by_geno = defaultdict(list)
    for seq_name in sequences:
        geno = get_genotype(seq_name)
        by_geno[geno].append(seq_name)

    test_seqs = []
    val_seqs = []
    train_seqs = []

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


def copy_pairs(pairs, out_dir, prefix=""):
    """Copy image+mask pairs to output directory in cellpose format."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for img_path, mask_path in pairs:
        name = f"{prefix}{os.path.basename(img_path).replace('.tif', '')}"
        dst_img = os.path.join(out_dir, f"{name}_img.tif")
        dst_mask = os.path.join(out_dir, f"{name}_masks.tif")
        shutil.copy2(img_path, dst_img)
        # Convert mask to cellpose format (0=bg, 1=cell)
        import tifffile
        mask = tifffile.imread(mask_path)
        if mask.max() == 255:
            mask = (mask > 127).astype(np.uint16)
        tifffile.imwrite(dst_mask, mask.astype(np.uint16))
        count += 1
    return count


def add_our_gt(out_dir):
    """Add our full-frame GT pairs to training set."""
    count = 0
    import tifffile
    for gtype in ["ctrl", "cko"]:
        pattern = os.path.join(TRAINING_DIR, f"our_{gtype}_gt_*.png")
        img_files = sorted(glob.glob(pattern))
        img_files = [f for f in img_files if "_masks" not in f]
        for img_path in img_files:
            mask_path = img_path.replace(".png", "_masks.png")
            if not os.path.exists(mask_path):
                continue
            name = os.path.basename(img_path).replace(".png", "")
            dst_img = os.path.join(out_dir, f"{name}_img.tif")
            dst_mask = os.path.join(out_dir, f"{name}_masks.tif")
            import cv2
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint16)
            tifffile.imwrite(dst_img, img)
            tifffile.imwrite(dst_mask, mask)
            count += 1
    return count


def add_ctc(out_dir):
    """Add CTC DIC-HeLa pairs to training set."""
    count = 0
    import tifffile, cv2
    pattern = os.path.join(TRAINING_DIR, "ctc_*_*.png")
    img_files = sorted(glob.glob(pattern))
    img_files = [f for f in img_files if "_masks" not in f]
    for img_path in img_files:
        mask_path = img_path.replace(".png", "_masks.png")
        if not os.path.exists(mask_path):
            continue
        name = os.path.basename(img_path).replace(".png", "")
        dst_img = os.path.join(out_dir, f"{name}_img.tif")
        dst_mask = os.path.join(out_dir, f"{name}_masks.tif")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint16)
        tifffile.imwrite(dst_img, img)
        tifffile.imwrite(dst_mask, mask)
        count += 1
    return count


def main():
    print("=== Preparing DIC Training Data ===\n")

    # Clean output
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    # Step 1: find and split VAMPIRE sequences
    sequences = find_vampire_sequences()
    print(f"Found {len(sequences)} VAMPIRE sequences, "
          f"{sum(len(v) for v in sequences.values())} total pairs")

    for seq, pairs in sorted(sequences.items()):
        geno = get_genotype(seq)
        print(f"  {seq}: {len(pairs)} pairs ({geno})")

    train_seqs, val_seqs, test_seqs = split_sequences(sequences)
    print(f"\nSplit: train={len(train_seqs)} val={len(val_seqs)} "
          f"test={len(test_seqs)} sequences")

    print(f"\nTest sequences (held out completely):")
    for s in test_seqs:
        print(f"  {s} ({get_genotype(s)}, {len(sequences[s])} pairs)")
    print(f"\nVal sequences:")
    for s in val_seqs:
        print(f"  {s} ({get_genotype(s)}, {len(sequences[s])} pairs)")

    # Step 2: copy VAMPIRE pairs
    train_dir = os.path.join(OUT_DIR, "train")
    val_dir = os.path.join(OUT_DIR, "val")
    test_dir = os.path.join(OUT_DIR, "test")

    n_train = 0
    for seq in train_seqs:
        # Cap per sequence to prevent domination
        pairs = sequences[seq]
        if len(pairs) > 120:
            pairs = random.sample(pairs, 120)
        n_train += copy_pairs(pairs, train_dir, prefix="vamp_")
    print(f"\nVAMPIRE train: {n_train} pairs")

    n_val = 0
    for seq in val_seqs:
        n_val += copy_pairs(sequences[seq], val_dir, prefix="vamp_")
    print(f"VAMPIRE val: {n_val} pairs")

    n_test = 0
    for seq in test_seqs:
        n_test += copy_pairs(sequences[seq], test_dir, prefix="vamp_")
    print(f"VAMPIRE test: {n_test} pairs")

    # Step 3: add our GT to train (all of it — it's in-domain for
    # regression testing, not VAMPIRE domain)
    n_gt = add_our_gt(train_dir)
    print(f"Our GT added to train: {n_gt} pairs")

    # Step 4: add CTC to train
    n_ctc = add_ctc(train_dir)
    print(f"CTC added to train: {n_ctc} pairs")

    # Summary
    train_count = len(glob.glob(os.path.join(train_dir, "*_img.tif")))
    val_count = len(glob.glob(os.path.join(val_dir, "*_img.tif")))
    test_count = len(glob.glob(os.path.join(test_dir, "*_img.tif")))

    print(f"\n=== Final Split ===")
    print(f"  Train: {train_count} pairs")
    print(f"  Val:   {val_count} pairs")
    print(f"  Test:  {test_count} pairs")
    print(f"  Total: {train_count + val_count + test_count} pairs")
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
