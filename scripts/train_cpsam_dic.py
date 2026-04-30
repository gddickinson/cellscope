"""Fine-tune cpsam on DIC data (Plan A).

Requires cellpose4 env (cellpose >= 4.1.1).
Fine-tunes the default cpsam model on VAMPIRE DIC pairs.

Output: data/models/cpsam_dic
"""
import os
import sys
import glob
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports, benchmark_data_root  # noqa
setup_imports()

TRAIN_DIR = "data/training/dic_splits/train"
OUT_MODEL = "data/models/cpsam_dic"
N_EPOCHS = 50
LR = 1e-5
BATCH_SIZE = 4


def load_training_data(data_dir):
    """Load all image+mask pairs from a directory."""
    import tifffile
    img_files = sorted(glob.glob(os.path.join(data_dir, "*_img.tif")))
    images = []
    masks = []
    for img_path in img_files:
        mask_path = img_path.replace("_img.tif", "_masks.tif")
        if not os.path.exists(mask_path):
            continue
        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)
        if img.ndim == 3:
            img = img[0]
        if mask.ndim == 3:
            mask = mask[0]
        images.append(img)
        masks.append(mask.astype(np.uint16))
    return images, masks


def main():
    print("=== Fine-tuning cpsam on DIC ===\n")

    import cellpose
    cp_ver = getattr(cellpose, '__version__', None) or cellpose.version
    print(f"cellpose version: {cp_ver}")
    if not cp_ver.startswith("4"):
        print("ERROR: Need cellpose 4.x for cpsam. Run in cellpose4 env.")
        sys.exit(1)

    print(f"Loading training data from {TRAIN_DIR}...")
    images, masks = load_training_data(TRAIN_DIR)
    print(f"Loaded {len(images)} training pairs")

    n_with_cells = sum(1 for m in masks if m.max() > 0)
    print(f"Pairs with cells: {n_with_cells}/{len(masks)}")

    print(f"\nTraining config:")
    print(f"  Base model: cpsam (default cellpose 4.x)")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Learning rate: {LR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Output: {OUT_MODEL}")

    from cellpose import models, train

    t0 = time.time()

    # Load the default cpsam model
    model = models.CellposeModel(gpu=True)

    print(f"\nStarting training...")
    try:
        new_model_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_data=images,
            train_labels=masks,
            save_path="data/models",
            n_epochs=N_EPOCHS,
            learning_rate=LR,
            batch_size=BATCH_SIZE,
            model_name="cpsam_dic",
            min_train_masks=1,
        )

        elapsed = time.time() - t0
        print(f"\nTraining complete in {elapsed/60:.1f} minutes")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        if hasattr(test_losses, '__len__') and len(test_losses) > 0:
            print(f"Final test loss: {test_losses[-1]:.4f}")
        print(f"Model saved: {new_model_path}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\nTraining failed after {elapsed/60:.1f} minutes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
