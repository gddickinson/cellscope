"""Train cellpose_dic_v2: CP3 model fine-tuned on DIC data.

Uses the dic_splits/train directory prepared by prepare_dic_training.py.
Fine-tunes from cyto3 base model with channels=[0,0] (grayscale).

Output: data/models/cellpose_dic_v2
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
OUT_MODEL = "data/models/cellpose_dic_v2"
N_EPOCHS = 100
LR = 0.005
BATCH_SIZE = 8


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
    print("=== Training cellpose_dic_v2 ===\n")

    print(f"Loading training data from {TRAIN_DIR}...")
    images, masks = load_training_data(TRAIN_DIR)
    print(f"Loaded {len(images)} training pairs")
    print(f"Image sizes: {set(img.shape for img in images[:20])}")

    # Check mask statistics
    n_with_cells = sum(1 for m in masks if m.max() > 0)
    print(f"Pairs with cells: {n_with_cells}/{len(masks)}")

    print(f"\nTraining config:")
    print(f"  Base model: cyto3")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Learning rate: {LR}")
    print(f"  Output: {OUT_MODEL}")

    from cellpose import models, train

    t0 = time.time()
    model = models.CellposeModel(gpu=True, model_type="cyto3")

    print(f"\nStarting training...")
    new_model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=masks,
        channels=[0, 0],
        save_path=os.path.dirname(OUT_MODEL),
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        model_name=os.path.basename(OUT_MODEL),
        min_train_masks=1,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    if hasattr(test_losses, '__len__') and len(test_losses) > 0:
        print(f"Final test loss: {test_losses[-1]:.4f}")
    print(f"Model saved: {new_model_path}")


if __name__ == "__main__":
    main()
