"""Train cellpose_dic_v3: CP3 model on standardized 448px DIC crops.

All training data has been resized/cropped to 448x448 to match
the tiled inference approach.
"""
import os, sys, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports, benchmark_data_root  # noqa
setup_imports()

TRAIN_DIR = "data/training/dic_splits_v3/train"
N_EPOCHS = 100
LR = 0.005


def load_data(data_dir):
    import tifffile
    img_files = sorted(glob.glob(os.path.join(data_dir, "*_img.tif")))
    images, masks = [], []
    for f in img_files:
        mf = f.replace("_img.tif", "_masks.tif")
        if not os.path.exists(mf):
            continue
        images.append(tifffile.imread(f))
        masks.append(tifffile.imread(mf).astype(np.uint16))
    return images, masks


def main():
    print("=== Training cellpose_dic_v3 (448px standardized) ===\n")
    images, masks = load_data(TRAIN_DIR)
    print("Loaded %d pairs, shape: %s" % (len(images), images[0].shape))
    n_cells = sum(1 for m in masks if m.max() > 0)
    print("With cells: %d/%d" % (n_cells, len(masks)))

    from cellpose import models, train
    model = models.CellposeModel(gpu=True, model_type="cyto3")

    t0 = time.time()
    print("\nTraining 100 epochs...")
    new_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=masks,
        channels=[0, 0],
        save_path="data/models",
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        batch_size=8,
        model_name="cellpose_dic_v3",
        min_train_masks=1,
    )
    elapsed = time.time() - t0
    print("\nDone in %.1f minutes" % (elapsed / 60))
    print("Final train loss: %.4f" % train_losses[-1])
    print("Model: %s" % new_path)


if __name__ == "__main__":
    main()
