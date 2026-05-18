# Brightness robustness — workflow

Phase 3.5 of the roadmap. Background: the current `cpsam_dic` v2 was
fine-tuned on shade-corrected DIC crops at one fixed exposure. On
recordings with even modest exposure / illumination drift it loses a
lot of IoU — the same failure mode the legacy piezo1_analysis project
hit on its `bright_dark` / `bright_bright` perturbations.

This document is the end-to-end workflow for measuring that gap and
closing it via a brightness-augmented retrain (`cpsam_dic_v4`).

## 1 — Build the brightness test set (one-time)

Generates 8 perturbation directories from the existing 1,129-pair
v3 test set. Fast, CPU only, ~1 minute:

```bash
python scripts/build_brightness_test.py
```

Output:

```
data/training/dic_splits_v3/test_brightness/
  clean/        (sanity baseline — identity copy)
  b_plus_30/    +30 brightness
  b_plus_60/    +60 brightness
  b_minus_30/   -30
  b_minus_60/   -60
  gamma_05/     gamma 0.5  (heavy contrast lift)
  gamma_18/     gamma 1.8  (heavy contrast crush)
  vignette/     2-D Gaussian shading bias
```

Visual sanity check: `results/brightness_eval/perturbation_preview.png`.

Each directory keeps the v3 file naming so `bench_cpsam_dic.py
--test-dir <subdir>` works unchanged.

## 2 — Bench the current model (the baseline)

Runs the bench across all perturbations in series and writes a
per-model summary + markdown table:

```bash
conda run -n cellpose4 python scripts/bench_brightness.py \
    --model data/models/cpsam_dic --label cpsam_dic_v2
```

Output: `results/brightness_eval/cpsam_dic_v2/{summary.json,
summary.md}` plus per-perturbation JSONs.

This is also the place to note **how bad the gap is** before
training. We expect retention < 0.6 on `b_minus_60` and `gamma_18`
(both push the image into a regime cpsam was never trained on).

## 3 — Build the augmented training set (one-time)

Generates the v4 brightness-augmented training set from v3 train.
~5 minutes CPU, ~21k pairs (8× v3):

```bash
python scripts/augment_brightness.py
```

Output: `data/training/dic_splits_v4_brightness/train/` with each
v3 source pair plus 7 brightness-perturbed copies. Same
augmentations as the test set so the model sees the exact failure
modes during training.

## 4 — Train cpsam_dic v4 (Colab, 6 h on T4 free)

Use `notebooks/train_cpsam_dic_v4_brightness_colab.ipynb`. Uploads
the v4 dataset to Drive, fine-tunes cpsam from scratch (lr=1e-5,
20 epochs, batch=1 to fit T4). Stratified subsampling (125 per
brightness variant) keeps each variant equally represented in the
training distribution.

The notebook's Step 5 does a quick 30-frame validation on both clean
and `b_plus_60` data so you can spot a training collapse before
spending the download bandwidth.

## 5 — Re-bench and compare

```bash
conda run -n cellpose4 python scripts/bench_brightness.py \
    --model data/models/cpsam_dic_v4 --label cpsam_dic_v4

python scripts/bench_brightness.py --compare cpsam_dic_v2 cpsam_dic_v4
```

Output: `results/brightness_eval/comparison.md` — side-by-side IoU
and retention.

**Ship criterion**: every perturbation must hit retention ≥ 0.8
without losing more than 0.02 mean IoU on `clean`. If met, promote
v4 to the default DIC model:

```bash
mv data/models/cpsam_dic data/models/cpsam_dic_v2_legacy
ln -s cpsam_dic_v4 data/models/cpsam_dic
```

## What we explicitly did NOT do

* No GUI changes — the brightness-augmented model is a drop-in
  replacement for the existing one.
* No new conda env — runs in `cellpose4` like the v2 model.
* No changes to the augmentation pipeline used inside cellpose's
  training loop. We're enriching the *input distribution* the model
  sees, not editing cellpose itself.
* The augmentation does not modify masks (geometry is preserved).
  Cellpose's internal flip/rotate handles geometric variation.
