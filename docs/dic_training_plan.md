# DIC-Specific Model Training Plans

## Context

CellScope's hybrid cpsam pipeline was optimized for phase-contrast
microscopy (Ignasi's recordings: 0.932 IoU). On Jesse's DIC recordings,
it achieves only 0.303–0.519 IoU because cpsam picks up DIC's textured
background as false positives. DIC (Differential Interference Contrast)
has fundamentally different contrast characteristics — cells appear as
raised/sunken relief rather than dark-on-bright.

Two training approaches can address this:

---

## Plan A: Fine-Tune cpsam on DIC Data

### Goal
Adapt cpsam's ViT backbone to DIC contrast so it stops detecting
background texture as cell material.

### Data Available
- **5,290 pre-aligned (DIC crop, mask) pairs** at
  `data/training/vampire/` across 3 genotypes (cKO, WT, GoF) and
  29 cell sequences
- Crop sizes vary: 242×282 to 576×688 (single-cell crops)
- Masks are binary (0/255), one cell per crop
- DIC crops are uint8, shade-corrected

### Approach

**Step 1: Prepare training data (1 day)**
- Convert VAMPIRE pairs to cellpose training format:
  `<name>_img.tif` + `<name>_masks.tif` (uint16 labels, 0=bg, 1=cell)
- Split by cell sequence (not frame) to prevent data leakage:
  - Train: 20 sequences (~3,700 pairs)
  - Validation: 5 sequences (~800 pairs)
  - Test: 4 sequences (~790 pairs, held out completely)
- Augment training set with:
  - Elastic deformation (α=150, σ=8) — already implemented
  - Motion blur (kernel 9, random angle) — already implemented
  - Defocus blur (σ∈[1,3]) — already implemented
  - Random intensity jitter (brightness ±20%, contrast ±30%)
  - Total: ~15,000 training pairs

**Step 2: Fine-tune cpsam (1-2 days compute)**
- Base: cellpose 4.1.1 default cpsam model (ViT-H backbone)
- Cellpose 4.x training API:
  ```python
  from cellpose import train
  model = train.train_seg(
      images, labels,
      model_type='cpsam',
      n_epochs=100,
      learning_rate=1e-5,  # low LR for fine-tuning
      weight_decay=1e-4,
      batch_size=8,
  )
  ```
- Freeze ViT encoder for first 20 epochs (adapter-only warmup),
  then unfreeze with 10× lower LR for encoder layers
- Monitor validation IoU every 5 epochs; early stop if no improvement
  for 20 epochs
- Expected training time: ~6-8 hours on MPS (M-series Mac)

**Step 3: Evaluate (0.5 day)**
- Per-frame IoU on held-out test sequences
- Compare against:
  - cpsam defaults (current, ~0.4 IoU on DIC)
  - cellpose_dic CP3 (current DIC champion, ~0.45 IoU)
  - Jesse's masks (ground truth)
- Per-genotype breakdown (WT vs cKO vs GoF)
- False positive rate (background detections)
- Overlay figures for visual QC

**Step 4: Ship (0.5 day)**
- Save as `data/models/cpsam_dic_finetuned`
- Add to GUI modality selector as "cpsam (DIC fine-tuned)"
- Update presets

### Risks
- Cellpose 4.x training API may not support cpsam fine-tuning directly
  (ViT encoder is frozen by default in cellpose 4). May need to use
  the underlying torch training loop.
- Small crop sizes (242×282) may not match cpsam's expected input scale.
  May need to pad or resize.
- DIC contrast varies between experiments (100_2018 vs 126_2019 vs
  135_2019 vs 240_2021). Fine-tuning on one experiment might not
  generalize to others → need cross-experiment validation.

### Expected Outcome
- IoU improvement from ~0.4 to 0.6-0.7 on DIC recordings
- Reduced false positive rate on DIC background
- May still need DeepSea refinement for best results

---

## Plan B: Train a DIC-Specific Cellpose 3 Detector

### Goal
Train a cellpose 3 model specifically for DIC endothelial cells using the
full VAMPIRE dataset. Unlike Plan A (fine-tuning cpsam), this trains a
lighter CNN-based model that runs in the CP3 env without needing the
ViT backbone.

### Data Available
Same 5,290 pairs as Plan A, plus:
- **244 full-frame our-GT pairs** (122 control + 122 cKO at 526×526)
- **168 CTC DIC-HeLa pairs** (different cell type but same modality)
- Total potential: ~5,700 DIC pairs

### Approach

**Step 1: Prepare mixed-scale training data (1 day)**
- Combine VAMPIRE crops (varied sizes) + our full-frame GT (526²) +
  CTC (varied)
- Key lesson from Test 19: naive mixing caused 88% of training to be
  Jesse's pre-cropped format → catastrophic failure on full-frame.
  Fix: **stratified sampling** — cap any single source at 30% of
  training set, oversample minority sources.
- Training mix target:
  - VAMPIRE crops: 1,500 (30%, sampled across all 29 sequences)
  - Our full-frame GT: 244 (5%, all included)
  - Augmented our-GT: 1,000 (20%, elastic+motion+defocus+intensity)
  - Augmented VAMPIRE: 1,500 (30%, same augmentations)
  - CTC: 168 (3%)
  - Hard negatives: 600 (12%, DIC background patches as negative examples)
  - Total: ~5,000 balanced pairs

**Step 2: Train with DIC-specific augmentation (4-6 hours)**
- Base model: cyto3 (general cellpose)
- Training script: extend `scripts/retrain_cellpose_v2.py`
  ```python
  model.train(
      images, labels,
      channels=[0, 0],  # grayscale
      n_epochs=100,
      learning_rate=0.005,
      model_name='cellpose_dic_v2',
  )
  ```
- DIC-specific augmentation additions:
  - **DIC shading simulation**: add a linear gradient across the image
    (DIC's characteristic illumination asymmetry)
  - **Relief inversion**: flip contrast (simulates cells that appear
    as depressions rather than protrusions — common in DIC when
    focusing through the sample)
  - **Texture noise**: add Perlin noise at the spatial frequency of
    DIC background texture

**Step 3: Evaluate (0.5 day)**
- Same evaluation protocol as Plan A
- Additionally test on our full-frame recordings (regression check)
- Compare against cellpose_dic (current) and cellpose_combined_robust

**Step 4: Ship (0.5 day)**
- Save as `data/models/cellpose_dic_v2`
- Integrate into DIC modality pipeline
- Update presets

### Risks
- Test 19 showed that mixing pre-cropped + full-frame data can hurt.
  Stratified sampling and augmentation should mitigate this.
- DIC background texture varies significantly between experiments.
  Hard-negative mining helps but may not generalize to unseen labs.
- cellpose_dic (current) was trained on a different dataset; the new
  model may regress on our recordings even as it improves on VAMPIRE.

### Expected Outcome
- IoU improvement from ~0.45 to 0.55-0.65 on VAMPIRE crops
- Maintained 122/122 detection on our full-frame recordings
- Faster inference than cpsam (CNN vs ViT)

---

## Plan C: Hybrid Approach (Recommended)

### Goal
Combine the best of both: fine-tuned cpsam for primary detection +
DIC-specific cellpose as fallback, with modality-aware preprocessing.

### Pipeline
```
Auto-detect modality (phase-contrast vs DIC)
  │
  ├─ Phase-contrast: cpsam → DeepSea union (current best, 0.932 IoU)
  │
  └─ DIC:
      1. Preprocessing: temporal median bg subtraction + high-pass
      2. Primary: cpsam_dic_finetuned (Plan A output)
      3. Fallback: cellpose_dic_v2 (Plan B output) for missed frames
      4. DeepSea union on all frames
      5. Post-processing: min_area filter, boundary confidence check
```

### Implementation Order
1. **Immediate**: Ship DIC preprocessing + cellpose_dic pipeline
   (no training needed, uses existing models)
2. **Week 1**: Train cellpose_dic_v2 (Plan B — faster, lower risk)
3. **Week 2**: Fine-tune cpsam on DIC (Plan A — higher potential,
   more complex)
4. **Week 3**: Benchmark all combinations, ship best as default

### Evaluation Criteria
- Mean IoU ≥ 0.6 on VAMPIRE held-out test sequences
- False positive rate < 5% (background detections)
- No regression on Ignasi phase-contrast (maintain 0.932)
- No regression on our full-frame DIC (maintain 122/122 detection)

---

## Data Preparation Checklist

- [ ] Build VAMPIRE train/val/test split by sequence (prevent leakage)
- [ ] Convert masks to cellpose format (uint16 labels)
- [ ] Generate augmented pairs (elastic, motion blur, defocus, intensity)
- [ ] Create hard-negative set (DIC background patches)
- [ ] Validate split: no sequence overlap between train/val/test
- [ ] Check class balance across genotypes in each split

## Infrastructure Needed

- `scripts/prepare_dic_training.py` — data preparation + splits
- `scripts/train_cpsam_dic.py` — Plan A fine-tuning (cellpose4 env)
- `scripts/train_cellpose_dic_v2.py` — Plan B training (cellpose env)
- `scripts/evaluate_dic_models.py` — standardized benchmark
- `core/modality.py` — auto-detection + per-modality pipeline routing
