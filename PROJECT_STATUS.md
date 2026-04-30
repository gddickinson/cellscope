# CellScope — Project Status

*Last updated: 2026-04-20*

## What is CellScope?

CellScope is an automated cell detection, tracking, and analysis
platform for DIC and phase-contrast time-lapse microscopy. It provides
end-to-end analysis from raw recordings to publication-ready figures
and statistical comparisons.

Built to analyze migrating keratinocytes from Holt et al. 2021 (eLife),
extensible to any single- or multi-cell DIC/phase-contrast recordings.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    main_suite.py                        │
│              Unified Launcher (tkinter)                 │
├───────────┬───────────┬──────────┬──────────┬──────────┤
│ Detection │   Batch   │ Tracking │   Mask   │ Training │
│ & Analysis│ Processing│ & Stats  │  Editor  │          │
│ (focused) │  (batch)  │(tracking)│ (editor) │(training)│
├───────────┴───────────┴──────────┴──────────┴──────────┤
│                   gui/ (shared)                         │
│        mask_editor, run_log, options panels             │
├────────────────────────────────────────────────────────┤
│                    core/ (34 modules)                   │
│  detection · tracking · morphology · edge dynamics      │
│  preprocessing · refinement · statistics · VAMPIRE      │
├────────────────────────────────────────────────────────┤
│                    output/                              │
│        masks.npz · metrics.json · plots · CSV           │
└────────────────────────────────────────────────────────┘
```

- **109 Python files** across 9 packages
- **5 specialized GUIs** + unified launcher
- **34 core modules** for detection, tracking, and analysis
- **Dual conda environments**: cellpose4 (cpsam ViT) + cellpose (CP3 fallback)

---

## Detection Pipelines

### Phase-Contrast (Ignasi's recordings)
```
cpsam (ViT) → DeepSea union → fallback (CP3) → gap fill
```
- **0.932 IoU** on 65 GT frames (best result)
- 100% detection across 5 recordings (485 frames)
- Auto-fallback rescues 1-4 frames per recording

### DIC (Jesse's VAMPIRE recordings)
```
cellpose_dic → preprocessing (temporal bg + HP) → retry → DeepSea
```
- **0.512 IoU** with cellpose_dic_v2 + preprocessing (best DIC result)
- +180% over cpsam on DIC (0.512 vs 0.183)
- Modality auto-detection routes to correct pipeline

### Multi-Cell
```
cpsam/cellpose → debris filter → DeepSea per-cell → Hungarian tracker
→ gap fill → post-process
```
- Division detection (mother→daughter tracking)
- 100% gap fill rate (41/41 gaps on tested recordings)
- TRA 0.929 on CTC DIC-C2DH-HeLa benchmark

---

## Models

| Model | Type | Trained On | Best For |
|-------|------|-----------|----------|
| cpsam (default) | ViT (CP4) | General microscopy | Phase-contrast |
| cellpose_dic | CNN (CP3) | Our DIC keratinocytes | Our full-frame DIC |
| cellpose_dic_v2 | CNN (CP3) | 2,812 DIC pairs (VAMPIRE+GT+CTC) | DIC with preprocessing |
| cellpose_dic_v3 | CNN (CP3) | 2,644 standardized 448px crops | Tiled DIC inference (training) |
| cellpose_robust_v2 | CNN (CP3) | 5,826 augmented pairs | Noisy/perturbed data |
| DeepSea | U-Net | Brightfield/phase-contrast | Boundary refinement |
| MedSAM | SAM-ViT | Biomedical images | Foundation model refinement |
| cpsam_dic | ViT (CP4) | — | DIC (planned, needs GPU training) |

---

## Benchmark Results

### Phase-Contrast Detection (Ignasi GT, 65 frames)
| Method | Mean IoU | >0.85 | Min |
|--------|----------|-------|-----|
| **cpsam + DeepSea** | **0.932** | 65/65 | 0.867 |
| cellpose + MedSAM + DeepSea | 0.907 | 55/65 | 0.629 |
| cpsam alone | 0.915 | 56/65 | — |

### DIC Detection (VAMPIRE, full-recording pipeline)
| Method | control | cKO | GoF | Mean |
|--------|---------|-----|-----|------|
| **cellpose_dic_v2 + preproc** | **0.874** | **0.503** | 0.159 | **0.512** |
| cellpose_robust | 0.652 | 0.352 | **0.434** | 0.479 |
| cellpose_dic + preproc | 0.493 | 0.456 | 0.191 | 0.380 |
| cpsam + DeepSea | — | — | — | 0.183 |

### Our Full-Frame GT (244 frames, regression check)
| Method | control | cKO | Mean |
|--------|---------|-----|------|
| **cellpose_robust** | **0.747** | **0.648** | **0.697** |
| cellpose_dic | 0.642 | 0.396 | 0.519 |
| cellpose_dic_v2 | 0.388 | 0.435 | 0.411 |

### Multi-Cell Tracking (CTC DIC-C2DH-HeLa)
| Tracker | DET | TRA | SEG |
|---------|-----|-----|-----|
| **Hungarian (ours)** | 0.936 | **0.929** | 0.860 |
| Trackastra | 0.935 | 0.847 | 0.860 |

---

## Analysis Capabilities

**20 graph types** including:
- Cell trajectory + speed over time
- MSD with diffusion model fit (D and α extraction)
- Area, circularity, aspect ratio timeseries
- Edge velocity kymograph (16 angular sectors)
- Protrusion/retraction rates
- VAMPIRE shape modes (PCA eigenshapes + K-means clustering)
- Shape mode distribution + heterogeneity (Shannon entropy)
- Frame quality assessment
- Bootstrap confidence intervals on all metrics

**Statistical comparison** (batch mode):
- 2 groups: t-test, Mann-Whitney, Cohen's d
- 3+ groups: ANOVA, Kruskal-Wallis, Bonferroni post-hoc
- Auto parametric/non-parametric selection via Shapiro-Wilk
- Box/violin plots with significance brackets

---

## GUI Features

### Detection & Analysis GUI (`main_focused.py`)
- Single-cell and multi-cell modes
- Modality selector: Auto / DIC / Phase-contrast
- Brightness/contrast adjustment, zoom, pan
- Frame navigator bar (green=detected, red=missed)
- ROI selector (rectangle/ellipse/polygon)
- Cancel button with elapsed time display
- Drag-and-drop file loading
- Project save/load (.cellscope format)
- 20 analysis graph types with resizable panels
- Export: masks, metrics, plots, MP4 overlay video, per-cell CSV

### Mask Editor (`main_editor.py`)
- Brush/eraser/polygon/fill tools
- Multi-cell labels (keys 1-9, spinbox for more)
- Per-cell colored overlay
- Relabel tool (connected component only)
- Separate brush/eraser size controls
- Send corrections back to main GUI

### Batch Processing (`main_batch.py`)
- Directory scan, recording tree
- Group-by-folder organization
- Per-recording and group CSV summaries

### Tracking & Comparison (`main_tracking.py`)
- Single recording: load masks → track → per-cell analysis
- Batch: multi-recording group comparison with statistical tests

### Training (`main_training.py`)
- Data preview thumbnails
- Live loss curve
- Augmentation options (noise, gamma, flip)

---

## Data Assets

| Dataset | Pairs | Format | Purpose |
|---------|-------|--------|---------|
| VAMPIRE (Jesse) | 5,290 | TIFF crops | DIC keratinocyte training |
| Our GT (control) | 122 | PNG 526² | Full-frame evaluation |
| Our GT (cKO) | 122 | PNG 526² | Full-frame evaluation |
| CTC DIC-HeLa | 168 | PNG 512² | Cross-domain benchmark |
| Augmented v2 | 5,269 | PNG | Robustness training |
| dic_splits_v3 (448px) | 4,302 | TIFF 448² | Standardized DIC training |

**29 VAMPIRE cell sequences** across 3 genotypes (control, cKO, GoF)
from 4 experiments (85, 100, 126, 135, 240).

---

## What's Working Well

- **Phase-contrast detection**: 0.932 IoU, 100% detection — production ready
- **Multi-cell tracking**: TRA 0.929 on CTC benchmark, division detection works
- **Analysis suite**: 20 graph types, VAMPIRE shape modes, statistical comparison
- **GUI**: 5 specialized apps covering the full workflow
- **Robustness**: cellpose_robust_v2 wins 9/11 perturbation tests

## Current Limitations

- **DIC detection**: 0.512 IoU best (vs 0.932 on phase-contrast) — DIC
  background texture still causes false positives
- **GoF genotype**: Hardest to detect (0.159-0.434 IoU depending on model)
- **cpsam fine-tuning**: ViT too slow on Apple Silicon MPS — needs CUDA GPU
  (Colab notebook ready at `notebooks/train_cpsam_dic_colab.ipynb`)
- **Multi-cell GT**: Limited ground truth for multi-cell evaluation
- **ID switching**: Touching cells still cause occasional identity swaps

---

## In Progress

1. **cellpose_dic_v3 training** — CP3 model on standardized 448px crops
   to match tiled inference (training now, ~2h)
2. **cpsam DIC fine-tuning** — Colab notebook ready, waiting for Drive
   sync to upload training data
3. **Tiled DIC inference** — `core/dic_tiled.py` tiles 1024² frames into
   448px patches matching the VAMPIRE training scale (+5-7% IoU on some
   recordings)

## Planned Next Steps

- Fine-tune cpsam on DIC via Colab (expected: 0.6-0.7 IoU on DIC)
- Evaluate dic_v3 with tiled inference on all VAMPIRE recordings
- Multi-cell ground truth expansion
- Boundary separation for touching cells
- False positive rejection improvements
- GitHub release preparation

---

## File Organization

```
cellscope/
├── core/              34 analysis modules
├── gui/               Shared GUI components
├── gui_focused/       Detection & Analysis GUI (12 files)
├── gui_batch/         Batch Processing GUI (3 files)
├── gui_tracking/      Tracking & Comparison GUI (6 files)
├── gui_editor/        Mask Editor GUI (3 files)
├── gui_training/      Model Training GUI (4 files)
├── output/            Result writers (2 files)
├── scripts/           Training, evaluation, testing (12 files)
├── notebooks/         Colab training notebook
├── docs/              Plans, manual, pipeline description
├── data/
│   ├── models/        cellpose_dic, dic_v2, robust, DeepSea
│   └── training/      dic_splits_v3 (448px standardized)
├── results/           Benchmark outputs
├── main_suite.py      Unified launcher
├── main_focused.py    Detection & Analysis GUI entry
├── main_batch.py      Batch Processing entry
├── main_tracking.py   Tracking & Comparison entry
├── main_editor.py     Mask Editor entry
├── main_training.py   Model Training entry
├── setup_wizard.py    Environment installer
├── config.py          Shared configuration
├── INTERFACE.md       Module map
├── INSTALLATION.md    Setup guide
└── README.md          User-facing documentation
```

## Environments

```bash
conda activate cellpose    # cellpose 3.1.1.1, torch 2.7 MPS, CP3 models
conda activate cellpose4   # cellpose 4.1.1, cpsam ViT backbone
```
