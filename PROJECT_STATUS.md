# CellScope — Project Status

*Last updated: 2026-04-30*

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

### DIC (Jesse's VAMPIRE / our keratinocytes)
```
cpsam_dic (CP4 ViT, via cellpose4 subprocess) → DeepSea union → fallback
```
- **0.795 IoU** on our 526² in-domain GT (cpsam_dic, current best)
- **0.697 IoU** on VAMPIRE held-out OOD crops (cpsam_dic, +0.42 over cellpose_dic_v3)
- +280% over default cpsam on DIC (0.697 vs 0.183)
- Modality auto-detection routes to correct pipeline. CP3 fine-tunes
  (cellpose_dic_v3, etc.) remain available as faster fallbacks.

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
| **cpsam_dic** ★ | ViT (CP4) | 1,000 DIC pairs (Colab fine-tune of base cpsam) | **DIC, current best** (0.795 in-domain, 0.697 OOD) |
| cpsam (default) | ViT (CP4) | General microscopy | Phase-contrast |
| cellpose_dic_v3 | CNN (CP3) | 2,644 standardized 448px crops | Faster DIC alternative |
| cellpose_dic_v2 | CNN (CP3) | 2,812 DIC pairs (VAMPIRE+GT+CTC) | Legacy DIC |
| cellpose_dic | CNN (CP3) | Our DIC keratinocytes | Original DIC fine-tune |
| cellpose_combined_robust | CNN (CP3) | 5,826 augmented pairs | Noisy / perturbed recordings |
| DeepSea | U-Net | Brightfield/phase-contrast | Boundary refinement |
| MedSAM | SAM-ViT | Biomedical images | Foundation model refinement |

---

## Benchmark Results

### Phase-Contrast Detection (Ignasi GT, 65 frames)
| Method | Mean IoU | >0.85 | Min |
|--------|----------|-------|-----|
| **cpsam + DeepSea** | **0.932** | 65/65 | 0.867 |
| cellpose + MedSAM + DeepSea | 0.907 | 55/65 | 0.629 |
| cpsam alone | 0.915 | 56/65 | — |

### DIC Detection — head-to-head (90 stratified frames per test)
| Test set | cellpose_dic_v3 | **cpsam_dic** | Δ |
|---|---:|---:|---:|
| our-GT 526² in-domain | 0.740 | **0.795** | **+0.055** |
| VAMPIRE held-out OOD crops | 0.279 | **0.697** | **+0.418** |
| Detection rate (our-GT) | 95% | **100%** | **+5pp** |

### VAMPIRE held-out per-genotype (cpsam_dic)
| Genotype | IoU | std |
|---|---:|---:|
| control | 0.897 | 0.083 |
| cKO | 0.490 | 0.390 |
| GoF | 0.703 | 0.150 |
| **mean** | **0.697** | 0.297 |

### Older DIC benchmarks (legacy CP3 models, kept for reference)
| Method | control | cKO | GoF | Mean |
|--------|---------|-----|-----|------|
| cellpose_dic_v2 + preproc | 0.874 | 0.503 | 0.159 | 0.512 |
| cellpose_combined_robust | 0.652 | 0.352 | 0.434 | 0.479 |
| cellpose_dic + preproc | 0.493 | 0.456 | 0.191 | 0.380 |
| cpsam (no fine-tune) + DeepSea | — | — | — | 0.183 |

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

- **GoF cKO heterogeneity**: cpsam_dic cKO std is 0.39 (vs 0.08 for
  control) — most cKO frames hit IoU 0.85+, but a tail of frames with
  thin filopodia or unusual morphology score below 0.4. Manual editing
  recovers these.
- **Multi-cell GT**: Limited ground truth for multi-cell evaluation
  beyond CTC DIC-C2DH-HeLa.
- **ID switching**: Touching cells still cause occasional identity
  swaps in the Hungarian tracker.
- **Brightness shifts**: All detectors lose ~30% IoU on `bright_dark` /
  `bright_bright` perturbations (would need brightness-augmented
  retraining to fix).

---

## In Progress

1. **Resume cpsam_dic fine-tune** — current model is from a partial
   ~6-epoch run that timed out on Colab. Resuming notebook is at
   `notebooks/resume_cpsam_dic_colab.ipynb`; expect another +0.02 IoU
   when the full 20-epoch run completes.
2. **Distribution polish** — install scripts, model bundles, and Drive
   downloader are wired (`install.{bat,sh}`, `download_models.py`,
   `make_models_bundle.py`, `make_dist.py`). Doc updates in flight.

## Planned Next Steps

- Multi-cell ground truth expansion (broader CTC test or new annotated
  Jesse recordings).
- Boundary separation for touching cells in dense recordings.
- Optional brightness-shift augmentation in next cpsam_dic retrain.
- GitHub release once distribution dry-run is complete.

---

## File Organization

```
cellscope/
├── install.bat / install.sh         Cross-platform installer
├── environment.yml                  cellpose env spec
├── environment-cellpose4.yml        cellpose4 env spec
├── download_models.py               Drive fetcher (cpsam_dic + bundle)
├── make_models_bundle.py            Maintainer: build models bundle
├── make_dist.py                     Maintainer: build dist zip
│
├── main_suite.py                    Unified launcher
├── main_focused.py                  Detection & Analysis
├── main_batch.py                    Batch Processing
├── main_tracking.py                 Tracking & Comparison
├── main_editor.py                   Mask Editor
├── main_training.py                 Model Training
│
├── core/                            34 analysis modules
├── gui/                             Shared GUI components
├── gui_focused/                     Detection & Analysis (12 files)
├── gui_batch/                       Batch Processing (3 files)
├── gui_tracking/                    Tracking & Comparison (6 files)
├── gui_editor/                      Mask Editor (3 files)
├── gui_training/                    Model Training (4 files)
├── output/                          Result writers
├── scripts/                         Training + evaluation + bench
│   └── _paths.py                    Project-root + benchmark-data helpers
├── notebooks/                       Colab training + resume notebooks
├── docs/                            Manual + recommendations + plans
├── data/
│   ├── models/                      cpsam_dic + CP3 fine-tunes + DeepSea
│   └── training/                    dic_splits_v3 (448px standardized)
├── results/                         Benchmark outputs
├── INTERFACE.md                     Module map
├── INSTALLATION.md                  Setup guide
├── PROJECT_STATUS.md                This file
└── README.md                        User-facing documentation
```

## Environments

Created automatically by `install.{bat,sh}` from `environment.yml` and
`environment-cellpose4.yml`:

```bash
conda activate cellpose    # cellpose 3.1.1.1, GUI + analysis pipeline + CP3 models
                            # ALWAYS launch the suite from this env.
conda activate cellpose4   # cellpose 4.1.1, cpsam ViT.
                            # Invoked automatically via subprocess
                            # whenever the pipeline needs cpsam_dic.
```
