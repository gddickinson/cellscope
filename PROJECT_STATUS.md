# CellScope — Project Status

*Last updated: 2026-05-27*

![CellScope pipeline](docs/figures/hero.png)

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
- **17 GUI-tunable pipeline parameters** threaded end-to-end through
  `core/unified_detection.detect_recording` (persistence_guard v2
  sub-params, mirror padding, SAM2 video gap-fill, max_gap_frames,
  min_track_length, DIC preprocess/retry, Cy5 fusion sub-thresholds,
  and the 6 previously-broken detection toggles)
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
- **0.826 IoU** on our 526² in-domain GT (cpsam_dic v2, current best)
- **0.754 IoU** on VAMPIRE held-out OOD crops (cpsam_dic v2, +0.475 over cellpose_dic_v3)
- +312% over default cpsam on DIC (0.754 vs 0.183)
- Modality auto-detection routes to correct pipeline. CP3 fine-tunes
  (cellpose_dic_v3, etc.) remain available as faster fallbacks.

### Multi-Cell
```
cpsam/cellpose → debris filter → DeepSea per-cell → Hungarian tracker
→ gap fill → post-process → division annotation (lineage)
```
- Division detection (mother→daughter tracking) — biology-aware
  annotator (`core/division_annotator.py`) sets `parent_id` +
  `division_score` + `division_frame` on daughter tracks; writes
  `divisions.json` sidecar next to `masks.npz`. **2 / 2 GT divisions
  caught across 9 GT recordings with 0 false positives** (Pos39_OT
  + Pos51_Y1).
  Multi-track daughter detection (2026-05-20): replaced the literal
  "track first-spawned this frame" check with "track first comes
  near the parent's split centroid this frame, having not been near
  it before". Handles Hungarian-tracked daughters that inherit an
  existing track ID at the split — a common pattern under raw cpsam
  multi-cell output that the legacy first-frame-only check missed.
  Restored Pos51_Y1's GT-evident division catch (lost after the
  cpsam_dic → cpsam routing change earlier the same day).
  Two false-positive classes filtered same day:
    * `MIN_PARENT_FRAMES_BEFORE_PEAK = 5` — kills early-recording
      tracker transients (caught Pos20_KO F3 FP: parent peak at F0
      with 0 prior frames)
    * `MIN_BASELINE_AREA_PX = 500` — kills noise-blob swellings
      where baseline ≈ 0 makes swelling ratio explode (caught
      Pos68_DMSO T11→T10 FP with baseline=20 px, swelling=48×)
  Filter relaxation 2026-05-17: pre-mitotic-balled check spans the
  [pre, post]-split window (PRE_STATE_LOOKBACK=3 + POST_STATE_WINDOW=3)
  and daughter persistence tolerates a 2-frame gap during the contact
  phase (`MAX_DAUGHTER_GAP_FRAMES=2`). Score-zero noise (mass-ratio
  violations zeroing the composite) filtered via
  `MIN_SCORE_TO_REPORT = 0.05`.
- 100% gap fill rate (41/41 gaps on tested recordings)
- TRA 0.929 on CTC DIC-C2DH-HeLa benchmark

---

## Models

| Model | Type | Trained On | Best For |
|-------|------|-----------|----------|
| **cpsam_dic** ★ | ViT (CP4) | 1,000 DIC pairs (Colab fine-tune of base cpsam) | **DIC, current best** (0.826 in-domain, 0.754 OOD; cpsam_dic v2) |
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
| Test set | cellpose_dic_v3 | cpsam_dic v1 | **cpsam_dic v2** | v2 vs v3 |
|---|---:|---:|---:|---:|
| In-domain 526² DIC GT | 0.740 | 0.795 | **0.826** | **+0.086** |
| Out-of-domain DIC crops | 0.279 | 0.697 | **0.754** | **+0.475** |
| Detection rate (in-domain) | 95% | 100% | **100%** | +5pp |
| Detection rate (OOD) | 42% | 88% | **96%** | +54pp |

### Out-of-domain per-genotype (cpsam_dic v2)
| Genotype | IoU | std |
|---|---:|---:|
| control | 0.871 | 0.105 |
| cKO | 0.678 | 0.291 |
| GoF | 0.713 | 0.159 |
| **mean** | **0.754** | 0.218 |

cpsam_dic (current) is the result of resuming Colab fine-tuning to
~20 epochs from a partial 6-epoch checkpoint. The Drive download URL
in `download_models.py` serves this same v2 file.

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
- **🔬 Test on frame** — preview detection on the currently displayed
  frame with current GUI settings; status bar reports cell count +
  runtime + density-aware extrapolation to full-recording runtime
  (sparse 1.5× / medium 2.0× / dense 2.5× post-proc multiplier).
  Single fastest way to tune parameters interactively.
- **17 GUI-tunable pipeline parameters**, grouped: Detection (model,
  min_area, expected_cells, search_radius, min_track_length, ROI),
  Refinement (DeepSea, TTA, cpsam-on-Cy5 union, fallback, mirror
  padding), Gap fill (toggle, SAM2 video sub-toggle, max_gap_frames),
  Cy5 multichannel (fusion, recovery, filter mode + 3
  persistence_guard sub-spinboxes), Tiling, DIC pipeline (preprocess,
  retry, 3 Cy5 fusion sub-thresholds). Default Cy5 filter:
  **persistence_guard** (see below).
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
- **Multi-cell tracking**: TRA 0.929 on CTC benchmark; biology-aware division annotator (`core/division_annotator.py`) detects pre-mitotic-swelling → balled → split → grown-daughter pattern, **catching 2 / 2 GT divisions across 13 IC295 + ignasi GT recordings** (Pos51_Y1, Pos39_OT). Pos39_OT initially regressed after mirror-pad rollout because padding caused cpsam to merge its dividing pair; restored via per-recording sidecar override `"use_mirror_pad": "off"` in the recording's `.ome.json`, honored by `scripts/run_pipeline_on_gt_recording.py`. Pad-off on Pos39_OT is strictly better (+0.005 F1, +4pp ID, division catch restored) at -0.008 IoU.
- **Cy5 false-positive filter (persistence_guard v2, 2026-05-25)**: 3-stage rule (mm-pass OR long-AND-moving) replaced the original strict `multi_metric` filter. +0.038 F1_focused aggregate across the 13-recording GT corpus. Big wins on weak-Cy5 conditions (GOF/OT/DMSO) where the original filter was dropping persistent real cells; persistent vignette/debris phantoms still killed by the static-track gate. **No per-recording overrides needed** — the global rule (vel<3 AND median consec IoU>0.85) cleanly separates phantoms from real cells via motion + shape stability. All sub-thresholds GUI-tunable.
- **17 GUI-tunable pipeline parameters threaded end-to-end**: every detection toggle and sub-threshold the user can see in the focused GUI now reaches `detect_recording` and the underlying functions. Three latent bugs (5 GUI toggles silently dropped onto hardcoded `DEFAULTS`, `min_track_length` hardcoded in 2 places, `postprocess_tracks` `min_frames=3` hardcoded) were uncovered + fixed during real-recording GUI driving for the 🔬 Test on frame feature.
- **🔬 Test on frame**: one-click detection preview on the currently displayed frame with current GUI parameters; status bar reports cell count, runtime, and density-aware full-recording extrapolation. Validated on both sparse (Pos10_WT, 3 cells in 28s) and dense (Pos68_DMSO, 14 cells in 40s) recordings.
- **GT aggregate**: 13 recordings, mean per-cell IoU **~0.83**, raw F1@.5 **0.81**, GT-focused F1@.5 **0.87**, ID consistency **94.55%** (278 annotated frames). Numbers refreshed 2026-05-25 after the persistence_guard v2 filter rollout. Aggregate F1_focused: **0.836 → 0.874 (+0.038)** vs the previous strict `multi_metric` filter, with no per-recording overrides needed. Biggest wins on the previously worst recordings: Pos31_GOF F1_focused 0.52 → **0.84**, Pos68_DMSO 0.58 → **0.75**, Pos20_KO 0.90 → **0.97**. Pos51_Y1 fully recovered to 1.00 (the static-track gate kills its persistent vignette phantom). One stubborn under-detection case remains — Pos21_KO (F1 0.66, IoU 0.677) — where cpsam still misses ~half the cells; next investigation is click-prompted SAM seeding.
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

1. **Pos21_KO under-detection** — only remaining stubborn case in the
   13-recording GT corpus (F1 0.66, IoU 0.677). Raw cpsam still misses
   roughly half the cells per frame on this Y-27632 / cKO field. Next:
   click-prompted SAM seeding + Cy5-peak auto-prompts.
2. **cpsam_dic v3 fine-tune** — current cpsam_dic v2 (0.826 in-domain,
   0.754 OOD) is from a 20-epoch resume. Further gains possible via
   harder-mining + augmentation expansion. Resuming notebook at
   `notebooks/resume_cpsam_dic_colab.ipynb`.
3. **Distribution polish** — install scripts, model bundles, and Drive
   downloader are wired (`install.{bat,sh}`, `download_models.py`,
   `make_models_bundle.py`, `make_dist.py`). Doc updates landed
   2026-05-25.

## Recently Completed

- **HTTP remote-control RPC on all 6 GUIs** (2026-05-27) — `CELLSCOPE_REMOTE=<port>`
  exposes a stdlib HTTP server inside the Qt event loop. Focused GUI gets
  the full 16-endpoint handler set (load_recording / detect / test_frame /
  analyze / export / etc.); batch / editor / training / tracking get
  `attach_minimal` (status + log). Suite (tkinter) runs the server in a
  daemon thread. Cross-thread dispatch via `pyqtSignal(dict, object)`.
  Used to drive the deployment-readiness systems test.
- **Hungarian tracker: IoU + area cost** (2026-05-27) — cost matrix
  augmented with mask overlap and area-difference terms. `track_w_dist`
  / `track_w_iou` / `track_w_area` plumbed through DEFAULTS. **Pos7-WT
  GT ID consistency 0.88 → 0.97** without affecting DET / SEG / IoU.
- **Mini batch portability fixes** (2026-05-27) — 9 bugs surfaced by the
  IC295 batch run on the Mac mini: `parse_n_channels` silently defaulting
  to 1 channel without `_metadata.txt` sidecar (now falls back to
  `.ome.json` or OME-XML SizeC); `save_unfiltered_detections` shape
  mismatch when downsample > 1 (root fix: iterate `tracks_raw + tracks`
  with id() dedupe); two-cell fusion artifact from Phase-4 mask
  propagation collision; auto-downsample threshold 900 → 1100 so 1024²
  stays at 1×; `merge_touching_splits` for cpsam over-segmentation;
  `reject_static_edge_blob_tracks` for vignette artefacts;
  `main_suite._find_conda` walks install roots and prefers one with
  `envs/cellpose4`; cpsam-pair subprocess timeout 600 → 1800s.
- **Persistence_guard v2 Cy5 filter** (2026-05-25) — replaced the
  strict `multi_metric` default after a 13-GT audit showed it was
  dropping 70 of 989 real cells. Validation: +0.038 F1_focused
  aggregate, +0.196 to +0.333 on the worst recordings (Pos68_DMSO,
  Pos31_GOF). Static-track gate (vel < 3 AND consec IoU > 0.85)
  cleanly separates phantoms from real cells without per-recording
  overrides.
- **17 GUI-tunable pipeline parameters** (2026-05-25) — full plumbing
  audit + the 🔬 Test on frame feature. Three latent GUI-bypassing
  bugs uncovered + fixed.
- **Mirror padding** (2026-05-21) — `use_mirror_pad="auto"` lifts
  aggregate IoU 0.847 → 0.858 on IC295.
- **IC295 multichannel pipeline + 13-recording GT** — Cy5 fusion +
  recovery + filter shipped; aggregate F1_focused 0.874.
- **Multi-track daughter detection** (2026-05-20) in the division
  annotator — catches all 2 GT divisions across the corpus.
- **IC293 phase-contrast** — 16 full-frame recordings × 5 conditions
  processed; results in `results/ignasi_new_full/`.

## Planned Next Steps

- Click-prompted SAM seeding for Pos21_KO-style under-detection.
- Phase 5 of multichannel (speculative): subcellular features
  (lamellipodia polarity, cortex ratio, stress-fibre density) — most
  likely to give a strong Piezo1 phenotype signal.
- Multi-cell ground truth expansion (broader CTC test or new annotated
  Jesse recordings).
- Boundary separation for touching cells in dense recordings.
- Brightness-augmented retrain (3.5 in roadmap; setup done).
- GitHub release once distribution dry-run is complete.

---

## GUI test coverage

**107/107 checks pass across 7 phases (A–G) covering 6 GUIs** —
see `results/comprehensive_gui_tests/FINAL_REPORT.md` and 63
screenshots in `results/comprehensive_gui_tests/screenshots/`.

| Phase | GUI(s) | Checks | Coverage |
|---|---|---:|---|
| A | Detection & Analysis (single-cell) | 59 | load → detect → analyze → 16 graph types → export, B/C, zoom, pan, frame nav |
| B | Detection & Analysis (multi-cell) | 8 | mode switch, multi detection, per-cell analytics, all 20 graphs, cell selector |
| C | ROI + Mask Editor integration | 9 | draw / persist / apply / clear ROI, mask editor open + send-to-GUI roundtrip |
| D | Batch GUI | 6 | directory scan, recording tree, settings widgets, params dict |
| E | Tracking GUI | 7 | load masks, Hungarian tracking, per-track analysis, track table, plots |
| F | Training + Mask Editor (standalone) | 7 | launch, scan data dir, dock panel |
| G | Parameter flow | 11 | params plumb through to detect dict; scale overrides; toggle behaviour |

Run via:
```bash
conda run -n cellpose4 python scripts/test_focused_gui.py       # Phase A
conda run -n cellpose4 python scripts/test_comprehensive_gui.py # Phases B-G
python scripts/aggregate_comprehensive_report.py                 # merge
```

All tests run headless via `QT_QPA_PLATFORM=offscreen`.

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
