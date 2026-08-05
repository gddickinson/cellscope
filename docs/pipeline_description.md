# CellScope — Pipeline Description

## Overview

CellScope is an automated analysis pipeline for DIC and phase-contrast
time-lapse microscopy of migrating cells. It combines deep learning-based
cell detection (Cellpose-SAM), foundation model refinement (DeepSea),
and classical tracking algorithms to produce per-cell timeseries of
migration, morphology, and edge dynamics.

## Cell Detection

### Primary detector: Cellpose-SAM (cpsam) — auto-routed

CellScope uses Cellpose-SAM (cpsam), a Cellpose variant that replaces
the CNN backbone with a Vision Transformer (ViT) image encoder from
the Segment Anything Model (SAM). The ViT encoder, pre-trained on a
large corpus of natural and microscopy images, provides strong
generalization to diverse cell types and imaging modalities without
per-experiment fine-tuning.

**Auto-select between raw cpsam and cpsam_dic**: a probe samples
the first 11 frames at the chosen detection resolution, counts cells
per frame, and uses the 75th-percentile count:

- < 1.5 cells/frame → **cpsam_dic** (DIC fine-tune, single-cell-
  biased — produces tighter boundaries on isolated cells)
- ≥ 1.5 cells/frame → **raw cpsam** (multi-cell ViT default —
  better at touching cells; cpsam_dic merges them)

Detection runs at default parameters (no diameter hint, no threshold
tuning) on each frame independently. The cpsam input is mirror-padded
by 50 px when detection-min-dim ≥ 1024 (auto-enabled) to handle cells
at the FoV edge.

### DeepSea union refinement

The raw cpsam masks undergo a union operation with predictions from
DeepSea (Zargari et al., 2023), a segmentation model specifically
trained on phase-contrast and brightfield time-lapse data. For each
frame:

1. DeepSea predicts a binary cell mask from the grayscale image
2. The cpsam mask and DeepSea mask are combined via pixel-wise OR
3. Binary hole-filling removes internal gaps
4. The largest connected component is retained (removing debris)

This union recovers boundary pixels that cpsam under-segments
(particularly thin filopodia) while the largest-CC step provides
automatic debris filtering.

### Fallback detection

For frames where cpsam returns insufficient cell area (below a
configurable threshold, default `min_area_px=200` from DEFAULTS),
the pipeline falls back to a secondary detector:

1. Cellpose (CNN backbone, cellpose 3.x) with tuned parameters
   (cellprob_threshold=-2.0, flow_threshold=0.0)
2. MedSAM bbox-refinement of the cellpose mask
3. DeepSea union (same as above)

This cascade runs via subprocess in a separate conda environment
(cellpose 3.x) to maintain compatibility with legacy CNN-trained
models, while the primary cpsam detector requires cellpose 4.x.

## Multi-Cell Tracking

### Hungarian assignment

For recordings containing multiple cells, CellScope extracts
per-frame instance labels from cpsam and assigns cross-frame
identity using the Hungarian algorithm (scipy linear_sum_assignment):

- Cost matrix: Euclidean distance between centroids
- Maximum allowed hop: 150 px × gap length (scales for missing frames)
- Gap tolerance: tracks survive up to `max_gap_frames` (default 15)
  consecutive unmatched frames — covers ~10 frames of biological
  transitions at 10-min/frame intervals
- New tracks spawn for unmatched detections (cells entering the
  field of view)
- Tracks shorter than `min_track_length` (default 3) are dropped

### Division detection

Cell division events are identified by a heuristic: when a new track
appears within the maximum hop distance of an existing track, and the
new cell's area is 20-90% of the parent's last measured area, the new
track is tagged as a daughter with a parent_id link.

### Gap filling

After tracking, internal gaps (frames where a tracked cell was
transiently undetected) are filled by a four-phase cascade:

1. **Phase 1** — re-run cpsam with test-time augmentation
   (`augment=True`, 4 rotations) on the gap frame, select detection
   nearest the interpolated centroid
2. **Phase 2** — if Phase 1 fails, fall back to cellpose
   + MedSAM + DeepSea (CP3 subprocess)
3. **Phase 3** — SAM2 video propagation from a flanking frame
   (memory attention follows cells through transitions where
   cpsam/cellpose can't detect them). Toggleable via
   `use_sam2_video_gap_fill` (default on; +1s per gap frame)
4. **Phase 4** — translation-only fill from the nearest flanking
   detection as a last resort

Tested fill rate: 100% on production recordings (32/32 gaps on
Pos3-WT, 9/9 on Pos2-WT, 41/41 across all GT recordings).

### Cy5 false-positive filter (multichannel only)

When a Cy5 fluorescence channel is present, the post-tracking
**persistence_guard** filter (default) removes false-positive
tracks using a three-stage rule:

1. **Pass** if the track passes ≥2 of 3 Cy5 cellularity metrics
   (score, inside/outside ratio, fraction positive)
2. **Drop** if short (lifetime < 35 frames) AND not passing the
   metrics — transient detections without Cy5 evidence
3. For long tracks that fail the Cy5 metrics, **drop only if
   STATIC** (mean per-frame centroid displacement < 3 px AND
   median consecutive-frame mask IoU > 0.85) — phantom signature:
   vignette artefacts and stuck debris are static AND shape-stable

Calibrated on a 13-recording GT corpus (989 GT cells). Achieves
F1_focused = 0.874 vs 0.836 for the original strict multi_metric
filter. All thresholds GUI-tunable.

## Per-Cell Analysis

### Migration

- **Instantaneous speed**: frame-to-frame centroid displacement
  divided by time interval (μm/min)
- **Total distance**: cumulative path length
- **Net displacement**: Euclidean distance from start to end
- **Persistence ratio**: net displacement / total distance (0-1;
  1 = perfectly straight)
- **Mean squared displacement (MSD)**: ensemble-averaged over
  overlapping windows, with standard error
- **Direction autocorrelation**: cosine similarity of displacement
  vectors at increasing lag (DiPer method)

### Morphology (per frame)

- **Area** (μm²): number of mask pixels × pixel area
- **Perimeter** (μm): contour length via scikit-image regionprops
- **Circularity**: 4π × area / perimeter² (1 = perfect circle)
- **Solidity**: area / convex hull area (measures concavity)
- **Aspect ratio**: major axis / minor axis of fitted ellipse
- **Eccentricity**: eccentricity of fitted ellipse (0 = circle,
  1 = line)

### Edge dynamics

Cell boundaries are represented in polar coordinates centered on
the centroid. The boundary is divided into angular sectors (default
16) and radial displacement between consecutive frames gives:

- **Edge velocity** per sector per frame (μm/min)
- **Protrusion velocity**: mean of positive (outward) velocities
- **Retraction velocity**: mean of negative (inward) velocities
- **Protrusion fraction**: fraction of edge extending outward
- **Kymograph**: (time × angle) heatmap of edge velocity

### VAMPIRE shape mode analysis

CellScope optionally performs VAMPIRE shape mode analysis (Phillip et
al., Nature Protocols 2021) to quantify morphological heterogeneity:

1. **Contour extraction**: the cell boundary is extracted from each
   frame's binary mask
2. **Registration**: each contour is resampled to 50 equidistant
   points, centroid-normalized, and rotationally aligned
3. **PCA decomposition**: principal component analysis on all
   registered contours yields eigenshapes -- the dominant axes of
   shape variation across the timeseries
4. **K-means clustering**: contours are assigned to K discrete shape
   modes (default K=4), each representing a morphological phenotype
5. **Shannon entropy**: the entropy of the mode frequency
   distribution quantifies heterogeneity. Higher entropy indicates
   more variable shape behavior; lower entropy indicates a cell
   that maintains a consistent morphology

This analysis produces four additional graph types: Shape Modes
(PCA scatter colored by cluster), Mode Distribution (histogram),
Mode Over Time (per-frame mode assignment), and Eigenshape
Variations (mean contour +/- principal components).

The `shape_entropy` metric is included in batch summary CSVs and
available as a comparison metric in the Tracking GUI.

## Statistical Comparison

For batch analysis, recordings are grouped by parent folder name
(corresponding to experimental conditions). CellScope computes
inter-group statistics:

- **2 groups**: Welch's t-test, Mann-Whitney U, Cohen's d effect size
- **3+ groups**: one-way ANOVA, Kruskal-Wallis, pairwise t-tests
  with Bonferroni correction

Results are displayed as box plots with individual data points and
significance brackets (*, **, ***).

## References

1. Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a
   generalist algorithm for cellular segmentation. Nature Methods
   2021;18:100-106.

2. Stringer C, Pachitariu M. Cellpose3: one-click image restoration
   for improved cellular segmentation. Nature Methods 2025.
   doi:10.1038/s41592-025-02595-5

3. Zargari A, et al. DeepSea is an efficient deep-learning model for
   single-cell segmentation and tracking in time-lapse microscopy.
   Cell Reports Methods 2023;3:100500.
   doi:10.1016/j.crmeth.2023.100500

4. Ma J, et al. Segment anything in medical images. Nature
   Communications 2024;15:654.

5. Holt JR, et al. Spatiotemporal dynamics of PIEZO1 localization
   controls keratinocyte migration during wound healing.
   eLife 2021;10:e65415. doi:10.7554/eLife.65415

6. Phillip JM, et al. A robust unsupervised machine-learning method to
   quantify the morphological heterogeneity of cells and nuclei.
   Nature Protocols 2021;16:754-774.
