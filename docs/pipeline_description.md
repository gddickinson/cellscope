# CellScope — Pipeline Description

## Overview

CellScope is an automated analysis pipeline for DIC and phase-contrast
time-lapse microscopy of migrating cells. It combines deep learning-based
cell detection (Cellpose-SAM), foundation model refinement (DeepSea),
and classical tracking algorithms to produce per-cell timeseries of
migration, morphology, and edge dynamics.

## Cell Detection

### Primary detector: Cellpose-SAM (cpsam)

CellScope uses Cellpose-SAM (cpsam), a Cellpose variant that replaces
the CNN backbone with a Vision Transformer (ViT) image encoder from
the Segment Anything Model (SAM). The ViT encoder, pre-trained on a
large corpus of natural and microscopy images, provides strong
generalization to diverse cell types and imaging modalities without
per-experiment fine-tuning.

Detection runs at default parameters (no diameter hint, no threshold
tuning) on each frame independently, producing per-cell instance
segmentation masks.

### DeepSea union refinement

The raw cpsam masks undergo a union operation with predictions from
DeepSea (Zargari et al., 2022), a segmentation model specifically
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
configurable threshold, default 500 px), the pipeline falls back to
a secondary detector:

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
- Gap tolerance: tracks survive up to 10 consecutive unmatched frames
- New tracks spawn for unmatched detections (cells entering the
  field of view)

### Division detection

Cell division events are identified by a heuristic: when a new track
appears within the maximum hop distance of an existing track, and the
new cell's area is 20-90% of the parent's last measured area, the new
track is tagged as a daughter with a parent_id link.

### Gap filling

After tracking, internal gaps (frames where a tracked cell was
transiently undetected) are filled by:

1. Interpolating the expected centroid from flanking frames
2. Re-running cpsam with test-time augmentation (4 rotations)
3. Selecting the detection closest to the expected position
4. If cpsam fails, falling back to the cellpose+MedSAM+DeepSea
   secondary pipeline

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

2. Pachitariu M, Stringer C. Cellpose 3: one-click image restoration
   for improved cellular segmentation. bioRxiv 2024.

3. Zargari A, et al. DeepSea: an efficient deep learning model for
   automated cell segmentation and tracking. Cell Reports Methods
   2022;2:100367.

4. Ma J, et al. Segment anything in medical images. Nature
   Communications 2024;15:654.

5. Holt CE, et al. Piezo1 regulates mechanotransduction in
   keratinocyte migration. eLife 2021.
