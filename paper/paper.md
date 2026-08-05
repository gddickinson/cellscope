---
title: 'CellScope: an end-to-end pipeline for cell detection, tracking, and morphometrics in label-free time-lapse microscopy'
tags:
  - Python
  - microscopy
  - bioimage analysis
  - cell tracking
  - cell segmentation
  - cell migration
authors:
  - name: George D. Dickinson
    orcid: 0000-0001-7711-0388
    affiliation: 1
affiliations:
  - name: Department of Neurobiology and Behavior, University of California, Irvine, USA
    index: 1
date: 5 August 2026
bibliography: paper.bib
---

# Summary

`CellScope` is a desktop application for quantifying cell behaviour in
label-free time-lapse microscopy. It takes a differential interference contrast
(DIC), phase-contrast, or multi-channel recording and produces tracked cell
outlines together with per-cell migration, morphology, and edge-dynamics
measurements, group-level statistical comparisons, and publication-ready
figures. The whole workflow — detection, tracking, curation, analysis, and
comparison — is driven from five task-specific graphical interfaces, so that
bench scientists can run it without writing code.

The pipeline combines a Cellpose-SAM segmentation backbone [@stringer2021;
@pachitariu2025] with a Hungarian-assignment tracker, a four-phase cascade for
filling gaps in tracks that includes SAM 2 video propagation [@ravi2024], and
optional refinement from DeepSea [@zargari2023] and MedSAM [@ma2024]. Shape
heterogeneity is quantified using VAMPIRE-style principal-component eigenshapes
[@phillip2021]. Every analysis run writes a `RUN_METADATA` record capturing the
source checksum, every parameter, environment versions, the git commit, and the
command needed to reproduce it.

# Statement of need

Fluorescent labelling perturbs the cells it measures. Studies of mechanically
sensitive processes — migration, spreading, mechanotransduction — therefore
often prefer label-free imaging, where cells are visible only through optical
path-length differences. That choice moves the difficulty into the analysis:
DIC and phase-contrast images have low, non-uniform contrast, cells routinely
touch and overlap, and mitotic rounding changes appearance so drastically that
identity is easily lost.

Excellent tools exist for parts of this problem. Cellpose [@stringer2021] and
its transformer successor [@pachitariu2025] segment single frames well;
TrackMate [@ershov2022] and btrack [@ulicna2021] link detections into
trajectories; CellProfiler [@stirling2021] provides batch measurement.
Assembling them into a reproducible study nonetheless leaves the biologist
responsible for several judgement calls that materially change the answer:
which segmentation model suits this recording, how to recover cells the
detector missed in some frames, how to keep an identity through mitotic
rounding, and — for multi-channel data — how to reject plausible-looking
detections that are optical artefacts rather than cells.

`CellScope` addresses that assembly problem directly, and makes the judgement
calls explicit, measured, and adjustable:

- **Automatic backbone selection.** Eleven frames are sampled and cells
  counted; sparse fields (75th-percentile count < 1.5 cells/frame) use a
  DIC-fine-tuned model that gives tighter boundaries, while crowded fields use
  the raw ViT model, which handles touching cells better.
- **Gap filling as a cascade.** Detection failures are repaired in four
  escalating phases — test-time augmentation, a CP3/MedSAM/DeepSea fallback,
  SAM 2 video propagation, and finally translation-only interpolation —
  achieving a 100% fill rate (41/41 gaps) on the tested recordings.
- **Artefact rejection for multi-channel data.** A `persistence_guard` filter
  distinguishes real cells from vignetting and debris using lifetime, motion,
  and frame-to-frame mask stability, rather than fluorescence intensity alone.
- **State-aware metrics.** Each cell-frame is classified as rounded,
  spread, or transitional, so that motility statistics are not confounded by
  the changing proportion of dividing cells in a population.

The tracker was compared against a transformer-based tracker
[@gallusser2024] on the Cell Tracking Challenge `DIC-C2DH-HeLa` benchmark
[@maska2023]. Combining centroid distance with mask intersection-over-union and
area difference in the assignment cost gives TRA 0.929 against 0.847, with DET
and SEG unchanged (0.936 and 0.860); on internal ground truth, track identity
consistency rose from 0.88 to 0.97 relative to a distance-only cost. Segmentation
accuracy on a 65-frame phase-contrast ground-truth set is a mean IoU of 0.932,
with every frame above 0.85.

`CellScope` was developed for a study of PIEZO1-dependent keratinocyte
migration and has been used to analyse DIC, phase-contrast, and DIC+actin
recordings of human induced pluripotent stem cell-derived cells
[@bertaccini2025]. It is not specific to that system: any label-free or
mixed-modality time-lapse in which cells must be segmented, tracked, and
compared across treatment groups is in scope. A companion package,
`cellscope_analysis`, provides a GUI-free API over the same results for
scripted analysis.

# Acknowledgements

CellScope builds on Cellpose, Cellpose-SAM, DeepSea, MedSAM, SAM 2, Trackastra,
and the VAMPIRE shape-analysis method, and I thank their authors for releasing
them openly. I thank Medha M. Pathak and Ian Parker for access to the
recordings used in development and validation.

Portions of the GUI scaffolding, pipeline integration, test harness, and
documentation were developed with assistance from Anthropic's Claude. All code
was reviewed and validated against ground-truth recordings before being
committed; design decisions, benchmark interpretation, and the biological
questions remain the author's responsibility.

# References
