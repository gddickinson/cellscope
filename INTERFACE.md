# CellScope — Interface Map

## Entry Points
- **main_suite.py** — Unified launcher (tkinter, runs from `cellpose` env)
- **main_focused.py** — Detection & Analysis GUI
- **main_batch.py** — Batch Processing GUI
- **main_tracking.py** — Tracking & Comparison GUI
- **main_editor.py** — Mask Editor GUI
- **main_training.py** — Model Training GUI

## Install + distribution
- **install.bat / install.sh** — create both conda envs (cellpose, cellpose4)
- **environment.yml** — `cellpose` env spec (CP3 + GUI + analysis pipeline)
- **environment-cellpose4.yml** — `cellpose4` env spec (cpsam ViT)
- **download_models.py** — fetch cpsam_dic + small-models bundle from Drive
- **make_models_bundle.py** — maintainer tool: zip the small models for upload
- **make_dist.py** — maintainer tool: zip the project for sharing
- **setup_wizard.py** — legacy tkinter installer (kept as alternative; `install.{bat,sh}` is now canonical)

## `core/` — Analysis Pipeline (35+ modules)

- **io.py** — `load_video`, `load_recording`, `find_recordings`
- **pipeline.py** — `detect()`, `refine()`, `analyze_recording()`
- **unified_detection.py** — `detect_recording()` — **single canonical detection entry point.** Performs alignment → downsample → threshold conversion → auto-pipeline-select → detect → Cy5 multi-metric filter → upscale labels back. Called from both `gui_focused/workers.py` (mode="auto") and `scripts/run_pipeline_on_gt_recording.py`, guaranteeing the GUI and runner produce identical output.
- **channel_alignment.py** — `align_cy5_to_dic()` measures DIC↔Cy5 offset via cellpose centroid matching (in `cellpose4` subprocess), `apply_offset_to_stack()` shifts via scipy.ndimage. Skips when |offset| < 1px.
- **detection.py** — `detect_cellpose`, `detect_cellpose_labels`, `detect_cellpose_tiled`
- **hybrid_cpsam.py** — `detect_hybrid_cpsam()` — single-cell cpsam + DeepSea + fallback
- **hybrid_cpsam_multi.py** — `detect_hybrid_cpsam_multi()` — multi-cell with tracking
- **hybrid_dic.py** — `detect_hybrid_dic()`, `detect_hybrid_dic_multi()` — DIC pipelines. When the selected DIC model is `cpsam_dic`, runs the cellpose step in the `cellpose4` env via subprocess (`_run_cpsam_dic_subprocess`); otherwise uses CP3 cellpose_dic + preprocessing + DeepSea + threshold retry directly.
- **modality.py** — `detect_modality()` — auto-detect DIC vs phase-contrast from image statistics; `get_pipeline_config()` returns per-modality settings
- **deepsea_multicell.py** — Per-cell DeepSea refinement preserving labels
- **medsam_deepsea_union.py** — MedSAM + DeepSea union (single-cell)
- **medsam_refine.py** — MedSAM bbox-prompt refinement
- **multi_cell.py** — `track_all_cells()` — Hungarian tracker
- **track_gap_fill.py** — Post-tracking gap fill with augmented re-detection
- **division_annotator.py** — post-hoc cell-division event detection. `find_candidates(labels, um_per_px)` scans a labels stack for the biology-aware division signature (pre-mitotic swelling → balled state → mask halves relative to peak → both daughters grow substantial → daughter persists ≥4 consecutive frames). `annotate_track_lineage(tracks, labels, um_per_px)` sets `parent_id`/`division_score`/`division_frame` on daughter tracks in the pipeline's track list. Called automatically by `hybrid_cpsam_multi` and `hybrid_dic` after postprocess; result dict gains `divisions: [...]` and `scripts/run_pipeline_on_gt_recording.py` writes a `divisions.json` sidecar.
- **tracking.py** — Speed, MSD, persistence, direction autocorrelation
- **morphology.py** — Area, perimeter, circularity, solidity, AR, eccentricity
- **edge_dynamics.py** — Edge velocity kymograph, protrusion/retraction
- **evaluation.py** — IoU, boundary confidence, area stability
- **statistics.py** — Group comparison: t-test, Mann-Whitney, ANOVA, Bonferroni
- **contour.py** — Contour extraction, Fourier smoothing, temporal smoothing
- **boundary_rf.py** — Random Forest boundary classifier
- **boundary_crf.py** — Dense CRF post-processing
- **refinement.py** — Full-stack refinement pipeline
- **alt_segmentation.py** — 15 classical segmentation methods
- **hybrid_rf.py** — Hybrid RF strategies
- **sam_refine.py** — SAM/SAM2 refinement
- **cascade_detect.py** — Three-stage cascade detection
- **crop_refine.py** — Global + per-cell cropping
- **preprocess.py** — Background subtraction, high-pass, debris filter
- **checkpoint.py** — Detection save/load
- **auto_params.py** — Automatic parameter selection
- **flow_quality.py** — Optical flow quality
- **membrane_quality.py** — Membrane texture metrics
- **vampire_analysis.py** — VAMPIRE shape mode analysis: contour extraction, PCA eigenshapes, K-means clustering, Shannon entropy heterogeneity (wraps vampire-analysis package)
- **gap_interp.py** — `interpolate_short_gaps()` + `plot_with_gaps()` — linearly fills NaN runs ≤ N frames in analysis timeseries (speed/area/etc.), draws filled samples dotted so they're never confused with measured points. Surfaced in the focused + tracking GUIs as a "Gap fill" combo (off / ≤1 / ≤2 / ≤3 / ≤5).
- **track_quality.py** — `compute_track_quality(track, n_total_frames, analysis_result)` returns a 0-1 composite quality score from frames-present + area stability + total path length. `quality_color(label)` maps "good"/"ok"/"poor" to pale-green/amber/red RGBA. Used by the Tracking GUI to color the track table.

## `gui/` — Shared Components
- **mask_editor.py** — Interactive mask editor (brush/eraser/polygon/fill, multi-cell labels)
- **mask_editor_multicell.py** — Per-cell color helpers, label utilities
- **run_log.py** — RunLogger + RunLogWidget (event logging)
- **workers.py** — DetectWorker, RefineWorker, BatchWorker
- **options/** — Shared parameter panels (params.py, detection_panel.py, refinement_panel.py, analysis_panel.py, presets.py, presets_widget.py, options_panel.py)

## `gui_focused/` — Detection & Analysis GUI
- **main_window.py** — FocusedMainWindow (state machine, ROI, drag-drop)
- **image_viewer.py** — ImageViewer + FrameNavigatorBar (B/C, zoom, pan)
- **pipeline_panel.py** — 5 stage buttons + mode selector
- **params_panel.py** — Context-sensitive parameters (modality selector: Auto/DIC/Phase-contrast)
- **analysis_view.py** — Summary/Graphs/Log tabs
- **analysis_plots.py** — 16 plot functions + GRAPH_REGISTRY (timeseries plots accept `gap_interp_max` kwarg for short-gap interpolation)
- **vampire_plots.py** — 4 VAMPIRE plots (Shape Modes scatter, Mode Distribution histogram, Mode Over Time, Eigenshape variations); split out so analysis_plots stays under the 500-line limit
- **export_dialog.py** — Export configuration dialog. When "Save masks" is ticked, also writes a `divisions.json` sidecar next to `masks.npz` (always present — empty `candidates` list if no divisions detected). Sidecar contains both the raw candidates from `core.division_annotator` and a compact `track_lineage` table mapping daughter-track-index → parent-track-index.
- **workers.py** — FocusedDetectWorker, FocusedAnalyzeWorker. `FocusedAnalyzeWorker` propagates each track's `parent_id`/`division_frame`/`division_score` into the per-cell `track_info` dict so the analysis view can display lineage.
- **roi_selector.py** — Rectangle/ellipse/polygon ROI
- **dialogs.py** — System info, shortcuts, about

## `gui_batch/` — Batch Processing GUI
- **batch_window.py** — Directory scan, recording tree, settings, progress
- **batch_worker.py** — QThread batch processing. Per recording writes the standard `output.results.write_recording_results` outputs (masks.npz, metrics.json, figures) PLUS a `divisions.json` sidecar with the same schema as the focused export.

## `gui_tracking/` — Tracking & Comparison GUI
- **tracking_window.py** — Main window with Single/Batch tabs
- **single_view.py** — Load masks, track, per-track analysis + plots. Track table's "Parent" column displays each daughter track's parent as a 1-based Track ID (with division frame + score in the tooltip).
- **batch_view.py** — Batch analysis + group statistical comparison
- **stats_plots.py** — Box/violin plots with significance brackets
- **batch_worker.py** — Batch tracking worker

## `gui_editor/` — Mask Editor GUI
- **editor_window.py** — Wraps MaskEditor + results dock panel
- **results_panel.py** — Metrics table + plot viewer

## `gui_training/` — Model Training GUI
- **training_window.py** — Data selection, config, live loss curve
- **training_worker.py** — QThread cellpose training
- **data_preview.py** — Thumbnail grid of training pairs

## `output/` — Result Writers
- **results.py** — `write_recording_results()` (masks, metrics, plots)
- **summary.py** — Batch CSV summaries

## `scripts/`
- **_paths.py** — Project-root resolver + `benchmark_data_root()` helper
  (env var `BENCHMARK_DATA_ROOT` overrides the default sibling lookup).
  Imported by every benchmark / training script via the standard preamble:
  ```python
  import sys, os
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  from _paths import setup_imports, benchmark_data_root
  setup_imports()
  ```
- **bench_cpsam_dic.py** — IoU benchmark for any cellpose model on a
  test directory (TIF or PNG with masks).
- **compare_cpsam_dic.py** — diff two bench JSONs side by side.
- **make_overlay_figures.py** — generate inspection overlays across
  recording types.
- **test_missed_cell_recovery.py** — compare default cpsam vs TTA vs
  multi-cell pipeline on missed-cell frames.
- **train_*.py / prepare_*.py** — model training + data prep scripts
  (dev only; require `BENCHMARK_DATA_ROOT` to point at the sibling
  `piezo1_analysis` project).
- **annotate_divisions.py** — runner for the `core.division_annotator`.
  Scans `pipeline_results/masks.npz` for division candidates and
  rejected near-misses across a set of recordings. Writes
  `divisions.json` + 9-frame strip PNGs (parent contour red,
  daughter contour cyan, peak frame and split frame marked) + a
  per-track area-over-time timeseries diagnostic per recording.
  Rejected events get colour-coded headers showing the rejection
  reason (`no_pre_balled`, `parent_not_substantial`,
  `no_nearby_new_track`, `daughter_not_substantial`,
  `daughter_transient`) so the user can audit whether the filters
  are correct. Composite summary at `results/divisions/summary.md`.
- **test_defaults_consistency.py** — Defaults-drift regression
  test. Verifies every GUI's initial widget values match
  `core.pipeline_defaults.DEFAULTS` and `core.cell_state.
  DEFAULT_THRESHOLDS`, and that every worker has `_PD` wired. 28
  checks. Run after editing any param-widget default. Catches the
  May-2026 class of "the same recording analysed via batch vs
  focused gave different debris filtering" bugs.
- **test_focused_gui.py** — Phase A of the GUI test suite: full
  single-cell load → detect → analyze → 16 graph types → export
  flow with screenshots. 59 checks. Runs headless via
  `QT_QPA_PLATFORM=offscreen`.
- **test_comprehensive_gui.py** — Phases B–G of the GUI test suite
  (multi-cell, ROI + mask editor, batch, tracking, training,
  parameter flow). 48 checks across 5 GUIs. Pass `--phase B|C|D|E|F|G`
  to run a single phase, `--phase all` (default) to run them all.
  Per-phase test bodies live in `scripts/comprehensive/`.
- **aggregate_comprehensive_report.py** — merge Phase A + Phases B–G
  reports into `results/comprehensive_gui_tests/FINAL_REPORT.{md,json}`.
- **comprehensive/** — per-phase test modules
  (`phase_b.py` … `phase_g.py`) plus `_common.py` with shared
  paths and helpers. Each phase imports `gui_focused` / `gui_batch`
  / `gui_tracking` / `gui_editor` / `gui_training` directly and
  drives the same widget paths the user clicks.

## `docs/`
- **user_manual.md** — How to use the GUIs (load → detect → edit → analyse → export).
- **recording_recommendations.md** — Per-modality / per-recording-type best practices.
- **pipeline_description.md** — Pipeline internals.
- **IMPROVEMENTS.md** — Research roadmap.

## `notebooks/` (maintainer only)
- **train_cpsam_dic_colab.ipynb** — Colab notebook for fine-tuning cpsam on DIC.
- **resume_cpsam_dic_colab.ipynb** — Resume training from a partial checkpoint.
- **train_cpsam_dic_v4_brightness_colab.ipynb** — Colab notebook for the brightness-augmented retrain (Phase 3.5). See `docs/brightness_robustness.md`.

## Brightness robustness (3.5)
- **scripts/build_brightness_test.py** — generates 8-perturbation test set from `dic_splits_v3/test`.
- **scripts/augment_brightness.py** — generates v4 brightness-augmented training set.
- **scripts/bench_brightness.py** — orchestrates `bench_cpsam_dic.py` across all perturbations + retention metric + `--compare` mode.

## Multi-channel (DIC + SiR-actin Cy5) pipeline — IC295 dataset
Source: `/Volumes/GeorgeDrive/ignasi/IC295/` (19 recordings × 1.6 GB,
2048×2048 × 97 frames × 3 channels: Cy5/SiR-actin, DIC 10x, None).
Conditions: WT (4), KO (3), GOF (3), Y1 (3), DMSO (3), OT (3).

The Cy5 channel is **SiR-actin** (fluorogenic F-actin probe, only
bright when bound to polymerised F-actin). User-confirmed:
no fully un-stained cells, large intensity range, occasional
non-cell fluorescence artefacts.

- **core/multichannel.py** — channel-aware loader (`load_recording_multi`),
  per-channel preprocessing (`to_uint8_dic`, `to_uint8_fluorescence`),
  Cy5 presence scoring (`cy5_presence_score`: z-score of inside-mask
  p75 vs local 30-px annulus median, robust MAD denominator → 0-1),
  AND-filter (`filter_dic_labels_by_cy5`), per-cell Cy5 features.
- **core/cy5_fallbacks.py** — three tiers of fail-safe recovery for
  cells DIC misses. Tier 2 (`recover_missed_cells_via_dic_crop`):
  Cy5+ regions without DIC masks → crop + re-run cpsam(DIC, TTA
  optional). Tier 3 (`cy5_gap_fill_for_track`): track has gap at
  frame N + Cy5 signal at interpolated centroid → fill via cpsam(Cy5
  crop) [default], `cy5_cleanup` (threshold + morphological), or
  `cy5_threshold` (raw threshold). cpsam-based strategies use cell-
  shape prior so fluorescence artefacts are rejected.
- **core/hybrid_multichannel.py** — orchestrator: wraps
  `hybrid_cpsam_multi` on DIC, then optional Cy5 fail-safe recovery
  (Tier 2), then track-level Cy5 filtering. Returns kept tracks +
  dropped tracks (debris) + raw labels + n_cy5_recovered.
- **scripts/test_multichannel_unit.py** — synthetic unit tests
  (bright/faint/debris/filopodia/edge/empty cases). All passing.
- **scripts/test_multichannel_pilot.py** — real-data pilot on a few
  IC295 frames; renders side-by-side comparison grid (DIC, Cy5,
  raw masks, filtered masks). Run after IC293 full pipeline frees
  cellpose4 GPU.
- **scripts/bench_multichannel.py** — benchmark vs hand-labelled GT;
  reports IoU DIC-only vs IoU after Cy5 filter, per-condition.
- **scripts/sample_gt_frames.py** — `--multichannel` flag added;
  saves `<name>_dic.png` (label this) + `_cy5.png` + `_composite.png`
  per candidate. New `_LABELLING_INSTRUCTIONS_MULTICHANNEL` template.
- **output/run_metadata.py** — `write_run_metadata(out_path, title,
  sections, rerun_cli, ...)` helper; auto-captures cellpose / torch /
  numpy / scipy / tifffile / python / platform versions. Used by all
  new analysis scripts to comply with the project's "always emit
  RUN_METADATA.md alongside results" convention.

See `docs/multichannel_analysis_plan.md` for the full plan + phased
implementation; `data/ic295_inspection/` for the channel inspection
PDF + contact sheet that informed the design.

## New Ignasi recordings — testing + GT sampling (IC293 single-channel)
Source: `/Users/george/Desktop/ignasi_cellscope_test_data` (16 recordings × 1.6 GB,
2048×2048 phase contrast at 0.65 µm/px, 10-min interval, 97 frames × 2 channels;
phase = channel 0). Conditions: WT (3), KO (3), GOF (3), Y1 (3), DMSO (4).

- **scripts/sample_gt_frames.py** — picks 4 frames per recording (5/34/62/91 by quantile),
  applies flat-field correction (subtract Gaussian-blurred background, σ=80) to remove
  the severe vignette, writes uint8 PNGs + manifest CSV + CLAHE contact sheet +
  `LABELLING.md`. Output: `data/ignasi_new_gt/candidates/`.
- **scripts/test_ignasi_new.py** — smoke-tests every .ome.tif: parses metadata, reads
  first/middle/last frames, runs cpsam on the middle frame in `cellpose4` env via
  subprocess, writes per-recording previews + `summary.csv` + `report.md`.
- **scripts/bench_ignasi_new.py** — once `*_masks.png` files exist, runs cpsam against
  every labelled candidate and reports IoU per condition (WT/KO/GOF/Y1/DMSO).
