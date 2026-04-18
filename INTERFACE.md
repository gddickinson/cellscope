# CellScope — Interface Map

## Entry Points
- **main_suite.py** — Unified launcher (tkinter, works from any env)
- **main_focused.py** — Detection & Analysis GUI
- **main_batch.py** — Batch Processing GUI
- **main_tracking.py** — Tracking & Comparison GUI
- **main_editor.py** — Mask Editor GUI
- **main_training.py** — Model Training GUI
- **setup_wizard.py** — Environment + model installer
- **make_dist.py** — Create distribution zip

## `core/` — Analysis Pipeline (32 modules)

- **io.py** — `load_video`, `load_recording`, `find_recordings`
- **pipeline.py** — `detect()`, `refine()`, `analyze_recording()`
- **detection.py** — `detect_cellpose`, `detect_cellpose_labels`, `detect_cellpose_tiled`
- **hybrid_cpsam.py** — `detect_hybrid_cpsam()` — single-cell cpsam + DeepSea + fallback
- **hybrid_cpsam_multi.py** — `detect_hybrid_cpsam_multi()` — multi-cell with tracking
- **deepsea_multicell.py** — Per-cell DeepSea refinement preserving labels
- **medsam_deepsea_union.py** — MedSAM + DeepSea union (single-cell)
- **medsam_refine.py** — MedSAM bbox-prompt refinement
- **multi_cell.py** — `track_all_cells()` — Hungarian tracker
- **track_gap_fill.py** — Post-tracking gap fill with augmented re-detection
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
- **params_panel.py** — Context-sensitive parameters
- **analysis_view.py** — Summary/Graphs/Log tabs
- **analysis_plots.py** — 20 plot functions + GRAPH_REGISTRY (includes 4 VAMPIRE plots: Shape Modes scatter, Mode Distribution histogram, Mode Over Time, Eigenshape variations)
- **export_dialog.py** — Export configuration dialog
- **workers.py** — FocusedDetectWorker, FocusedAnalyzeWorker
- **roi_selector.py** — Rectangle/ellipse/polygon ROI
- **dialogs.py** — System info, shortcuts, about

## `gui_batch/` — Batch Processing GUI
- **batch_window.py** — Directory scan, recording tree, settings, progress
- **batch_worker.py** — QThread batch processing

## `gui_tracking/` — Tracking & Comparison GUI
- **tracking_window.py** — Main window with Single/Batch tabs
- **single_view.py** — Load masks, track, per-track analysis + plots
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
