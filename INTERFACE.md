# CellScope — Interface Map

## Entry Points
- **main_suite.py** — Unified launcher (tkinter, runs from `cellpose` env)
- **main_focused.py** — Detection & Analysis GUI
- **main_batch.py** — Batch Processing GUI
- **main_tracking.py** — Tracking & Comparison GUI
- **main_editor.py** — Mask Editor GUI
- **main_training.py** — Model Training GUI
- **main_annotate.py** — Annotation / Labelling GUI — hand-label cell-frames into classes (rounded/spread, …) for training; review + edit existing labels. `gui_annotate/annotate_window.py` (table of cell-frames + zoomable DIC-crop viewer with r/s/u keys, auto-advance + auto-save) + `gui_annotate/annotate_data.py` (`LabelStore` CSV round-trip + `CropRenderer` with LRU recording cache). Reads/writes the IC295 state_labels CSV so labels feed `scripts/ic295_label_states.py train`. RPC port 8772.

## Remote control (all GUIs)
Every GUI launched with `CELLSCOPE_REMOTE=<port>` exposes an HTTP RPC
server for scripted / agent-driven testing. Default ports: focused 8765
(full handler set), batch 8766, editor 8767, training 8769, tracking 8770,
suite 8771. See `gui_focused/remote_control.py` and `CLAUDE.md` for usage.

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
- **unified_detection.py** — `detect_recording()` — **single canonical detection entry point.** Performs alignment → downsample → threshold conversion → auto-pipeline-select → detect (with optional Cy5 fusion + recovery inline) → Cy5 persistence_guard filter → upscale labels back. Accepts **16 explicit kwargs** (use_deepsea, use_gap_fill, use_sam2_video_gap_fill, max_gap_frames, min_track_length, use_tta, use_cpsam_cy5_union, use_fallback, use_mirror_pad, use_preprocess, use_retry, cy5_filter_mode, cy5_filter_threshold, cy5_pg_min_lifetime, cy5_pg_static_velocity_px, cy5_pg_static_shape_iou, cy5_fusion_jaccard_thresh, cy5_fusion_max_overlap_frac, cy5_fusion_augment_cpsam, use_bfloat16) — None for any kwarg falls back to `DEFAULTS`. Called from both `gui_focused/workers.py` (mode="auto") and `scripts/run_pipeline_on_gt_recording.py`, guaranteeing the GUI and runner produce identical output. **`use_bfloat16`** (DEFAULTS True = bf16, cellpose default) is an opt-in precision flag: uncheck / `--fp32` → fp32, ~1.26× faster on Apple Silicon (MPS); no effect on CUDA. Validated as a wash on per-frame IoU but not yet F1-certified, so default stays bf16. Affects raw-cpsam detection + all gap-fill; the cpsam_dic subprocess stays bf16.
- **detection_presets.py** — named speed↔quality bundles for the focused GUI's "Detection preset" dropdown: **Fast** (skips gap-fill, 3× downsample, fp32, no TTA/DeepSea/fallback/mirror-pad), **Medium** (cheap gap-fill only — no SAM2/CP3 — fp32), **Default (Balanced)** (no overrides = `DEFAULTS`), **Highest Quality** (full res, all gap-fill phases, full-frame always-augment, TTA + DeepSea + fallback + mirror-pad). Per the single-source rule a preset records only *deltas* vs `DEFAULTS`; `preset_values(name)` resolves the full knob set, `apply_preset_to_panel(panel, name)` drives the existing widgets (no new pipeline plumbing). Selecting a preset resets all of `PRESET_PARAMS`, so it's deterministic; the user can still tweak individual options afterward.
- **channel_alignment.py** — `align_cy5_to_dic()` measures DIC↔Cy5 offset via cellpose centroid matching (in `cellpose4` subprocess), `apply_offset_to_stack()` shifts via scipy.ndimage. Skips when |offset| < 1px.
- **detection.py** — `detect_cellpose`, `detect_cellpose_labels`, `detect_cellpose_tiled`
- **hybrid_cpsam.py** — `detect_hybrid_cpsam()` — single-cell cpsam + DeepSea + fallback
- **hybrid_cpsam_multi.py** — `detect_hybrid_cpsam_multi()` — multi-cell with tracking
- **hybrid_dic.py** — `detect_hybrid_dic()`, `detect_hybrid_dic_multi()` — DIC pipelines. When the selected DIC model is `cpsam_dic`, runs the cellpose step in the `cellpose4` env via subprocess (`_run_cpsam_dic_subprocess`); otherwise uses CP3 cellpose_dic + preprocessing + DeepSea + threshold retry directly.
- **modality.py** — `detect_modality()` — auto-detect DIC vs phase-contrast from image statistics; `get_pipeline_config()` returns per-modality settings
- **deepsea_multicell.py** — Per-cell DeepSea refinement preserving labels
- **medsam_deepsea_union.py** — MedSAM + DeepSea union (single-cell)
- **medsam_refine.py** — MedSAM bbox-prompt refinement
- **multi_cell.py** — `track_all_cells()` — Hungarian tracker (accepts `min_track_length` + `max_gap_frames` + cost-matrix weights `w_dist` / `w_iou` / `w_area`). Cost combines `w_dist · distance + w_iou · (1-IoU) · max_hop + w_area · |Δarea|/baseline · max_hop`. Defaults 1.0 / 0.5 / 0.3 — calibrated against Pos7-WT GT (ID consistency 0.88 → 0.97 vs distance-only). The 3 weights live in `DEFAULTS.track_w_dist / track_w_iou / track_w_area`.
- **detection_postprocess.py** — `merge_touching_splits(labels, area_ratio=0.3, dilate_px=2)` — merges touching cpsam over-segmentation fragments where the smaller fragment is < area_ratio of its larger neighbour and they touch within `dilate_px`. Lands before tracking so cells get a single ID.
- **track_gap_fill.py** — Post-tracking gap fill cascade: cpsam(augment) → CP3 + MedSAM + DeepSea → SAM2 video propagation → translation-only fill. 100% fill rate on tested recordings (41/41 gaps). **Phase 1 cascades fast→slow**: cpsam on an adaptive crop, no-augment (`DEFAULTS.gap_fill_crop` / `gap_fill_augment`), with augment-on-miss then full-frame-on-miss fallbacks so recall ≥ the old always-full/always-augment path. GT-validated no quality loss (`scripts/bench_gap_fill.py`): crop 12.5× at full res; `augment=False` 2.1× at production 1024² (the lever that survives downsampling, since augment is call-count not pixels). `fill_track_gaps(..., use_crop, use_augment, stats_out, cpsam_model)` logs per-phase timing + gaps-filled. `stats_out` (per-phase `{time_s, filled}`) is now surfaced in the result dict as `gap_fill_stats` and written to `RUN_METADATA.json` `extra.gap_fill_stats` by every detect writer (enables the per-phase ablation). `cpsam_model` lets a caller pass an already-loaded **plain raw** cpsam model to reuse for Phase 1 (skips a second cold-load; `hybrid_cpsam_multi` passes its detection model when not a fine-tune — GT-verified bit-identical). NB cpsam runs on **MPS (the Apple GPU)** via `gpu=True` on Apple Silicon — ~43× faster than CPU; the pipeline is GPU-bound, not CPU-bound.
- **track_postprocess.py** — `postprocess_tracks(tracks, frames, min_frames)` — 6-stage track-level cleanup: (1) reject_false_positives, (2) reject_edge_artifact_tracks (small near-FoV-edge tracks), (3) reject_edge_sliver_detections (vignette-bar mis-detections), (4) reject_static_edge_blob_tracks (large static blobs near FoV edges — vignette / illumination artefacts with min_area_px=2000, edge_band_px=120, max_velocity_px=3.0, min_shape_iou=0.85), (5) smooth_bouncing_ids (per-frame ID smoothing for transient single-frame swaps), (6) remove_empty_tracks below `min_frames` (default 3, must match `min_track_length` for single-frame test-on-frame to work).
- **cy5_filter.py** — `apply_cy5_filter(tracks, mode, threshold, pg_min_lifetime, pg_static_velocity_px, pg_static_shape_iou)` — Tier 4 false-positive filter for tracked Cy5-multichannel runs. 11 strategies: off / persistence_guard (default) / conservative / conservative_strict / threshold / adaptive / adaptive_loose / multi_metric / composite_score / consensus / temporal_stability. **`persistence_guard_filter(tracks, min_lifetime=35, static_velocity_px=3.0, static_shape_iou=0.85, ...)`** — 3-stage rule: (1) keep if multi-metric ≥2/3 pass; (2) drop short tracks (lifetime < min_lifetime) that fail; (3) for long-lived tracks failing the metrics, drop only if STATIC (mean velocity < static_velocity_px AND median consecutive-frame mask IoU > static_shape_iou). Calibrated against full 13-GT corpus (F1_focused +0.038 vs strict multi_metric).
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
- **edge_filter.py** — `mask_touches_edge(mask_bool, margin=0)` / `bbox_touches_edge(r0,r1,c0,c1,h,w,margin=0)` — detect cell masks truncated by the image border. MUST be given a FULL-FRAME mask (a bbox crop touches its own border by construction). The single primitive behind edge-truncation exclusion: such cell-frames are voided to `unknown` for shape/state, but still counted + tracked.
- **cell_state.py** — **binary** cell-frame classification: `rounded` / `spread` / `unknown`. PRIMARY rule (learned from hand labels, 0.90 CV vs the old 0.60 — see `scripts/ic295_eval_state_rule.py`): `rounded` iff `area_um2 ≤ rounded_area_um2` (960) AND `eccentricity ≤ rounded_eccentricity` (0.85) — needs the recording's `um_per_px`, so `classify_state(metrics, thresholds, um_per_px=...)` and `classify_track_states(stack, thresholds, um_per_px=...)` take the scale (threaded from `state_analysis`, `mask_metrics`, `gui_batch/batch_worker`). FALLBACK when no scale is known: legacy `circ ≥ rounded_circ` (0.80) AND `solid ≥ rounded_solid` (0.92). `DEFAULT_THRESHOLDS` (canonical) holds all of `rounded_area_um2`/`rounded_eccentricity`/`rounded_circ`/`rounded_solid`/`min_area_px`/`edge_margin_px`. Edge-truncated cell-frames (`shape_metrics_for_mask` sets `edge_touch`, via `core.edge_filter`) are `unknown` so they drop out of every per-state aggregation; `classify_track_states` returns a per-frame `edge` array. Deprecated `STATE_BALLED`/`STATE_ATTACHED`/`STATE_TRANSITIONAL` aliases (→ rounded/spread/spread) keep legacy state scripts importing. Helpers: `classify_track_states`, `state_segments`, `state_fraction`, `per_state_means`.
- **state_analysis.py** — `annotate_state(result, cell_stack, um, dt)` — **single source of truth** for cell-state metrics, shared by `scripts/ic295_analyze_one.py` AND `gui_focused/workers.py`. Fixes the time-in-state confound: every per-frame metric is computed SEPARATELY over the cell's rounded frames and its spread frames — `mean_speed_{rounded,spread}`, `persistence_*`, `straightness_*`, `mean_area_um2_*`, `mean_circularity_*`, `mean_solidity_*`, `mean_aspect_ratio_*`, `mean_eccentricity_*` — plus the lifetime `frac_rounded`/`frac_spread`. Wraps `core.cell_state` + `core.motility_state`.
- **cell_metrics_table.py** — `per_cell_row(c, ti)` + `aggregate_recording(rows, n_div)` + `write_per_cell_csv(rows, path)` — the flat per-cell CSV schema (lifetime cols: `frac_rounded`/`frac_spread`/`frames_tracked`/division*; + the per-state rounded/spread metrics — the state-MIXED whole-track averages are intentionally excluded) + recording-level mean/median/std aggregation, shared by `scripts/ic295_analyze_one.py` and `gui_focused/export_dialog.py` so a focused-GUI export drops straight into `scripts/ic295_compare.py`.
- **mask_metrics.py** — `compute_label_metrics(labels, um, dt)` — fast per-cell metrics computed straight from a label stack (per-track scalars + per-frame arrays + per-frame state codes), with NO full Analyze run. Powers the "Colour masks by result" overlay in both the focused viewer and the mask editor.

## `gui/` — Shared Components
- **mask_editor.py** — Interactive mask editor (brush/eraser/polygon/fill, multi-cell labels). "Colour:" dropdown + ↻ recompute colours each cell by a metric (state / speed / circularity / % rounded / …) via `gui.metric_coloring`. Keyboard shortcuts: tools `B/E/P/F/R/X`, `N` new cell, `1–0` cell select, `Ctrl+Z/S`, `Ctrl+Shift+F` filter, `Ctrl+T` trim edges, `Ctrl+K` clean, `Ctrl+0` fit, `F1`/`?` shortcuts popup. Help menu via `gui.editor_help`.
- **help_menu.py** — **reusable Help menu for every CellScope Qt GUI.** `install_help_menu(window, app_name, shortcuts, guide_path=None, doc_url=None)` adds (or extends) a Help menu — Keyboard Shortcuts… popup (built from the per-GUI `shortcuts` list), optional in-app markdown guide, Online Documentation ↗, About — creating the menu bar if absent, appending to an existing Help menu without duplicating items, and registering F1 / ? (ApplicationShortcut). `ShortcutsDialog` / `GuideDialog` / `show_shortcuts` / `show_guide`. Used by the editor (via `editor_help`), tracking, annotation, batch, and training GUIs (the focused GUI has its own richer Help menu).
- **editor_help.py** — thin mask-editor Help config over `gui.help_menu`: the editor's `SHORTCUTS` reference + `docs/mask_editor_guide.md` path; `install_help_menu(window)` / `show_shortcuts(parent)` that `gui.mask_editor` imports.
- **mask_editor_multicell.py** — Per-cell color helpers, label utilities. `render_label_overlay(..., color_lut=None)` — the optional `color_lut` ({cell_id:(r,g,b)}) drives the colour-by-result overlay.
- **metric_coloring.py** — Colour-by-result infrastructure shared by the focused viewer + mask editor: `METRICS` registry (Cell ID / Cell state / per-track scalars / per-frame values), `MetricColorizer` (value→RGB via matplotlib colormaps; `STATE_COLORS` for the categorical state metric), and `MetricLegend` (gradient-bar / state-swatch key widget).
- **run_log.py** — RunLogger + RunLogWidget (event logging)
- **workers.py** — DetectWorker, RefineWorker, BatchWorker
- **options/** — Shared parameter panels (params.py, detection_panel.py, refinement_panel.py, analysis_panel.py, presets.py, presets_widget.py, options_panel.py)

## `gui_focused/` — Detection & Analysis GUI
- **main_window.py** — FocusedMainWindow (state machine, ROI, drag-drop). Hosts `_on_test_frame()` which runs detection on the currently displayed frame with current GUI parameters, times it, and reports a density-aware extrapolation to full-recording runtime (1.5× sparse / 2.0× medium / 2.5× dense post-proc multiplier). Test-on-frame now runs the same `_on_detect` path with `min_track_length=1` and skips multi-frame stages, and emits a warning when the selected mode (single/multi) doesn't match the recording's density probe. Also hosts the remote-control handler set when `CELLSCOPE_REMOTE=<port>` is set (16 endpoints: status/log/load_recording/load_pipeline_results/load_project/clear_all/set_param/set_frame/set_view/set_mode/detect/test_frame/analyze/save_screenshot/save_project/export). closeEvent warns on unsaved results.
- **remote_control.py** — HTTP RPC server module (stdlib `BaseHTTPRequestHandler` inside the Qt event loop). `RemoteControlServer(QObject)` dispatches commands across threads via `pyqtSignal(dict, object)` so handlers run on the GUI thread. `attach(window, handlers, port)` for the full-featured focused GUI, `attach_minimal(window, gui_type, default_port)` for the simpler GUIs (status + log only). `parse_remote_env()` reads `CELLSCOPE_REMOTE=<port>` from the environment.
- **image_viewer.py** — ImageViewer + FrameNavigatorBar (B/C, zoom, pan). "Colour by:" dropdown + legend recolour cells by an analysis result (cell state / mean speed / persistence / % rounded / per-frame area·circularity·speed) via `gui.metric_coloring`; metrics come from `core.mask_metrics` (no Analyze needed) and refresh when masks change.
- **pipeline_panel.py** — 5 stage buttons + mode selector + Cancel / Undo Detect / **🔬 Test on frame** / Clear All toolbar row. `btn_test_frame` auto-gates on detect-stage availability; emits `test_frame_clicked` signal.
- **params_panel.py** — Context-sensitive parameters (modality selector: Auto/DIC/Phase-contrast). A top-of-page **Detection preset** dropdown (Fast / Medium / Default (Balanced) / Highest Quality) one-click-sets the speed↔quality knobs via `core.detection_presets`. Exposes **19 wired pipeline parameters** in 6 grouped sections: Detection (model, min_area, expected_cells, search_radius, min_track_length, ROI), Refinement steps (DeepSea, TTA, cpsam-on-Cy5 union, fallback, mirror-pad), Gap fill (toggle + SAM2 video sub-toggle + **Gap-fill crop (fast)** + **Gap-fill always augment** revert toggles, all gated on Gap fill, max_gap_frames), Cy5 multichannel (fusion, recovery, filter mode dropdown + 3 persistence_guard sub-spinboxes gated on Persistence guard mode), Tiling, DIC pipeline (preprocess, retry, 3 Cy5 fusion sub-thresholds). All values reach `detect_recording` end-to-end via `get_detect_params()`.
- **analysis_view.py** — Summary/Graphs/Log tabs. Multi-cell Summary now shows a recording-level aggregate block (n_cells, division rate, mean speed, speed-by-state) plus per-cell state composition + per-frame speed-by-state, using `core.cell_metrics_table`.
- **analysis_plots.py** — 16 plot functions + GRAPH_REGISTRY (timeseries plots accept `gap_interp_max` kwarg for short-gap interpolation)
- **vampire_plots.py** — 4 VAMPIRE plots (Shape Modes scatter, Mode Distribution histogram, Mode Over Time, Eigenshape variations); split out so analysis_plots stays under the 500-line limit
- **division_plots.py** — cell-division graphs registered in GRAPH_REGISTRY: **Cell Lineage Tree** (per-cell lifelines + parent→daughter division connectors with score) and **Division Timeline** (per-event score stems + cumulative-divisions step). Driven purely from per-cell `track_info` (parent_id / division_frame / division_score) + `shape_timeseries`, so they render whenever divisions were detected (inline or by `core.division_annotator`). Division detection is gated by the params panel's "Detect divisions" toggle (`get_division_params`, default on) → `FocusedAnalyzeWorker(division_params=…)`.
- **share_export.py** — **Export Shareable Image** (File menu / Ctrl+Shift+I): compact PNG/JPEG (current frame), MP4/animated-GIF/Montage (all frames), each overlay independently switchable (mask fill, contours, cell IDs, tracks, timestamp, scale bar) + a max-dimension downscale + JPEG quality, for small files that drop into a slide/chat. Reuses `mask_editor_multicell.render_label_overlay` + `overlays.draw_overlays`. `ShareImageDialog` + standalone `render_share_frame`/`save_static`/`save_mp4`/`save_gif`/`make_montage` helpers.
- **export_dialog.py** — Export configuration dialog. When "Save masks" is ticked, also writes a `divisions.json` sidecar next to `masks.npz` (always present — empty `candidates` list if no divisions detected). Sidecar contains both the raw candidates from `core.division_annotator` and a compact `track_lineage` table mapping daughter-track-index → parent-track-index. With "Metrics (.json)" ticked in multi-cell mode also writes `per_cell.csv` + `recording_summary.json` (IC295 schema, via `core.cell_metrics_table`) so exports feed `scripts/ic295_compare.py` directly.
- **workers.py** — FocusedDetectWorker, FocusedAnalyzeWorker. `FocusedAnalyzeWorker` propagates each track's `parent_id`/`division_frame`/`division_score` into the per-cell `track_info` dict so the analysis view can display lineage. State classification is **on by default** (`DEFAULTS.compute_state_classification`); `_annotate_with_state` delegates to `core.state_analysis.annotate_state` (shared with the IC295 batch).
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
- **IC295 batch analysis** (`ic295_*.py`) — long-running detect →
  manual-review → analyze → compare pipeline for the IC295 dataset.
  See `ic295_analysis/README.md` for usage.
  - **ic295_common.py** — shared utilities: drive inventory, condition
    parsing, `by_condition/<cond>/<label>/` paths, atomic
    `progress.json` writes, `.cellscope` project writer, recording-
    folder setup (symlinks + sidecar), priority queue (existing drive
    detections first, then round-robin to balance n).
  - **ic295_detect_one.py** — Phase-1 per-recording worker. Adopts
    existing drive masks instantly; falls through to full
    `unified_detection.detect_recording` only when needed. Writes
    `pipeline_results/masks.npz` + `divisions.json` + `RUN_METADATA` +
    `<label>.cellscope`. Idempotent.
  - **ic295_analyze_one.py** — Phase-2 per-recording worker. Loads
    (possibly user-edited) masks, rebuilds tracks, runs
    `annotate_track_lineage` + per-cell `analyze_recording` +
    cell-state classification, aggregates to a single-row recording
    summary. Writes `analysis.json` + `per_cell.csv` +
    `recording_summary.json`. Idempotent.
  - **ic295_batch.py** — driver. `--phase=detect|analyze|both`. Lock
    file, SIGTERM-safe, per-recording subprocess isolation, atomic
    progress updates, `--retry-failed`, `--limit`, `--label`.
  - **ic295_status.py** — read-only state reporter (safe while batch
    runs): per-condition state counts + ETA + currently-running +
    `--failed` tails.
  - **ic295_compare.py** — Phase 3. **Each recording = one experiment**
    (n = recordings/condition). For each metric: per-condition
    mean/SEM/n, Kruskal-Wallis across conditions + pairwise
    Mann-Whitney (Bonferroni), box+scatter plots. Metric set is binary-
    state: lifetime (n_cells, n_divisions, division_rate, frac_rounded)
    + per-state rounded + per-state spread — never a state-mixed
    average. Outputs `per_recording.csv` + `per_treatment.csv` +
    `stats.json` + `plots/*.png` under `ic295_analysis/compare/`.
  - **ic295_compare_pooled.py** — cell-POOLED mirror of `ic295_compare`
    (each CELL = one sample, n = cells/condition). Reuses the same stats
    helpers; only the replication unit differs. ⚠️ pseudoreplication
    (cells within a recording aren't independent) — for side-by-side
    comparison only. Outputs under `ic295_analysis/compare_pooled/`.
    `--min-valid-frames N` drops cells with too few in-view frames.
  - **ic295_compare_arms.py** — **design-aware comparison (the correct
    primary for hypothesis testing).** The conditions are TWO independent
    experiments each with its own control: GENETIC (control WT → GOF, KO)
    and DRUG (vehicle DMSO → YODA1/Y1, OT), plus the VEHICLE check WT vs
    DMSO. Each arm gets its OWN Kruskal-Wallis + pairwise Mann-Whitney,
    **Bonferroni-corrected within the arm** (not the flat all-15-pairwise,
    which tests meaningless cross-arm contrasts like GOF vs OT and
    over-corrects). Reads the `per_recording.csv` / `per_cell_pooled.csv`
    that `ic295_compare`/`_pooled` wrote. Outputs `stats_arms.json` +
    `plots_arms/<metric>.png` (genetic | drug panels, control highlighted,
    test-vs-control stars, vehicle p in title) in both compare dirs.
    `--level recording|pooled|both`. Key finding: the VEHICLE effect (WT
    vs DMSO) is itself significant, so drug effects must be read vs DMSO.
  - **ic295_plot_mean_sem.py** — bar (mean) + SEM error bars + individual
    points per condition, for every metric, at both levels (points =
    recordings or cells). Writes `plots_mean_sem/` in each compare dir.
  - **ic295_histograms.py** — per-metric histograms split by condition
    (overlaid step, shared bins), at both levels — recording
    (`compare/histograms/`) and cell (`compare_pooled/histograms/`).
    Shape metrics get the rounded threshold drawn.
  - **ic295_plot_utils.py** — `apply_ybreak(fig, draw, values, …)`: shared
    helper that auto-breaks the y-axis (small outlier panel + large bulk
    panel + diagonal marks) when high outliers would squish the data
    (inlier range < ½ the full range); else a single axes. Used by the
    box+strip (`ic295_compare`), mean±SEM, and arm plots.
  - **ic295_track_data.py** — **shared per-cell track collector + cache**
    (`collect`, `load_or_build`) used by BOTH `ic295_flower_plots` and
    `ic295_motility_stats`. One mask-load pass per recording builds an
    enriched per-cell record: recording `label`/`cond`, origin-centred
    `traj` + absolute `cents` (µm, NaN gaps), per-frame `states`,
    per-frame local density (`n_neighbors` within 100 µm + nearest-
    neighbour `nn_dist`), per-frame `area_frames`, and the scalar
    `speed`/`distance`/`netdisp`/`area`. Versioned pickle cache
    (`compare/flower_plots/_track_cache.pkl`, `CACHE_VERSION`) so the ~1 h
    reload stays out of the plot/stat iterate loop; `--rebuild` recollects.
  - **ic295_flower_plots.py** — origin-centred **track (flower) plots** +
    per-cell **motility** by condition (consumes `ic295_track_data`). Each
    cell's full trajectory (both states) is translated to the origin and
    overlaid per condition on a shared equal x/y axis
    (`flower_{all,rounded_only,spread_only}.png`); plus per-cell box+strip
    plots (`{speed,distance,netdisp}_{all,rounded,spread}.png`) over three
    groupings (all / whole-track-rounded / whole-track-spread), per-cell
    **area-vs-speed** scatter, and ensemble **MSD(τ)** for full-recording
    cells: `_ensemble_msd(stat)` does mean±SEM OR **median+bootstrap-CI**
    (robust to the skew). Emitted all-treatment + per **arm** (genetic |
    drug), linear/log-log/median (`msd_by_treatment*`, `msd_{genetic,
    drug}{,_loglog,_median}.png`). Under `compare/flower_plots/`.
  - **ic295_motility_stats.py** — **design-correct + confounder-aware**
    motility/dispersal stats (consumes `ic295_track_data`). (b) reduces
    each recording's full-duration cells to one value → arm-structured
    test (reuses `ic295_compare_arms`) on net displacement, endpoint MSD,
    speed, frac-spread, and crowding. Confounders: STATE (paired within-
    cell spread-vs-rounded speed + frac_spread as its own metric), CONTACT
    (speed-vs-density Spearman + paired isolated-vs-crowded speed +
    per-treatment density), PSEUDOREPLICATION (recording-level numpy OLS
    adjusting treatment effect for frac_spread + density — the dependency-
    free stand-in for a cell-level LMM). Writes `stats_arms_motility.json`,
    `plots_arms/*.png`, `speed_vs_density.png`, `REPORT.md` under
    `compare/motility_stats/`.
  - **ic295_state_features.py** — multi-feature diagnostic for the
    rounded/spread boundary: per cell-frame computes area (µm²),
    **rel_area** (footprint vs the cell's own 90th-pctl area),
    circularity, solidity, extent, eccentricity, aspect_ratio, convexity;
    reports each feature's bimodality coefficient + the feature
    distributions split by the **deployed** rounded/spread call (the
    deployed-rule threshold lines drawn on the area + eccentricity
    panels). Finding (IC295): **no feature is statistically bimodal** (all
    Sarle BC < 0.56 — the classes overlap as a continuum, which is why the
    cut is fitted from labels, not read off a trough); rel_area cleanly
    separates the call (median 0.17 rounded vs 0.84 spread). Under
    `compare/state_features/`.
  - **ic295_label_states.py** — `sample` / `label` / `train`. Since no
    single feature is bimodal, this learns the boundary from a few dozen
    hand-labelled examples: `sample` k-means-picks a morphologically
    diverse set of cell-frames, renders DIC crops + a numbered montage +
    `labels.csv`; `label` is an interactive labeller (r/s/u keys);
    `train` fits a logistic-regression + depth-2 tree on the shape
    features and reports CV accuracy + feature weights + a rule. Under
    `ic295_analysis/state_labels/`.
  - **ic295_state_diagnostic.py** — **confirms the rounded/spread cut**
    against ALL mask data. Loads every masks.npz, recomputes per-frame
    **area (µm²) + eccentricity** (the deployed rule's features) for every
    classifiable (non-edge) cell-frame, and plots their distributions
    (pooled + per condition) with the `DEFAULT_THRESHOLDS` cuts drawn + a
    2D area-vs-ecc density with the rounded region boxed + a `summary.txt`
    (% frames rounded, per gate, per condition). Under
    `compare/state_diagnostic/`. (IC295: area is visually bimodal with the
    960 µm² cut in the trough; KO 39% rounded → OT 4%.)
  - **ic295_eval_state_rule.py** — **scores the deployed rule against the
    hand labels** (`state_labels/labels.csv`). Shipped-rule confusion
    matrix, per-feature ROC-AUC + best single threshold, 5-fold CV of
    logistic / decision-tree models, an interpretable depth-2 tree, and
    `--um`/`--no-plots`. Emits the fitted `rounded_area_um2` +
    `rounded_eccentricity` (re-fit as labels grow) and writes
    label-grounded validation figures (decision boundary, per-feature
    histograms by hand label, confusion, AUC bar) to
    `compare/state_rule_validation/`. Current 279-label result: the
    deployed rule scores acc 0.93 / rounded-recall 0.90 (vs the old
    circ/solid rule's 0.60 / 0.14).
  - **ic295_compare_edits.py** — "how did manual edits change the
    numbers?" For any recording with a `masks_original.npz` beside the
    current `masks.npz`: backs up the edited state, swaps in the
    originals, re-runs `ic295_analyze_one.py`, saves the result as
    `recording_summary_original.json`, then restores the edited state
    byte-for-byte. Skips if the original masks are missing or the
    `*_original.json` already exists. `[label]` to target one.
  - **ic295_copy_from_lab.py** — disaster recovery: copy source
    TIFFs from `/Volumes/pathaklab/...` to local `_cache/` when the
    primary drive fails. Driven by `progress.json` (detect.state ==
    'done' by default). Atomic `.tmp`+rename copy with size verify
    and exponential backoff retries. Synthesizes `.ome.json`
    sidecars and repoints `by_condition/<cond>/<label>/` symlinks
    at the local cache. Idempotent / resume-friendly.
- **gui/mask_editor_sam2_point.py** — SAM2 point-and-click cell
  detection for the mask editor. Picks "sam2" in the tool palette,
  one click on a missed cell adds a mask with the active cell ID.
  Runs SAM2 (Hiera Tiny) on a 512×512 crop around the click for
  100-200 ms latency. Guards reject too-small / too-large blobs
  and clicks outside the predicted mask. Apply uses the editor's
  existing undo stack (Cmd+Z restores).
- **_paths.py** — Project-root resolver + `benchmark_data_root()` helper
  (env var `BENCHMARK_DATA_ROOT` overrides the default sibling lookup).
  Imported by every benchmark / training script via the standard preamble:
  ```python
  import sys, os
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  from _paths import setup_imports, benchmark_data_root
  setup_imports()
  ```
- **bench_gap_fill.py** — gap-fill Phase-1 benchmark: deletes known
  masks from reviewed tracks (GT), runs full-frame vs adaptive-crop
  cpsam on the same synthetic gaps, scores fill-rate + IoU-vs-GT + time.
  Gates `DEFAULTS.gap_fill_crop`.
- **bench_cpsam_dic.py** — IoU benchmark for any cellpose model on a
  test directory (TIF or PNG with masks).
- **compare_cpsam_dic.py** — diff two bench JSONs side by side.
- **make_overlay_figures.py** — generate inspection overlays across
  recording types.
- **make_single_cell_example.py** — crop ONE cell's track (+margin,
  longest contiguous present-run) out of a recording + label stack into
  a small, fast, single-cell example under `data/examples/`. Used to
  rebuild `test_focused_gui.py`'s recording after the source drive
  failed (`data/examples/single_cell_crop_wt/`).
- **fix_dead_symlinks.py** — drive-failure cleanup: repoint dangling
  symlinks to local copies (`_cache` / `by_condition`), materialize a
  dead `results/` link as a real dir, or remove the rest (recording
  each removed target to `DEAD_SYMLINK_RECOVERY.md`). `--dry-run`
  previews. Touches only dangling symlinks — never real files or
  `gt_masks/*.png`.
- **recompress_recordings.py** — losslessly re-compress recording
  TIFFs (Deflate by default) to shrink storage with ZERO effect on
  results. Per file: read → write compressed `.tmp` → **verify
  bit-for-bit page-by-page + unchanged `detect_channels`** → only then
  atomic `os.replace`. Never removes an original without a verified
  replacement. `--dry-run` / `--force` / `--keep-backup` / `--codec`.
  On the IC295 16-bit uncompressed masters: ~1.97× (2.45 GB → 1.25 GB)
  with identical `load_recording` frames + identical cpsam labels.
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
  single-cell load → detect → analyze → 16 graph types → colour-by-
  result (all metric_coloring options + legend) → export flow with
  screenshots. 64 checks. Runs headless via
  `QT_QPA_PLATFORM=offscreen`. Recording resolves from CLI arg →
  `$CELLSCOPE_TEST_RECORDING` → bundled `data/examples/single_cell_crop_wt`
  → `single_cell_phase_WT` (no longer a hard-coded path).
- **_gui_verify.py** — fast headless test-drive of *every* GUI +
  *every* focused-GUI option, complementing the per-pipeline depth of
  `test_focused_gui.py`. 85 checks: all 5 GUI main windows construct;
  the gap-fill `crop`+`augment` toggles are present, default to
  `DEFAULTS`, gate on `use_gap_fill`, their revert flows reach
  `get_detect_params()`, and the detect worker receives both;
  colour-by every `metric_coloring` metric (focused viewer + mask
  editor + legend); Analyze on loaded masks → every registered graph
  (incl. lineage + division timeline); share-image `_export()` (PNG +
  JPEG) via the dialog's real path. Drives loaded masks (no detection)
  so it runs in ~1 min. The canonical multi-cell recording is
  `ic295_analysis/by_condition/WT/Pos7-WT`.
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
