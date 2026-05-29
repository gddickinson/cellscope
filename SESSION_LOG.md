# CellScope — Session Log

Chronological log of substantive changes + investigations. Per
`CLAUDE.md`, append a short entry whenever a non-trivial change
lands. Source of project memory; "why does X work this way?" should
have an answer here or in the linked commit.

Format: **DATE — short title** with bullets describing what changed
+ key numbers + link to driving commits/files. Most recent first.

---

## 2026-05-29 — Multichannel TIFF without `_metadata.txt` no longer loads as a line

`core/io.py::detect_channels` previously fell back from `_metadata.txt`
straight to OME-XML `SizeC` — and many of our `.ome.tif`s don't carry
`SizeC`. Without the Micro-Manager sidecar the chain returned 1,
the focused GUI skipped the channel-chooser dialog, and
`load_video`'s `arr.mean(axis=-1)` collapsed the wrong axis on
`(N, C, H, W)` arrays, leaving the viewer with a `(N, C, H)` "stack"
that rendered as a thin horizontal line.

Added a `tifffile.series.axes` lookup in `detect_channels` (before the
OME-XML fallback): when the series axes string contains `"C"`, that
dimension's size IS the channel count, read straight from the OME-TIFF
structure. Verified: IC295 `.ome.tif` returns 3 with and without
`_metadata.txt`; single-channel example `.tif`s (`axes="QYX"`) still
return 1.

Deliberately did NOT use `.ome.json::n_channels` as a fallback:
IC295 sidecars declare 2 (Cy5+DIC) but the physical file has 3
pages-per-frame (Cy5/DIC/blank), so `.ome.json` would misalign the
loader's page stride.

Discovered after the `gt_review/` cleanup exposed it (recordings
there had `.ome.json` but no `_metadata.txt` symlink); pre-existing
bug, not a regression. `gt_review/` also got `_metadata.txt` symlinks
to the drive originals as belt-and-suspenders.

## 2026-05-29 — Drag-drop + menu loader for masks.npz files

Focused GUI: dropping a `.npz` file onto the window now loads it as
pipeline results (parent dir → `on_load_pipeline_results`), alongside
the existing recording (`.mp4/.avi/.mov/.tif`) and project
(`.cellscope`) drop targets. Companion `File → Open Masks File…` menu
item picks the `.npz` directly (the existing `Open Pipeline Results…`
picks a folder). Both routes funnel into the same loader, so
`RUN_METADATA`-driven recording resolution + the "discard unsaved
results?" confirmation work the same as before — handy for the new
`gt_review/<rec>/pipeline_results/masks.npz` files that have no
metadata sidecar (the loader will prompt for the recording).

## 2026-05-29 — Clean data/ to GT/examples/models only; results → local gt_review/

Decluttered `data/`: removed all pipeline-run + evaluation artifacts
(`pipeline_results/`, suffixed `pipeline_results_*/`, `evaluation/`,
`evaluation_old_cpsam_dic/`, `channel_alignment/`, per-recording
`*_masks.npz` + `*.cellscope`, and top-level `gt_evaluation_summary.md`)
across `ic295_gt_full` + `legacy_gt`. 196 tracked files removed from git.

`data/` now holds only GT (`gt_masks/`, `GT_FRAMES.txt`, recordings +
sidecars), the GT registry (`GT_INDEX.md`, `gt_index.json`), examples,
and models — plus the drive convenience symlinks.

New **local, gitignored `gt_review/`** aggregates everything for viewing
+ analysis in CellScope: per recording, the source recording (symlink to
the drive) + sidecar + `pipeline_results/masks.npz` (symlink to the mini's
run on `GeorgeDrive/ignasi/IC295{,_batch2}/processed/<label>/`) + the
moved `evaluation/` reports. 10 IC295 GT + 3 legacy recordings.

GT was backed up first (`data/gt_backups/gt_2026-05-29_164049.tar.gz`,
328 files) per the CLAUDE.md GT-protection rule; a scripted dry-run +
safety assertion confirmed no `gt_masks/`/recording/sidecar was ever in
the delete set. The removed results remain recoverable from git history
and (for masks) from the drive.

## 2026-05-28 — Pre-download raw cpsam ViT in download_models.py

`download_models.py` previously fetched only `cpsam_dic` (Drive) +
the small-models bundle. The **raw default cpsam ViT** (cellpose
builtin, `~/.cellpose/models/cpsam`, ~1.1 GB) — used by the
auto-select probe and raw-cpsam detection — was left to cellpose to
download silently on first inference, with no GUI progress bar. On a
fresh lab-PC install that first-use download looked exactly like a
hang (compounded by CPU-only inference; see below).

Fix: new `fetch_cpsam_builtin()` calls cellpose's own
`models.cache_CPSAM_model_path()` (CP4-only; fetches from HuggingFace
with a progress bar, idempotent) so the version + cache path always
match what cellpose expects — no Drive re-hosting. Wired into `main()`
under `do_cpsam`, surfaced in `report_status()` / `--check-only`, and
gracefully skips with a hint when run from the CP3 `cellpose` env.
Also fixed two stale `conda activate cellpose` → `cellpose4` env
references in the script (cellpose4 is canonical).

Context: diagnosed why the lab PC's "Test on frame" appeared to hang.
Root cause is CPU-only inference (GPU was admin-locked at install):
benchmarked cpsam on a 1024² frame — **CPU 297s/frame (922s w/ TTA)
vs MPS 6.8s (21s): ~44× slower**, effectively non-functional
interactively, not a bug. A multi-cell Test-on-frame compounds it
(separate `conda run` probe subprocess + its own CPU inference, then
in-process detection inference). Fix for the lab PC is enabling the
GPU; the silent raw-cpsam download was a second, independent hang
contributor that this commit removes.

## 2026-05-27 — HTTP remote-control RPC for all 6 GUIs

Commits `000860b`, `928e2e2`, `b1764a1`. New `gui_focused/remote_control.py`
exposes a stdlib HTTP server (BaseHTTPRequestHandler) inside the Qt event
loop. `CELLSCOPE_REMOTE=<port>` env var enables it on launch; commands are
dispatched cross-thread via `pyqtSignal(dict, object)` so handlers run on
the GUI thread.

Coverage:
- `main_focused.py` (port 8765) — full handler set: load_recording,
  load_pipeline_results, load_project, clear_all, set_param, set_frame,
  set_view, set_mode, detect, test_frame, analyze, save_screenshot,
  save_project, export
- `main_editor.py` (8767), `main_batch.py` (8766), `main_training.py`
  (8769), `main_tracking.py` (8770), `main_suite.py` (8771): minimal
  `attach_minimal()` wiring — status + log endpoints. Suite is tkinter,
  not Qt; its server runs in a daemon thread.

Purpose: lets external scripts and agents drive the GUI for automated
testing, regression checks, and reproducible screenshots without
QTest-style internal widget manipulation. Used to exercise every endpoint
during the deployment readiness check.

Key implementation note: `RemoteControlServer(QObject)` tolerates
non-QObject parents via `super().__init__(parent if isinstance(parent,
QObject) else None)` — `EditorWindow` is a QMainWindow but some test
fixtures construct lighter parents.

## 2026-05-27 — Hungarian tracker: IoU + area in cost matrix

Commit `60ea19c`. `core/multi_cell.track_all_cells` cost matrix now
combines:
- `w_dist · raw_distance` (always; default 1.0)
- `w_iou · (1 - iou) · max_hop_px` (default 0.5 — strongest non-distance
  signal; mask overlap is highly discriminative for touching cells)
- `w_area · |Δarea|/baseline · max_hop_px` (default 0.3 — penalises
  identity-swap candidates with very different cell size)

New `DEFAULTS.track_w_dist / track_w_iou / track_w_area` plumbed through
`unified_detection` → `hybrid_*_multi` → `track_all_cells`. Validation on
Pos7-WT GT (mini-side): **ID consistency 0.88 → 0.97** with no change to
DET / SEG / IoU. No GUI changes — defaults work out of the box.

The mini's diagnosis was that bouncing was sustained drift, not single-
frame bounce, so `smooth_bouncing_ids` (commit `225064c`, wired into
`postprocess_tracks`) helped some but not enough; the cost-matrix change
was the structural fix.

## 2026-05-27 — Cluster of bug fixes uncovered by the mini IC295 batch run

Commits `bb84460`, `971c62f`, `b7aca1d`, `4f1c374`, `f0ff603`, `b1764a1`.
The mini's first GT batch attempt crashed in several places that the local
single-recording dev workflow never touched.

- `parse_n_channels` silently defaulted to 1 channel when
  `_metadata.txt` was absent; tifffile then crashed several layers deep on
  3-channel TIFFs. Fixed in `core/multichannel.py` with a fallback chain:
  metadata.txt → `.ome.json` `n_channels` (authoritative) → OME-XML SizeC
  (heuristic).
- `save_unfiltered_detections` crashed with shape mismatch when
  downsample > 1: `tracks_raw` is a mixed-shape list (kept references
  stay upscaled, dropped copies stay downsampled). First two attempts
  (`4f1c374`, `b7aca1d`) chased the wrong list; root fix iterates
  `tracks_raw + tracks` with `id()` dedupe.
- `Downsample` dropdown launch crash from `UnboundLocalError`: a local
  `from PyQt5.QtWidgets import QComboBox` inside `_build_detection_page`
  shadowed the module-level import. Removed.
- Test-on-frame button stayed greyed after loading a recording.
- Two-cell fusion artifact (commit `ba858a1`): gap-fill Phase 4 was
  blindly translating the last-known mask forward into frames where it
  overlapped another track. Added collision rejection. Companion fix:
  `DOWNSAMPLE_SMALL_PX` 900 → 1100 so 1024² recordings stay at 1× by
  default — cpsam's merging bias on the downsampled version was a
  contributor.

## 2026-05-27 — Other quality-of-life landings

Mixed commit cluster from the deployment-readiness cycle.

- `35e4b53` — screenshot menu actions; `search_radius` wired through
  to multi-cell tracker.
- `b8519b7` — 🔬 Test on frame predicts Detect path; warns when current
  mode (single/multi) doesn't match recording density.
- `7795409` — `Open Pipeline Results…` loader in the focused GUI loads
  existing batch outputs (masks.npz + run metadata) without re-running.
- `3952cd3` — persists pre-Cy5-filter detections to disk and renders
  dropped cells in the viewer for audit.
- `24d15fe` — close / clear / reload warn when results are unsaved.
- `0c0d939` — DeepSea per-cell refinement: forbidden_mask parameter
  prevents a cell expanding into its neighbour's pixels.
- `5a59f88` — DIC↔Cy5 fusion no longer carves a fragment out of the
  DIC label when the Cy5 mask overlaps; it absorbs into the existing
  DIC label instead. Removed the spurious fragment class that surfaced
  in Pos68_DMSO.
- `391b7ea` — merge touching split fragments to fix cpsam
  over-segmentation (`core/detection_postprocess.merge_touching_splits`).
- `85b83e1` — `reject_static_edge_blob_tracks` in
  `core/track_postprocess.py` kills vignette / illumination artefacts
  near FoV edges that pass other filters.
- `ece9780` — `main_suite._find_conda` walks install roots and prefers
  one with `envs/cellpose4`. Fixes mini deployments with both
  `~/anaconda3` and `~/miniconda3` where `cellpose4` only exists in the
  latter.
- `1f56774` — IC295 batch aggregation + genotype-comparison scripts
  for the mini run.
- `714e803` — `channel_alignment` cpsam-pair subprocess timeout
  600s → 1800s for 2048² recordings.

## 2026-05-25 — Doc audit across all 5 top-level docs

Synced README, INTERFACE, PROJECT_STATUS, INSTALLATION, CLAUDE with
the persistence_guard v2 + 17-param plumbing state. Commits
`4b2c395` → `3303174`.

Key fixes:
- README pipeline diagram had structural error: Cy5 filter shown
  BEFORE tracking, but actually runs AFTER (inside the post-detection
  block in `unified_detection.py`). Reordered + added missing
  `POST-PROCESS TRACKS` stage + mirror padding bullet in DETECTION.
- INSTALLATION had **backwards** env recommendation: said "always run
  from cellpose env, never cellpose4" — but cellpose4 is canonical
  (cpsam_dic + raw cpsam both need CP4). Fresh installs following the
  doc verbatim would have launched into the wrong env. 8 references
  fixed.
- CLAUDE.md doc'd Phase-2 gap fill running in `cellpose4` env; it
  actually runs in `cellpose` (CP3 models). Would have misled
  future agents debugging gap fill.
- 4 stale probe stats in CLAUDE.md: "samples 5" → 11; "median" →
  p75; "cpsam_dic probe" → raw cpsam; "multi_metric default" →
  persistence_guard.
- GT aggregate table in README refreshed against post-v2
  pipeline_results: F1_focused 0.836 → 0.874 corpus-wide.

## 2026-05-25 — persistence_guard v2 + 17-param GUI plumbing + 🔬 Test on frame

Commit `b11196e`. 90 files, +1871 / −560 lines.

**Cy5 filter swap** — replaced strict `multi_metric` default with new
`persistence_guard` v2 (3-stage rule: mm-pass OR long+moving).
13-recording GT audit revealed the old filter was dropping 70 of 989
real cells (-7.1% recall), catastrophic on weak-Cy5 conditions
(Pos31_GOF -50%, Pos44_OT -23%, Pos68_DMSO -22%). v2 keeps tracks
that pass ≥2/3 multi-metric OR are long-lived (≥35 frames) AND
moving (vel<3 px/frame fails when median consec mask IoU>0.85 ↔
static-phantom signature). On the full corpus: F1 0.783→0.802
(+0.019), F1_focused 0.836→0.874 (+0.038). Pos51_Y1 fully recovered
to canon (the static gate kills its persistent vignette phantom).
No per-recording overrides needed.

**GUI plumbing audit** — 17 detection params now reach
`detect_recording` end-to-end. Found + fixed 3 latent bugs where
GUI silently dropped user values onto hardcoded `DEFAULTS`:
- `detect_recording` hardcoded `use_deepsea` / `use_gap_fill` /
  `use_tta` / `use_cpsam_cy5_union` / `use_fallback` / `use_mirror_pad`
- `min_track_length` hardcoded to 3 in both `hybrid_cpsam_multi`
  and `hybrid_dic_multi` (the "Min track length" GUI control was
  silently no-op for years)
- `postprocess_tracks` hardcoded `min_frames=3`, dropping all 1-frame
  tracks — only surfaced when the new test-frame feature tried to
  preview a single frame

**🔬 Test on frame** — new toolbar button that runs detection on
the currently displayed frame with current GUI parameters, times it,
reports a density-aware extrapolation to full-recording runtime
(sparse 1.5× / medium 2.0× / dense 2.5× post-proc multiplier).
Validated on Pos10_WT (sparse, 3 cells in 28s → 1.1h est) and
Pos68_DMSO (dense, 14 cells in 40s → 2.7h est). The density
multipliers were calibrated against the canonical full-pipeline
runtimes of those recordings.

**Data**: on-disk `pipeline_results/masks.npz` for all 10 Cy5 GT
recordings rebuilt offline with v2 (previous saved as
`masks_pre_v2.npz` alongside).

## 2026-05-23 — Method D (cpsam-on-Cy5 union) flipped off

Commit `dfae5cb`. Bbox-detection validation showed +0.11-0.26 F1 on
dense recordings (Pos68_DMSO, Pos31_GOF), but full-pipeline re-run
showed those gains evaporated — the downstream multi-metric Cy5
filter was rejecting the Cy5-side detections because their signal
characteristics didn't match what the filter expected. Default OFF
until the filter is tuned (resolved 2 days later by replacing the
filter entirely with persistence_guard — see 2026-05-25).

Cost when enabled: ~1.5× runtime. Code retained behind
`use_cpsam_cy5_union=True` flag for future experiments.

## 2026-05-22 — Method D added (cpsam-on-Cy5 union)

Commit `d340c37`. Added optional stage: run cpsam directly on the Cy5
channel as a parallel detection, union-merge with DIC-detected cells
via NMS. Aimed at recovering cells dim in DIC but visible in
fluorescence. Wired through `core/hybrid_cpsam_multi.py`. Shipped
default-ON; reverted next day (see above).

## 2026-05-21 — 4 new GT recordings + GT-focused F1 + Pos39_OT override + mirror-pad rollout

Commits `300776e` → `d8dbd20`. Most productive day of the cycle.

- **Mirror padding rollout**: re-ran all 9 GT recordings with the
  new `use_mirror_pad="auto"` (enabled when min detection-dim ≥
  1024 px). Aggregate IoU 0.847 → 0.858 (+0.011). Per-recording IoU
  improved on all 6 IC295 + all 3 legacy ignasi.
- **GT-focused F1 metric** (`scripts/evaluate_against_gt.py`):
  excludes predictions with zero IoU vs any GT from the FP count —
  the predictions that are real cells in the field but just weren't
  annotated. Fixes the apparent regression on ignasi recordings
  (single-cell GT but multi-cell field): F1 0.63/0.66 → focused
  1.00/1.00.
- **Pos39_OT per-recording override**: sidecar JSON sets
  `"use_mirror_pad": "off"` because padding causes cpsam to merge
  the recording's dividing cell pair (~F73-F80) into a single mass.
  Pad-off restores the division catch (T7→T9 F75 ✓) and is strictly
  better on this recording (+0.005 F1, +4pp ID, division caught) at
  -0.008 IoU. Pattern reusable for any recording where padding
  shows similar trade-offs.
- **4 new GT recordings**: Pos10_WT, Pos21_KO, Pos31_GOF, Pos44_OT
  (10 frames each). Brings corpus to 12 then 13 recordings.

## 2026-05-20 — Auto-select probe rewrite + multi-track daughter detection + Pos68_DMSO GT

Commits `5e7b718` → `9a9c47c`. The day's theme was getting the
density probe + division annotator to handle real multi-cell scenes.

- **Probe model**: cpsam_dic → raw cpsam. cpsam_dic was probing 9-14
  cells/frame recordings as single-cell because of its merging bias.
- **Probe sample count**: 5 → 11 frames.
- **Probe aggregator**: median → p75. Empty frames at the start of a
  recording were dragging multi-cell counts under threshold.
- **Multi-track division annotator**: replaced the literal "track
  first spawned this frame" check with "track first comes near the
  parent's split centroid this frame, having not been near it
  before". Handles Hungarian-tracked daughters that inherit an
  existing track ID at the split (a common pattern under raw cpsam
  multi-cell output that the legacy first-frame-only check missed).
  Restored Pos51_Y1's division catch lost in the routing change.
- Edge-sliver filter (`core/track_postprocess.py::reject_edge_sliver_detections`)
  zeroes vignette-bar mis-detections at the FoV edge.
- Mask editor: palette + spinbox cap extended to 30 cell IDs;
  brush rendering changed to O(1) overlay + defer contours mid-stroke
  (was unbearable above ~15 cells).
- Pos68_DMSO GT added (11 frames — the densest IC295 recording).

## 2026-05-18 — Division annotator end-to-end + Pos51_Y1 GT

Commits `0478733` → `5c37b1f`. Cell-division annotator
(`core/division_annotator.py`) wired into `hybrid_cpsam_multi` +
`hybrid_dic`. Writes `divisions.json` sidecar containing
candidates + a `track_lineage` table. GUIs surface lineage in
per-cell views; export writes lineage to the standard outputs.

Catches **2 / 2 GT divisions** in the corpus (Pos39_OT, Pos51_Y1).
Rejection visualisation added (renders 9-frame strip PNGs per
candidate with reasons for rejected near-misses under
`results/divisions/<recording>/`).

## Older (pre-cycle) — see git log

Major prior milestones (pre-2026-05-15):
- 13-recording GT corpus assembled (IC295 + legacy ignasi).
- cpsam_dic v2 fine-tune (1,000 DIC pairs, Colab) — 0.826 in-domain,
  0.754 OOD on the cropped split.
- Multichannel pipeline (DIC + Cy5 fusion + recovery + Tier-4 filter)
  built and unit-tested.
- 4-phase track gap-fill cascade (cpsam-augment → CP3 + MedSAM +
  DeepSea → SAM2 video propagation → translation-only).
- VAMPIRE shape modes + cell-state classification + per-state
  motility analytics.
- 5 specialised GUIs (focused / batch / tracking / editor / training)
  + unified suite launcher.

`git log --oneline --since="2026-04-01" -- core/ gui_focused/` for
the full timeline.
