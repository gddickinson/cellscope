# CellScope — agent notes

## Ground-truth data — DO NOT DELETE

All hand-labelled GT lives under `data/ic295_gt_full/` and
`data/legacy_gt/`. **`data/GT_INDEX.md` is the canonical registry** —
re-generate it any time you add/remove a GT folder by running
`python scripts/audit_gt.py`. The audit script also writes a
machine-readable `data/gt_index.json`.

**Periodic backup**: `python scripts/backup_gt.py` writes a tarball
to `data/gt_backups/gt_<YYYY-MM-DD>_<HHMMSS>.tar.gz` containing
every mask PNG + the index. Run this whenever new GT lands.

**Do not delete** any folder containing `gt_masks/*.png` without
explicit user confirmation. The PNGs are real files (not symlinks)
and easily worth several days of manual labelling work each.

## Two rules you MUST follow

### 1. Pipeline defaults live in ONE place

All defaults for **every CellScope GUI** (`gui_focused/`, `gui_batch/`,
`gui_tracking/`) and the multichannel pipeline (`hybrid_dic.py`,
`hybrid_cpsam_multi.py`, `dic_cy5_fusion.py`) come from
**`core/pipeline_defaults.py::DEFAULTS`**. State-classification
thresholds come from **`core/cell_state.py::DEFAULT_THRESHOLDS`** —
both GUIs that expose them (`gui_focused/params_panel.py`,
`gui_batch/batch_window.py`) read from there.

> **State rule (learned, 2026-06-08).** `rounded` is now decided by a
> rule fit to the hand labels: `area_um2 ≤ rounded_area_um2` (960) AND
> `eccentricity ≤ rounded_eccentricity` (0.85) — size/footprint, not
> circularity (0.90 CV vs the old circ/solid rule's 0.60; see
> `scripts/ic295_eval_state_rule.py`). It needs the recording's
> `um_per_px`, so `classify_state` / `classify_track_states` take a
> `um_per_px=` arg (threaded from `state_analysis`, `mask_metrics`,
> `gui_batch/batch_worker`). With no scale they fall back to the legacy
> `rounded_circ`/`rounded_solid` gate. The GUI threshold panels still
> expose the fallback circ/solid widgets — **TODO: surface
> `rounded_area_um2` + `rounded_eccentricity` there too** (analysis
> already uses them from `DEFAULT_THRESHOLDS`). Re-fit the thresholds
> from new labels with `ic295_eval_state_rule.py` (prints the values).

- `core/pipeline_defaults.py::DEFAULTS` is the canonical source for
  detection / tracking / refinement / Cy5 / VAMPIRE defaults.
- All widget initial values (every `setValue` / `setChecked` on a
  param widget) MUST come from `DEFAULTS.<field>`. **Do not hardcode
  values** — change `DEFAULTS` and the widgets track it.
- All `params.get("key", FALLBACK)` calls in workers MUST use
  `DEFAULTS.<field>` as the fallback (not bare literals). This
  prevents drift between the panel's initial value and the worker's
  behaviour when a key is missing from the params dict (e.g. when an
  older saved project is loaded or a script forgets to pass a key).
- Pipeline functions accept overrides but default to `None` and fall
  back to `DEFAULTS` internally. **Do not pass parameter overrides in
  scripts unless you have a documented reason** — that's how the
  May-2026 GUI-vs-script mismatch happened (`min_area_px=200` vs GUI
  `=500`, hardcoded `cellpose_dic` vs GUI `cpsam_dic`).
- For recording-aware physical-unit thresholds, call
  `DEFAULTS.pixel_thresholds(um_per_px, time_interval_min)` — the
  tracking GUI's `_on_track` uses this so tracking parameters auto-
  scale with the recording's pixel size.
- The legacy `config.py` at the project root still serves the older
  pipeline (`core/auto_params.py`, `core/refinement.py`, etc.). Leave
  it alone unless you're editing legacy refinement.

When proposing a default change:
- Edit only `core/pipeline_defaults.py`.
- Smoke-test the GUI (`scripts/test_focused_gui.py`) before claiming
  the change works.
- Note the change in `SESSION_LOG.md`.

### 2. Every analysis run MUST write RUN_METADATA.{md,json}

`core/run_metadata.py::write_run_metadata(out_dir, ...)` writes both
a human-readable `RUN_METADATA.md` and a machine-readable
`RUN_METADATA.json` containing:

- source recording path + checksum + n_frames + um_per_px
- pipeline function name + mode
- **all** params used + a diff against `DEFAULTS` (only the keys that
  deviated from defaults)
- env info: conda env, python, cellpose, numpy, scipy, skimage,
  tifffile, torch versions
- git commit hash if the cellscope repo is git-tracked
- timestamp_started / timestamp_finished / runtime_seconds
- exact shell command to reproduce the run

**Required at every analysis entry point**:
- `gui_focused/export_dialog.py::_on_export` — always writes, even
  if no other checkboxes are ticked
- `gui_batch/batch_worker.py` — should write one per recording in
  the batch loop
- Any new `scripts/*.py` that runs detection or analysis — must call
  `write_run_metadata` before exiting

When you add a new analysis script: add the call. Don't ship without
it. Reviewers should reject PRs that produce results without metadata.

If you find existing results directories without `RUN_METADATA.md`,
they were produced before this requirement landed (May 2026) or by a
broken path — flag them and treat the params as unknown.

---

## File-size policy
Keep every Python file under 500 lines (project-wide rule from
`/Users/george/.claude/CLAUDE.md`). If a refactor would push a file
past 500, split first.

## INTERFACE.md
`@INTERFACE.md` is the navigation map. Read it before opening source
files. Update it when you add or rename modules.

## SESSION_LOG.md
Append a short entry whenever you make a non-trivial change. Source
of project memory; treated as authoritative for "why does X work
this way?".

## Environments
- `cellpose` — torch 2.7, cellpose 3.1.1.1, vampire-analysis 0.0.1,
  sam2 (any version), hosts CP3 models.
- `cellpose4` — clone of `cellpose` with cellpose 4.1.1, hosts cpsam
  + cpsam_dic (CP4 ViT). Default env for evaluation runs because the
  auto-selector may pick either backbone.

**Run pipeline scripts from `cellpose4`** by default. Both
`hybrid_dic_multi` (cpsam_dic) and `hybrid_cpsam_multi` (raw cpsam)
load CP4 ViT models natively there. CP3 fallback steps subprocess
out to the `cellpose` env automatically.

## Detection backbone — auto-selected per recording

For multi-cell scenes, **`cpsam_dic` (the DIC fine-tune) merges
touching cells**. Diagnosed on IC293 Pos3 (3 cells/frame): cpsam_dic
F1 0.65, raw cpsam F1 0.90 — same recording, same defaults, only the
backbone differs.

`core/pipeline_defaults.py::select_pipeline_for_recording(frames)`
samples **11 evenly-spaced frames** (`N_SAMPLE_FRAMES_FOR_AUTO`),
runs **raw cpsam** (no fine-tune bias — switched 2026-05-20 because
cpsam_dic was probing a 9-14-cells/frame recording as single-cell
and mis-routing it), and returns:

- `('cpsam_dic', model_path, info)` when **p75 cell count** < 1.5 →
  cleaner boundaries on isolated cells
- `('cpsam', None, info)` when p75 ≥ 1.5 → handles touching cells

p75 instead of median means a few empty frames at the start of a
recording don't pull a multi-cell scene's count below threshold.

`scripts/run_pipeline_on_gt_recording.py` calls this automatically.
The choice + sample counts are recorded in `RUN_METADATA.json` under
`extra.auto_select` so any evaluation can be reproduced exactly.

## Unified detection path (GUI ↔ script)

`core/unified_detection.detect_recording(...)` is the **single
canonical detection function** used by:

- `gui_focused/workers.py::FocusedDetectWorker` when `mode == "auto"`
  (the default the main_window picks for any DIC multi-cell recording)
- `scripts/run_pipeline_on_gt_recording.py` for batch / evaluation
  runs

It performs the full chain in fixed order:

1. measure + apply DIC↔Cy5 alignment (when multichannel)
2. resolve downsample factor + downsample stacks
3. convert physical-unit thresholds to per-recording px
4. auto-select cpsam_dic vs raw cpsam by sampled cell density
5. run the chosen detection pipeline (which internally does:
   detection → tracking → 4-phase gap fill → post-process tracks)
6. annotate Cy5 metrics + apply **`persistence_guard`** filter (default;
   was `multi_metric` before 2026-05-25) — multichannel only
7. upscale labels back to original resolution

**`detect_recording` accepts 17 explicit kwargs** for fine-grained
overrides (use_deepsea, use_gap_fill, use_sam2_video_gap_fill,
max_gap_frames, min_track_length, use_tta, use_cpsam_cy5_union,
use_fallback, use_mirror_pad, use_preprocess, use_retry,
cy5_filter_mode, cy5_filter_threshold, cy5_pg_min_lifetime,
cy5_pg_static_velocity_px, cy5_pg_static_shape_iou,
cy5_fusion_jaccard_thresh, cy5_fusion_max_overlap_frac,
cy5_fusion_augment_cpsam, use_bfloat16). All default to `None` → fall
back to `DEFAULTS`. The focused GUI's params panel exposes all 18.
`use_bfloat16` (DEFAULTS True = bf16, cellpose default) is opt-in: set
False (`--fp32` / GUI uncheck) for fp32 — ~1.26× faster on Apple
Silicon (MPS, no effect on CUDA), GT-validated as a per-frame wash but
not yet F1-certified, so the default stays bf16. Affects raw-cpsam
detection + gap-fill only (the cpsam_dic subprocess stays bf16).

### 🔬 Single-cell curation (opt-in, step 7.5 of detect_recording)

For recordings hand-cropped to ONE target cell under known priors
(isolated, flattened, non-dividing, present every frame, roughly
centred), `detect_recording` runs `core/single_cell_curation.py::
curate_single_cell` AFTER detection (at working resolution, before the
upscale) to **assemble that one cell's track label-agnostically**:
texture-based rejection of debris + uniform optical shadows (in-mask DIC
std), ID-stitching across tracker label switches, SAM2 recovery of
missing frames, and tracking through rounded (balled-up) states. It
FLAGS (never drops) exceptions — division / two persistent cells /
unrecoverable frames — in `result["curation"]` + `auto.single_cell_
curation`. Controlled by 6 kwargs (`single_cell_curation` +
`sc_present_every_frame` / `sc_no_dividing` / `sc_isolated` /
`sc_roughly_centered` / `sc_expected_cell_area_um2`), all defaulting to
None → `DEFAULTS`. **`DEFAULTS.single_cell_curation` is False, so
ordinary multi-cell detection is completely unchanged.** The focused
GUI's params panel exposes all six (Detection tab, "Single-cell
curation" group); the 1-frame **Test on frame** preview forces it OFF
(it's a multi-frame operation). Built + validated on the IC293
EC-migration crops (2026-06-17; see `core/single_cell_curation.py`,
`ic293_analysis/CURATION_DECISIONS.md`).

**Any change to detection defaults belongs in `detect_recording` or
its dependencies (`pipeline_defaults`, `channel_alignment`,
`hybrid_dic`, `hybrid_cpsam_multi`, `cy5_filter`).** Don't fork the
logic into script-only or GUI-only paths.

Launch the GUI from the **`cellpose4`** env — both `cpsam_dic`
(CP4 ViT fine-tune) and raw cpsam need cellpose 4.x. The runner
script needs cellpose4 for the same reason.

```bash
conda run -n cellpose4 python main_focused.py
```

### 🔬 Test on frame — preview detection on the displayed frame

The focused GUI has a toolbar button (between Cancel and Undo Detect)
that runs detection on the currently displayed frame with current
GUI parameters, times it, and reports a density-aware extrapolation
to full-recording runtime (sparse 1.5× / medium 2.0× / dense 2.5×
post-proc multiplier). **Use this when validating parameter changes
— skip the 1-3 hour full-recording detect cycle.**

The handler (`gui_focused/main_window.py::_on_test_frame`) forces
`min_track_length=1` and skips multi-frame stages (Cy5 filter, gap
fill) since they're not meaningful on 1 frame. When adding new
detection-related kwargs, remember to forward them in this handler
too — otherwise the preview will silently use defaults while the
production `_on_detect` honours the GUI's value.

## Auto-downsample

`run_pipeline_on_gt_recording.py` takes `--downsample auto` (default),
`--downsample N` (integer ≥ 1), or `--downsample off`.

`core/pipeline_defaults.resolve_downsample(spec, frame_shape)` is the
single decision point. Auto mode uses `max(H, W)` of the recording:

| max(H, W) | Factor | Rationale |
|---|:---:|---|
| < 1100 | 1 (no change) | cells get too small for cpsam at ds≥2 (Pos0 F1 0.92 → 0.82 was the lesson). Threshold raised from 900 → **`DOWNSAMPLE_SMALL_PX = 1100`** so 1024² recordings keep full res |
| 1100–1500 | 2 | medium recordings get ~3× speedup, ~3% IoU cost (Pos3) |
| ≥ 1500 | 2 | large recordings get ~5× speedup with NO IoU cost (Pos7_WT actually IMPROVED with ds=2) |

Tunable per-run via `--downsample-small-px` / `--downsample-large-px`.

## Physical-unit thresholds

`PipelineDefaults` exposes both pixel (`min_area_px=200`) and
physical-unit (`min_area_um2=85.0`) defaults. Call
`DEFAULTS.pixel_thresholds(um_per_px=..., time_interval_min=...)` to
get the per-recording px values; falls back to raw px defaults when
scale is unknown. The runner does this conversion automatically using
the recording's `.ome.json` sidecar metadata.

Important physical defaults:

| Default | Value | Why |
|---|---|---|
| `min_area_um2` | 85 µm² | catches rounded endothelial cells (~80-150 µm²) |
| `expected_cell_diameter_um` | 30 µm | typical spread endothelial cell; passed to cellpose for cross-scope robustness |
| `max_hop_um_per_min` | 15 µm/min | upper bound on endothelial cell motion |
| `search_radius_um` | 100 µm | tracker gap-fill window |

For a hypothetical 0.4 µm/px microscope, the converter gives
`min_area_px=531, cell_diameter_px=75` — automatically scaled.

## Edge-truncated cells — excluded from shape/state, kept for counts

A cell whose mask reaches the image border is only partially in view,
so its outline and every shape metric (area, circularity, solidity, …)
and the rounded/spread state derived from them are unreliable. Such a
cell-frame is **excluded from SHAPE + STATE analysis but still counted
and still tracked** (its centroid anchors identity).

- `core/edge_filter.py::mask_touches_edge(mask, margin)` is the single
  primitive. **Only ever call it on a FULL-FRAME mask** — a bbox crop
  touches its own border, so every cell would read as edge. Use
  `bbox_touches_edge(...)` when you already cropped but kept the
  full-frame bbox + frame size.
- The chokepoint is `core/cell_state.py::shape_metrics_for_mask`
  (sets `edge_touch`) → `classify_state` voids the frame to `unknown`.
  Because every per-state aggregation keys off state, this makes shape
  means / `frac_rounded` / per-state speed exclude edge frames with no
  extra filtering. **Don't add a parallel edge check elsewhere** —
  route new shape code through `shape_metrics_for_mask`.
- Margin lives in `core/cell_state.py::DEFAULT_THRESHOLDS["edge_margin_px"]`
  (rule 1; default 0 = literal border contact). It flows into
  RUN_METADATA's defaults-diff automatically (rule 2).
- Per-cell bookkeeping (in `state_analysis.annotate_state` →
  `per_cell.csv`): `n_frames_edge`, `n_frames_classified`,
  `frac_in_view`. A cell never cleanly in view gets
  `frac_rounded/spread = None` (so it can't bias means). Drop such
  cells from pooled comparisons with `ic295_compare_pooled.py
  --min-valid-frames N`.
- GUI overlay state code **3** = edge (amber, `gui/metric_coloring.py`).
- **Hand labels are exempt.** The annotation GUI flags edge crops
  (amber) but never discards a user's spread/rounded label — the
  filter gates only AUTO-computed shape/state.

## Remote control RPC

Every CellScope GUI exposes an HTTP RPC server when launched with
`CELLSCOPE_REMOTE=<port>` set in the environment. Use it for automated
testing, agent-driven workflows, and reproducible screenshots without
QTest-style internal widget manipulation.

Default port assignments (one per GUI so they can run concurrently):

| GUI | Port | Module |
|---|---:|---|
| `main_focused.py` | 8765 | `gui_focused/remote_control.py` (full handler set) |
| `main_batch.py` | 8766 | `attach_minimal()` |
| `main_editor.py` | 8767 | `attach_minimal()` |
| `main_training.py` | 8769 | `attach_minimal()` |
| `main_tracking.py` | 8770 | `attach_minimal()` |
| `main_suite.py` | 8771 | tkinter, daemon-thread server |
| `main_annotate.py` | 8772 | `attach_minimal()` |
| `main_review.py` | 8773 | `gui_review/` + `attach_minimal()` |

```bash
CELLSCOPE_REMOTE=8765 python main_focused.py &
curl -s http://127.0.0.1:8765/status | jq
curl -s -X POST http://127.0.0.1:8765/load_recording \
    -H 'Content-Type: application/json' \
    -d '{"path": "data/examples/test_small_multichannel/test_small_multichannel.ome.tif"}'
curl -s -X POST http://127.0.0.1:8765/detect
```

**Teardown — the RPC server holds the port until the process dies.**
A surviving GUI process keeps `LISTEN`-ing on its port, so the next
launch fails to bind, runs headless, and your `curl /status` is
silently answered by the **stale** process (old code, maybe a stale
recording loaded) — the smoke looks green but tested nothing. Tear down
**per-name** (`pkill -f` uses an *extended* regex, so `\|` alternation
matches NOTHING — one substring per call):

```bash
for s in main_focused.py main_batch.py main_editor.py \
         main_training.py main_tracking.py; do pkill -f "$s"; done
# verify the port is now served by a FRESH pid + fresh state:
lsof -nP -iTCP:8765 -sTCP:LISTEN          # who owns 8765?
curl -s http://127.0.0.1:8765/status      # focused should show recording_loaded:false
```

If a just-launched focused GUI reports a recording already loaded, you
are hitting a zombie — kill it and relaunch.

Focused GUI endpoints: `status`, `log`, `load_recording`,
`load_pipeline_results`, `load_project`, `clear_all`, `set_param`,
`set_frame`, `set_view`, `set_mode`, `detect`, `test_frame`, `analyze`,
`save_screenshot`, `save_project`, `export`.

Implementation: stdlib `BaseHTTPRequestHandler` inside the Qt event loop.
Cross-thread dispatch via `pyqtSignal(dict, object)` so handlers run on
the GUI thread. `RemoteControlServer(QObject)` tolerates non-QObject
parents (cells through `super().__init__(parent if isinstance(parent,
QObject) else None)`).

When adding a new GUI feature that needs scripted testing, add an
RPC endpoint alongside the user-facing action and document it here.

## IC295 batch (detect → review → analyze → compare)

`scripts/ic295_*.py` runs the 65-recording IC295 dataset end-to-end
with a **manual review checkpoint** between detection and analysis.
See `ic295_analysis/README.md` for usage. Folder layout
`ic295_analysis/by_condition/<cond>/<label>/` — each recording's
results sit in a self-contained folder with a `.cellscope` project
file that drag-loads in the focused GUI for mask editing.

Conventions to preserve when extending:

- Detection uses `core.unified_detection.detect_recording` with all
  kwargs defaulted (so they fall back to **`DEFAULTS`** — rule 1).
  Do not hardcode pipeline params in the IC295 scripts.
- Each recording's `pipeline_results/` gets a `RUN_METADATA.json`
  (rule 2). The detect script writes it; `ic295_analyze_one.py`
  refreshes with analysis timing.
- `pipeline_results/masks.npz` is a **real file** (copy from drive
  in the adopt path, or written fresh by detection) — never a
  symlink — so user mask edits via the focused GUI don't overwrite
  the canonical drive run.
- The rounded/spread state cut is **learned from the hand labels**
  (`ic295_analysis/state_labels/labels.csv`) — see the "State rule"
  note under rule 1. Validate / re-fit with
  `scripts/ic295_eval_state_rule.py` (writes
  `compare/state_rule_validation/`); `ic295_state_diagnostic.py` +
  `ic295_state_features.py` give the all-mask-data views. **If you
  change `rounded_area_um2` / `rounded_eccentricity`, re-run
  `ic295_analyze_one` on every recording before re-comparing** — the
  comparison reads the per-recording `frac_rounded` / per-state means.
- **Experimental design — comparisons are arm-structured, not flat.**
  Two independent experiments, each with its own control: GENETIC
  (control **WT** → GOF, KO) and DRUG (vehicle **DMSO** → YODA1/Y1, OT),
  plus the VEHICLE check WT vs DMSO. Cross-arm contrasts (e.g. GOF vs
  OT) are meaningless. `scripts/ic295_compare_arms.py` is the
  design-correct primary (per-arm KW + within-arm Bonferroni); the flat
  `ic295_compare.py` all-pairwise output is exploratory only — don't
  quote its cross-arm pairwise p-values. The vehicle effect (WT vs
  DMSO) is significant on this corpus, so **drug effects are read vs
  DMSO, not WT**.
- **Motility/dispersal is its own pipeline, off a shared enriched track
  cache.** `ic295_track_data.py` collects per-cell tracks + per-frame
  state + per-frame local density once (`_track_cache.pkl`,
  `CACHE_VERSION`; `--rebuild` after re-detect/edit). `ic295_flower_plots`
  (flowers, MSD — always check the **median**, the mean MSD is outlier-
  driven here) and `ic295_motility_stats` (the design-correct test) both
  read it. **Same arm structure + recording-as-unit rule as the shape
  comparison** — `ic295_motility_stats` reuses `ic295_compare_arms`'s
  machinery, reduces each recording's full-duration cells to one value,
  and reports α / Fürth D,P (`ic295_motility_models`) plus explicit
  confounder controls (state via frac_spread, contact via density, and
  pseudoreplication via recording-OLS + a statsmodels LMM — **statsmodels
  is installed in `cellpose4`**). Finding (n=8): **no treatment changes
  any motility metric; the IC295 phenotype is in shape/state, not
  migration** — state + crowding dominate what motility variation exists.
  Don't re-quote raw ensemble-MSD rank order as a result.
- The driver's per-recording subprocess isolation, lock file,
  SIGTERM handler, and atomic `progress.json` are load-bearing for
  multi-day runs. Don't simplify them away.
- The condition-grouped folder layout
  (`by_condition/<cond>/<label>/`) is **for manual-review
  ergonomics** — the user navigates by treatment to spot artifacts
  in context. Don't flatten it.
- `ic295_analysis/` is **gitignored except its README** so the local
  state (symlinks to the drive, large CSVs, plots) never ships to git
  but the usage doc does.

## Track gap-fill cascade

> **🚩 FLAGGED FOR AUDIT (2026-05-31).** IC295 batch timing shows the
> cascade is the **dominant cost** of multi-cell detection: ~50 min
> base + **~17 min per cell** (fit on 7 completed real detections, R²
> very high). Pos2-WT (11 cells) took 234 min, nearly all of it
> per-cell cascade work. After the current batch finishes, profile
> each phase + ablate to determine which phases are pulling their
> weight — see Priority 0 in [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md)
> for the plan (instrumentation, per-phase yield, subprocess
> batching, confident-detection short-circuit, possible removal of
> low-yield phases). Target: ≥ 30 % wall-time reduction with no F1
> loss on the GT corpus.
>
> **UPDATE (2026-06-05).** Dominant cost found + fixed: Phase 1 ran
> full-frame `cpsam(augment=True)` (~76 s/gap on 2048²) just to find
> one cell near a known centroid. Now segments an adaptive **crop**
> around the interpolated centroid (`DEFAULTS.gap_fill_crop=True`,
> full-frame fallback on the rare crop-miss). GT benchmark
> (`scripts/bench_gap_fill.py`, reviewed masks as truth): **12.5×**
> faster (6.1 vs 76 s/gap), **0** good fills dropped, shared-fill IoU
> Δ −0.002. Per-phase timing now instrumented (`stats_out`). Revert
> via `gap_fill_crop=False` (GUI checkbox / `detect_recording` kwarg /
> `--no-gap-fill-crop`).
>
> **CAVEAT (end-to-end, 2026-06-05).** The 12.5× is at full 2048² res;
> production **downsamples to 1024²** where cpsam cost is NOT
> pixel-bound (ViT normalises to target diameter), so the measured
> end-to-end gain is small: Pos10-WT gap-fill 1.2× (46.7→39.8 min),
> total 1.09×. Crop stays ON (safe; *improves* Phase-1 fill rate).
>
> **`augment=False` (2026-06-05).** augment is a call count (not
> pixels), so it cuts cost even when the crop can't engage. GT
> benchmark at production 1024²: crop+noaug vs crop+aug **2.1×**, 0
> good fills dropped, IoU Δ −0.011. Shipped `DEFAULTS.gap_fill_augment
> =False` — Phase 1 now cascades **no-augment → augment-on-miss →
> full-frame-on-miss** (recall ≥ old path). Revert: `gap_fill_augment
> =True` (GUI "always augment" / `--gap-fill-always-augment`). Phase
> 2/3 ablation + confident-track short-circuit still open. Phase 2
> already batched.
>
> **Stringent re-validation (2026-06-06).** crop+noaug (the shipped
> default) re-benchmarked vs the original full+aug across **4
> recordings spanning density 2→19 cells/frame incl. the densest
> (Pos68-DMSO)** — **ALL PASS**: 0 good fills dropped on any, shared-
> fill IoU within ±0.02, ~16–19× Phase-1 speedup, equal-or-more gaps
> filled. The Pos7-only result generalises. Per-phase `{time_s,
> filled}` now persists to `RUN_METADATA.json` `extra.gap_fill_stats`
> (unblocks the ablation); Phase 1 also reuses the detection model
> (`fill_track_gaps(cpsam_model=...)`, GT-verified bit-identical).

When the Hungarian tracker assigns identity, a track may have internal
gaps (frames where detection failed). `core/track_gap_fill.fill_track_gaps`
runs a four-phase cascade to recover the cell:

1. **Phase 1 — cpsam(augment=True)**: re-run cpsam at the gap frame
   with 4-rotation TTA. Catches detections cpsam missed at the
   default orientation.
2. **Phase 2 — CP3 fallback**: cellpose+MedSAM+DeepSea via subprocess
   in the **`cellpose`** env (CP3 models). Different model family;
   catches different failures.
3. **Phase 3 — SAM2 video propagation**: use `core/sam2_video.py` to
   propagate the most recent flanking mask through the gap using
   SAM2's memory attention. **This is the key win for cells that
   biologically retract / dim too much for any single-frame
   detector.** Tagged in `track["sam2_propagated_frames"]`.
4. **Phase 4 — simple mask translation**: carry the last-known mask
   forward, translated to the interpolated centroid. Last resort.
   Tagged in `track["propagated_frames"]`.

Each phase only operates on gaps the previous phase left unfilled.
Tags let downstream analytics distinguish "real" detections from
propagated estimates if needed.

Defaults are configurable via `core/pipeline_defaults.py`:
- `max_gap_frames` (15) — how long the tracker keeps a track alive
  through a gap before declaring it dead.
- `use_sam2_video_gap_fill` (True) — enable Phase 3.
