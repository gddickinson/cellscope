# CellScope — Session Log

Chronological log of substantive changes + investigations. Per
`CLAUDE.md`, append a short entry whenever a non-trivial change
lands. Source of project memory; "why does X work this way?" should
have an answer here or in the linked commit.

Format: **DATE — short title** with bullets describing what changed
+ key numbers + link to driving commits/files. Most recent first.

---

## 2026-06-12 — Motility/dispersal: design-correct + confounder-aware stats

Prompted by an over-read of the ensemble MSD curves ("KO and OT reduce
dispersal vs their controls"). The mean MSD is outlier-driven on this
skewed data, so the rank order isn't trustworthy:

- **Median MSD** (`_ensemble_msd(stat="median")`, bootstrap CI) **inverts**
  the genetic arm — KO becomes the *highest* median, not lowest — and keeps
  OT lowest in the drug arm. So the KO read was a mean artifact; the OT
  read is the more defensible one. Both CIs overlap → neither significant.

Built the proper test stack (new modules, all from one enriched cache):
- **`ic295_track_data.py`** — shared per-cell collector + versioned cache
  (`CACHE_VERSION=2`), now carrying recording label + per-frame state +
  per-frame local density (neighbours within 100 µm, nearest-neighbour
  distance). `ic295_flower_plots` refactored onto it.
- **`ic295_motility_models.py`** — `msd_alpha` (log-log slope), `furth_fit`
  (2D persistent-random-walk → motility coeff D + persistence time P),
  `fit_lmm` (statsmodels MixedLM; **statsmodels installed into cellpose4**).
- **`ic295_motility_stats.py`** — (b) per-recording arm test (recording =
  unit, no pseudoreplication) over 11 metrics; confounders for STATE
  (paired spread-vs-rounded + frac_spread metric), CONTACT (speed-vs-
  density Spearman, paired isolated-vs-crowded, per-treatment density),
  and PSEUDOREPLICATION (recording-level OLS + cell-level LMM adjusting for
  state + density). Writes `compare/motility_stats/{REPORT.md,
  stats_arms_motility.json, speed_vs_density.png, plots_arms/}`.

**Result (n=8/condition): NO treatment changes any motility/dispersal
metric** — survives none of per-recording Bonferroni, OLS, or LMM. What IS
significant is confounders: **state dominates motility** (LMM frac_spread
p=0.001 genetic / <0.001 drug — the biggest single driver of dispersal),
and **weak contact inhibition** (speed vs crowding ρ=−0.12, p=0.020, but
*between-cell* — the within-cell paired isolated-vs-crowded test is ns).

Caveats recorded in REPORT.md: full-duration cohort censors the most motile
cells (leave FOV → dispersal conservative); the rounded-faster-than-spread
paired result is likely centroid noise on small rounded masks (LMM's
negative frac_spread coefficient is consistent with that); the drug-arm LMM
random effect is singular (use the OLS there). **Takeaway: the IC295
phenotype is in cell shape/state, not migration** — and migration variation
is driven by state + crowding, not treatment.

Bugfix during validation: a recording with < 3 full-duration cells has no
PRW fit (NaN D/P); `mannwhitneyu` propagated the NaN, blanking the D/P/
nn_dist rows to `n/a`. Non-finite per-recording values are now filtered
before the arm test.

---

## 2026-06-12 — Arm-split MSD plots (genetic | drug)

Added two per-arm ensemble-MSD plots to `ic295_flower_plots.py` alongside
the existing all-treatments pair, so MSD respects the experimental design
(only conditions sharing a control are overlaid). `_plot_msd` gained
`conds=` + `arm_label=` params; `_MSD_ARMS` drives the two extra calls in
`main()`. Each panel autoscales to its own arm (the drug arm tops out
~25k µm² vs the genetic arm's ~41k, so the shared-axis version hid the
drug-arm structure). New files: `msd_genetic{,_loglog}.png`,
`msd_drug{,_loglog}.png`. Same full-recording cohort + 60 lag bins as the
combined plot. Re-plots from `_track_cache.pkl` in ~5 s (`--from-cache`).

Reading: genetic — WT≈GOF (high), KO lower; drug — Y1 (high) > DMSO > OT.
SEM bands overlap within each arm, consistent with the n=8 recording-level
stats showing no significant frac_rounded/motility arm effect.

---

## 2026-06-08 — Add recordings to raise n (target n=8/condition, +19)

The arms are underpowered at n=4–6/condition (most real effects show as
trends; only the pooled pseudoreplicated view clears Bonferroni). Decided
to balance every condition to **n=8** by adding 19 of the 36 pending
recordings (WT+2, KO+3, GOF+3, OT+3, Y1+4, DMSO+4 — lowest-numbered Pos
per condition ≈ acquisition order).

Source = the Pathak lab share
`/Volumes/pathaklab/.../IC295_ECmigrationwithSirActin/IC295__1` (the 72
TRUE originals: uint16, **uncompressed**, full multi-position OME metadata,
2.45 GB each). GeorgeDrive is dead, so `inventory_drive()` works off
`_cache` — a recording becomes analysable simply by landing there.

Safe two-step pipeline (existing tooling, **never touches the originals**):
1. `ic295_copy_from_lab.py --label <Pos>` — opens the lab file read-only,
   verified atomic byte-copy → `_cache`, synthesises the `.ome.json`
   sidecar, repoints the `by_condition/` symlink. ~4.5 min/recording.
2. `recompress_recordings.py <file> --codec deflate` — lossless deflate
   with **page-by-page bit-identity + channel-count verification** before
   atomic replace (original untouched on any failure). 2.45 → ~1.23 GB
   (1.99×), matching the existing `_cache` format exactly.

De-risk (Pos63-DMSO) PASSED: byte-format-identical to Pos60-DMSO
(1.23 GB, deflate, single-series), sidecar + symlink present,
`inventory_drive()` now sees 30. Remaining 18 copying in the background
(`_runs/copy_recompress.log`); 155 GiB free, transient disk bounded to one
2.45 GB raw file at a time.

Follow-on (the slow part, NOT yet run): `ic295_detect_one` (1–3 h GPU
each) → `ic295_analyze_one` → re-run `ic295_compare_arms` at n=8.

## 2026-06-08 — Arm-aware comparison (respect the experimental design)

The conditions are TWO independent experiments each with its own control,
not one 6-way comparison: GENETIC (control WT → GOF, KO) and DRUG (vehicle
DMSO → YODA1/Y1, OT), plus the VEHICLE check WT vs DMSO. The flat
`ic295_compare` (6-way KW + all-15-pairwise-Bonferroni) tests meaningless
cross-arm contrasts (GOF vs OT) and over-corrects.

`scripts/ic295_compare_arms.py` (new): per-arm KW + pairwise MWU
**Bonferroni-corrected within the arm**; vehicle = single MWU. Reuses the
`ic295_compare`/`_pooled` stats helpers, reads their `per_recording.csv` /
`per_cell_pooled.csv`. Writes `stats_arms.json` + `plots_arms/<metric>.png`
(genetic | drug panels, control highlighted, test-vs-control stars,
vehicle p in title) in both compare dirs. `--level recording|pooled|both`.

Results (recording-level, n=4–6/arm-condition; pooled corroborates with
more power):
- **VEHICLE effect (WT vs DMSO) is significant** — frac_rounded p=0.019,
  spread circularity p=0.019, spread solidity p=0.010, n_cells p=0.031.
  ⇒ drug effects must be read vs DMSO, not WT. WT/DMSO are separate
  recordings so this could be vehicle OR batch/seeding — flagged.
- GENETIC: spread-eccentricity omnibus p=0.049 (KO vs WT 0.052 borderline;
  pooled KO vs WT 0.026) — KO spread cells more compact. GOF n.s.
- DRUG: n_cells omnibus p=0.018, rounded-circularity omnibus p=0.045;
  pooled sharpens it — Y1 & OT both vs DMSO on rounded circularity
  (p=0.006 / 0.009), Y1 on rounded area (0.040).
- The flat analysis' dramatic "OT lowest frac_rounded" is, vs its proper
  control DMSO, only a non-significant downward trend (recording level) —
  it looked dramatic only because the flat test compared it across arms.

Docs: INTERFACE (script entry), ic295_analysis/README (design table +
"read drug effects vs DMSO" warning), CLAUDE (arm-structured convention).

## 2026-06-08 — Re-analyse 29 recordings + regenerate all state diagnostics

Re-ran Phase 2 (`ic295_analyze_one`) on all **29 reviewed recordings**
with the edge filter + learned classifier, then regenerated every
comparison/diagnostic so the whole `compare/` tree is consistent with the
deployed rule. All 28 (+Pos60) succeeded, no failures.

Cross-condition result — the rounded/spread axis is now meaningful (was
uniformly ~0 under the old rule). `frac_rounded` omnibus Kruskal-Wallis
**p=0.004** (recording-level); ordering **KO ≈ GOF > WT > Y1 ≈ DMSO >
OT**, with **OT the clear outlier (~5 % rounded vs KO ~40 %)**. No
pairwise survives Bonferroni at recording level (n=4–6/condition,
underpowered); pooled (n=cells, pseudoreplicated) clears it for
OT-vs-{GOF,KO,WT}. Also significant (recording-level): spread-cell
circularity/solidity (GOF/KO rounder), spread-cell persistence (OT/Y1
most directional), n_cells (DMSO densest). The per-frame `state_diagnostic`
view agrees exactly (KO 39 % → OT 4 %).

Results-folder updates (all under the gitignored `ic295_analysis/`):
- `compare/` + `compare_pooled/` (stats, per_*.csv, `plots/`,
  `histograms/`) — fresh.
- `plots_mean_sem/` — refreshed from the new CSVs.
- **`state_rule_validation/`** (NEW, `ic295_eval_state_rule.py --plots`)
  — label-grounded: decision boundary (area_um2 vs ecc, hand labels
  coloured, misclassified circled), per-feature histograms by hand label,
  confusion matrix, per-feature AUC bar. **Deployed rule vs 279 labels:
  acc 0.93 / rounded-recall 0.90** (old circ/solid: 0.60 / 0.14).
- **`state_diagnostic/`** — rewritten from circ/solid to the deployed
  **area_um2 + eccentricity** (edge frames excluded). Pooled area is
  visually bimodal with the 960 µm² cut in the trough. Stale circ/solid
  PNGs removed.
- **`state_features/`** — `current_rounded` re-pointed to the deployed
  rule; threshold lines + titles reframed off the old "circularity
  misses" narrative. Honest nuance surfaced: ALL features Sarle
  BC < 0.56 (continuum, not bimodal) → the cut is fitted from labels, not
  a natural trough; rel_area separates the call median 0.17 vs 0.84.

Provenance recorded in `ic295_analysis/state_labels/README.md` (local).
Docs updated: INTERFACE (eval + diagnostic scripts), CLAUDE (state rule),
ic295_analysis/README (classifier + validation + re-fit recipe).

## 2026-06-08 — Learn the rounded/spread rule from the hand labels

Evaluated the shipped state rule (`circ ≥ 0.80 AND solid ≥ 0.92`) against
the 279 hand labels (`scripts/ic295_eval_state_rule.py`, new): **acc
0.60, rounded-recall 0.14** — it recognised only 18 of 130 hand-rounded
cells. The labeller's rounded/spread split is driven by SIZE / footprint
collapse, NOT circularity:

- single-feature agreement: area AUC 0.90 (rel_area 0.78) > circularity
  0.88 (best circ threshold ~0.68, not 0.80) > solidity 0.76; aspect
  ratio useless (0.51).
- scale-invariant shape alone caps at ~0.80 CV; **physical area_um2 +
  eccentricity (depth-2 tree) reaches 0.90 CV** — size is necessary.
- 24/279 labelled frames are edge-truncated and ALL 24 are hand-`spread`
  (truncated cells read large/elongated); the edge filter voids them
  from the AUTO pipeline (labels kept).

Deployed (4017f92) as two interpretable, scope-robust thresholds (no
pickled model / sklearn at inference):

    rounded iff area_um2 ≤ 960 AND eccentricity ≤ 0.85

- `DEFAULT_THRESHOLDS` gains `rounded_area_um2` (960) +
  `rounded_eccentricity` (0.85). `classify_state` / `classify_track_
  states` take `um_per_px`; physical rule when scale is known, else the
  legacy circ/solid gate (unchanged with no scale).
- `um_per_px` threaded through `annotate_state`, `mask_metrics`,
  `gui_batch/batch_worker` (real scale or None, never the 1.0 sentinel).
  `division_annotator` stays on the legacy fallback (its per-frame call
  passes no scale; detection-time division sidecars unchanged).
- `ic295_eval_state_rule.py` emits the fitted thresholds (re-fit as more
  labels land). GUI panels still show the fallback circ/solid widgets —
  TODO surface area/ecc.

Validated: deployed rule on the labels acc 0.928 / precision 0.944 /
**recall 0.900** (was 0.14); real Pos60-DMSO per-cell `frac_rounded` now
0.0–1.0 (mean 0.18) vs the old all-zeros; `test_focused_gui` 64/64.
**Re-analysing all 29 reviewed recordings** with edge + this classifier
(`_runs/reanalyze_edge_clf.log`) before re-running the comparisons.

## 2026-06-08 — Exclude edge-truncated cells from shape / state

Cells cut by the image border are only partially visible, so their
outline — and every shape metric + the rounded/spread state derived
from it — is unreliable. They must NOT pollute shape/state analysis,
but they DO still exist: keep them in cell counts and in their tracks
(the centroid still anchors identity).

Mechanism — one flag, one chokepoint, automatic propagation:
- `core/edge_filter.py` (new): `mask_touches_edge(mask, margin=0)` +
  `bbox_touches_edge(...)`. Must be given a FULL-FRAME mask (a bbox
  crop touches its own border). Verified every active analysis path
  measures `labels == cid` (full-frame), so `mask.shape` IS the frame.
- `core/cell_state.py`: `shape_metrics_for_mask` sets
  `metrics["edge_touch"]`; `classify_state` voids edge frames to
  `unknown`; `classify_track_states` returns a per-frame `edge` array.
  New `DEFAULT_THRESHOLDS["edge_margin_px"] = 0` (literal border).
- Because every per-state aggregation keys off state, edge→`unknown`
  makes shape means, `frac_rounded`/`frac_spread`, and per-state speed
  exclude edge frames with NO change to the aggregation maths.
- `core/state_analysis.py`: new per-cell `n_frames_edge`,
  `n_frames_classified`, `frac_in_view`; `frac_rounded/spread = None`
  when a cell is never cleanly in view (so it can't bias means);
  motion steps touching an edge frame (either end) dropped — a
  truncated centroid is biased inward → inflated displacement.
- `core/mask_metrics.py` + `gui/metric_coloring.py`: edge frames get
  overlay state code **3** (amber "edge (excluded)"); legend is
  data-driven so it picks it up. `cell_metrics_table.per_cell_row`
  surfaces the 3 new fields.
- Labelling: `_frame_feats` (ic295_state_features) emits `edge_touch`
  before its bbox crop; new label CSVs carry an `edge_touch` column.
  Existing hand labels are KEPT verbatim — user is confident in the
  spread/rounded call even on cropped/mis-outlined cells, so the
  filter gates only AUTO-computed shape/state, never the labels.
- `scripts/ic295_compare_pooled.py`: `--min-valid-frames N` drops
  cells with too few in-view classifiable frames (reports the count
  dropped — no silent cap); histograms inherit via per_cell_pooled.csv.
- Annotation GUI: amber outline + "⚠ edge-truncated — shape excluded"
  banner on edge crops.

Validated: `mask_touches_edge` unit tests; synthetic
`classify_track_states`/`annotate_state` (edge→unknown, fully-truncated
→None); real Pos60-DMSO — **n_cells 9 unchanged**, 5/9 cells have edge
frames, the known truncated cell 10 now `frac_in_view=0.19` (shape
excluded). `test_focused_gui.py` 64/64 pass (code-3 colour safe).

Re-run `ic295_analyze_one` to refresh per_cell.csv with the new
columns + edge exclusion before re-comparing.

## 2026-06-08 — Binary cell state (rounded / spread) + per-state metrics

Reworked cell-state analysis to fix a real confound: a whole-track
average (e.g. overall mean speed) is a TIME-WEIGHTED blend of the two
states, so a condition that merely spends more time rounded looked
slower/rounder even if its cells behaved identically within each state.

**State model — 3+compounds → binary.** `core/cell_state.py` now
classifies each cell-frame `rounded` / `spread` / `unknown`. Strict
rounded definition kept (circ ≥ 0.80 AND solid ≥ 0.92); the old
`attached` + `transitional` merge into `spread`. Dropped the compound
cohorts (`unattached`, `non_balled`). `DEFAULT_THRESHOLDS` →
`rounded_circ` / `rounded_solid` (the `attached_*` knobs are gone).
Deprecated `STATE_BALLED`/`STATE_ATTACHED`/`STATE_TRANSITIONAL` aliases
kept so the ~9 legacy standalone state scripts still import.

**Every per-frame metric is now state-stratified.** `state_analysis.py`
emits, for rounded AND spread each: `mean_speed`, `persistence`,
`straightness`, `mean_area_um2`, `mean_circularity`, `mean_solidity`,
`mean_aspect_ratio`, `mean_eccentricity`. `cell_metrics_table.py`'s
`per_cell_row` = lifetime (state-independent: `frac_rounded`,
`frac_spread`, `frames_tracked`, division*) + the per-state columns;
the state-MIXED whole-track averages are deliberately dropped from the
comparison-facing table.

**Comparisons** (`ic295_compare.py` + pooled) now test ONLY lifetime
metrics (n_cells, n_divisions, division_rate, % time rounded) and
per-state metrics — never a state-mixed average. Example from Pos7-WT:
one cell moves 6.4 µm/min while rounded vs 1.3 µm/min spread (5×), yet
is spread 99% of the time — a whole-track mean (~1.4) hid that entirely.

**Terminology + GUI.** `balled→rounded`, `attached/unballed→spread`
everywhere user-facing: colour-by ("Cell state (rounded / spread)",
"% time rounded"), 2-colour state palette (green spread / red rounded),
params + batch threshold panels (rounded circ/solid only). Fixed
`division_annotator` (pre-mitotic rounding gate counted balled OR
transitional = via aliases = every frame; now counts STATE_ROUNDED) —
affects future detections only (analysis uses cached divisions.json).

Decisions confirmed with the user: terms **rounded / spread**; boundary
**keep the strict rounded definition** (transitional → spread).

Re-analyzed all 29 recordings → re-ran both comparisons + mean±SEM
plots on the binary schema.

**Histograms + threshold confirmation.** Added `ic295_histograms.py`
(per-metric histograms split by condition, recording + cell levels) and
`ic295_state_diagnostic.py` (loads every masks.npz, per-frame circ +
solid distributions with the rounded cut drawn + a 2D circ-vs-solid
density). Finding: only **2.9% of cell-frames are rounded**, and the cut
is driven almost entirely by **circularity** (2.9% pass circ≥0.80 vs 28%
pass solid≥0.92 — solidity is nearly redundant in that corner). The
circ-vs-solid space is a **continuous spread→rounded ridge with no
bimodal valley**, so 0.80 is a defensible *conservative operational*
threshold isolating the clearly-rounded (mitotic/dying) tail, not a
natural cluster boundary — lowering circ toward ~0.70 would capture more
partial-rounding if desired.

## 2026-06-07 — Detection speed↔quality presets in the focused GUI

Added a **Detection preset** dropdown at the top of the focused GUI's
params panel — a one-click bundle over the speed/quality knobs
(downsample, gap-fill phases, TTA, DeepSea, mirror-pad, CP3 fallback,
cpsam precision):

- **Fast** — fastest possible (skips the whole gap-fill cascade, 3×
  downsample, fp32, no TTA/DeepSea/fallback/mirror-pad); expect missed
  cells.
- **Medium** — cheap gap-fill only (crop+no-augment, drops SAM2 + CP3),
  fp32; noticeably faster than Default, modest quality cost.
- **Default (Balanced)** — the canonical validated pipeline (no
  overrides = `DEFAULTS`).
- **Highest Quality** — full resolution, all gap-fill phases, full-frame
  always-augment, TTA + DeepSea + fallback + mirror-pad on; slowest.

Design honours rule 1: presets live in **`core/detection_presets.py`**
and record only *deltas* vs `DEFAULTS` (baseline filled from `_PD`);
`apply_preset_to_panel` just drives the existing widgets — no new
pipeline plumbing, no new `detect_recording` kwargs. Selecting a preset
resets the full `PRESET_PARAMS` set (deterministic), and the user can
still tweak any option afterward. (Both `params_panel.py` 917→ and
`pipeline_defaults.py` are already >500 lines, so the preset logic went
in a new focused module rather than growing them.)

Verified: `_test_presets.py` 17/17 (each preset round-trips through
`get_detect_params()`; Default == fresh defaults; ladder monotone),
`_gui_verify.py` **91/91** (new B4b preset section), canonical
`test_focused_gui.py` 64/64, defaults-consistency pass.

## 2026-06-06 — Live-RPC-smoke gotcha: stale GUI processes hold their ports

Full GUI test-drive re-run after the fp32/gap-fill plumbing: headless
harness **85/85**, canonical `test_focused_gui.py` **64/64**, live RPC
smoke on all 5 GUIs green — but only after catching a teardown bug.

**Gotcha:** `pkill -f "main_focused.py\|main_batch.py\|..."` does NOT
work — `pkill -f` matches an *extended* regex where `\|` is a literal
escaped pipe, so it matches nothing and the GUI processes survive. They
keep `LISTEN`-ing on their RPC ports (8765–8770), so a *later* live
smoke's fresh launches fail to bind, run headless, and the `curl
/status` you get back is silently answered by the **old** process
(running old code — the focused one even still had a recording loaded
from the previous drive). The smoke looks green but tested nothing.

**Correct pattern** when running the live RPC smoke:
- Tear down per-name: `for s in main_focused.py main_batch.py
  main_editor.py main_training.py main_tracking.py; do pkill -f "$s";
  done` (one substring per call — these match).
- After launch, VERIFY the port owner is a fresh PID + state is fresh:
  `lsof -nP -iTCP:8765 -sTCP:LISTEN` and check focused `/status` shows
  `recording_loaded: false`. If a recording is already loaded on a
  just-launched focused GUI, you're hitting a stale process.
- The RPC server binds on launch; a port already taken means a zombie.

## 2026-06-06 — fp32 opt-in toggle (MPS speedup) + stringent re-validation of gap-fill defaults

**fp32-on-MPS exposed as an opt-in `DEFAULTS.use_bfloat16` flag
(default True = bf16 = unchanged behaviour).** Earlier benchmark found
fp32 ~1.26× faster than bf16 on the Apple GPU, but the broad 5-recording
validation (density 2→19) showed quality was a *noisy wash* — not
certifiable loss-free at the tracked-output level (NO-GO on flipping the
default). So per Option 2 it's wired as opt-in: threaded `use_bfloat16`
through `pipeline_defaults` → `detect_recording` → `hybrid_cpsam_multi`
/ `hybrid_dic` → model creation + `fill_track_gaps` (gap-fill model);
GUI checkbox ("cpsam bfloat16 (uncheck=fp32, faster on Mac)") + worker
forwarding (auto + legacy + GapFillWorker + `_on_test_frame`) +
`run_pipeline_on_gt_recording.py --fp32`. The cpsam_dic subprocess
detection stays bf16 (different model, fp32 unvalidated there).
Verified: default unchanged, `use_bfloat16=False` → real `torch.float32`
net on MPS, defaults-consistency passes. A full-pipeline F1 vs GT (on
the NVIDIA box) is the gate for any future default flip.

**Stringent re-validation of the shipped gap-fill defaults
(crop+noaug) — ALL PASS across 4 recordings incl. the densest.** The
crop + augment=False wins were previously validated on Pos7-WT only.
Re-ran the GT synthetic-gap benchmark (delete known masks → known
truth) at production 1024², shipped `crop+noaug` vs original
`full+aug`, spanning density:

| recording | density (p75) | Phase-1 speedup | good fills dropped | shared-fill IoU Δ | fills |
|---|---|---|---|---|---|
| Pos0-WT | 2 | 18.7× | 0 | −0.003 | 7→8 |
| Pos41-OT | 5 | 17.4× | 0 | +0.018 | 8→8 |
| Pos20-KO | 8 | 16.2× | 0 | −0.017 | 7→8 |
| Pos68-DMSO | 19 (densest) | 18.6× | 0 | −0.020 | 8→8 |

Every recording: **0 good fills dropped**, shared-fill IoU within
±0.02 (bar −0.03), and crop+noaug filled **equal-or-MORE** gaps
(never fewer). The Pos7-only result generalises — no revert needed.

## 2026-06-06 — Gap-fill per-phase stats persisted + model-share; cpsam already on Apple GPU (MPS)

**Discovery (corrects a wrong assumption): the pipeline is GPU-bound,
not CPU-bound.** On this M1 Max, `CellposeModel(gpu=True)` resolves to
`device=mps` — cpsam has been running on the Apple GPU all along
(cellpose 4.x `assign_device` checks CUDA, absent here, then MPS).
Measured at production 1024²: **MPS 6.9 s/frame vs CPU 296.5 s/frame
(~43×)**. Both conda envs report MPS available; every detection path
uses `gpu=True`, so it's consistent (main detect, gap-fill, cpsam_dic
subprocess, probe, CP3 fallback). Consequence: CPU-side levers (thread
tuning, multiprocessing, eval batching) are moot. Remaining Apple-GPU
lever to investigate = **fp16 on MPS** (needs GT validation — changes
numerics). Bigger jump still needs a faster (NVIDIA) GPU.

**Per-phase gap-fill stats now persisted.** `fill_track_gaps` already
computed `{time_s, filled}` per phase but only logged it. Threaded
`stats_out` up through `hybrid_cpsam_multi` / `hybrid_dic` →
`detect_recording` (result gains `gap_fill_stats`) → into
`RUN_METADATA.json` `extra.gap_fill_stats` for all three writers
(`run_pipeline_on_gt_recording.py`, `ic295_detect_one.py`,
`gui_focused/export_dialog.py`). Unblocks the per-phase ablation: the
next detection run on any machine yields the data for free.

**Model-share micro-opt.** Gap-fill Phase 1 now reuses the
already-loaded raw cpsam model (`fill_track_gaps(cpsam_model=...)`)
instead of cold-loading a second one — gated to a *plain* raw model
(never a CPSAM_PRETRAINED fine-tune). GT-verified bit-identical: 4 real
Pos7-WT fills (823/945/888/1683 px) matched fresh-model exactly.

**Probe-reuse micro-opt — deferred (not bit-safe).** The auto-select
probe runs `eval(augment=False)` with no mirror-padding while the full
pass pads large frames, so reusing probe masks would change output on
padded recordings. Not free; not shipped.

## 2026-06-05 — Skip-doomed-tracks short-circuit ruled unsafe + full GUI test-drive

No product-behaviour change — an investigation conclusion + a
verification sweep.

**Skip-doomed-tracks gap-fill short-circuit — investigated, rejected.**
The remaining gap-fill lever after `augment=False` was: skip the
cascade for tracks the post-filter would drop anyway (≈ wasted cpsam
on phantoms). Ruled out *analytically* — both survival gates count
**present frames** (`track_postprocess.remove_empty_tracks`
min_frames=3 and `cy5_filter.persistence_guard_filter` min_lifetime=35,
both via `stack.any(axis=(1,2)).sum()` / equivalent), and gap-fill
*increases* that count. So gap-fill can rescue a borderline track past
either gate — which is precisely its documented Phase-3 job (cells that
retract / dim). Skipping it would change which cells survive (the
filters were calibrated *with* gap-fill on the GT corpus) → recall
risk, no safe pre-filter proxy. See `docs/IMPROVEMENTS.md` Priority 0.
Net: the durable gap-fill win remains `augment=False`; the real
remaining speedup is GPU.

**Full GUI test-drive — everything green.** Verified all five GUIs +
every focused-GUI option after the recent gap-fill / colour-by /
division / share work:
- `scripts/_gui_verify.py` (new headless harness): **85/85** — all 5
  GUI windows construct; gap-fill `crop`+`augment` toggles present,
  defaulted to `DEFAULTS`, gate on `use_gap_fill`, revert flows reach
  params, detect worker receives both; colour-by all 14 metrics
  (viewer + editor + legend); Analyze → all 22 graphs incl. lineage +
  division timeline; share `_export()` PNG + JPEG.
- Live-process RPC smoke: all 5 GUIs boot + answer `/status`; focused
  loaded a 97-frame recording + ran real single-frame detection over
  HTTP.
- Canonical `scripts/test_focused_gui.py`: **64/64** (real detect →
  analyze → graphs → export). `test_defaults_consistency.py`: pass.

Also committed the orphaned `scripts/ic295_compare_edits.py`
(re-analyse pre-edit masks → `*_original.json` to quantify how manual
edits changed downstream analysis numbers).

## 2026-06-05 — Gap-fill Phase 1: `augment=False` (the lever that works at production res)

Follow-up to the crop work. The end-to-end run showed the crop's win is
**track-cell-size-dependent** (the adaptive window grows with
`expected_area`: clean reviewed masks → 12×; raw production tracks →
~1.2×). So I tested the other Phase-1 cost: `augment=True` (4-rotation
TTA = 4 forward passes). Unlike pixels, augment is a **call count**, so
it cuts time even when the crop can't engage.

GT benchmark at **production resolution** (`scripts/bench_gap_fill.py
--downsample 2`, reviewed Pos7-WT masks, 18 gaps):

| variant | fill | mean IoU | s/gap |
|---|---|---|---|
| full+aug (old) | 14/18 | 0.831 | 21.5 |
| crop+aug (was default) | 15/18 | 0.810 | 2.4 |
| **crop+noaug (new default)** | **15/18** | **0.799** | **1.2** |

`crop+noaug` vs `crop+aug`: **2.1×**, same fill rate, **0** good fills
(IoU≥0.5) dropped, shared-fill IoU Δ **−0.011**. crop+noaug vs full+aug:
**~18×** at 1024².

Adopted as `DEFAULTS.gap_fill_augment=False` with the same safety as the
crop: Phase 1 now tries **no-augment first → augment fallback on miss →
full-frame+augment fallback**, so recall is provably ≥ the old
always-augment path; only misses pay the extra cost. Threaded
`detect_recording → detect_hybrid_{cpsam,dic}_multi → fill_track_gaps`,
exposed as a "Gap-fill always augment" GUI checkbox + `--gap-fill-always-
augment` runner flag (revert to old behaviour). Defaults-consistency
green. Caveat: synthetic gaps are easier than real ones, but the augment
fallback covers any hard-gap miss; hard-gap *fill quality* untested
beyond the Δ −0.011 — worth an eventual real-detection GT spot-check.

## 2026-06-05 — Gap-fill Phase 1: 12.5× faster, GT-validated no quality loss

Attacked the Priority-0 bottleneck — the 4-phase track gap-fill cascade
(`core/track_gap_fill.py`), which dominated multi-cell detection at
~17 min/cell.

**Found** (via new per-phase instrumentation): Phase 1 ran
`cpsam.eval(full_frame, augment=True)` per gap on the whole 2048² image
— **~76 s/gap** — to pick the single cell near an already-interpolated
centroid. Phase 1 only accepts a cell within `search_radius` of that
centroid, so an adaptive **crop** around it provably contains every
acceptable candidate.

**Fixed**: `_gap_crop_window` + crop-aware `_try_primary_cpsam`
(`DEFAULTS.gap_fill_crop=True`). The chosen mask maps back to
full-frame coords so all size/collision guards are unchanged. A
full-frame fallback fires only when the crop misses, so recall is
provably ≥ the old path. Per-phase timing now logs + flows to a
`stats_out` dict (the open Priority-0 "instrument" step).

**Validated against reviewed masks as GT** (`scripts/bench_gap_fill.py`:
delete known masks from reviewed tracks, run full vs crop on the same
gaps, score fill-rate + IoU-vs-GT + time). Pos7-WT, 20 synthetic gaps:
- crop **6.1 s/gap vs full 75.9 s/gap = 12.5×**
- **0** good fills (IoU≥0.5) dropped; the 1 crop-miss was a weak full
  fill (IoU 0.38)
- shared-fill IoU vs GT within **−0.002** of full; mask agreement 0.967
- Verdict: crop matches full on every good fill → default ON.

**End-to-end reality check** (Pos10-WT, 4 cells, 118 gaps, full
production pipeline): the 12.5× is real at **full 2048² resolution**,
but production **auto-downsamples to 1024²**, where cpsam's cost is NOT
pixel-bound (the ViT normalises to a target cell diameter), so the crop
only gave **gap-fill 1.2× (46.7→39.8 min), end-to-end 1.09×** (81.3→74.4
min). The crop is still safe and *improved* Phase-1 fill rate (33 vs 19
gaps, shifting work off slower SAM2) — so it stays ON — but it is NOT
the several-fold end-to-end win I first claimed. Gap-fill is still ~half
of detect time, dominated by Phase 1 running `cpsam(augment=True)` on
118 gaps. The real remaining lever at production resolution is the
**number of Phase-1 calls / `augment` (4×)**, not input pixels — next
target (needs the same GT validation as the crop). Phase 2/3 ablation
also open (`docs/IMPROVEMENTS.md` Priority 0). Measured via
`scripts/_bench_detect_e2e.py` (isolate base once, time gap fill both
ways on identical inputs). CLAUDE.md audit flag updated.

**Revert option** (default stays on the new crop path): `gap_fill_crop`
is now a first-class pipeline kwarg threaded
`detect_recording → detect_hybrid_{cpsam,dic}_multi → fill_track_gaps`
(None → `DEFAULTS.gap_fill_crop=True`; `False` reverts to full-frame).
Exposed as a "Gap-fill crop (fast)" checkbox on the focused GUI
Detection tab (gated on Gap fill) and as `--no-gap-fill-crop` on
`scripts/run_pipeline_on_gt_recording.py`. Defaults-consistency green.

## 2026-06-04 — Lossless re-compress of the recording masters (−35 GB)

The IC295 `.ome.tif` masters were 16-bit, 3-channel, **uncompressed**
(~2.45 GB each; one channel is the empty "None" channel). Re-saving
them with a lossless codec keeps the pixels bit-for-bit identical, so
every analysis result is unchanged — it just stores the same data in
fewer bytes. (PNG/JPEG would be wrong for the analysis path: JPEG is
8-bit + lossy; the masters carry ~13 real bits. Sharing-format question
came up while building the shareable-image export.)

`scripts/recompress_recordings.py` — per file: read → write
Deflate-compressed `.tmp` → **verify every page bit-for-bit identical +
`detect_channels` unchanged** → only then atomic `os.replace`. Never
removes an original without a verified replacement. `--dry-run` /
`--force` / `--keep-backup` / `--codec`.

Proof before trusting it (Pos0-WT, via the real loader + detector):
DIC + Cy5 frames bit-identical, channels/µm-per-px identical, and
**cpsam detection labels identical (4 cells on frame 48)** vs the
uncompressed original. Lossless ⇒ identical deterministic results,
confirmed end-to-end.

Batch over all 29 `_cache` masters: **70.93 GB → 35.88 GB (1.98×,
saved 35 GB)**, 0 errors. Files are gitignored so no git change; the
`data/ic295_gt_full/` + `gt_review/` symlinks point to the same paths
so they keep resolving (spot-checked: detect_channels=3, load_recording
OK, µm/px 0.6523 from sidecar). Masters also exist on the lab share if
ever needed.

## 2026-06-04 — Shareable image export + cell-division graphs (focused GUI)

Two requested focused-GUI additions.

**Shareable image export** (`gui_focused/share_export.py`; File → Export
Shareable Image…, Ctrl+Shift+I). Produces *small* files for sharing,
distinct from the archival full-res overlay TIFF:
- Current frame → PNG or JPEG (quality slider); all frames → MP4 (mp4v),
  animated GIF, or a single Montage PNG/JPEG (evenly-sampled grid).
- Each overlay independently switchable — mask fill, contours, cell IDs,
  tracks (trails), timestamp, scale bar — initialised from the viewer's
  current toggles + overlay settings.
- Max-dimension downscale (default 1024 px) + JPEG quality keep files
  tiny: a 1024-px JPEG of a frame ≈ 37 KB, a 12-frame montage ≈ 0.7 MB.
- Reuses `mask_editor_multicell.render_label_overlay` (mask/contour/ID)
  + `overlays.draw_overlays` (timestamp/scale bar baked in) so the look
  matches the viewer. Timestamp/scale-bar checkboxes auto-disable when
  the recording lacks dt / µm-per-px.

**Cell-division analysis + graphs** (`gui_focused/division_plots.py`):
- Two new multi-cell graphs in GRAPH_REGISTRY: **Cell Lineage Tree**
  (per-cell lifelines with parent→daughter division connectors labelled
  by division score) and **Division Timeline** (per-event score stems +
  cumulative-divisions step). Both derive from per-cell `track_info`
  (parent_id / division_frame / division_score) + the area timeseries,
  so they need no extra data and gracefully show "No divisions detected"
  when empty.
- New **"Detect divisions"** analysis toggle (params panel, default on) →
  `get_division_params()` → `FocusedAnalyzeWorker(division_params=…)`,
  which gates the `annotate_track_lineage` re-run for loaded masks.

Verified headless (15 checks): both plots render with/without divisions,
the toggle flips, and PNG/JPEG/MP4/GIF/Montage all export (JPEG < PNG,
JPEG < 400 KB). Screenshots confirmed the lineage connector + score, the
timeline stems + cumulative curve, and overlays baking correctly into a
montage. `image_viewer.py`/`mask_editor.py` untouched here; new logic
lives in the two new modules (both < 500 lines).

## 2026-06-04 — Dead-symlink sweep + test recording repoint (drive-failure cleanup)

The failed GeorgeDrive left **66 dangling symlinks**. Two fixes:

**Test recording.** `scripts/test_focused_gui.py` hard-coded an IC293
cropped WT on the dead drive. Now:
- `scripts/make_single_cell_example.py` — reusable generator that crops
  ONE cell's track (+margin, longest contiguous present-run) out of a
  recording + label stack. Used reviewed IC295 masks
  (`by_condition/WT/Pos10-WT`) + cached pixels to make
  `data/examples/single_cell_crop_wt/` (cell 1, 81 frames, 282×378,
  present in 100% of frames, ~8.6 MB) + its `_masks.npz` + `.json`.
- The test now resolves the recording from CLI arg → `$CELLSCOPE_TEST_
  RECORDING` → the bundled crop → `single_cell_phase_WT`; loads it
  channel-aware; navigates frames relative to length (was hard-coded
  frame 50); and falls back off a dead `results/` symlink. **Full run:
  59/59 pass** (detect 157 s, all 16 graphs, export, consistency).

**All dead symlinks fixed or removed** via
`scripts/fix_dead_symlinks.py` (`--dry-run` supported):
- 50 repointed to local copies — IC295 source/metadata → `_cache`;
  `gt_review/*/pipeline_results/masks.npz` → the reviewed
  `by_condition/.../masks.npz` (the original batch2 detections died
  with the drive, so gt_review now reflects the reviewed masks).
- 1 materialized — `results` dead symlink → real local dir.
- 15 removed — dead dir aliases (`data/training`, `data/ic295_inspection`,
  `data/ignasi_new_gt`, `data/ic295_gt`, `fiji_export_test`), uncached
  sources (Pos53_Y1, Pos69_DMSO — both had EMPTY `gt_masks/`), and the
  3 IC293 GT-recording pointers whose pixels are gone. **No real files
  or `gt_masks/*.png` were deleted** — only dangling symlinks. Every
  removed link's original target is recorded in
  `DEAD_SYMLINK_RECOVERY.md` for restoration if the drive returns.
- Result: 0 dead symlinks (108 remain, all alive). Follow-up: re-run
  `python scripts/audit_gt.py` to refresh `data/GT_INDEX.md`, since a
  few GT folders lost their (dead) recording pointer.

## 2026-06-04 — Colour masks by result + focused-GUI analysis parity

Two requested features, built on shared modules so the GUIs and the
IC295 batch never drift again.

**1. Colour masks by result** (both focused viewer + mask editor).
A "Colour by" dropdown recolours each cell by a measured value
instead of its ID:
- *Cell state* — balled (red) / attached (green) / transitional
  (amber), per frame (the categorical metric the balled-vs-non-balled
  review needs).
- per-track scalars — mean speed, persistence, net displacement,
  total distance, mean area/circularity/solidity, % time balled,
  frames tracked (continuous matplotlib colormaps).
- per-frame values — area / circularity / speed.
A `MetricLegend` widget under the canvas draws the colour key
(gradient bar + min/max, or state swatches).
- `core/mask_metrics.py::compute_label_metrics` — fast per-cell
  metrics straight from the label stack, so colouring works right
  after detection / loading (NO Analyze run). Reuses
  `cell_state.classify_track_states` (one regionprops pass → shape +
  state) + `tracking.extract_centroids`.
- `gui/metric_coloring.py` — `METRICS` registry, `MetricColorizer`
  (value→RGB, `STATE_COLORS`), `MetricLegend`. Shared by both GUIs.
- `gui/mask_editor_multicell.render_label_overlay(..., color_lut=)` —
  optional per-cell colour override (editor path).
- Wired into `gui_focused/image_viewer.py` (per-lab colour in
  `_render_frame`; cache invalidated in `update_masks`) and
  `gui/mask_editor.py` (combo + ↻ recompute + `color_lut` in
  `_redraw`). Default "Cell ID" keeps the native palette — the metric
  compute only runs when a metric is selected.

**2. Focused GUI now performs every analysis the project uses.**
The IC295 per-recording script computed per-state metrics the focused
GUI didn't. Closed the gap by extracting the script's logic into
shared core modules used by BOTH:
- `core/state_analysis.py::annotate_state` — union of the script's
  `_annotate_state` and the worker's `_annotate_with_state`: adds the
  compound `unattached` / `non_balled` states, per-frame speed
  variants (`*_pf`), and per-state straightness that the GUI lacked,
  while keeping the extended `balled_`/`attached_` display keys.
  `gui_focused/workers.py` + `scripts/ic295_analyze_one.py` both call
  it → identical per-state metrics.
- `core/cell_metrics_table.py::{per_cell_row, aggregate_recording,
  write_per_cell_csv}` — moved out of the IC295 script. The focused
  export (`export_dialog.py`) now also writes `per_cell.csv` +
  `recording_summary.json` in the same schema, so a GUI export feeds
  `ic295_compare.py` directly.
- State classification is now **on by default**:
  `DEFAULTS.compute_state_classification` flipped to `True` (canonical
  source), and the focused panel's `compute_states` now reads from it
  (was a hardcoded `False` — a rule-1 drift; also wired
  `vampire_clusters` / `compute_vampire` to DEFAULTS).
- `analysis_view.py` multi-cell Summary gained a recording-aggregate
  block (cells / division rate / mean speed / speed-by-state) + per-
  cell state composition + per-frame speed-by-state.

Verified: defaults-consistency test (28 checks) passes with
`compute_states=True`; headless smoke drove ImageViewer (all colour-by
options) + MaskEditor (colour-by + refresh) + AnalysisView (state +
aggregate) + ExportDialog (per_cell.csv 37 cols + recording_summary)
+ FocusedAnalyzeWorker multi with state on by default. (Full
`test_focused_gui.py` not run — its WT recording lived on the failed
GeorgeDrive and isn't local.)

Note: `image_viewer.py` (≈900 lines) and `mask_editor.py` (≈2.4k)
remain over the 500-line budget — pre-existing; new logic was pushed
into the shared modules to keep additions minimal. Splitting those two
is a separate refactor.

## 2026-06-01 — SAM2 point-and-click cell-detect tool (`gui/mask_editor_sam2_point.py`)

New mask-editor tool addressing the most common review pain point:
adding a cell that the original cellpose pass missed. Pick "sam2"
in the tool palette, left-click anywhere on the missed cell, and a
mask appears under the active cell ID from the spinner.

Why SAM2-on-crop over alternatives:
- One click vs cellpose-on-rectangle (lower cognitive load)
- Click is an explicit "this is where" prompt — no missing
- SAM2 encoder is O(N²); 512×512 crop is ~16× faster than full
  2048×2048 frame → 100-200 ms per click with no first-click warm-up
  surprise (which a frame-cache approach would have)
- Reuses the existing SAM2 tiny checkpoint already in the repo for
  video gap-fill — no new model dependency

Guards (refuse with clear status message):
- min/max area: 200 ≤ area ≤ 50000 px. The max guard catches the
  silent failure mode where SAM2 segments a featureless background
  region as one giant blob — observed in smoke testing: a click on
  empty space returned a 260k-px mask with score 0.97 → would have
  silently corrupted the labels.
- click must lie inside the predicted mask
- active cell ID must not already exist in this frame

On success: existing undo stack handles regret (Cmd+Z restores
frame state). Editor changes minimal: one item added to the tool
radio list, one elif in canvas mousePressEvent, one ~35-line
handler `run_sam2_at` on the editor class. New module is 189 lines.

Commit 38f4056.

## 2026-06-01 — Drive failure recovery: copy source TIFFs from Pathak lab share

GeorgeDrive (Seagate Backup Plus, the IC295 primary store) went into
a SCSI-INQUIRY wedge mid-batch (2026-05-31 ~23 Z). USB enumerates
cleanly but `IOUSBMassStorageDriver` never publishes an `IOMedia`
node — `diskutil list external` hangs, Disk Utility shows zero
devices when the drive is plugged in. Survived cold-boot + cable
swap; symptom is below `diskarbitrationd`, in the storage stack.
Drive is functionally dead until cable/enclosure/recovery service
resolves it.

Two changes to keep working:

1. `scripts/ic295_common.py::inventory_drive()` falls back to scan
   `CACHE_DIR` when both DRIVE_SOURCES are unreachable. Lets the 5
   prefetched TIFFs (Pos16-KO + Pos28/41/52/62 from the
   non-WT/KO/DMSO conditions) keep flowing through the existing
   pipeline. Detect-and-analyze on those 5 finished cleanly while
   the drive was gone (commit 837b4fe). New per-condition totals:
   WT 6, KO 5, GOF 5, Y1 4, OT 5, DMSO 4 → 29 detected, all six
   conditions at n ≥ 4.

2. `scripts/ic295_copy_from_lab.py` — recovery script to copy
   master TIFFs from the lab share
   (`/Volumes/pathaklab/Lab/Ignasi/IC295_ECmigrationwithSirActin/IC295__1`)
   to local `_cache/`. Driven by `progress.json`, defaults to all
   `detect.state == 'done'` labels. Atomic `.tmp` + rename copy,
   size verify, exponential-backoff retries, idempotent. Synthesizes
   `.ome.json` sidecars (lab source has none — all IC295 recordings
   share identical microscope metadata so they template trivially
   from any known-good one). Repoints `by_condition/<cond>/<label>/`
   symlinks at the local cache so existing pipeline + GUI work
   transparently. Smoke-tested end-to-end on Pos68-DMSO: 2.45 GB
   copied, sidecar synthesized, symlinks updated, 0 failures.
   Observed throughput ~2.6 MB/s on WiFi → 24-label full run is
   ~6h; wired ethernet should cut that to ~30 min.

Pos60-DMSO and Pos61-DMSO are flagged for re-analysis (mask edits
applied during the review session before the drive crashed). They'll
be re-analyzable once their TIFFs land via the copy script.

Plan after recovery: review the 29 detected recordings (currently
2 accepted + 1 reset-pending + 26 unreviewed), kick off interim
`ic295_compare.py` on the 27 already-analyzed recordings to see
treatment signals, and decide whether to detect more beyond n=4 or
freeze the analysis at the current cohort.

## 2026-05-31 — Flag: 4-phase gap-fill cascade dominates IC295 detect cost

Live timing data from the in-progress IC295 batch (7 real detections
complete) gave a clean linear fit:

  detect_minutes = ~50 + ~17 * max_cells_per_frame   (R² very high)

For a 12-cell recording (Pos60-DMSO, Pos2-WT, etc), 75-80 % of the
~3.5 h detect time is per-cell work, and the 4-phase track gap-fill
cascade (`core/track_gap_fill.fill_track_gaps`) is the bulk of it
(cpsam-augment / CP3-MedSAM-DeepSea subprocess / SAM2 video / mask
translation). Phase 2 in particular pays ~5-10 s conda env warmup per
gap call — likely amortizable.

Flagged for post-batch investigation:
- `docs/IMPROVEMENTS.md` Priority 0: per-phase instrumentation +
  ablation study + subprocess batching + confident-detection
  short-circuit + possible removal of low-yield phases. Target ≥ 30 %
  wall-time reduction without losing F1 on the 13-recording GT
  corpus.
- `CLAUDE.md` "Track gap-fill cascade" section: added a 🚩 marker
  pointing future agents at the audit.

A 30 % reduction on the cascade is ~1-2 days off the IC295 batch
(65 recordings × ~50 min saved each), so the optimization payoff is
concrete.

## 2026-05-30 — IC295 analysis-run operations guide (`docs/ic295_analysis_run.md`)

Created a dedicated operations guide for the long-running IC295 batch:
the three concurrent daemons (`ic295_batch.py --phase=detect`,
`ic295_analyze_watch.py`, `ic295_prefetch.py`) with their separate
lock files and roles, full replication from scratch, monitoring +
control + crash recovery commands, the manual mask-review checkpoint
workflow, the full directory layout, the per-recording + treatment-
comparison output schema (including the K-W + pairwise MWU /
Bonferroni stats), troubleshooting, and an honest M1-Max time
estimate (~5-6 days with prefetcher). Tracked in git.

`ic295_analysis/README.md` updated: cross-links the new guide, and the
quick-start launch block now reflects the three-daemon setup
(nohup-detached, separate lock files, graceful kill loop).

## 2026-05-30 — Sync top-level docs with recent landings

Updated README, PROJECT_STATUS, docs/user_manual, CLAUDE to reflect:
the `masks.npz` drag-drop + Open Masks File menu, the `Open Pipeline
Results` 4-step recording-resolution chain, `detect_channels`'s
`tifffile.series.axes` fallback (multichannel TIFFs work without
`_metadata.txt`), track + lineage rebuild on load, and the new IC295
batch pipeline. CLAUDE gained an "IC295 batch" section codifying the
conventions to preserve when extending (`DEFAULTS`-only params,
RUN_METADATA per recording, masks.npz as real file not symlink, etc).

INTERFACE.md already had the IC295 scripts added at build time
(commit `51173f0`).

## 2026-05-30 — IC295 batch analysis pipeline (detect → review → analyze → compare)

Six new scripts under `scripts/` for the full IC295 treatment-comparison
workflow, designed around a manual review checkpoint between detection
and analysis:

- `ic295_common.py` — shared utilities: drive inventory, condition
  parsing, `by_condition/<cond>/<label>/` path conventions, atomic
  progress.json writes, `.cellscope` project-file writer, recording-
  folder setup (symlinks + sidecar copy), priority queue (existing
  drive detections first, then round-robin over conditions to balance n).
- `ic295_detect_one.py` — Phase 1 per-recording worker. **Adopts**
  existing drive masks instantly (copy, not symlink, so user edits
  don't overwrite canonical); falls through to full
  `unified_detection.detect_recording` only when no drive masks exist.
  Writes pipeline_results/masks.npz + divisions.json + RUN_METADATA +
  `<label>.cellscope` project file. Idempotent.
- `ic295_analyze_one.py` — Phase 2 per-recording worker. Reads
  (possibly user-edited) masks, rebuilds tracks, runs
  `annotate_track_lineage` + per-cell `analyze_recording` + cell-state
  classification, aggregates to a single-row recording summary. Writes
  analysis.json (arrays stripped) + per_cell.csv + recording_summary.json.
- `ic295_batch.py` — long-running driver. `--phase=detect|analyze|both`.
  Lock file prevents double-start; SIGTERM finishes current and exits;
  per-recording **subprocess isolation** (cellpose OOM / segfault on one
  doesn't kill the driver); atomic progress.json updates;
  `--retry-failed`, `--limit`, `--label` flags.
- `ic295_status.py` — read-only state reporter (safe to run while a
  driver is going): per-condition state counts, ETA, currently-running,
  optional `--failed` tail of last error per failure.
- `ic295_compare.py` — Phase 3 compiler. For each metric: per-condition
  mean / SEM / n, Kruskal-Wallis across conditions, pairwise
  Mann-Whitney with Bonferroni correction, box+scatter plot. Outputs
  per_recording.csv + per_treatment.csv + stats.json + plots/*.png.

Folder: `ic295_analysis/` (gitignored). Each recording's `.cellscope`
file drag-loads in the focused GUI for review/edit between Phase 1 and
Phase 2; `Save Project` overwrites masks.npz in place, and Phase 2
reads whatever is at that path.

Inventory: 65 unique recordings (19 in IC295/ + 46 in IC295_batch2/,
dedup Pos51-Y1). Per condition: WT/KO/GOF/OT 11, Y1 10, DMSO 11.
12 already have drive masks from the mini's IoU+area batch run.

See `ic295_analysis/README.md` for the full workflow.

## 2026-05-29 — Cell-division lineage now populated for loaded masks too

Detection pipelines (`hybrid_cpsam_multi`, `hybrid_dic`) have always
called `core.division_annotator.annotate_track_lineage` after
post-processing, setting `parent_id`/`division_frame`/`division_score`
on daughter tracks and attaching a `divisions` list to the result —
which `analysis_view._populate_summary_multi` then surfaces as
"X division(s) detected" + per-cell parent links. But this only fired
on the detect path; loading `masks.npz` (e.g. from `gt_review/`) left
all tracks lineage-less, so analyze showed 0 divisions even on
recordings with known events (Pos39_OT, Pos51_Y1).

Two-stage fix:
- **`on_load_pipeline_results`** — if a `divisions.json` sidecar is
  next to the masks (pipeline / export writes one), apply its
  `track_lineage` table to the rebuilt tracks instantly + store
  `candidates` on `detect_result["divisions"]`.
- **`FocusedAnalyzeWorker`** — at the start of the multi-cell branch,
  if `detect_result` has no `divisions` key (i.e. neither detection
  nor the load fast-path populated it) and labels are available, call
  `annotate_track_lineage(tracks, labels, um_per_px)` to compute it
  before per-cell analysis. ~16 s on a 97-frame 2048² recording on
  the M1 Max — too slow to do synchronously on every mask drop, but
  fine inside the already-blocking analyze run.

No UI changes needed — the existing `analysis_view` summary already
renders lineage from `track_info.parent_id`. Verified on Pos39_OT
(10 cells): annotate finds 2 division candidates, sets parent_id on
daughter tracks at indices [7, 8].

## 2026-05-29 — Loaded `masks.npz` is now analyzable (multi-cell)

`FocusedAnalyzeWorker`'s multi-cell branch walks
`self.detect_result["tracks"]` for per-cell analysis. The
`on_load_pipeline_results` loader populated `detect_result["masks"]`
+ `detect_result["labels"]` but never `tracks`, so clicking Analyze
on a loaded `masks.npz` reported "0 cells detected" (the worker
iterates an empty list).

Added a `_rebuild_tracks_from_labels(labels)` helper in
`gui_focused/project_handlers.py`: one track per unique non-zero
ID in the (N, H, W) int32 stack, with `track["stack"]` = the
per-frame boolean mask and `track["first_frame"]` = the first
frame the cell appears in. Matches what `FocusedAnalyzeWorker`
expects, so analyze runs end-to-end without re-detecting.

Verified on `gt_review/ignasi_control/pipeline_results/masks.npz`
(3 cells, 15 frames) — 3 tracks reconstructed with correct shapes
+ first_frames; edge cases (None / all-zeros) return `[]`.

## 2026-05-29 — `Open Pipeline Results` no longer prompts when the recording is obvious

`on_load_pipeline_results` previously had a single recording-resolution
path — read `RUN_METADATA.json`'s `video_path` — and prompted the user
to locate the recording manually if that failed. With the new
`gt_review/<rec>/pipeline_results/masks.npz` layout (no RUN_METADATA
sidecar there), every mask load triggered the "The recording referenced
by RUN_METADATA (unknown) is not accessible — Locate the recording
file:" prompt, even when the user had *just dragged the recording in*
or it was sitting right next to the masks folder.

Resolution chain is now:
  1. `RUN_METADATA.json::source_recording.video_path` (existing).
  2. **NEW**: a sibling recording in `pr_dir`'s parent
     (handles `gt_review/<rec>/<rec>.ome.tif` next to
     `gt_review/<rec>/pipeline_results/masks.npz`).
  3. **NEW**: the currently-loaded recording, if any — user already
     told us what to overlay onto by loading it first.
  4. Prompt (only as a last resort).

Also: when the resolved video matches what's already loaded, skip the
redundant `_load_path` reload — matters on multi-GB recordings.

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
