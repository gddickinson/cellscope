# IC295 batch analysis

Long-running detect + analyze + compare pipeline for the IC295 dataset
(`/Volumes/GeorgeDrive/ignasi/IC295{,_batch2}`). Two-phase: **detect**
first (slow, GPU), pause for manual review/edits in the focused GUI,
then **analyze** + **compare** (fast). Each phase is restart-safe and
crash-isolated.

> **Full operations guide:**
> [`docs/ic295_analysis_run.md`](../docs/ic295_analysis_run.md) covers
> the three concurrent daemons (detect driver + analyze watcher +
> prefetcher), monitoring + control + crash recovery, the manual mask
> review workflow, the final-output schema, and partial-comparison
> recipes. This README is a quick-start.

## Quick start

```bash
# === Long-running mode (recommended for the full 65-recording batch) ===
# Launch all three daemons detached; they run concurrently with
# separate lock files. Real per-recording time: ~3.5 h detect /
# ~2.5 min analyze on M1 Max. See docs/ic295_analysis_run.md for the
# full guide (monitoring, stop/start, partial comparison, etc).

nohup bash -lc 'conda run -n cellpose4 python scripts/ic295_batch.py --phase detect' \
  > ic295_analysis/_runs/driver.log 2>&1 &
disown
nohup bash -lc 'conda run -n cellpose4 python scripts/ic295_analyze_watch.py' \
  > ic295_analysis/_runs/analyze_watch.log 2>&1 &
disown
nohup bash -lc 'conda run -n cellpose4 python scripts/ic295_prefetch.py' \
  > ic295_analysis/_runs/prefetch.log 2>&1 &
disown

# === Monitor ===
conda run -n cellpose4 python scripts/ic295_status.py            # state table
conda run -n cellpose4 python scripts/ic295_status.py --failed   # error tails
tail -f ic295_analysis/_runs/driver.log                          # live driver

# === Stop everything gracefully ===
for L in lock.txt analyze.lock prefetch.lock; do
  PID=$(awk '{print $1}' ic295_analysis/_runs/$L 2>/dev/null)
  [ -n "$PID" ] && kill "$PID"
done

# === Manual mask review between Phase 1 and Phase 2 ===
# Drag ic295_analysis/by_condition/<cond>/<label>/<label>.cellscope
# into the focused GUI, edit masks, Save Project. The Phase 2 watcher
# reads whatever's at pipeline_results/masks.npz on its next poll.

# === Phase 3 — treatment comparison ===
# Runs whenever you're satisfied with the cell counts; re-runnable.
conda run -n cellpose4 python scripts/ic295_compare.py

# === Foreground mode (smaller jobs, debugging) ===
conda run -n cellpose4 python scripts/ic295_batch.py --phase detect --limit 3
conda run -n cellpose4 python scripts/ic295_batch.py --phase analyze --analyze-only-detected
conda run -n cellpose4 python scripts/ic295_detect_one.py Pos21-KO --force
conda run -n cellpose4 python scripts/ic295_analyze_one.py Pos21-KO --force
```

## Directory layout

```
ic295_analysis/
├── _runs/                           # state (NOT analysis data)
│   ├── progress.json                # per-recording {detection, analysis} state
│   ├── lock.txt                     # presence ⇒ a driver is running
│   └── logs/<label>.<phase>.log     # full subprocess output per recording
├── by_condition/<cond>/<label>/     # one folder per recording, grouped
│   ├── <label>.cellscope            # drag into focused GUI to review/edit
│   ├── IC295__1_MMStack_<label>.ome.tif      # symlink → drive
│   ├── IC295__1_MMStack_<label>.ome.json     # sidecar copy
│   ├── IC295__1_MMStack_<label>_metadata.txt # symlink → drive
│   ├── pipeline_results/
│   │   ├── masks.npz                # REAL file (user edits land here)
│   │   ├── fusion_diagnostic.png    # if produced
│   │   ├── divisions.json
│   │   └── RUN_METADATA.json
│   ├── analysis.json                # Phase 2 per-cell results
│   ├── per_cell.csv                 # Phase 2 flat per-cell metrics
│   └── recording_summary.json       # Phase 2 aggregated single row
└── compare/                         # Phase 3
    ├── per_recording.csv            # one row per recording, all metrics
    ├── per_treatment.csv            # mean / SEM / n per (condition, metric)
    ├── stats.json                   # K-W + pairwise MWU (Bonferroni)
    ├── plots/<metric>.png           # box+scatter per condition
    ├── plots_mean_sem/<metric>.png  # mean ± SEM + individual points
    ├── histograms/<metric>.png      # per-metric, split by condition
    ├── state_rule_validation/       # rounded/spread rule vs hand labels
    ├── state_diagnostic/            # rounded cut vs ALL mask data
    └── state_features/              # multi-feature boundary diagnostic
# (compare_pooled/ mirrors compare/ with the CELL as the unit; its
#  --min-valid-frames N drops cells with too few in-view frames)
```

## Manual editing in the focused GUI

After Phase 1 finishes a recording, drag
`ic295_analysis/by_condition/<cond>/<label>/<label>.cellscope` into
the focused GUI (or `File → Open Project`). It loads the recording +
masks; use the mask editor to fix artifacts, then `File → Save Project`
to overwrite `pipeline_results/masks.npz` in place.

Phase 2 reads whatever's at that path — your edits, or the original
detection.

## Crash recovery + restart

Just re-run the driver (`python scripts/ic295_batch.py --phase ...`).
It will:

1. Refuse to start if the lock file is present (another driver running).
   If you're sure none is, remove `ic295_analysis/_runs/lock.txt`.
2. Read `progress.json` and skip recordings already marked `done`.
3. Treat `failed` recordings as not pending — pass `--retry-failed` to
   re-queue them.
4. Resume from the next pending recording in the priority queue.

Each recording runs in a subprocess, so a hard crash (cellpose OOM,
segfault, drive disconnect) leaves the driver alive — it marks that
recording `failed` and moves on. SIGTERM / Ctrl-C finishes the current
recording cleanly, then exits.

## Priority queue

The driver processes recordings in this order:

1. All with existing drive detections (12 currently — fast adopt path).
2. Undetected recordings round-robined across conditions so n stays
   balanced (Y1 → DMSO → KO → GOF → OT → Y1 → DMSO → ...).

Pass `--limit N` to cap how many pending recordings the driver
processes in one run. Pass `--label <label>` to run a specific one.

## Per-recording cost (rough)

- Adopt-from-drive: < 1 s (file copy).
- Full detection (no existing masks): 1–3 h / recording on M1 Max GPU
  (mostly cellpose). Driver-safe but the recording itself is slow.
- Analysis: 30–60 s / recording (mostly `annotate_track_lineage` for
  the division signature scan + per-cell `analyze_recording`).
- Treatment comparison (Phase 3): seconds total across all recordings.

## Experimental design — which comparisons are valid

The six conditions are **two independent experiments**, each with its own
control, plus a vehicle check:

| arm | control | tests | question |
|---|---|---|---|
| **genetic** | **WT** | GOF, KO | effect of the genetic perturbation |
| **drug** | **DMSO** (vehicle) | YODA1 (Y1), OT | effect of the drug |
| **vehicle** | — | WT vs DMSO | does the vehicle alone shift cells? |

**Cross-arm contrasts are meaningless** (e.g. GOF vs OT — different
controls, different experiments). Use **`scripts/ic295_compare_arms.py`**
(the design-correct primary): each arm gets its own Kruskal-Wallis +
pairwise Mann-Whitney, **Bonferroni-corrected within the arm**; the
vehicle is a single MWU. `ic295_compare.py` (flat 6-way + all-pairwise)
is kept only as an exploratory all-conditions view — its all-15-pairwise
Bonferroni over-corrects and mixes arms, so don't read its pairwise
p-values as the result.

```bash
conda run -n cellpose4 python scripts/ic295_compare.py          # collect CSVs
conda run -n cellpose4 python scripts/ic295_compare_pooled.py    # (for pooled)
conda run -n cellpose4 python scripts/ic295_compare_arms.py      # the real test
# → compare{,_pooled}/stats_arms.json + plots_arms/<metric>.png
```

⚠️ **The vehicle effect (WT vs DMSO) is significant** on this corpus
(frac_rounded, spread circularity/solidity) — so **read drug effects vs
DMSO, not WT**. (WT and DMSO are separate recordings, so this may be a
true vehicle effect or a batch/seeding difference; either way the drug
arm's control is DMSO.)

## Treatment comparison details

Each recording counts as one experimental replicate. Per-cell metrics
are aggregated within each recording (mean / median / std across cells)
to one row in `recording_summary.json`. Then in Phase 3:

- For each metric: group recording-level values by condition.
- Compute `n / mean / SEM / std / median` per condition.
- **Kruskal-Wallis** across all 6 conditions (non-parametric — robust
  on small `n` and skewed distributions typical of motility metrics).
- **Pairwise Mann-Whitney** between every condition pair, with
  **Bonferroni correction** on `p_raw`.
- A box plot (with individual recording dots overlaid) per metric.

The default metric set lives at the top of `scripts/ic295_compare.py`;
override with `--metrics name1,name2,...`. The `per_recording.csv`
keeps every field, so any further analysis can use the full data.

## Motility & dispersal (tracks, MSD, confounders)

Track-level migration analysis runs off a **shared enriched cache** built
once from every recording's masks (`scripts/ic295_track_data.py`,
`compare/flower_plots/_track_cache.pkl`, `CACHE_VERSION`): per cell it
stores the trajectory, per-frame state, and per-frame local density.
Rebuild after re-detecting/editing masks:

```bash
conda run -n cellpose4 python scripts/ic295_track_data.py --rebuild   # ~1 h
conda run -n cellpose4 python scripts/ic295_flower_plots.py --from-cache   # ~5 s
conda run -n cellpose4 python scripts/ic295_motility_stats.py --from-cache
conda run -n cellpose4 python scripts/ic295_motility_plots.py --from-cache
```

The three figures worth showing (`ic295_motility_plots.py` →
`compare/motility_stats/`): **`motility_forest.png`** (covariate-adjusted
treatment effects ± 95% CI — every interval crosses 0 = the null result),
**`motility_covariates.png`** (net displacement vs time-spread and vs
crowding — both negative, treatments intermix: dispersal is structured by
state + crowding, not treatment), and **`msd_mean_vs_median.png`** (genetic
arm: KO is lowest by mean but highest by median — why the ensemble mean
misleads). Plus `speed_vs_density.png` (contact inhibition) from the stats
script.

**Like-vs-like check** (`ic295_motility_matched.py --from-cache`): removes
the state + crowding confounders *by design* rather than by model — cells
are compared to their arm control only within strata of the same
state-class × neighbour count (coarsened exact matching + van Elteren
stratified test), and also via control-baseline residual **normalization**
(continuous). Outputs `matched_forest.png`, `matched_strata.png`,
`MATCHED.md`. All ns: **covariate adjustment (LMM), matching, and
normalization independently agree on the null** — the result is not an
artefact of comparing unlike cells.

- **`ic295_flower_plots.py`** → `compare/flower_plots/`: origin-centred
  flower plots, per-cell speed/distance/net-disp, area-vs-speed, and
  ensemble **MSD(τ)** for full-recording cells — mean±SEM *and*
  **median+bootstrap-CI** (the mean is outlier-driven on this skewed
  data; always check the median), all-treatment + per-arm.
- **`ic295_motility_stats.py`** → `compare/motility_stats/REPORT.md`:
  the **design-correct** test — each recording's full-duration cells are
  reduced to one value (recording = unit, no pseudoreplication), then the
  arm-structured test (reuses `ic295_compare_arms`) over 11 metrics incl.
  MSD exponent α and the persistent-random-walk D & P
  (`ic295_motility_models.py`). Confounders are handled explicitly:
  **STATE** (frac_spread + paired within-cell speed), **CONTACT/density**
  (Spearman + paired isolated-vs-crowded), and **pseudoreplication**
  (recording-level OLS + cell-level statsmodels LMM, both adjusting for
  state + density).

**Result on this corpus (n=8/condition): no treatment changes any
motility/dispersal metric** (survives none of Bonferroni / OLS / LMM).
What *is* significant is the confounders — **time-in-state dominates
motility** (LMM frac_spread p≤0.001) and there's weak (between-cell)
contact inhibition. Read together with the shape/state findings: the
IC295 phenotype is in cell **shape/state, not migration**.

## Rounded / spread state classifier (learned from hand labels)

Each cell-frame is classified `rounded` / `spread` / `unknown`. The rule
(`core.cell_state.DEFAULT_THRESHOLDS`) was **learned from hand labels**,
not hand-set:

> **rounded iff `area_um2 ≤ 960` AND `eccentricity ≤ 0.85`**

The labeller's rounded/spread split is driven by **size / footprint
collapse**, not circularity. The old `circ ≥ 0.80 AND solid ≥ 0.92` rule
recognised only 14 % of hand-rounded cells; the deployed rule scores
**acc 0.93 / rounded-recall 0.90** on the 279 labels (decision-boundary
quality — the labelled set is class-balanced, not natural prevalence).

Edge-truncated cell-frames (mask cut by the image border) are voided to
`unknown` — they're still counted + tracked, but their shape is
unreliable so they're excluded from shape/state metrics. A cell never
cleanly in view reports `frac_rounded = None`; drop thin cells from the
pooled comparison with `ic295_compare_pooled.py --min-valid-frames N`.

Re-validate / re-fit (e.g. after labelling more in the annotation GUI):

```bash
# score the deployed rule + emit re-fitted thresholds + validation plots
conda run -n cellpose4 python scripts/ic295_eval_state_rule.py
# all-mask-data view of the cut (area µm² + eccentricity distributions)
conda run -n cellpose4 python scripts/ic295_state_diagnostic.py
# multi-feature boundary diagnostic
conda run -n cellpose4 python scripts/ic295_state_features.py
```

`ic295_analysis/state_labels/` holds `labels.csv` + crops + a local
`README.md` (gitignored) recording the labelling + validation provenance.
**Changing the thresholds means re-running Phase 2 (`ic295_analyze_one`)
on every recording** before the comparisons are valid again.
