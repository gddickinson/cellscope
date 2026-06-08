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
