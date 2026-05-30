# IC295 batch analysis

Long-running detect + analyze + compare pipeline for the IC295 dataset
(`/Volumes/GeorgeDrive/ignasi/IC295{,_batch2}`). Two-phase: **detect**
first (slow, GPU), pause for manual review/edits in the focused GUI,
then **analyze** + **compare** (fast). Each phase is restart-safe and
crash-isolated.

## Quick start

```bash
# Phase 1 — detection. Adopts existing drive masks instantly;
# for the rest runs the full detect_recording pipeline (~2–3 h /
# recording on M1 Max GPU). Stop at any time with Ctrl-C.
conda run -n cellpose4 python scripts/ic295_batch.py --phase detect

# Watch progress in another shell:
conda run -n cellpose4 python scripts/ic295_status.py
conda run -n cellpose4 python scripts/ic295_status.py --failed   # tails of errors

# ── Optional: open by_condition/<cond>/<label>/<label>.cellscope
# in the focused GUI to review & edit masks before analysis. ──

# Phase 2 — analysis. Reads the (possibly edited) masks.npz,
# rebuilds tracks, annotates divisions, runs per-cell analytics +
# state classification. Fast (~30–60 s / recording).
conda run -n cellpose4 python scripts/ic295_batch.py --phase analyze

# Phase 3 — treatment comparison. Aggregates per-recording metrics
# by condition, runs Kruskal-Wallis + pairwise Mann-Whitney
# (Bonferroni), writes CSVs + box plots.
conda run -n cellpose4 python scripts/ic295_compare.py
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
    └── plots/<metric>.png           # box+scatter per condition
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
