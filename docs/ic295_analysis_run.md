# IC295 batch analysis — operations guide

The IC295 batch processes the **65 multichannel DIC + Cy5 keratinocyte
recordings** across 6 Piezo1-genotype conditions (WT, KO, GOF, Y1, OT,
DMSO) into a treatment-comparison dataset, with a **manual mask-review
checkpoint between detection and analysis** so segmentation artifacts
get fixed before they pollute the comparison.

Three-phase pipeline:

| Phase | What it does | Per-recording cost | Total cost (M1 Max) |
|---|---|---|---|
| **1 — Detect** | Run the canonical multichannel pipeline (cpsam DIC + Cy5 fusion + tracker + post-process + division annotation) per recording. | ~3.5 h (drive-bound) | ~5-6 days for the 53 undetected + adopt 12 existing |
| **review** | Drag `<label>.cellscope` into the focused GUI, fix any artifacts, `Save Project`. | minutes if needed | manual |
| **2 — Analyze** | Per-cell metrics (speed, MSD, shape, edge, state classification) + recording-level aggregation. Reads possibly-edited masks. | ~2.5 min | ~2 h (runs in parallel with Phase 1 via the watcher) |
| **3 — Compare** | Cross-treatment statistics (Kruskal-Wallis + pairwise Mann-Whitney with Bonferroni) + box+scatter plots. | seconds total | run once at the end |

Each recording is a separate biological replicate; per-cell metrics
are aggregated within a recording (mean / median / std) before
cross-condition stats. Cells within a recording are not independent
(same dish, same animal), so the recording is the unit of analysis.

---

## The three concurrent daemons

The running system has **three independent daemons** with separate lock
files. They are designed to run together:

| Daemon | Script | Lock | Polls | Purpose |
|---|---|---|---|---|
| **Detect driver** | `scripts/ic295_batch.py --phase detect` | `_runs/lock.txt` | per-recording | Subprocess-isolated detection in priority order |
| **Analyze watcher** | `scripts/ic295_analyze_watch.py` | `_runs/analyze.lock` | 10 min | Runs Phase 2 analysis on newly-completed detections, sequentially (uses ~7 GB RAM each) |
| **Prefetcher** | `scripts/ic295_prefetch.py` | `_runs/prefetch.lock` | 2 min | Caches the next 5 source TIFs from the USB drive to local SSD (~25-30% Phase 1 speedup) |

All three are **restart-safe**: state lives in `_runs/progress.json`
(atomic writes), and each daemon reads it on startup, skips `done`
recordings, and picks up at the next `pending` one. Lock files
prevent same-daemon double-starts; nothing prevents the three from
running concurrently (this is intentional).

---

## How to replicate / set up from scratch

### Prerequisites

- Drive mounted at `/Volumes/GeorgeDrive` with `ignasi/IC295/` and
  `ignasi/IC295_batch2/` subfolders containing the source `.ome.tif`s.
- `cellpose4` conda env (see [`INSTALLATION.md`](../INSTALLATION.md)).
- Repo cloned to a path with ≥ 30 GB free on the local SSD
  (the prefetcher refuses to start below that).
- macOS / Linux with `bash` (Windows would need analogous nohup + jobs
  syntax).

### One-shot setup

```bash
cd /path/to/cellscope
mkdir -p ic295_analysis/_runs/logs
```

That's it — the per-recording folders, cache dir, and progress.json all
get created on demand by the daemons.

### Launch all three daemons

```bash
cd /path/to/cellscope

# 1. detect driver
nohup bash -lc 'conda run -n cellpose4 python scripts/ic295_batch.py --phase detect' \
  > ic295_analysis/_runs/driver.log 2>&1 &
disown

# 2. analyze watcher
nohup bash -lc 'conda run -n cellpose4 python scripts/ic295_analyze_watch.py' \
  > ic295_analysis/_runs/analyze_watch.log 2>&1 &
disown

# 3. prefetcher (local-SSD cache for the slow USB drive)
nohup bash -lc 'conda run -n cellpose4 python scripts/ic295_prefetch.py' \
  > ic295_analysis/_runs/prefetch.log 2>&1 &
disown
```

`nohup` + `disown` detaches each from the shell so they survive the
terminal closing. Lock files claim each daemon's slot.

### Practical reminders before the long run

- **Keep the Mac plugged in.** A 5-6 day batch on battery is not going
  to end well.
- **Prevent sleep.** Either System Settings → Lock Screen → "Never"
  while plugged in, or in a shell:
  `caffeinate -dimsu -t 604800 &` (a week).
- **Keep `/Volumes/GeorgeDrive` mounted.** If it ever unmounts, the
  currently-running detection fails (the driver moves on); reconnect
  the drive and re-run with `--retry-failed` later. The prefetcher
  declines to cache while the drive is gone but doesn't crash.

---

## Monitoring

### Aggregate state, by condition

```bash
conda run -n cellpose4 python scripts/ic295_status.py
```

Output is per-phase: done / running / failed / pending counts in total
and broken down by condition, the average per-recording time, ETA, and
the currently-running recording. Safe to run while the batch goes.

`--failed` adds the tail of the last error per failed recording.

### Live logs

```bash
tail -f ic295_analysis/_runs/driver.log              # detect driver
tail -f ic295_analysis/_runs/analyze_watch.log       # phase 2 watcher
tail -f ic295_analysis/_runs/prefetch.log            # cache daemon
tail -f ic295_analysis/_runs/logs/<label>.detect.log   # one recording's detect run
tail -f ic295_analysis/_runs/logs/<label>.analyze.log  # one recording's analysis
```

### Cache + disk

```bash
ls ic295_analysis/_cache/             # what's staged
du -sh ic295_analysis/_cache/         # cache total size
df -h /                               # local SSD free space
```

---

## Stopping

Graceful — finishes current work and exits:

```bash
# Stop one
kill $(awk '{print $1}' ic295_analysis/_runs/lock.txt)         # detect
kill $(awk '{print $1}' ic295_analysis/_runs/analyze.lock)     # analyze watcher
kill $(awk '{print $1}' ic295_analysis/_runs/prefetch.lock)    # prefetcher

# Stop all three
for L in lock.txt analyze.lock prefetch.lock; do
  PID=$(awk '{print $1}' ic295_analysis/_runs/$L 2>/dev/null)
  [ -n "$PID" ] && kill "$PID"
done
```

The detect driver's SIGTERM handler **finishes the currently running
recording** before exiting cleanly. The watcher and prefetcher exit
after their current copy / current analysis. None of them leave
partial state — the next start picks up where they left off.

If a daemon dies unexpectedly and leaves a stale lock file:

```bash
rm ic295_analysis/_runs/lock.txt        # or analyze.lock / prefetch.lock
```

then relaunch normally.

---

## Per-recording control (no daemons)

```bash
# Process one recording end-to-end manually
conda run -n cellpose4 python scripts/ic295_detect_one.py Pos21-KO
conda run -n cellpose4 python scripts/ic295_analyze_one.py Pos21-KO

# Force re-process (override the "already done" skip)
... --force

# Skip cell-state classification (faster, matches focused-GUI default)
... --no-state                          # analyze_one only

# Run a fresh detection even when the drive already has masks
... --detect-anyway                     # detect_one only
```

---

## Driver flags worth knowing

```bash
python scripts/ic295_batch.py --phase detect --label Pos21-KO    # only this one
python scripts/ic295_batch.py --phase detect --limit 5           # cap to 5 recordings
python scripts/ic295_batch.py --phase detect --retry-failed      # re-queue failed
python scripts/ic295_batch.py --phase analyze --analyze-only-detected  # skip not-yet-detected
```

The analyze watcher and prefetcher also have flags:

```bash
python scripts/ic295_analyze_watch.py --interval 600 --once
python scripts/ic295_prefetch.py --lookahead 5 --interval 120 \
       --disk-floor 50 --disk-refuse 30 --once
```

---

## Manual mask review (between Phase 1 and Phase 2)

After detection finishes for a recording, you can fix any
mis-segmentations before Phase 2 runs. The **`ic295_review.py`**
script wraps this in a safe, state-tracked workflow:

```bash
# See the queue — counts pending / accepted / edited / skipped per condition
conda run -n cellpose4 python scripts/ic295_review.py status
conda run -n cellpose4 python scripts/ic295_review.py status -v   # full list

# Open the next pending recording in the focused GUI (auto-loaded)
conda run -n cellpose4 python scripts/ic295_review.py next

# Or pick a specific one
conda run -n cellpose4 python scripts/ic295_review.py open Pos21-KO

# After you've edited a few, re-run Phase 2 on the ones you changed
conda run -n cellpose4 python scripts/ic295_review.py reanalyze-pending
```

What it does, per `open`/`next`:

1. **Refuses** to open a recording the analyze watcher is currently
   running on (avoids the read-during-save race).
2. **Backs up** `pipeline_results/masks.npz` → `masks_original.npz`
   (once per recording, never overwritten — your ground-zero is safe).
3. **Launches** the dedicated **`main_editor.py`** as a subprocess
   with the recording + masks as CLI args, so both load on startup.
   The mask-editor UI is leaner than the focused GUI (no Detect
   button to click by accident, no Analyze stage).
4. **Waits** for you to close the GUI window.
5. **Captures** a before/after MD5 + per-cell area diff if you saved
   changes; marks the recording `accepted` (no change) or `edited`
   (with diff stats + `needs_reanalysis: true`); appends to
   `_runs/review_audit.log`.

**Compare original vs edits inside the GUI** — drag
`pipeline_results/masks_original.npz` onto the window to view the
pre-edit masks (the editor's drag-drop reloads the mask layer
without changing the recording). Drag `masks.npz` back to see your
current edits. Drag-drop loads NEVER repoint the save target —
`Ctrl+S` always writes to the file the editor was launched against,
so dragging the original in for comparison won't accidentally
overwrite it.

**Save with `Ctrl+S`** — when the editor was launched against a
specific `masks.npz` (as `ic295_review.py` does), `Ctrl+S` overwrites
that file in place, preserving its original key (`labels` or `masks`)
and dtype. The standalone "Save Masks" button does the same. (Open
the editor without a `mask_path` and `Ctrl+S` falls back to the
existing GT-folder dialog used for ground-truth labeling.)

### Bulk-cleanup tools

For removing artifacts across many frames at once, the editor has
three additions that complement single-pixel brush / eraser work:

- **🔍 Filter Cells…** button (or `Ctrl+Shift+F`) — opens a dialog
  with per-frame filters (min/max area in pixels, min distance to
  frame edge) and per-track filters (min lifetime in frames, max
  mean velocity for static phantoms). Tick whichever criteria
  apply, click **Preview** to see exactly which cell IDs and which
  frames would be removed (with reasons listed), then **Apply**.
  Per-frame filters remove individual instances; per-track filters
  remove the entire cell ID from every frame. Undo entries are
  pushed per affected frame.
- **🗑 Delete by Cell ID…** button — prompts for a cell ID and
  removes it from every frame it appears in. Equivalent to the
  Shift+Click shortcut below but discoverable in the toolbar.
- **Shift+Click in the delete tool** — single-shortcut variant of
  the above. Click any pixel of a cell with the `delete` tool +
  Shift held → that cell's track is wiped across all frames.

To discover good filter thresholds for a specific recording, use
the **`analyze-edits`** subcommand of `ic295_review.py`:

```bash
conda run -n cellpose4 python scripts/ic295_review.py analyze-edits <label>
```

This compares your current `masks.npz` against the original backup
(`masks_original.npz`) and reports:
- Every cell ID with its category (`REMOVED` / `TRIMMED` / `kept`)
- Per-cell stats: lifetime, mean area, min distance to frame edge,
  mean velocity
- Filter threshold suggestions where there's clean separation
  between touched and kept cells

Once you've established good thresholds on a representative
recording, the same numbers usually apply to other recordings in
the same condition.

**State** lives in `ic295_analysis/_runs/review_state.json` (per
condition × recording) + an append-only audit log. The review tool
uses its own `_runs/review.lock` so two review sessions can't open
the same recording concurrently — but it doesn't conflict with the
detect driver / analyze watcher / prefetcher.

**Other subcommands:**

```bash
# Diff between the current and the original for one recording
conda run -n cellpose4 python scripts/ic295_review.py diff Pos21-KO

# Manually mark a recording's status (e.g. you reviewed in the GUI
# directly without going through the tool):
conda run -n cellpose4 python scripts/ic295_review.py mark Pos21-KO --status accepted
```

### Without the review tool (direct GUI workflow)

If you'd rather drive the GUI manually:

1. Drag `ic295_analysis/by_condition/<cond>/<label>/<label>.cellscope`
   into the focused GUI (or `File → Open Project`).
2. The recording + detection overlay loads.
3. Click `Edit → Edit Masks` (or the Edit pipeline stage button) to open
   the mask editor; paint/erase as needed.
4. `File → Save Project` (Ctrl-S) — overwrites
   `pipeline_results/masks.npz` in place.

The Phase 2 watcher reads whatever's at that path on its next poll.
If `analysis.json` already existed, the watcher skips it — re-analyze
manually:

```bash
conda run -n cellpose4 python scripts/ic295_analyze_one.py <label> --force
```

This direct path doesn't get the backup or diff capture — `ic295_review.py`
is recommended for the systematic pass through the corpus.

---

## What lives where

```
ic295_analysis/
├── _runs/                              # daemon state (not analysis data)
│   ├── progress.json                   # per-recording {detect, analyze} state
│   ├── lock.txt                        # detect driver lock
│   ├── analyze.lock                    # analyze watcher lock
│   ├── prefetch.lock                   # prefetcher lock
│   ├── driver.log                      # detect driver stdout
│   ├── analyze_watch.log               # watcher stdout
│   ├── prefetch.log                    # prefetcher stdout
│   └── logs/<label>.{detect|analyze}.log    # per-recording subprocess output
├── _cache/                             # local-SSD staging (~2.3 GB × N)
│   ├── IC295__1_MMStack_<label>.ome.tif
│   ├── IC295__1_MMStack_<label>.ome.json
│   └── IC295__1_MMStack_<label>_metadata.txt
├── by_condition/<cond>/<label>/        # per-recording, self-contained
│   ├── <label>.cellscope               # drag into focused GUI
│   ├── IC295__1_MMStack_<label>.ome.tif        # SYMLINK → drive (durable)
│   ├── IC295__1_MMStack_<label>.ome.json       # sidecar copy
│   ├── IC295__1_MMStack_<label>_metadata.txt   # symlink → drive
│   ├── pipeline_results/               # real files (not symlinks)
│   │   ├── masks.npz                   # ← Phase 1 detection
│   │   ├── divisions.json              # ← Phase 1 lineage
│   │   ├── RUN_METADATA.json
│   │   ├── fusion_diagnostic.png       # multichannel only
│   │   └── (mini-era extras: masks_unfiltered.npz, filter_decisions.json,
│   │       analysis_summary.json, per_cell.csv  — for adopted recordings)
│   ├── analysis.json                   # ← Phase 2 per-cell results
│   ├── per_cell.csv                    # ← Phase 2 flat per-cell metrics
│   └── recording_summary.json          # ← Phase 2 single-row aggregate
└── compare/                            # ← Phase 3 cross-treatment
    ├── per_recording.csv               # one row per recording, all metrics
    ├── per_treatment.csv               # n / mean / SEM / std / median per (condition, metric)
    ├── stats.json                      # K-W + pairwise MWU (Bonferroni)
    └── plots/<metric>.png              # box + scatter per condition
```

The whole `ic295_analysis/` tree is **gitignored** except this guide
and `ic295_analysis/README.md` — local state, never ships to git.

---

## Final outputs

When all three phases finish you'll have:

### Per-recording (under `by_condition/<cond>/<label>/`)

- **`pipeline_results/masks.npz`** — int32 label stack `(N, H, W)`,
  one ID per cell, per frame.
- **`pipeline_results/divisions.json`** — division candidates +
  daughter-track lineage.
- **`analysis.json`** — full per-cell analytics, arrays stripped (~few
  KB per cell): speed (mean, time series stats), MSD, autocorrelation,
  persistence, shape summary (area, circularity, solidity, aspect
  ratio, eccentricity), edge dynamics (protrusion / retraction
  velocity), boundary confidence, area stability, cell-state
  classification fractions and per-state speeds.
- **`per_cell.csv`** — flat one-row-per-cell with the headline metrics.
- **`recording_summary.json`** — single-row aggregate (mean / median /
  std / n per metric across cells in this recording). This is the row
  Phase 3 consumes.
- **`<label>.cellscope`** — drag into the focused GUI to view + edit.

### Treatment comparison (under `compare/`)

- **`per_recording.csv`** — N rows (one per recording with a
  `recording_summary.json`), every metric.
- **`per_treatment.csv`** — long-form: `(condition, metric, n, mean,
  sem, std, median)`. Easy to pivot in pandas / Excel.
- **`stats.json`** — for each metric:
  - `kw`: Kruskal-Wallis statistic + p-value across all conditions
    that have ≥ 2 recordings (non-parametric — robust on small n + the
    skewed distributions typical of motility metrics).
  - `pairs`: every pairwise Mann-Whitney U test with raw p and
    Bonferroni-corrected p; `sig_bonf: true` if `p_bonf < 0.05`.
- **`plots/<metric>.png`** — box plot per condition with individual
  recording dots overlaid + Kruskal-Wallis p-value in the title.

### Default metric set in the comparison

Listed at the top of `scripts/ic295_compare.py::DEFAULT_METRICS`:

- Mean / median cell speed (µm/min)
- Persistence ratio
- Total distance traveled (µm)
- Net displacement (µm)
- Mean cell area (µm²)
- Mean circularity / solidity / aspect ratio
- Mean protrusion / retraction velocity
- State fractions (% balled / % attached)
- Per-state mean speed (speed when balled, speed when attached)
- Division rate (divisions / cell)
- n_cells, n_divisions

`per_recording.csv` includes **every** column from
`recording_summary.json`, not just these — pass
`--metrics name1,name2,...` to `ic295_compare.py` to plot/test a
different set.

---

## Partial comparisons (you don't need to wait for all 65)

The priority queue round-robins conditions so n grows balanced. Once
each condition has ≥ 4 done, you can stop the driver and get a
provisional treatment comparison:

```bash
# 1. Stop the driver (and watcher / prefetcher if you want)
kill $(awk '{print $1}' ic295_analysis/_runs/lock.txt)

# 2. Make sure Phase 2 caught up — wait for the watcher to clear its
#    queue, or trigger a single pass:
conda run -n cellpose4 python scripts/ic295_analyze_watch.py --once

# 3. Generate comparison
conda run -n cellpose4 python scripts/ic295_compare.py
```

You can re-run `ic295_compare.py` any number of times — it just reads
whatever `recording_summary.json` files exist.

---

## Troubleshooting

**"Another driver appears to be running" but I'm sure none is** — a
crash left a stale lock. `rm ic295_analysis/_runs/<lock-file>` and
relaunch.

**Drive unmounted mid-run** — the in-flight detection fails (driver
records and moves on). Re-mount the drive. The prefetcher refuses
copies while the drive is gone but doesn't die. To re-run the failed
recording:
```bash
conda run -n cellpose4 python scripts/ic295_batch.py --phase detect --retry-failed
```

**Disk getting full** — the prefetcher stops adding when free space
< 50 GB (`--disk-floor`). Eviction on detection-done frees space
automatically. To force-clear the cache (safe — `best_video_path`
falls back to the drive):
```bash
rm ic295_analysis/_cache/*
```

**A recording loads as "a line" in the GUI** — multichannel TIFF
loaded without `_metadata.txt` and OME-XML lacks `SizeC`. Should be
fixed by [the `tifffile.series.axes` change in `detect_channels`
(commit `1fd36ee`)] — open a fresh session and try again.

**Mini's `per_cell.csv` differs from ours** — the 7 adopted recordings
that came from the mini's IoU+area batch have **both** `per_cell.csv`
files in their `pipeline_results/` (mini-era) and at the recording-
folder top level (Phase 2). Column names differ (mini uses
`diffusion_D_um2_per_min`, ours uses `total_distance` + `persistence`)
but they should agree on shared columns. Substantial divergence on a
shared column suggests a real pipeline change worth investigating.

**Watcher not picking up newly-finished detections** — 10-min poll by
default. To force a check without restarting the long-running watcher,
run a single-pass watcher in another shell:
```bash
conda run -n cellpose4 python scripts/ic295_analyze_watch.py --once
```
(This will refuse to start while the long-running one holds the lock —
intentional. The one-shot mode is for when no watcher is running.)
Or just wait 10 min.

---

## Honest time estimate (M1 Max, ~3.5 h/recording observed)

| Phase | Per recording | Remaining 53 (or analyze all 65) |
|---|---|---|
| **Phase 1** — detect, drive-bound | ~3.5 h | ~5-6 days continuous (with prefetcher) |
| **Phase 2** — analyze, per-cell | ~2.5 min | ~2 h total, runs in parallel via the watcher |
| **Phase 3** — compare, stats + plots | seconds total | run once at the end |

The prefetcher reduces Phase 1 by ~25-30% by eliminating the USB-drive
read for source TIFs (the dominant I/O cost). Without it, expect
~7-8 days.

You don't need to wait for all 65 — after ~3 days the round-robin will
have n ≥ 4 per condition, which is enough for a meaningful treatment
comparison. Stop the driver and run `ic295_compare.py` whenever you're
satisfied.

---

## See also

- [`ic295_analysis/README.md`](../ic295_analysis/README.md) — quick-start in-folder.
- [`CLAUDE.md`](../CLAUDE.md) — the "IC295 batch" section codifies
  conventions for extending the batch scripts.
- [`INTERFACE.md`](../INTERFACE.md) — module index; the `scripts/`
  section lists every `ic295_*.py` script.
- [`docs/pipeline_description.md`](pipeline_description.md) — what
  the detection pipeline actually does, per stage.
