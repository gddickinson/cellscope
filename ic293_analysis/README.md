# IC293 single-cell-crop analysis

The IC295 detect → analyze → compare battery, re-targeted at Ignasi's
hand-cropped **IC293 endothelial-cell** crops. Everything here is
generated locally and gitignored except this README and `PROVENANCE.md`.

## What's different from IC295 (read this first)

| | IC295 | IC293 |
|---|---|---|
| recording | a field of view, **many cells** | **one hand-cropped cell** |
| channels | DIC + Cy5 (SirActin) | **DIC only** (no fluorescence) |
| cell type | keratinocytes | **endothelial cells** |
| duration | ~97 frames, full | **30–97 frames**, may be partial |
| statistical unit | the recording | the **position** (field of view) |

Three consequences are baked into the IC293 scripts:

1. **Single-channel detection.** Crops load with no channel args
   (`cy5_frames=None`), so every Cy5 step in `detect_recording`
   (alignment, annotation, persistence-guard) is guarded off. The
   auto-selector picks **cpsam_dic** for these 1-cell scenes (tighter
   boundaries on isolated cells). No cells are wrongly deleted.

   **One cell per crop — extras are artifacts.** Each crop is hand-cropped
   to exactly one target cell; anything else cellpose detects (debris,
   fragments) is an artifact. Analysis applies a transparent
   **primary-cell selection** (`select_primary_cell`: persistence × size
   × start-centrality) that keeps the one real cell and drops the rest —
   `masks.npz` is left untouched (artifacts stay visible/editable in the
   GUI review). Crops where the runner-up scores close are flagged
   `selection_ambiguous` in `recording_summary.json` — review those.

2. **Position is the unit.** Each crop is one cell; several crops share
   an original field of view (`Pos{N}`). The **position** is the
   cluster for pseudoreplication — the stats reduce each metric to one
   value per position (median over its crops), the exact analog of
   IC295's "recording is the unit". 78 crops → **55 positions**
   (genetic: WT 12 / GOF 7 / KO 10; drug: DMSO 9 / Y1 8 / OT 9).

3. **State rule is keratinocyte-derived → state metrics are
   EXPLORATORY.** The rounded/spread rule (area ≤ 960 µm² AND ecc ≤
   0.85) was fit to IC295 keratinocyte hand-labels. There are no IC293
   labels, so `frac_spread` / per-state shape are reported in a
   separate, flagged block. The **primary** read is the state-agnostic
   motility/size metrics (speed, net displacement, area, MSD, α, Fürth
   D/P), which don't depend on that rule.

> **Cross-dataset caveat:** only **within-IC293** arm contrasts are
> meaningful. Absolute values are not comparable to IC295 (different
> cell type, no actin channel).

## Pipeline

```bash
# Phase 0 — staging (already done; reads the lab share read-only).
conda run -n cellpose4 python scripts/ic293_stage_crops.py        # → _cache/

# Phase 1 — detect all 78 crops (~4 h; cpsam_dic, single-channel).
conda run -n cellpose4 python scripts/ic293_batch.py --phase detect
conda run -n cellpose4 python scripts/ic293_status.py             # monitor
conda run -n cellpose4 python scripts/ic293_status.py --failed    # error tails

#   → MANUAL REVIEW CHECKPOINT: drag any
#     by_condition/<cond>/<label>/<label>.cellscope into the focused GUI
#     (conda run -n cellpose4 python main_focused.py) to inspect / edit
#     masks before analysis. Edits write back to masks.npz (a real file).

# Phase 2 — analyze (fast; per-cell metrics + state).
conda run -n cellpose4 python scripts/ic293_batch.py --phase analyze

# Phase 3 — compare (design-correct, arm-structured).
conda run -n cellpose4 python scripts/ic293_track_data.py --rebuild   # cache
conda run -n cellpose4 python scripts/ic293_motility_stats.py         # PRIMARY
conda run -n cellpose4 python scripts/ic293_compare.py                # state (exploratory)
conda run -n cellpose4 python scripts/ic293_flower_plots.py           # figures
```

Single recording (debug): `ic293_detect_one.py <label>` /
`ic293_analyze_one.py <label>` (e.g. `Pos0-WT`, `Pos11_cell2-WT`).

## Outputs

```
ic293_analysis/
  _cache/                         staged crops (.ome.tif + .ome.json + _metadata.txt)
  by_condition/<cond>/<label>/    per-crop project folder
    pipeline_results/masks.npz    detection (edit in focused GUI)
    pipeline_results/RUN_METADATA.{md,json}   detect-phase provenance
    per_cell.csv, recording_summary.json      analysis
    <label>.cellscope             drag-load project file
  compare/
    motility_stats/REPORT.md      PRIMARY result (state-agnostic, arm-structured)
    motility_stats/plots_arms/*   per-metric genetic|drug arm panels
    compare/stats_arms_shape.json state/shape (EXPLORATORY)
    flower_plots/*                flower + MSD figures
  PROVENANCE.md                   staging provenance (env, git, per-crop table)
```

## Conventions to preserve

- Detection uses `core.unified_detection.detect_recording` with all
  kwargs defaulted (rule 1 — no hardcoded pipeline params).
- Every detect/analyze writes `RUN_METADATA.{md,json}` (rule 2).
- `masks.npz` is a real file (never a symlink) so GUI edits don't touch
  anything canonical.
- Stats are **arm-structured** (genetic control WT, drug vehicle DMSO);
  cross-arm contrasts are meaningless.
- `ic293_analysis/` is gitignored except this README + `PROVENANCE.md`.
