# IC295 single-cell-crop analysis

A **derived** dataset: individual, **non-dividing** single cells cropped out of
the multi-cell IC295 recordings and analysed the same way as the IC293
single-cell crops. Everything here is generated from `ic295_analysis/` by two
scripts; **nothing in `ic295_analysis/` is modified**.

## Why

The IC295 population result was: *the genetic/drug phenotype is in cell
**shape/state**, not **migration**; the motility variation that exists is driven
by **state + crowding**, not treatment.* This dataset isolates individual
non-dividing cells — removing the crowding/contact confound and division events
— and asks whether that conclusion still holds for clean single cells. It is the
single-cell analogue of IC293. These are **endothelial cells** (IC295 = "EC
migration"), the same cell type as IC293 — but they are IC295's **own** cells,
the ones the rounded/spread state rule was fitted on
(`ic295_analysis/state_labels`), so the state metrics here are on the same
footing as the main IC295 population analysis (where state is a **primary**
readout). IC293 is a *separate* endothelial experiment with no hand labels of
its own, which is why its scripts flag the *transferred* state rule exploratory.

## How it was built

1. **`scripts/ic295sc_crop_cells.py`** (reads `ic295_analysis/`, read-only).
   For every IC295 recording with `pipeline_results/masks.npz`, it keeps each
   tracked cell that is:
   - **non-dividing** — not a parent or daughter in that recording's
     `divisions.json` (`track_lineage`; cell label = `track_index + 1`);
   - **tracked long enough** — present for **≥ 30 contiguous frames** (the floor
     is *the shortest track in the IC293 analysis set*, which is 30; the IC295
     recordings are 97 frames, so most crops are far longer — median ~full
     recording), with present/span ≥ 0.90;
   - **not heavily edge-truncated** — touches the field-of-view border in
     ≤ 50 % of its frames.

   It then crops the **DIC** video and **that one cell's mask** to the cell's
   trajectory bounding-box + a 60 px margin, over the cell's present-frame span,
   so each crop is *one cell, present every frame* — the IC293 prior. The
   **masks are reused** from the validated + reviewed IC295 detection (the cells
   are **not** re-detected). Other cells that fall inside the crop window appear
   in the video but are **not** in the mask, so the analysis measures only the
   target cell.

   Output per crop (IC293 on-disk layout, so the whole IC293 battery runs on it
   unchanged):
   ```
   _cache/<label>.ome.tif            # uint8 DIC, axes TYX, single channel
   _cache/<label>.ome.json           # um_per_px, time_interval_min, n_frames
   by_condition/<cond>/<label>/
       <label>.ome.tif -> _cache     # symlink
       <label>.cellscope             # drag-loads in the focused GUI
       pipeline_results/masks.npz    # the single cell, relabeled to 1
       pipeline_results/RUN_METADATA.{md,json}   # provenance (source rec,
                                     # cell id, crop bbox, frame span)
   crops_manifest.csv                # one row per crop
   ```
   Label grammar: `<SrcPos>_cell<ID>-<COND>` (e.g. `Pos7_cell3-WT`) — the IC293
   label grammar, so the condition/position parsers work unchanged. The
   **source recording is the unit of replication** (the `PosN` parsed from the
   label, unique across conditions — the exact analogue of IC293's "position").

2. **`scripts/ic295sc_run.py`** sets `IC293_ANALYSIS_ROOT` to this folder and
   runs the **identical** IC293 battery (no code forks — see the env-override
   note in `scripts/ic293_common.py`):
   - `ic293_analyze_one.py` per crop → `per_cell.csv`, `recording_summary.json`,
     `analysis.json`, `RUN_METADATA`;
   - `ic293_compare.py` → `compare/per_position.csv`, `stats_arms_shape.json`,
     `compare/plots_arms/`;
   - `ic293_track_data.py --rebuild` → `compare/flower_plots/_track_cache.pkl`;
   - `ic293_motility_stats.py` → `compare/motility_stats/{stats_arms_motility.json,
     REPORT.md, plots_arms/}`;
   - `ic293_flower_plots.py` → `compare/flower_plots/*.png`.

## Reproduce

```bash
conda run -n cellpose4 python scripts/ic295sc_crop_cells.py            # build crops
conda run -n cellpose4 python scripts/ic295sc_run.py --jobs 4          # analyse
# tweak selection:
conda run -n cellpose4 python scripts/ic295sc_crop_cells.py --dry-run  # plan only
conda run -n cellpose4 python scripts/ic295sc_crop_cells.py \
    --min-frames 30 --margin-px 60 --min-contig 0.9 --max-edge-frac 0.5
```

## Design notes / caveats

- **Unit of replication = source IC295 recording** (not the cell). The stats
  reduce a recording's crops to one value (per-position reduction) and add a
  cell-level mixed model with `(1 | position)` — same machinery as IC293/IC295,
  no pseudoreplication.
- **Two-arm design** (genetic WT→GOF/KO; drug DMSO→Y1/OT; vehicle WT vs DMSO),
  arm-structured tests, exactly as IC295/IC293.
- **Endothelial cells; state rule applies on the IC295 footing.** These are
  endothelial cells (IC295 EC-migration) — and they are IC295's OWN cells, the
  ones the rounded/spread rule was fitted on (`ic295_analysis/state_labels`), so
  the state-split metrics are a **primary** read here, exactly as in the main
  IC295 population analysis. The auto-generated `compare/motility_stats/REPORT.md`
  prints the IC293 "EXPLORATORY (state rule on EC)" wording verbatim — that
  caveat is about IC293 (a *separate* endothelial experiment with no labels of
  its own, where the rule is *transferred*); it does not apply to this dataset
  of IC295's own cells. The numbers are identical regardless.
- **No crowding term.** Each crop is one isolated cell, so local density is ~0
  by construction; the motility stats drop the density covariate (as IC293
  does). Removing crowding is part of the point of this dataset.
- **DIC only.** The Cy5 channel is dropped — the masks already exist, so Cy5 is
  not needed for re-analysis, and this matches the IC293 single-channel format.
- **Non-destructive.** This folder is built entirely from `ic295_analysis/`;
  that folder is never written. This whole directory is git-ignored (derived
  microscopy data, public repo) except this README.
