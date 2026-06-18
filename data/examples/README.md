# Demo recordings

Five compact example recordings for trying out the GUI. Each is a
trimmed + DEFLATE-compressed slice of a real endothelial cell recording —
small enough to ship in the repo (~108 MB total) but big enough to
exercise the full pipeline end-to-end.

To use: launch any GUI (`python main_focused.py`, `python main_batch.py`,
etc.), then *File ▸ Open Recording* and pick the `.tif` or `.ome.tif`
inside one of these folders.

## What's included

| Folder | File | Frames | Size | Pipeline | Demonstrates |
|---|---|---:|---:|---|---|
| `single_cell_phase_WT/` | `single_cell_phase_WT.tif` | 50 × 438×759 | 13 MB | Single-cell, phase contrast | Cleanest case — one cell tracked across 50 frames; trajectory + speed + edge-dynamics graphs |
| `multi_cell_DIC_WT/` | `multi_cell_DIC_WT.tif` | 50 × 512×512 | 10 MB | Multi-cell, DIC | Hungarian tracker on 4-6 cells per frame; per-cell graphs |
| `multi_cell_DIC_KO/` | `multi_cell_DIC_KO.tif` | 50 × 512×512 | 10 MB | Multi-cell, DIC | KO genotype counterpart of `multi_cell_DIC_WT` — try both then run a 2-condition comparison in the Tracking GUI |
| `multichannel_DIC_Cy5_WT/` | `multichannel_DIC_Cy5_WT.ome.tif` | 30 × 512×512 × 2ch | 14 MB | Multi-cell, DIC + Cy5 actin filter | Multichannel pipeline (small) — Cy5 = ch0, DIC = ch1. Drops debris that lacks F-actin |
| **`multichannel_DIC_Cy5_DMSO_busy/`** ★ | `multichannel_DIC_Cy5_DMSO_busy.ome.tif` | 40 × 1024×1024 × 2ch | 61 MB | Multi-cell, DIC + Cy5 actin filter | **Busy scene (28+ cells per frame)** — IC295 batch2 Pos69 DMSO. Demonstrates the Cy5 filter and Hungarian tracker on a dense field. Channel chooser: **Cy5 = ch0, DIC = ch1** |

Each folder also contains a JSON sidecar
(`{name}.json` / `{name}.ome.json`) with `name`, `um_per_px`,
`time_interval_min`, `condition`, and (for the multichannel demo) the
channel labels. The GUI auto-loads these so you don't have to set
pixel size or frame interval manually.

## Sources

These are derived from real data on the external drive at
`/Volumes/GeorgeDrive/cellscope_data/`:

- `single_cell_phase_WT` — first 50 frames of `Ignasi C1-IC293 Pos0-WT`
  cropped phase-contrast (uint16 → uint8 via p1/p99 rescale)
- `multi_cell_DIC_WT` / `KO` — first 50 frames of Jesse `pos17_wt` /
  `pos59_ko` OME-TIFF, downscaled 2× (1024 → 512), uint16 → uint8
- `multichannel_DIC_Cy5_WT` — first 30 frames of IC295 `Pos0_WT`,
  downscaled 4× (2048 → 512), both channels interleaved as the GUI's
  multichannel loader expects (`[T0_C0, T0_C1, T1_C0, …]`)
- `multichannel_DIC_Cy5_DMSO_busy` — first 40 frames of IC295 **batch2**
  `Pos69_DMSO` (the densest scene in the dataset — max 36 cells/frame,
  mean 28). Downscaled 2× (2048 → 1024) to preserve cell detail in the
  busy field. Same interleaved layout as above.

## Re-generating

The build script lives at `/tmp/build_demo_recordings.py` (also
preserved in the chat history). To re-create or edit:

```bash
conda run -n cellpose python /tmp/build_demo_recordings.py
```

It pulls from the symlinked external-drive sources, so the drive must
be mounted.

## Suggested demo flow during a presentation

1. Load **single_cell_phase_WT** → Detect → Analyze → page through the
   trajectory/speed/MSD/edge-kymograph graphs. (~30 s of work.)
2. Load **multi_cell_DIC_WT** → switch pipeline mode to *Multi Cell
   (hybrid_cpsam_multi)* in the Pipeline panel → Detect → Analyze →
   show per-cell graphs and the per-track summary.
3. Load **multichannel_DIC_Cy5_DMSO_busy** (recommended — busiest +
   most recent). Pick Cy5 = ch0, DIC = ch1 in the channel chooser.
   Tick *Cy5 filter (Tier 4)* → *Multi-metric* in the Detection params.
   Run Detect — the filter drops debris that lacks F-actin signal.
   (`multichannel_DIC_Cy5_WT` is the smaller / sparser alternative for
   quick tests.)
4. Open the Tracking & Stats GUI (`python main_tracking.py`) → load
   `multi_cell_DIC_WT` and `multi_cell_DIC_KO` as separate groups for a
   2-condition comparison.
