# RUN_METADATA — Pos51_Y1

**Schema version**: 1.0.0
**Started**: 2026-05-21T05:08:26
**Finished**: 2026-05-21T05:08:26
**Runtime**: 4334 s
**Pipeline**: `hybrid_cpsam_multi` (mode = multi)

## Source recording

- **video_path**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos51_Y1/IC295__1_MMStack_Pos51-Y1.ome.tif`
- **n_frames**: 97
- **has Cy5 channel**: True
- **um_per_px**: 0.6523
- **time_interval_min**: 10.0
- **checksum**: `sha256-64mb:0c79178dcfa22cdd`

## Results

- **n_tracks**: 2
- **n_cy5_fusion_added** (pre-tracking): 28
- **n_analysis_cells**: 0
  - fusion source breakdown: both = 2

## Parameters deviating from defaults

  - **auto_selected_pipeline**: `cpsam` (not in defaults)
  - **downsample**: `2` (not in defaults)
  - **downsample_reason**: `auto: max dim 2048 ≥ 1500; ds=2 gives ~5× speedup with minimal accuracy loss` (not in defaults)
  - **downsample_spec**: `auto` (not in defaults)
  - **model_path**: `None` (not in defaults)
  - **use_cy5_recovery**: `False` (default `True`)

## Environment

- **conda env**: `cellpose4`
- **python**: `3.10.17`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **cellpose**: `?`
- **numpy**: `2.0.2`
- **tifffile**: `2023.2.28`
- **scipy**: `1.15.3`
- **skimage**: `0.25.2`
- **torch**: `2.7.0`

## Reproducibility

- **git commit**: `9dd8c371336e`
- **rerun command**:

```bash
conda run -n cellpose4 python scripts/run_pipeline_on_gt_recording.py data/ic295_gt_full/Pos51_Y1
```

## Output directory

`/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos51_Y1/pipeline_results`

---
*Generated automatically. If this file is missing alongside results,
the run did not write metadata — see core/run_metadata.py for the
expected API. This is a requirement: every analysis path (GUI export,
batch worker, programmatic script) MUST call `write_run_metadata`.*
