# RUN_METADATA — Pos7_WT

**Schema version**: 1.0.0
**Started**: 2026-05-16T04:59:47
**Finished**: 2026-05-16T04:59:47
**Runtime**: 32359 s
**Pipeline**: `hybrid_cpsam_multi` (mode = multi)

## Source recording

- **video_path**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos7_WT/IC295__1_MMStack_Pos7-WT.ome.tif`
- **n_frames**: 97
- **has Cy5 channel**: True
- **um_per_px**: 0.6523
- **time_interval_min**: 10.0
- **checksum**: `sha256-64mb:136fb2b1b212972c`

## Results

- **n_tracks**: 17
- **n_cy5_fusion_added** (pre-tracking): 111
- **n_analysis_cells**: 0
  - fusion source breakdown: both = 12, cy5_only = 1, dic_only = 4

## Parameters deviating from defaults

  - **auto_selected_pipeline**: `cpsam` (not in defaults)
  - **model_path**: `None` (not in defaults)
  - **use_cy5_recovery**: `False` (default `True`)

## Environment

- **conda env**: `cellpose4`
- **python**: `3.10.17`
- **platform**: `macOS-26.4.1-arm64-arm-64bit`
- **cellpose**: `?`
- **numpy**: `2.0.2`
- **tifffile**: `2023.2.28`
- **scipy**: `1.15.3`
- **skimage**: `0.25.2`
- **torch**: `2.7.0`

## Reproducibility

- **git commit**: `(not a git repo or git unavailable)`
- **rerun command**:

```bash
conda run -n cellpose python scripts/run_pipeline_on_gt_recording.py data/ic295_gt_full/Pos7_WT
```

## Output directory

`/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos7_WT/pipeline_results`

---
*Generated automatically. If this file is missing alongside results,
the run did not write metadata — see core/run_metadata.py for the
expected API. This is a requirement: every analysis path (GUI export,
batch worker, programmatic script) MUST call `write_run_metadata`.*
