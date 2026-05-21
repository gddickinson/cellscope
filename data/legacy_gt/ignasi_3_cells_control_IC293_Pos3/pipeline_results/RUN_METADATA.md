# RUN_METADATA — IC293__1_MMStack_Pos3-WT.ome-cropped.tif

**Schema version**: 1.0.0
**Started**: 2026-05-21T07:39:52
**Finished**: 2026-05-21T07:39:52
**Runtime**: 594 s
**Pipeline**: `hybrid_cpsam_multi` (mode = multi)

## Source recording

- **video_path**: `/Users/george/claude_test/cellscope/data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/IC293__1_MMStack_Pos3-WT.ome-cropped.tif`
- **n_frames**: 97
- **has Cy5 channel**: False
- **um_per_px**: 1.0
- **time_interval_min**: 1.0
- **checksum**: `sha256-64mb:bc295e2e4012eb40`

## Results

- **n_tracks**: 3
- **n_cy5_fusion_added** (pre-tracking): 0
- **n_analysis_cells**: 0
  - fusion source breakdown: dic_only = 3

## Parameters deviating from defaults

  - **auto_selected_pipeline**: `cpsam` (not in defaults)
  - **cy5_filter_mode**: `None` (default `multi_metric`)
  - **downsample**: `2` (not in defaults)
  - **downsample_reason**: `auto: max dim 1028 in [900, 1500); ds=2 keeps cells resolvable while giving a ~3× speedup` (not in defaults)
  - **downsample_spec**: `auto` (not in defaults)
  - **model_path**: `None` (not in defaults)

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
conda run -n cellpose4 python scripts/run_pipeline_on_gt_recording.py data/legacy_gt/ignasi_3_cells_control_IC293_Pos3
```

## Output directory

`/Users/george/claude_test/cellscope/data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/pipeline_results`

---
*Generated automatically. If this file is missing alongside results,
the run did not write metadata — see core/run_metadata.py for the
expected API. This is a requirement: every analysis path (GUI export,
batch worker, programmatic script) MUST call `write_run_metadata`.*
