# RUN_METADATA — ignasi_3_cells_control_IC293_Pos3 (cpsam variant)

**Schema version**: 1.0.0
**Started**: 2026-05-15T19:41:00
**Finished**: 2026-05-15T19:41:00
**Runtime**: 1254 s
**Pipeline**: `hybrid_cpsam_multi` (mode = multi)

## Source recording

- **video_path**: `data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/IC293__1_MMStack_Pos3-WT.ome-cropped.tif`
- **n_frames**: 97
- **has Cy5 channel**: False
- **um_per_px**: 0.6523
- **time_interval_min**: 10.0
- **checksum**: `sha256-64mb:bc295e2e4012eb40`

## Results

- **n_tracks**: 4
- **n_cy5_fusion_added** (pre-tracking): 0
- **n_analysis_cells**: 0
  - fusion source breakdown: dic_only = 4

## Parameters deviating from defaults

  - **model_used**: `raw cpsam (vit_h)` (not in defaults)

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
python scripts/rerun_pos3_with_cpsam.py
```

## Output directory

`/Users/george/claude_test/cellscope/data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/pipeline_results_cpsam`

---
*Generated automatically. If this file is missing alongside results,
the run did not write metadata — see core/run_metadata.py for the
expected API. This is a requirement: every analysis path (GUI export,
batch worker, programmatic script) MUST call `write_run_metadata`.*
