# RUN_METADATA — ignasi_15frames_for_GT.tif

**Schema version**: 1.0.0
**Started**: 2026-05-15T18:59:22
**Finished**: 2026-05-15T18:59:22
**Runtime**: 62 s
**Pipeline**: `hybrid_dic_multi` (mode = multi)

## Source recording

- **video_path**: `/Users/george/claude_test/cellscope/data/legacy_gt/ignasi_control/ignasi_15frames_for_GT.tif`
- **n_frames**: 15
- **has Cy5 channel**: False
- **um_per_px**: 1.0
- **time_interval_min**: 1.0
- **checksum**: `sha256-64mb:0e4674307a0310c5`

## Results

- **n_tracks**: 1
- **n_cy5_fusion_added** (pre-tracking): 0
- **n_analysis_cells**: 0
  - fusion source breakdown: dic_only = 1

## Parameters deviating from defaults

  - **model_path**: `data/models/cpsam_dic` (not in defaults)

## Environment

- **conda env**: `cellpose`
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
conda run -n cellpose python scripts/run_pipeline_on_gt_recording.py data/legacy_gt/ignasi_control
```

## Output directory

`/Users/george/claude_test/cellscope/data/legacy_gt/ignasi_control/pipeline_results`

---
*Generated automatically. If this file is missing alongside results,
the run did not write metadata — see core/run_metadata.py for the
expected API. This is a requirement: every analysis path (GUI export,
batch worker, programmatic script) MUST call `write_run_metadata`.*
