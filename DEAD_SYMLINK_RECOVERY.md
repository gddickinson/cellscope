# Dead-symlink recovery record

Generated 2026-06-04 10:01 by `scripts/fix_dead_symlinks.py` after the GeorgeDrive external disk failed.

If the drive is ever recovered, the **Removed** links below can be recreated with `ln -s <original target> <path>`. No real files or `gt_masks/*.png` were deleted — only dangling symlinks.

## Removed (no local copy existed)

| path | original target |
|---|---|
| `data/ic295_gt` | `/Volumes/GeorgeDrive/cellscope_data/cellscope/data/ic295_gt` |
| `data/ic295_gt_full/Pos53_Y1/IC295__1_MMStack_Pos53-Y1.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos53-Y1.ome.tif` |
| `data/ic295_gt_full/Pos53_Y1/IC295__1_MMStack_Pos53-Y1_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos53-Y1_metadata.txt` |
| `data/ic295_gt_full/Pos69_DMSO/IC295__1_MMStack_Pos69-DMSO.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos69-DMSO.ome.tif` |
| `data/ic295_gt_full/Pos69_DMSO/IC295__1_MMStack_Pos69-DMSO_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos69-DMSO_metadata.txt` |
| `data/ic295_inspection` | `/Volumes/GeorgeDrive/cellscope_data/cellscope/data/ic295_inspection` |
| `data/ignasi_new_gt` | `/Volumes/GeorgeDrive/cellscope_data/cellscope/data/ignasi_new_gt` |
| `data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/IC293__1_MMStack_Pos3-WT.ome-cropped.tif` | `/Users/george/claude_test/piezo1_analysis/data/ignasi/IC293__1_MMStack_Pos3-WT.ome-cropped.tif` |
| `data/legacy_gt/ignasi_control/ignasi_15frames_for_GT.tif` | `/Volumes/GeorgeDrive/cellscope_data/piezo1_analysis/data/ignasi/ignasi_15frames_for_GT.tif` |
| `data/legacy_gt/ignasi_control_full/C1-IC293__1_MMStack_Pos0-WT.ome-1cropped.tif` | `/Volumes/GeorgeDrive/cellscope_data/piezo1_analysis/data/ignasi/C1-IC293__1_MMStack_Pos0-WT.ome-1cropped.tif` |
| `data/training` | `/Volumes/GeorgeDrive/cellscope_data/cellscope/data/training` |
| `fiji_export_test` | `/Volumes/GeorgeDrive/cellscope_data/cellscope/fiji_export_test` |
| `gt_review/ignasi_3_cells_control_IC293_Pos3/IC293__1_MMStack_Pos3-WT.ome-cropped.tif` | `/Users/george/claude_test/cellscope/data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/IC293__1_MMStack_Pos3-WT.ome-cropped.tif` |
| `gt_review/ignasi_control/ignasi_15frames_for_GT.tif` | `/Users/george/claude_test/cellscope/data/legacy_gt/ignasi_control/ignasi_15frames_for_GT.tif` |
| `gt_review/ignasi_control_full/C1-IC293__1_MMStack_Pos0-WT.ome-1cropped.tif` | `/Users/george/claude_test/cellscope/data/legacy_gt/ignasi_control_full/C1-IC293__1_MMStack_Pos0-WT.ome-1cropped.tif` |

## Repointed to local copies

| path | old target | new target |
|---|---|---|
| `data/ic295_gt_full/Pos10_WT/IC295__1_MMStack_Pos10-WT.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos10-WT.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos10-WT.ome.tif` |
| `data/ic295_gt_full/Pos10_WT/IC295__1_MMStack_Pos10-WT_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos10-WT_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos10-WT_metadata.txt` |
| `data/ic295_gt_full/Pos20_KO/IC295__1_MMStack_Pos20-KO.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos20-KO.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos20-KO.ome.tif` |
| `data/ic295_gt_full/Pos20_KO/IC295__1_MMStack_Pos20-KO_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos20-KO_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos20-KO_metadata.txt` |
| `data/ic295_gt_full/Pos21_KO/IC295__1_MMStack_Pos21-KO.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos21-KO.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos21-KO.ome.tif` |
| `data/ic295_gt_full/Pos21_KO/IC295__1_MMStack_Pos21-KO_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos21-KO_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos21-KO_metadata.txt` |
| `data/ic295_gt_full/Pos30_GOF/IC295__1_MMStack_Pos30-GOF.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos30-GOF.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos30-GOF.ome.tif` |
| `data/ic295_gt_full/Pos30_GOF/IC295__1_MMStack_Pos30-GOF_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos30-GOF_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos30-GOF_metadata.txt` |
| `data/ic295_gt_full/Pos31_GOF/IC295__1_MMStack_Pos31-GOF.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos31-GOF.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos31-GOF.ome.tif` |
| `data/ic295_gt_full/Pos31_GOF/IC295__1_MMStack_Pos31-GOF_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos31-GOF_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos31-GOF_metadata.txt` |
| `data/ic295_gt_full/Pos39_OT/IC295__1_MMStack_Pos39-OT.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295/IC295__1_MMStack_Pos39-OT.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos39-OT.ome.tif` |
| `data/ic295_gt_full/Pos39_OT/IC295__1_MMStack_Pos39-OT_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295/IC295__1_MMStack_Pos39-OT_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos39-OT_metadata.txt` |
| `data/ic295_gt_full/Pos44_OT/IC295__1_MMStack_Pos44-OT.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos44-OT.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos44-OT.ome.tif` |
| `data/ic295_gt_full/Pos44_OT/IC295__1_MMStack_Pos44-OT_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos44-OT_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos44-OT_metadata.txt` |
| `data/ic295_gt_full/Pos51_Y1/IC295__1_MMStack_Pos51-Y1.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295/IC295__1_MMStack_Pos51-Y1.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos51-Y1.ome.tif` |
| `data/ic295_gt_full/Pos51_Y1/IC295__1_MMStack_Pos51-Y1_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295/IC295__1_MMStack_Pos51-Y1_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos51-Y1_metadata.txt` |
| `data/ic295_gt_full/Pos68_DMSO/IC295__1_MMStack_Pos68-DMSO.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos68-DMSO.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos68-DMSO.ome.tif` |
| `data/ic295_gt_full/Pos68_DMSO/IC295__1_MMStack_Pos68-DMSO_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos68-DMSO_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos68-DMSO_metadata.txt` |
| `data/ic295_gt_full/Pos7_WT/IC295__1_MMStack_Pos7-WT.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos7-WT.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos7-WT.ome.tif` |
| `data/ic295_gt_full/Pos7_WT/IC295__1_MMStack_Pos7-WT_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos7-WT_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos7-WT_metadata.txt` |
| `gt_review/Pos10_WT/IC295__1_MMStack_Pos10-WT.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos10-WT.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos10-WT.ome.tif` |
| `gt_review/Pos10_WT/IC295__1_MMStack_Pos10-WT_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos10-WT_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos10-WT_metadata.txt` |
| `gt_review/Pos10_WT/pipeline_results/masks.npz` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/processed/Pos10-WT/pipeline_results/masks.npz` | `ic295_analysis/by_condition/WT/Pos10-WT/pipeline_results/masks.npz` |
| `gt_review/Pos20_KO/IC295__1_MMStack_Pos20-KO.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos20-KO.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos20-KO.ome.tif` |
| `gt_review/Pos20_KO/IC295__1_MMStack_Pos20-KO_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos20-KO_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos20-KO_metadata.txt` |
| `gt_review/Pos20_KO/pipeline_results/masks.npz` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/processed/Pos20-KO/pipeline_results/masks.npz` | `ic295_analysis/by_condition/KO/Pos20-KO/pipeline_results/masks.npz` |
| `gt_review/Pos21_KO/IC295__1_MMStack_Pos21-KO.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos21-KO.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos21-KO.ome.tif` |
| `gt_review/Pos21_KO/IC295__1_MMStack_Pos21-KO_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos21-KO_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos21-KO_metadata.txt` |
| `gt_review/Pos21_KO/pipeline_results/masks.npz` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/processed/Pos21-KO/pipeline_results/masks.npz` | `ic295_analysis/by_condition/KO/Pos21-KO/pipeline_results/masks.npz` |
| `gt_review/Pos30_GOF/IC295__1_MMStack_Pos30-GOF.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos30-GOF.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos30-GOF.ome.tif` |
| `gt_review/Pos30_GOF/IC295__1_MMStack_Pos30-GOF_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos30-GOF_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos30-GOF_metadata.txt` |
| `gt_review/Pos30_GOF/pipeline_results/masks.npz` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/processed/Pos30-GOF/pipeline_results/masks.npz` | `ic295_analysis/by_condition/GOF/Pos30-GOF/pipeline_results/masks.npz` |
| `gt_review/Pos31_GOF/IC295__1_MMStack_Pos31-GOF.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos31-GOF.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos31-GOF.ome.tif` |
| `gt_review/Pos31_GOF/IC295__1_MMStack_Pos31-GOF_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos31-GOF_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos31-GOF_metadata.txt` |
| `gt_review/Pos31_GOF/pipeline_results/masks.npz` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/processed/Pos31-GOF/pipeline_results/masks.npz` | `ic295_analysis/by_condition/GOF/Pos31-GOF/pipeline_results/masks.npz` |
| `gt_review/Pos39_OT/IC295__1_MMStack_Pos39-OT.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295/IC295__1_MMStack_Pos39-OT.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos39-OT.ome.tif` |
| `gt_review/Pos39_OT/IC295__1_MMStack_Pos39-OT_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295/IC295__1_MMStack_Pos39-OT_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos39-OT_metadata.txt` |
| `gt_review/Pos39_OT/pipeline_results/masks.npz` | `/Volumes/GeorgeDrive/ignasi/IC295/processed/Pos39-OT/pipeline_results/masks.npz` | `ic295_analysis/by_condition/OT/Pos39-OT/pipeline_results/masks.npz` |
| `gt_review/Pos44_OT/IC295__1_MMStack_Pos44-OT.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos44-OT.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos44-OT.ome.tif` |
| `gt_review/Pos44_OT/IC295__1_MMStack_Pos44-OT_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos44-OT_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos44-OT_metadata.txt` |
| `gt_review/Pos44_OT/pipeline_results/masks.npz` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/processed/Pos44-OT/pipeline_results/masks.npz` | `ic295_analysis/by_condition/OT/Pos44-OT/pipeline_results/masks.npz` |
| `gt_review/Pos51_Y1/IC295__1_MMStack_Pos51-Y1.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295/IC295__1_MMStack_Pos51-Y1.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos51-Y1.ome.tif` |
| `gt_review/Pos51_Y1/IC295__1_MMStack_Pos51-Y1_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295/IC295__1_MMStack_Pos51-Y1_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos51-Y1_metadata.txt` |
| `gt_review/Pos51_Y1/pipeline_results/masks.npz` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/processed/Pos51-Y1/pipeline_results/masks.npz` | `ic295_analysis/by_condition/Y1/Pos51-Y1/pipeline_results/masks.npz` |
| `gt_review/Pos68_DMSO/IC295__1_MMStack_Pos68-DMSO.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos68-DMSO.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos68-DMSO.ome.tif` |
| `gt_review/Pos68_DMSO/IC295__1_MMStack_Pos68-DMSO_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos68-DMSO_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos68-DMSO_metadata.txt` |
| `gt_review/Pos68_DMSO/pipeline_results/masks.npz` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/processed/Pos68-DMSO/pipeline_results/masks.npz` | `ic295_analysis/by_condition/DMSO/Pos68-DMSO/pipeline_results/masks.npz` |
| `gt_review/Pos7_WT/IC295__1_MMStack_Pos7-WT.ome.tif` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos7-WT.ome.tif` | `ic295_analysis/_cache/IC295__1_MMStack_Pos7-WT.ome.tif` |
| `gt_review/Pos7_WT/IC295__1_MMStack_Pos7-WT_metadata.txt` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/IC295__1_MMStack_Pos7-WT_metadata.txt` | `ic295_analysis/_cache/IC295__1_MMStack_Pos7-WT_metadata.txt` |
| `gt_review/Pos7_WT/pipeline_results/masks.npz` | `/Volumes/GeorgeDrive/ignasi/IC295_batch2/processed/Pos7-WT/pipeline_results/masks.npz` | `ic295_analysis/by_condition/WT/Pos7-WT/pipeline_results/masks.npz` |

## Materialized as real directories

- `results` (was → `/Volumes/GeorgeDrive/cellscope_data/cellscope/results`)
