# GT evaluation aggregate

_9 recordings, generated Thu May 21 07:55:00 CDT 2026_

| Recording | GT frames | Mean IoU | F1@.5 | Mean TP/frame | Mean FN | Mean FP | ID consistency | Perfect tracks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pos20_KO | 10 | 0.842 | 0.88 | 6.2 | 1.3 | 0.3 | 92.28% | 7/9 |
| Pos30_GOF | 10 | 0.860 | 0.86 | 4.8 | 1.5 | 0.0 | 100.00% | 5/5 |
| Pos39_OT | 10 | 0.863 | 0.94 | 6.6 | 0.8 | 0.1 | 93.75% | 7/8 |
| Pos51_Y1 | 10 | 0.866 | 1.00 | 1.3 | 0.0 | 0.0 | 100.00% | 2/2 |
| Pos68_DMSO | 11 | 0.762 | 0.56 | 8.6 | 8.8 | 2.9 | 88.37% | 8/15 |
| Pos7_WT | 10 | 0.851 | 0.77 | 8.3 | 2.5 | 2.5 | 92.22% | 7/9 |
| ignasi_3_cells_control_IC293_Pos3 | 97 | 0.820 | 0.87 | 2.6 | 0.4 | 0.3 | 93.01% | 1/3 |
| ignasi_control | 15 | 0.924 | 0.63 | 1.0 | 0.0 | 1.2 | 100.00% | 1/1 |
| ignasi_control_full | 65 | 0.932 | 0.66 | 1.0 | 0.0 | 1.0 | 100.00% | 1/1 |

## Per-recording reports

- **Pos20_KO** — `data/ic295_gt_full/Pos20_KO/evaluation/report.md`
- **Pos30_GOF** — `data/ic295_gt_full/Pos30_GOF/evaluation/report.md`
- **Pos39_OT** — `data/ic295_gt_full/Pos39_OT/evaluation/report.md`
- **Pos51_Y1** — `data/ic295_gt_full/Pos51_Y1/evaluation/report.md`
- **Pos68_DMSO** — `data/ic295_gt_full/Pos68_DMSO/evaluation/report.md`
- **Pos7_WT** — `data/ic295_gt_full/Pos7_WT/evaluation/report.md`
- **ignasi_3_cells_control_IC293_Pos3** — `data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/evaluation/report.md`
- **ignasi_control** — `data/legacy_gt/ignasi_control/evaluation/report.md`
- **ignasi_control_full** — `data/legacy_gt/ignasi_control_full/evaluation/report.md`

## Aggregate (across recordings)

- Mean per-cell IoU: **0.858** (across 238 annotated frames)
- Mean F1 @ IoU≥0.5: **0.80**
- Mean ID consistency: **95.51%**
