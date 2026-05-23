# GT evaluation aggregate

_13 recordings, generated Sat May 23 09:30:22 CDT 2026_

| Recording | GT frames | Mean IoU | F1@.5 | F1@.5 focused | Mean TP/frame | Mean FN | Mean FP | OOS pred | ID consistency | Perfect tracks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pos10_WT | 10 | 0.863 | 0.84 | 0.97 | 2.9 | 0.1 | 1.0 | 0.9 | 93.33% | 2/3 |
| Pos20_KO | 10 | 0.842 | 0.88 | 0.90 | 6.2 | 1.3 | 0.3 | 0.3 | 92.28% | 7/9 |
| Pos21_KO | 10 | 0.799 | 0.61 | 0.61 | 2.8 | 3.5 | 0.2 | 0.0 | 91.33% | 3/5 |
| Pos30_GOF | 10 | 0.860 | 0.86 | 0.86 | 4.8 | 1.5 | 0.0 | 0.0 | 100.00% | 5/5 |
| Pos31_GOF | 10 | 0.873 | 0.52 | 0.52 | 1.3 | 2.4 | 0.0 | 0.0 | 100.00% | 2/2 |
| Pos39_OT | 10 | 0.855 | 0.95 | 0.96 | 6.9 | 0.5 | 0.3 | 0.2 | 97.78% | 8/9 |
| Pos44_OT | 10 | 0.881 | 0.74 | 0.79 | 3.5 | 1.9 | 0.6 | 0.6 | 98.00% | 4/5 |
| Pos51_Y1 | 10 | 0.866 | 1.00 | 1.00 | 1.3 | 0.0 | 0.0 | 0.0 | 100.00% | 2/2 |
| Pos68_DMSO | 11 | 0.762 | 0.56 | 0.58 | 8.6 | 8.8 | 2.9 | 1.6 | 88.37% | 8/15 |
| Pos7_WT | 10 | 0.851 | 0.77 | 0.85 | 8.3 | 2.5 | 2.5 | 2.0 | 92.22% | 7/9 |
| ignasi_3_cells_control_IC293_Pos3 | 97 | 0.820 | 0.87 | 0.87 | 2.6 | 0.4 | 0.3 | 0.0 | 93.01% | 1/3 |
| ignasi_control | 15 | 0.924 | 0.63 | 1.00 | 1.0 | 0.0 | 1.2 | 1.2 | 100.00% | 1/1 |
| ignasi_control_full | 65 | 0.932 | 0.66 | 1.00 | 1.0 | 0.0 | 1.0 | 1.0 | 100.00% | 1/1 |

## Per-recording reports

- **Pos10_WT** — `data/ic295_gt_full/Pos10_WT/evaluation/report.md`
- **Pos20_KO** — `data/ic295_gt_full/Pos20_KO/evaluation/report.md`
- **Pos21_KO** — `data/ic295_gt_full/Pos21_KO/evaluation/report.md`
- **Pos30_GOF** — `data/ic295_gt_full/Pos30_GOF/evaluation/report.md`
- **Pos31_GOF** — `data/ic295_gt_full/Pos31_GOF/evaluation/report.md`
- **Pos39_OT** — `data/ic295_gt_full/Pos39_OT/evaluation/report.md`
- **Pos44_OT** — `data/ic295_gt_full/Pos44_OT/evaluation/report.md`
- **Pos51_Y1** — `data/ic295_gt_full/Pos51_Y1/evaluation/report.md`
- **Pos68_DMSO** — `data/ic295_gt_full/Pos68_DMSO/evaluation/report.md`
- **Pos7_WT** — `data/ic295_gt_full/Pos7_WT/evaluation/report.md`
- **ignasi_3_cells_control_IC293_Pos3** — `data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/evaluation/report.md`
- **ignasi_control** — `data/legacy_gt/ignasi_control/evaluation/report.md`
- **ignasi_control_full** — `data/legacy_gt/ignasi_control_full/evaluation/report.md`

## Aggregate (across recordings)

- Mean per-cell IoU: **0.856** (across 278 annotated frames)
- Mean F1 @ IoU≥0.5: **0.76**
- Mean F1 @ IoU≥0.5 focused: **0.84** (excludes predictions with no GT overlap from FP count)
- Mean ID consistency: **95.87%**
