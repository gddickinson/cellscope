# GT evaluation aggregate

_8 recordings, generated Mon May 18 20:41:10 CDT 2026_

| Recording | GT frames | Mean IoU | F1@.5 | Mean TP/frame | Mean FN | Mean FP | ID consistency | Perfect tracks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pos20_KO | 10 | 0.839 | 0.85 | 6.2 | 1.3 | 0.9 | 95.00% | 7/8 |
| Pos30_GOF | 10 | 0.848 | 0.84 | 4.7 | 1.6 | 0.2 | 92.00% | 3/5 |
| Pos39_OT | 10 | 0.855 | 0.95 | 6.9 | 0.5 | 0.3 | 97.78% | 8/9 |
| Pos51_Y1 | 10 | 0.753 | 0.90 | 1.1 | 0.2 | 0.1 | 83.33% | 1/2 |
| Pos7_WT | 10 | 0.844 | 0.82 | 8.5 | 2.3 | 1.3 | 100.00% | 9/9 |
| ignasi_3_cells_control_IC293_Pos3 | 97 | 0.820 | 0.87 | 2.6 | 0.4 | 0.3 | 93.01% | 1/3 |
| ignasi_control | 15 | 0.890 | 0.80 | 0.8 | 0.2 | 0.0 | 100.00% | 1/1 |
| ignasi_control_full | 65 | 0.897 | 0.92 | 0.9 | 0.1 | 0.0 | 100.00% | 1/1 |

## Per-recording reports

- **Pos20_KO** — `data/ic295_gt_full/Pos20_KO/evaluation/report.md`
- **Pos30_GOF** — `data/ic295_gt_full/Pos30_GOF/evaluation/report.md`
- **Pos39_OT** — `data/ic295_gt_full/Pos39_OT/evaluation/report.md`
- **Pos51_Y1** — `data/ic295_gt_full/Pos51_Y1/evaluation/report.md`
- **Pos7_WT** — `data/ic295_gt_full/Pos7_WT/evaluation/report.md`
- **ignasi_3_cells_control_IC293_Pos3** — `data/legacy_gt/ignasi_3_cells_control_IC293_Pos3/evaluation/report.md`
- **ignasi_control** — `data/legacy_gt/ignasi_control/evaluation/report.md`
- **ignasi_control_full** — `data/legacy_gt/ignasi_control_full/evaluation/report.md`

## Aggregate (across recordings)

- Mean per-cell IoU: **0.843** (across 227 annotated frames)
- Mean F1 @ IoU≥0.5: **0.87**
- Mean ID consistency: **95.14%**
