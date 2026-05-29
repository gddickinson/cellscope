# Evaluation report

**Recording**: `/Users/george/cellscope/data/ic295_gt_full/Pos20_KO`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 9

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 7.2 | 0.3 | 0.6 | 0.94 | 0.98 |
| IoU≥0.5 | 7.2 | 0.3 | 0.6 | 0.94 | 0.98 |
| IoU≥0.7 | 6.8 | 0.7 | 1.0 | 0.89 | 0.92 |

- **Mean per-cell IoU (matched)**: 0.845
- **Median per-cell IoU (matched)**: 0.873
- **Out-of-scope predictions/frame**: 0.6

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **97.53%**
- **GT cells with perfect 1.0 consistency**:
  8 / 9

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 2 | 2 | 2 | 1.00 |
| 5 | 10 | 4 | 1.00 |
| 6 | 10 | 1 | 1.00 |
| 7 | 10 | 5 | 1.00 |
| 8 | 7 | 6 | 1.00 |
| 9 | 10 | 7 | 1.00 |
| 1 | 9 | 8 | 1.00 |
| 10 | 5 | 11 | 1.00 |
| 3 | 9 | 2 | 0.78 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 9 | 7 | 7 | 0 | 2 | 0.88 |
| F10 | 8 | 9 | 8 | 1 | 0 | 0.94 |
| F20 | 7 | 7 | 7 | 0 | 0 | 1.00 |
| F30 | 7 | 8 | 7 | 1 | 0 | 0.93 |
| F40 | 7 | 7 | 7 | 0 | 0 | 1.00 |
| F50 | 8 | 9 | 8 | 1 | 0 | 0.94 |
| F60 | 8 | 8 | 8 | 0 | 0 | 1.00 |
| F70 | 7 | 8 | 7 | 1 | 0 | 0.93 |
| F80 | 7 | 8 | 7 | 1 | 0 | 0.93 |
| F90 | 7 | 7 | 6 | 1 | 1 | 0.86 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
