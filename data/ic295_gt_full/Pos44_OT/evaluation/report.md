# Evaluation report

**Recording**: `/Users/george/cellscope/data/ic295_gt_full/Pos44_OT`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 7

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 4.6 | 0.8 | 2.3 | 0.75 | 0.92 |
| IoU≥0.5 | 4.5 | 0.9 | 2.4 | 0.73 | 0.90 |
| IoU≥0.7 | 4.5 | 0.9 | 2.4 | 0.73 | 0.90 |

- **Mean per-cell IoU (matched)**: 0.872
- **Median per-cell IoU (matched)**: 0.884
- **Out-of-scope predictions/frame**: 2.3

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **98.57%**
- **GT cells with perfect 1.0 consistency**:
  6 / 7

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 1 | 1.00 |
| 2 | 10 | 3 | 1.00 |
| 5 | 10 | 4 | 1.00 |
| 6 | 4 | 9 | 1.00 |
| 8 | 1 | 7 | 1.00 |
| 4 | 1 | 5 | 1.00 |
| 3 | 10 | 5 | 0.90 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 5 | 6 | 4 | 2 | 1 | 0.73 |
| F10 | 4 | 7 | 4 | 3 | 0 | 0.73 |
| F20 | 5 | 6 | 3 | 3 | 2 | 0.55 |
| F30 | 4 | 6 | 4 | 2 | 0 | 0.80 |
| F40 | 5 | 6 | 4 | 2 | 1 | 0.73 |
| F50 | 5 | 7 | 4 | 3 | 1 | 0.67 |
| F60 | 6 | 8 | 5 | 3 | 1 | 0.71 |
| F70 | 6 | 8 | 5 | 3 | 1 | 0.71 |
| F80 | 7 | 8 | 6 | 2 | 1 | 0.80 |
| F90 | 7 | 7 | 6 | 1 | 1 | 0.86 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
