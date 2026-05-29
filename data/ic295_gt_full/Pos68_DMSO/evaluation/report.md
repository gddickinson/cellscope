# Evaluation report

**Recording**: `/Users/george/cellscope/data/ic295_gt_full/Pos68_DMSO`
**GT frames evaluated**: 11
**GT cells (unique IDs)**: 21

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 14.5 | 3.0 | 4.1 | 0.77 | 0.81 |
| IoU≥0.5 | 13.6 | 3.8 | 4.9 | 0.73 | 0.76 |
| IoU≥0.7 | 11.4 | 6.1 | 7.2 | 0.61 | 0.64 |

- **Mean per-cell IoU (matched)**: 0.756
- **Median per-cell IoU (matched)**: 0.830
- **Out-of-scope predictions/frame**: 3.5

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **90.32%**
- **GT cells with perfect 1.0 consistency**:
  16 / 21

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 3 | 1.00 |
| 2 | 9 | 1 | 1.00 |
| 3 | 10 | 2 | 1.00 |
| 6 | 10 | 4 | 1.00 |
| 7 | 10 | 7 | 1.00 |
| 11 | 10 | 12 | 1.00 |
| 12 | 9 | 9 | 1.00 |
| 13 | 4 | 11 | 1.00 |
| 14 | 10 | 14 | 1.00 |
| 15 | 8 | 16 | 1.00 |
| 18 | 10 | 5 | 1.00 |
| 5 | 7 | 17 | 1.00 |
| 20 | 8 | 18 | 1.00 |
| 9 | 6 | 10 | 1.00 |
| 17 | 6 | 15 | 1.00 |
| 4 | 4 | 23 | 1.00 |
| 21 | 3 | 22 | 0.67 |
| 8 | 10 | 21 | 0.60 |
| 10 | 10 | 8 | 0.60 |
| 16 | 10 | 19 | 0.60 |
| 19 | 2 | 6 | 0.50 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 18 | 16 | 14 | 2 | 4 | 0.82 |
| F10 | 18 | 15 | 14 | 1 | 4 | 0.85 |
| F20 | 19 | 18 | 15 | 3 | 4 | 0.81 |
| F30 | 17 | 19 | 15 | 4 | 2 | 0.83 |
| F35 | 1 | 20 | 0 | 20 | 1 | 0.00 |
| F40 | 19 | 18 | 14 | 4 | 5 | 0.76 |
| F50 | 20 | 21 | 17 | 4 | 3 | 0.83 |
| F60 | 20 | 21 | 17 | 4 | 3 | 0.83 |
| F70 | 21 | 19 | 17 | 2 | 4 | 0.85 |
| F80 | 20 | 19 | 13 | 6 | 7 | 0.67 |
| F90 | 19 | 18 | 14 | 4 | 5 | 0.76 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
