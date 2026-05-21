# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos68_DMSO`
**GT frames evaluated**: 11
**GT cells (unique IDs)**: 15

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 8.9 | 8.5 | 2.6 | 0.58 | 0.60 |
| IoU≥0.5 | 8.6 | 8.8 | 2.9 | 0.56 | 0.58 |
| IoU≥0.7 | 7.6 | 9.8 | 3.9 | 0.50 | 0.51 |

- **Mean per-cell IoU (matched)**: 0.762
- **Median per-cell IoU (matched)**: 0.839
- **Out-of-scope predictions/frame**: 1.6

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **88.37%**
- **GT cells with perfect 1.0 consistency**:
  8 / 15

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 3 | 1.00 |
| 6 | 10 | 4 | 1.00 |
| 7 | 10 | 7 | 1.00 |
| 13 | 2 | 10 | 1.00 |
| 9 | 6 | 9 | 1.00 |
| 4 | 1 | 1 | 1.00 |
| 15 | 1 | 15 | 1.00 |
| 16 | 2 | 15 | 1.00 |
| 3 | 10 | 2 | 0.90 |
| 2 | 9 | 1 | 0.89 |
| 11 | 10 | 11 | 0.80 |
| 10 | 10 | 8 | 0.70 |
| 18 | 10 | 13 | 0.70 |
| 19 | 3 | 6 | 0.67 |
| 8 | 10 | 14 | 0.60 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 18 | 12 | 11 | 1 | 7 | 0.73 |
| F10 | 18 | 11 | 9 | 2 | 9 | 0.62 |
| F20 | 19 | 11 | 10 | 1 | 9 | 0.67 |
| F30 | 17 | 11 | 10 | 1 | 7 | 0.71 |
| F35 | 1 | 11 | 0 | 11 | 1 | 0.00 |
| F40 | 19 | 12 | 9 | 3 | 10 | 0.58 |
| F50 | 20 | 12 | 9 | 3 | 11 | 0.56 |
| F60 | 20 | 12 | 8 | 4 | 12 | 0.50 |
| F70 | 21 | 12 | 10 | 2 | 11 | 0.61 |
| F80 | 20 | 12 | 9 | 3 | 11 | 0.56 |
| F90 | 19 | 11 | 10 | 1 | 9 | 0.67 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
