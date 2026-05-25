# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos68_DMSO`
**GT frames evaluated**: 11
**GT cells (unique IDs)**: 21

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 13.9 | 3.5 | 3.3 | 0.77 | 0.77 |
| IoU≥0.5 | 13.4 | 4.1 | 3.8 | 0.74 | 0.75 |
| IoU≥0.7 | 11.1 | 6.4 | 6.1 | 0.61 | 0.62 |

- **Mean per-cell IoU (matched)**: 0.775
- **Median per-cell IoU (matched)**: 0.835
- **Out-of-scope predictions/frame**: 1.9

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **86.46%**
- **GT cells with perfect 1.0 consistency**:
  11 / 21

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 3 | 1.00 |
| 6 | 10 | 4 | 1.00 |
| 7 | 10 | 7 | 1.00 |
| 10 | 7 | 8 | 1.00 |
| 14 | 10 | 13 | 1.00 |
| 15 | 8 | 15 | 1.00 |
| 5 | 6 | 17 | 1.00 |
| 12 | 4 | 9 | 1.00 |
| 20 | 6 | 18 | 1.00 |
| 9 | 6 | 10 | 1.00 |
| 17 | 6 | 14 | 1.00 |
| 3 | 10 | 2 | 0.90 |
| 2 | 10 | 1 | 0.80 |
| 11 | 10 | 12 | 0.80 |
| 4 | 5 | 23 | 0.80 |
| 18 | 10 | 16 | 0.70 |
| 13 | 3 | 11 | 0.67 |
| 19 | 3 | 6 | 0.67 |
| 21 | 3 | 19 | 0.67 |
| 8 | 10 | 21 | 0.60 |
| 16 | 9 | 19 | 0.56 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 18 | 14 | 14 | 0 | 4 | 0.88 |
| F10 | 18 | 15 | 13 | 2 | 5 | 0.79 |
| F20 | 19 | 17 | 16 | 1 | 3 | 0.89 |
| F30 | 17 | 17 | 16 | 1 | 1 | 0.94 |
| F35 | 1 | 17 | 0 | 17 | 1 | 0.00 |
| F40 | 19 | 18 | 14 | 4 | 5 | 0.76 |
| F50 | 20 | 21 | 16 | 5 | 4 | 0.78 |
| F60 | 20 | 21 | 16 | 5 | 4 | 0.78 |
| F70 | 21 | 18 | 17 | 1 | 4 | 0.87 |
| F80 | 20 | 16 | 12 | 4 | 8 | 0.67 |
| F90 | 19 | 15 | 13 | 2 | 6 | 0.77 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
