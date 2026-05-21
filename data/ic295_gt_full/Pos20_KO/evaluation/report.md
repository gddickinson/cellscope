# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos20_KO`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 8

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 |
|---:|---:|---:|---:|---:|
| IoU≥0.3 | 6.2 | 1.3 | 0.8 | 0.85 |
| IoU≥0.5 | 6.2 | 1.3 | 0.8 | 0.85 |
| IoU≥0.7 | 5.7 | 1.8 | 1.3 | 0.78 |

- **Mean per-cell IoU (matched)**: 0.839
- **Median per-cell IoU (matched)**: 0.870

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **95.00%**
- **GT cells with perfect 1.0 consistency**:
  7 / 8

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 2 | 2 | 3 | 1.00 |
| 3 | 8 | 4 | 1.00 |
| 5 | 10 | 5 | 1.00 |
| 6 | 10 | 2 | 1.00 |
| 7 | 10 | 6 | 1.00 |
| 8 | 7 | 7 | 1.00 |
| 10 | 5 | 9 | 1.00 |
| 1 | 10 | 1 | 0.60 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 9 | 7 | 7 | 0 | 2 | 0.88 |
| F10 | 8 | 7 | 7 | 0 | 1 | 0.93 |
| F20 | 7 | 7 | 6 | 1 | 1 | 0.86 |
| F30 | 7 | 7 | 6 | 1 | 1 | 0.86 |
| F40 | 7 | 7 | 6 | 1 | 1 | 0.86 |
| F50 | 8 | 9 | 7 | 2 | 1 | 0.82 |
| F60 | 8 | 8 | 7 | 1 | 1 | 0.88 |
| F70 | 7 | 7 | 5 | 2 | 2 | 0.71 |
| F80 | 7 | 5 | 5 | 0 | 2 | 0.83 |
| F90 | 7 | 6 | 6 | 0 | 1 | 0.92 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
