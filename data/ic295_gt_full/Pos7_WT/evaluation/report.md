# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos7_WT`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 9

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 |
|---:|---:|---:|---:|---:|
| IoU≥0.3 | 8.6 | 2.2 | 1.1 | 0.84 |
| IoU≥0.5 | 8.5 | 2.3 | 1.2 | 0.83 |
| IoU≥0.7 | 7.7 | 3.1 | 2.0 | 0.75 |

- **Mean per-cell IoU (matched)**: 0.844
- **Median per-cell IoU (matched)**: 0.875

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **100.00%**
- **GT cells with perfect 1.0 consistency**:
  9 / 9

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 3 | 10 | 3 | 1.00 |
| 4 | 10 | 2 | 1.00 |
| 5 | 10 | 4 | 1.00 |
| 6 | 10 | 5 | 1.00 |
| 7 | 10 | 6 | 1.00 |
| 8 | 9 | 8 | 1.00 |
| 9 | 10 | 7 | 1.00 |
| 1 | 9 | 1 | 1.00 |
| 11 | 8 | 10 | 1.00 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 10 | 7 | 7 | 0 | 3 | 0.82 |
| F10 | 10 | 10 | 7 | 3 | 3 | 0.70 |
| F20 | 11 | 10 | 9 | 1 | 2 | 0.86 |
| F30 | 11 | 10 | 9 | 1 | 2 | 0.86 |
| F40 | 11 | 10 | 9 | 1 | 2 | 0.86 |
| F50 | 11 | 10 | 9 | 1 | 2 | 0.86 |
| F60 | 11 | 10 | 9 | 1 | 2 | 0.86 |
| F70 | 11 | 10 | 9 | 1 | 2 | 0.86 |
| F80 | 11 | 11 | 9 | 2 | 2 | 0.82 |
| F90 | 11 | 9 | 8 | 1 | 3 | 0.80 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
