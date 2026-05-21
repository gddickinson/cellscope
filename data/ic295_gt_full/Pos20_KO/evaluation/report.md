# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos20_KO`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 9

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 |
|---:|---:|---:|---:|---:|
| IoU≥0.3 | 6.2 | 1.3 | 0.3 | 0.88 |
| IoU≥0.5 | 6.2 | 1.3 | 0.3 | 0.88 |
| IoU≥0.7 | 5.8 | 1.7 | 0.7 | 0.83 |

- **Mean per-cell IoU (matched)**: 0.842
- **Median per-cell IoU (matched)**: 0.872

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **92.28%**
- **GT cells with perfect 1.0 consistency**:
  7 / 9

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 2 | 2 | 2 | 1.00 |
| 5 | 10 | 4 | 1.00 |
| 6 | 10 | 1 | 1.00 |
| 7 | 10 | 5 | 1.00 |
| 8 | 7 | 6 | 1.00 |
| 10 | 5 | 10 | 1.00 |
| 9 | 1 | 6 | 1.00 |
| 3 | 8 | 2 | 0.75 |
| 1 | 9 | 7 | 0.56 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 9 | 6 | 6 | 0 | 3 | 0.80 |
| F10 | 8 | 7 | 7 | 0 | 1 | 0.93 |
| F20 | 7 | 6 | 6 | 0 | 1 | 0.92 |
| F30 | 7 | 6 | 6 | 0 | 1 | 0.92 |
| F40 | 7 | 6 | 6 | 0 | 1 | 0.92 |
| F50 | 8 | 8 | 7 | 1 | 1 | 0.88 |
| F60 | 8 | 7 | 7 | 0 | 1 | 0.93 |
| F70 | 7 | 7 | 7 | 0 | 0 | 1.00 |
| F80 | 7 | 6 | 5 | 1 | 2 | 0.77 |
| F90 | 7 | 6 | 5 | 1 | 2 | 0.77 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
