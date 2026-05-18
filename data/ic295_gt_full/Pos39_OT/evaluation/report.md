# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos39_OT`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 9

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 |
|---:|---:|---:|---:|---:|
| IoU≥0.3 | 7.0 | 0.4 | 0.2 | 0.96 |
| IoU≥0.5 | 6.9 | 0.5 | 0.3 | 0.95 |
| IoU≥0.7 | 6.9 | 0.5 | 0.3 | 0.95 |

- **Mean per-cell IoU (matched)**: 0.855
- **Median per-cell IoU (matched)**: 0.864

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **97.78%**
- **GT cells with perfect 1.0 consistency**:
  8 / 9

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 6 | 1 | 1.00 |
| 2 | 10 | 2 | 1.00 |
| 3 | 10 | 3 | 1.00 |
| 4 | 10 | 4 | 1.00 |
| 7 | 10 | 6 | 1.00 |
| 9 | 10 | 5 | 1.00 |
| 6 | 2 | 7 | 1.00 |
| 10 | 2 | 8 | 1.00 |
| 5 | 10 | 7 | 0.80 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 7 | 7 | 7 | 0 | 0 | 1.00 |
| F10 | 7 | 7 | 7 | 0 | 0 | 1.00 |
| F20 | 7 | 7 | 7 | 0 | 0 | 1.00 |
| F30 | 7 | 7 | 7 | 0 | 0 | 1.00 |
| F40 | 7 | 7 | 7 | 0 | 0 | 1.00 |
| F50 | 7 | 7 | 6 | 1 | 1 | 0.86 |
| F60 | 7 | 7 | 6 | 1 | 1 | 0.86 |
| F70 | 7 | 7 | 6 | 1 | 1 | 0.86 |
| F80 | 9 | 8 | 8 | 0 | 1 | 0.94 |
| F90 | 9 | 8 | 8 | 0 | 1 | 0.94 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
