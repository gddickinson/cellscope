# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos39_OT`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 8

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 |
|---:|---:|---:|---:|---:|
| IoU≥0.3 | 6.6 | 0.8 | 0.1 | 0.94 |
| IoU≥0.5 | 6.6 | 0.8 | 0.1 | 0.94 |
| IoU≥0.7 | 6.6 | 0.8 | 0.1 | 0.94 |

- **Mean per-cell IoU (matched)**: 0.863
- **Median per-cell IoU (matched)**: 0.867

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **93.75%**
- **GT cells with perfect 1.0 consistency**:
  7 / 8

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 5 | 1 | 1.00 |
| 2 | 10 | 2 | 1.00 |
| 4 | 10 | 4 | 1.00 |
| 5 | 10 | 7 | 1.00 |
| 7 | 10 | 6 | 1.00 |
| 9 | 9 | 5 | 1.00 |
| 10 | 2 | 9 | 1.00 |
| 3 | 10 | 3 | 0.50 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 7 | 7 | 7 | 0 | 0 | 1.00 |
| F10 | 7 | 7 | 7 | 0 | 0 | 1.00 |
| F20 | 7 | 7 | 7 | 0 | 0 | 1.00 |
| F30 | 7 | 7 | 6 | 1 | 1 | 0.86 |
| F40 | 7 | 7 | 7 | 0 | 0 | 1.00 |
| F50 | 7 | 6 | 6 | 0 | 1 | 0.92 |
| F60 | 7 | 6 | 6 | 0 | 1 | 0.92 |
| F70 | 7 | 6 | 6 | 0 | 1 | 0.92 |
| F80 | 9 | 7 | 7 | 0 | 2 | 0.88 |
| F90 | 9 | 7 | 7 | 0 | 2 | 0.88 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
