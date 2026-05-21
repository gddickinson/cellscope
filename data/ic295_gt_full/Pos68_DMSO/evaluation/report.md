# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos68_DMSO`
**GT frames evaluated**: 11
**GT cells (unique IDs)**: 14

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 |
|---:|---:|---:|---:|---:|
| IoU≥0.3 | 7.9 | 9.5 | 3.3 | 0.53 |
| IoU≥0.5 | 7.5 | 10.0 | 3.7 | 0.50 |
| IoU≥0.7 | 6.7 | 10.7 | 4.5 | 0.45 |

- **Mean per-cell IoU (matched)**: 0.761
- **Median per-cell IoU (matched)**: 0.830

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **84.29%**
- **GT cells with perfect 1.0 consistency**:
  8 / 14

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 3 | 1.00 |
| 2 | 9 | 1 | 1.00 |
| 3 | 10 | 2 | 1.00 |
| 7 | 10 | 6 | 1.00 |
| 9 | 6 | 9 | 1.00 |
| 4 | 1 | 1 | 1.00 |
| 20 | 1 | 7 | 1.00 |
| 17 | 1 | 7 | 1.00 |
| 18 | 10 | 4 | 0.90 |
| 10 | 10 | 7 | 0.80 |
| 8 | 10 | 13 | 0.60 |
| 11 | 10 | 11 | 0.50 |
| 13 | 2 | 10 | 0.50 |
| 19 | 2 | 5 | 0.50 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 18 | 11 | 10 | 1 | 8 | 0.69 |
| F10 | 18 | 10 | 8 | 2 | 10 | 0.57 |
| F20 | 19 | 10 | 8 | 2 | 11 | 0.55 |
| F30 | 17 | 11 | 9 | 2 | 8 | 0.64 |
| F35 | 1 | 12 | 0 | 12 | 1 | 0.00 |
| F40 | 19 | 12 | 8 | 4 | 11 | 0.52 |
| F50 | 20 | 12 | 7 | 5 | 13 | 0.44 |
| F60 | 20 | 11 | 8 | 3 | 12 | 0.52 |
| F70 | 21 | 11 | 8 | 3 | 13 | 0.50 |
| F80 | 20 | 12 | 7 | 5 | 13 | 0.44 |
| F90 | 19 | 11 | 9 | 2 | 10 | 0.60 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
