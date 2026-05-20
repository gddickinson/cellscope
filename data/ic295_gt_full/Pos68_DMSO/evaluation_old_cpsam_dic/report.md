# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos68_DMSO`
**GT frames evaluated**: 11
**GT cells (unique IDs)**: 13

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 |
|---:|---:|---:|---:|---:|
| IoU≥0.3 | 7.2 | 10.3 | 1.0 | 0.52 |
| IoU≥0.5 | 6.3 | 11.2 | 1.9 | 0.46 |
| IoU≥0.7 | 3.1 | 14.4 | 5.1 | 0.22 |

- **Mean per-cell IoU (matched)**: 0.653
- **Median per-cell IoU (matched)**: 0.662

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **93.08%**
- **GT cells with perfect 1.0 consistency**:
  10 / 13

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 2 | 1 | 1 | 1.00 |
| 3 | 10 | 2 | 1.00 |
| 7 | 10 | 6 | 1.00 |
| 8 | 10 | 8 | 1.00 |
| 10 | 10 | 7 | 1.00 |
| 18 | 10 | 4 | 1.00 |
| 9 | 6 | 10 | 1.00 |
| 14 | 1 | 11 | 1.00 |
| 4 | 1 | 12 | 1.00 |
| 16 | 1 | 13 | 1.00 |
| 1 | 10 | 1 | 0.90 |
| 11 | 10 | 9 | 0.70 |
| 19 | 2 | 5 | 0.50 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 18 | 9 | 9 | 0 | 9 | 0.67 |
| F10 | 18 | 7 | 5 | 2 | 13 | 0.40 |
| F20 | 19 | 7 | 7 | 0 | 12 | 0.54 |
| F30 | 17 | 7 | 5 | 2 | 12 | 0.42 |
| F35 | 1 | 7 | 0 | 7 | 1 | 0.00 |
| F40 | 19 | 8 | 7 | 1 | 12 | 0.52 |
| F50 | 20 | 9 | 7 | 2 | 13 | 0.48 |
| F60 | 20 | 8 | 7 | 1 | 13 | 0.50 |
| F70 | 21 | 10 | 8 | 2 | 13 | 0.52 |
| F80 | 20 | 10 | 8 | 2 | 12 | 0.53 |
| F90 | 19 | 8 | 6 | 2 | 13 | 0.44 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
