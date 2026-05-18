# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos30_GOF`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 5

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 |
|---:|---:|---:|---:|---:|
| IoU≥0.3 | 4.7 | 1.6 | 0.2 | 0.84 |
| IoU≥0.5 | 4.7 | 1.6 | 0.2 | 0.84 |
| IoU≥0.7 | 4.4 | 1.9 | 0.5 | 0.79 |

- **Mean per-cell IoU (matched)**: 0.848
- **Median per-cell IoU (matched)**: 0.858

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **92.00%**
- **GT cells with perfect 1.0 consistency**:
  3 / 5

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 1 | 1.00 |
| 3 | 7 | 3 | 1.00 |
| 4 | 10 | 4 | 1.00 |
| 2 | 10 | 2 | 0.80 |
| 5 | 10 | 5 | 0.80 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 6 | 5 | 5 | 0 | 1 | 0.91 |
| F10 | 6 | 5 | 5 | 0 | 1 | 0.91 |
| F20 | 7 | 5 | 5 | 0 | 2 | 0.83 |
| F30 | 7 | 5 | 5 | 0 | 2 | 0.83 |
| F40 | 6 | 5 | 5 | 0 | 1 | 0.91 |
| F50 | 6 | 5 | 5 | 0 | 1 | 0.91 |
| F60 | 6 | 5 | 5 | 0 | 1 | 0.91 |
| F70 | 7 | 6 | 4 | 2 | 3 | 0.61 |
| F80 | 6 | 4 | 4 | 0 | 2 | 0.80 |
| F90 | 6 | 4 | 4 | 0 | 2 | 0.80 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
