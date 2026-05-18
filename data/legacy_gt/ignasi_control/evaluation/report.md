# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/legacy_gt/ignasi_control`
**GT frames evaluated**: 15
**GT cells (unique IDs)**: 1

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 |
|---:|---:|---:|---:|---:|
| IoU≥0.3 | 0.8 | 0.2 | 0.0 | 0.80 |
| IoU≥0.5 | 0.8 | 0.2 | 0.0 | 0.80 |
| IoU≥0.7 | 0.8 | 0.2 | 0.0 | 0.80 |

- **Mean per-cell IoU (matched)**: 0.890
- **Median per-cell IoU (matched)**: 0.893

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **100.00%**
- **GT cells with perfect 1.0 consistency**:
  1 / 1

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 12 | 1 | 1.00 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F1 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F2 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F3 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F4 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F5 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F6 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F7 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F8 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F9 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F10 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F11 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F12 | 1 | 0 | 0 | 0 | 1 | 0.00 |
| F13 | 1 | 0 | 0 | 0 | 1 | 0.00 |
| F14 | 1 | 0 | 0 | 0 | 1 | 0.00 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
