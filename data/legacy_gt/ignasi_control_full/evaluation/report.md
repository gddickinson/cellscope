# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/legacy_gt/ignasi_control_full`
**GT frames evaluated**: 65
**GT cells (unique IDs)**: 1

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 |
|---:|---:|---:|---:|---:|
| IoU≥0.3 | 0.9 | 0.1 | 0.0 | 0.92 |
| IoU≥0.5 | 0.9 | 0.1 | 0.0 | 0.92 |
| IoU≥0.7 | 0.9 | 0.1 | 0.0 | 0.92 |

- **Mean per-cell IoU (matched)**: 0.897
- **Median per-cell IoU (matched)**: 0.897

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **100.00%**
- **GT cells with perfect 1.0 consistency**:
  1 / 1

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 60 | 1 | 1.00 |

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
| F12 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F13 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F14 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F15 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F16 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F17 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F18 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F19 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F20 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F21 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F22 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F23 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F24 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F25 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F26 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F27 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F28 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F29 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F30 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F31 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F32 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F33 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F34 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F35 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F36 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F37 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F38 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F41 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F43 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F46 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F48 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F49 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F52 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F54 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F61 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F68 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F69 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F70 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F75 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F76 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F77 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F81 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F82 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F84 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F85 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F86 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F87 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F88 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F89 | 1 | 0 | 0 | 0 | 1 | 0.00 |
| F92 | 1 | 0 | 0 | 0 | 1 | 0.00 |
| F93 | 1 | 0 | 0 | 0 | 1 | 0.00 |
| F94 | 1 | 0 | 0 | 0 | 1 | 0.00 |
| F96 | 1 | 0 | 0 | 0 | 1 | 0.00 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
