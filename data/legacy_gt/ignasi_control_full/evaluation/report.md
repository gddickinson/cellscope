# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/legacy_gt/ignasi_control_full`
**GT frames evaluated**: 65
**GT cells (unique IDs)**: 1

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 |
|---:|---:|---:|---:|---:|
| IoU≥0.3 | 1.0 | 0.0 | 1.0 | 0.66 |
| IoU≥0.5 | 1.0 | 0.0 | 1.0 | 0.66 |
| IoU≥0.7 | 1.0 | 0.0 | 1.0 | 0.66 |

- **Mean per-cell IoU (matched)**: 0.932
- **Median per-cell IoU (matched)**: 0.940

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **100.00%**
- **GT cells with perfect 1.0 consistency**:
  1 / 1

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 65 | 2 | 1.00 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F1 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F2 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F3 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F4 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F5 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F6 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F7 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F8 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F9 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F10 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F11 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F12 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F13 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F14 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F15 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F16 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F17 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F18 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F19 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F20 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F21 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F22 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F23 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F24 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F25 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F26 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F27 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F28 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F29 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F30 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F31 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F32 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F33 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F34 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F35 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F36 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F37 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F38 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F41 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F43 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F46 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F48 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F49 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F52 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F54 | 1 | 3 | 1 | 2 | 0 | 0.50 |
| F61 | 1 | 3 | 1 | 2 | 0 | 0.50 |
| F68 | 1 | 3 | 1 | 2 | 0 | 0.50 |
| F69 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F70 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F75 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F76 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F77 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F81 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F82 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F84 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F85 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F86 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F87 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F88 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F89 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F92 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F93 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F94 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F96 | 1 | 2 | 1 | 1 | 0 | 0.67 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
