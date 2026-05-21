# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/legacy_gt/ignasi_control`
**GT frames evaluated**: 15
**GT cells (unique IDs)**: 1

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 1.0 | 0.0 | 1.2 | 0.63 | 1.00 |
| IoU≥0.5 | 1.0 | 0.0 | 1.2 | 0.63 | 1.00 |
| IoU≥0.7 | 1.0 | 0.0 | 1.2 | 0.63 | 1.00 |

- **Mean per-cell IoU (matched)**: 0.924
- **Median per-cell IoU (matched)**: 0.935
- **Out-of-scope predictions/frame**: 1.2

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **100.00%**
- **GT cells with perfect 1.0 consistency**:
  1 / 1

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 15 | 2 | 1.00 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F1 | 1 | 2 | 1 | 1 | 0 | 0.67 |
| F2 | 1 | 3 | 1 | 2 | 0 | 0.50 |
| F3 | 1 | 3 | 1 | 2 | 0 | 0.50 |
| F4 | 1 | 3 | 1 | 2 | 0 | 0.50 |
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

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
