# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos39_OT`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 9

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 6.3 | 1.1 | 0.3 | 0.90 | 0.91 |
| IoU≥0.5 | 6.2 | 1.2 | 0.4 | 0.89 | 0.90 |
| IoU≥0.7 | 6.2 | 1.2 | 0.4 | 0.89 | 0.90 |

- **Mean per-cell IoU (matched)**: 0.857
- **Median per-cell IoU (matched)**: 0.867
- **Out-of-scope predictions/frame**: 0.2

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

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
| 7 | 10 | 5 | 1.00 |
| 8 | 3 | 7 | 1.00 |
| 6 | 2 | 6 | 1.00 |
| 10 | 2 | 8 | 1.00 |
| 5 | 10 | 6 | 0.80 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 7 | 6 | 6 | 0 | 1 | 0.92 |
| F10 | 7 | 6 | 6 | 0 | 1 | 0.92 |
| F20 | 7 | 6 | 6 | 0 | 1 | 0.92 |
| F30 | 7 | 6 | 6 | 0 | 1 | 0.92 |
| F40 | 7 | 6 | 6 | 0 | 1 | 0.92 |
| F50 | 7 | 6 | 5 | 1 | 2 | 0.77 |
| F60 | 7 | 7 | 6 | 1 | 1 | 0.86 |
| F70 | 7 | 7 | 6 | 1 | 1 | 0.86 |
| F80 | 9 | 8 | 8 | 0 | 1 | 0.94 |
| F90 | 9 | 8 | 7 | 1 | 2 | 0.82 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
