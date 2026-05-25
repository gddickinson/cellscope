# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos21_KO`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 9

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 4.9 | 1.4 | 1.3 | 0.76 | 0.79 |
| IoU≥0.5 | 4.3 | 2.0 | 1.9 | 0.66 | 0.69 |
| IoU≥0.7 | 3.5 | 2.8 | 2.7 | 0.55 | 0.57 |

- **Mean per-cell IoU (matched)**: 0.677
- **Median per-cell IoU (matched)**: 0.793
- **Out-of-scope predictions/frame**: 0.4

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **79.59%**
- **GT cells with perfect 1.0 consistency**:
  4 / 9

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 1 | 1.00 |
| 4 | 1 | 3 | 1.00 |
| 9 | 7 | 4 | 1.00 |
| 8 | 5 | 7 | 1.00 |
| 2 | 10 | 2 | 0.90 |
| 3 | 3 | 9 | 0.67 |
| 5 | 8 | 5 | 0.62 |
| 6 | 7 | 6 | 0.57 |
| 7 | 5 | 5 | 0.40 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 6 | 4 | 3 | 1 | 3 | 0.60 |
| F10 | 5 | 4 | 2 | 2 | 3 | 0.44 |
| F20 | 4 | 5 | 3 | 2 | 1 | 0.67 |
| F30 | 4 | 5 | 3 | 2 | 1 | 0.67 |
| F40 | 6 | 5 | 2 | 3 | 4 | 0.36 |
| F50 | 7 | 7 | 6 | 1 | 1 | 0.86 |
| F60 | 7 | 8 | 4 | 4 | 3 | 0.53 |
| F70 | 8 | 8 | 7 | 1 | 1 | 0.88 |
| F80 | 8 | 8 | 7 | 1 | 1 | 0.88 |
| F90 | 8 | 8 | 6 | 2 | 2 | 0.75 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
