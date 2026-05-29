# Evaluation report

**Recording**: `/Users/george/cellscope/data/ic295_gt_full/Pos21_KO`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 9

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 4.8 | 1.5 | 0.9 | 0.77 | 0.80 |
| IoU≥0.5 | 4.0 | 2.3 | 1.7 | 0.63 | 0.65 |
| IoU≥0.7 | 3.4 | 2.9 | 2.3 | 0.55 | 0.56 |

- **Mean per-cell IoU (matched)**: 0.680
- **Median per-cell IoU (matched)**: 0.801
- **Out-of-scope predictions/frame**: 0.3

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **84.44%**
- **GT cells with perfect 1.0 consistency**:
  5 / 9

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 1 | 1.00 |
| 4 | 1 | 3 | 1.00 |
| 9 | 8 | 4 | 1.00 |
| 8 | 5 | 7 | 1.00 |
| 3 | 2 | 9 | 1.00 |
| 2 | 10 | 2 | 0.90 |
| 6 | 5 | 6 | 0.60 |
| 7 | 5 | 5 | 0.60 |
| 5 | 8 | 8 | 0.50 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 6 | 4 | 3 | 1 | 3 | 0.60 |
| F10 | 5 | 4 | 2 | 2 | 3 | 0.44 |
| F20 | 4 | 4 | 2 | 2 | 2 | 0.50 |
| F30 | 4 | 4 | 2 | 2 | 2 | 0.50 |
| F40 | 6 | 5 | 2 | 3 | 4 | 0.36 |
| F50 | 7 | 7 | 6 | 1 | 1 | 0.86 |
| F60 | 7 | 6 | 4 | 2 | 3 | 0.61 |
| F70 | 8 | 8 | 7 | 1 | 1 | 0.88 |
| F80 | 8 | 8 | 7 | 1 | 1 | 0.88 |
| F90 | 8 | 7 | 5 | 2 | 3 | 0.67 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
