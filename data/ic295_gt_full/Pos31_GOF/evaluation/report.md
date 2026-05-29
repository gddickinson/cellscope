# Evaluation report

**Recording**: `/Users/george/cellscope/data/ic295_gt_full/Pos31_GOF`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 5

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 3.2 | 0.5 | 1.5 | 0.75 | 0.91 |
| IoU≥0.5 | 3.0 | 0.7 | 1.7 | 0.70 | 0.85 |
| IoU≥0.7 | 2.5 | 1.2 | 2.2 | 0.59 | 0.72 |

- **Mean per-cell IoU (matched)**: 0.777
- **Median per-cell IoU (matched)**: 0.847
- **Out-of-scope predictions/frame**: 1.4

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **88.89%**
- **GT cells with perfect 1.0 consistency**:
  3 / 5

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 2 | 1.00 |
| 3 | 8 | 4 | 1.00 |
| 2 | 3 | 6 | 1.00 |
| 5 | 9 | 1 | 0.78 |
| 4 | 3 | 1 | 0.67 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 4 | 4 | 3 | 1 | 1 | 0.75 |
| F10 | 3 | 4 | 2 | 2 | 1 | 0.57 |
| F20 | 3 | 4 | 2 | 2 | 1 | 0.57 |
| F30 | 3 | 4 | 2 | 2 | 1 | 0.57 |
| F40 | 2 | 4 | 2 | 2 | 0 | 0.67 |
| F50 | 4 | 4 | 3 | 1 | 1 | 0.75 |
| F60 | 4 | 5 | 3 | 2 | 1 | 0.67 |
| F70 | 4 | 6 | 3 | 3 | 1 | 0.60 |
| F80 | 5 | 6 | 5 | 1 | 0 | 0.91 |
| F90 | 5 | 6 | 5 | 1 | 0 | 0.91 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
