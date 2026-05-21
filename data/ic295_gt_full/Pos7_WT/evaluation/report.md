# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos7_WT`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 9

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 8.4 | 2.4 | 2.4 | 0.78 | 0.86 |
| IoU≥0.5 | 8.3 | 2.5 | 2.5 | 0.77 | 0.85 |
| IoU≥0.7 | 7.8 | 3.0 | 3.0 | 0.72 | 0.80 |

- **Mean per-cell IoU (matched)**: 0.851
- **Median per-cell IoU (matched)**: 0.874
- **Out-of-scope predictions/frame**: 2.0

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **92.22%**
- **GT cells with perfect 1.0 consistency**:
  7 / 9

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 3 | 10 | 4 | 1.00 |
| 4 | 10 | 3 | 1.00 |
| 6 | 10 | 6 | 1.00 |
| 7 | 10 | 7 | 1.00 |
| 8 | 10 | 9 | 1.00 |
| 1 | 9 | 1 | 1.00 |
| 11 | 5 | 10 | 1.00 |
| 9 | 10 | 8 | 0.70 |
| 5 | 10 | 5 | 0.60 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 10 | 8 | 7 | 1 | 3 | 0.78 |
| F10 | 10 | 11 | 7 | 4 | 3 | 0.67 |
| F20 | 11 | 11 | 9 | 2 | 2 | 0.82 |
| F30 | 11 | 11 | 9 | 2 | 2 | 0.82 |
| F40 | 11 | 12 | 9 | 3 | 2 | 0.78 |
| F50 | 11 | 12 | 9 | 3 | 2 | 0.78 |
| F60 | 11 | 11 | 9 | 2 | 2 | 0.82 |
| F70 | 11 | 12 | 8 | 4 | 3 | 0.70 |
| F80 | 11 | 10 | 8 | 2 | 3 | 0.76 |
| F90 | 11 | 10 | 8 | 2 | 3 | 0.76 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
