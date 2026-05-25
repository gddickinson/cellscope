# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos7_WT`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 10

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 9.6 | 1.2 | 2.5 | 0.84 | 0.92 |
| IoU≥0.5 | 9.5 | 1.3 | 2.6 | 0.83 | 0.91 |
| IoU≥0.7 | 8.9 | 1.9 | 3.2 | 0.78 | 0.85 |

- **Mean per-cell IoU (matched)**: 0.847
- **Median per-cell IoU (matched)**: 0.873
- **Out-of-scope predictions/frame**: 2.0

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **89.25%**
- **GT cells with perfect 1.0 consistency**:
  7 / 10

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 2 | 9 | 5 | 1.00 |
| 3 | 10 | 4 | 1.00 |
| 4 | 10 | 3 | 1.00 |
| 6 | 10 | 7 | 1.00 |
| 7 | 10 | 8 | 1.00 |
| 8 | 10 | 10 | 1.00 |
| 1 | 9 | 1 | 1.00 |
| 9 | 10 | 9 | 0.70 |
| 11 | 8 | 11 | 0.62 |
| 5 | 10 | 6 | 0.60 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 10 | 9 | 8 | 1 | 2 | 0.84 |
| F10 | 10 | 12 | 8 | 4 | 2 | 0.73 |
| F20 | 11 | 12 | 10 | 2 | 1 | 0.87 |
| F30 | 11 | 12 | 10 | 2 | 1 | 0.87 |
| F40 | 11 | 11 | 9 | 2 | 2 | 0.82 |
| F50 | 11 | 13 | 10 | 3 | 1 | 0.83 |
| F60 | 11 | 14 | 10 | 4 | 1 | 0.80 |
| F70 | 11 | 14 | 10 | 4 | 1 | 0.80 |
| F80 | 11 | 12 | 10 | 2 | 1 | 0.87 |
| F90 | 11 | 12 | 10 | 2 | 1 | 0.87 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
