# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos21_KO`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 5

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 2.8 | 3.5 | 0.2 | 0.61 | 0.61 |
| IoU≥0.5 | 2.8 | 3.5 | 0.2 | 0.61 | 0.61 |
| IoU≥0.7 | 2.4 | 3.9 | 0.6 | 0.53 | 0.53 |

- **Mean per-cell IoU (matched)**: 0.799
- **Median per-cell IoU (matched)**: 0.868
- **Out-of-scope predictions/frame**: 0.0

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **91.33%**
- **GT cells with perfect 1.0 consistency**:
  3 / 5

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 1 | 1.00 |
| 4 | 1 | 3 | 1.00 |
| 8 | 5 | 4 | 1.00 |
| 2 | 10 | 2 | 0.90 |
| 3 | 3 | 5 | 0.67 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 6 | 3 | 3 | 0 | 3 | 0.67 |
| F10 | 5 | 2 | 2 | 0 | 3 | 0.57 |
| F20 | 4 | 2 | 2 | 0 | 2 | 0.67 |
| F30 | 4 | 2 | 2 | 0 | 2 | 0.67 |
| F40 | 6 | 2 | 2 | 0 | 4 | 0.50 |
| F50 | 7 | 3 | 3 | 0 | 4 | 0.60 |
| F60 | 7 | 4 | 3 | 1 | 4 | 0.55 |
| F70 | 8 | 4 | 4 | 0 | 4 | 0.67 |
| F80 | 8 | 4 | 4 | 0 | 4 | 0.67 |
| F90 | 8 | 4 | 3 | 1 | 5 | 0.50 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
