# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos51_Y1`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 2

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 1.3 | 0.0 | 0.0 | 1.00 | 1.00 |
| IoU≥0.5 | 1.3 | 0.0 | 0.0 | 1.00 | 1.00 |
| IoU≥0.7 | 1.3 | 0.0 | 0.0 | 1.00 | 1.00 |

- **Mean per-cell IoU (matched)**: 0.866
- **Median per-cell IoU (matched)**: 0.872
- **Out-of-scope predictions/frame**: 0.0

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **100.00%**
- **GT cells with perfect 1.0 consistency**:
  2 / 2

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 1 | 1.00 |
| 2 | 3 | 2 | 1.00 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F10 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F20 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F30 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F40 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F50 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F60 | 1 | 1 | 1 | 0 | 0 | 1.00 |
| F70 | 2 | 2 | 2 | 0 | 0 | 1.00 |
| F80 | 2 | 2 | 2 | 0 | 0 | 1.00 |
| F90 | 2 | 2 | 2 | 0 | 0 | 1.00 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
