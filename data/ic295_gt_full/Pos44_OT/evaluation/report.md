# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos44_OT`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 5

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 3.5 | 1.9 | 0.6 | 0.74 | 0.79 |
| IoU≥0.5 | 3.5 | 1.9 | 0.6 | 0.74 | 0.79 |
| IoU≥0.7 | 3.5 | 1.9 | 0.6 | 0.74 | 0.79 |

- **Mean per-cell IoU (matched)**: 0.881
- **Median per-cell IoU (matched)**: 0.889
- **Out-of-scope predictions/frame**: 0.6

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **98.00%**
- **GT cells with perfect 1.0 consistency**:
  4 / 5

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 2 | 10 | 2 | 1.00 |
| 5 | 10 | 3 | 1.00 |
| 6 | 4 | 6 | 1.00 |
| 4 | 1 | 4 | 1.00 |
| 3 | 10 | 4 | 0.90 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 5 | 4 | 3 | 1 | 2 | 0.67 |
| F10 | 4 | 4 | 3 | 1 | 1 | 0.75 |
| F20 | 5 | 3 | 3 | 0 | 2 | 0.75 |
| F30 | 4 | 3 | 3 | 0 | 1 | 0.86 |
| F40 | 5 | 3 | 3 | 0 | 2 | 0.75 |
| F50 | 5 | 4 | 3 | 1 | 2 | 0.67 |
| F60 | 6 | 5 | 4 | 1 | 2 | 0.73 |
| F70 | 6 | 5 | 4 | 1 | 2 | 0.73 |
| F80 | 7 | 5 | 4 | 1 | 3 | 0.67 |
| F90 | 7 | 5 | 5 | 0 | 2 | 0.83 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
