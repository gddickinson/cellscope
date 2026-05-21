# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos10_WT`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 3

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 3.0 | 0.0 | 0.9 | 0.87 | 1.00 |
| IoU≥0.5 | 2.9 | 0.1 | 1.0 | 0.84 | 0.97 |
| IoU≥0.7 | 2.9 | 0.1 | 1.0 | 0.84 | 0.97 |

- **Mean per-cell IoU (matched)**: 0.863
- **Median per-cell IoU (matched)**: 0.891
- **Out-of-scope predictions/frame**: 0.9

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **93.33%**
- **GT cells with perfect 1.0 consistency**:
  2 / 3

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 2 | 1.00 |
| 2 | 10 | 1 | 1.00 |
| 3 | 10 | 3 | 0.80 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 3 | 4 | 3 | 1 | 0 | 0.86 |
| F10 | 3 | 4 | 3 | 1 | 0 | 0.86 |
| F20 | 3 | 4 | 3 | 1 | 0 | 0.86 |
| F30 | 3 | 4 | 3 | 1 | 0 | 0.86 |
| F40 | 3 | 4 | 3 | 1 | 0 | 0.86 |
| F50 | 3 | 4 | 3 | 1 | 0 | 0.86 |
| F60 | 3 | 4 | 2 | 2 | 1 | 0.57 |
| F70 | 3 | 4 | 3 | 1 | 0 | 0.86 |
| F80 | 3 | 4 | 3 | 1 | 0 | 0.86 |
| F90 | 3 | 3 | 3 | 0 | 0 | 1.00 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
