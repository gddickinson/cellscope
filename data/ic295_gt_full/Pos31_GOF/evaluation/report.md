# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos31_GOF`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 2

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 1.3 | 2.4 | 0.0 | 0.52 | 0.52 |
| IoU≥0.5 | 1.3 | 2.4 | 0.0 | 0.52 | 0.52 |
| IoU≥0.7 | 1.3 | 2.4 | 0.0 | 0.52 | 0.52 |

- **Mean per-cell IoU (matched)**: 0.873
- **Median per-cell IoU (matched)**: 0.892
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
| F0 | 4 | 1 | 1 | 0 | 3 | 0.40 |
| F10 | 3 | 1 | 1 | 0 | 2 | 0.50 |
| F20 | 3 | 1 | 1 | 0 | 2 | 0.50 |
| F30 | 3 | 1 | 1 | 0 | 2 | 0.50 |
| F40 | 2 | 1 | 1 | 0 | 1 | 0.67 |
| F50 | 4 | 1 | 1 | 0 | 3 | 0.40 |
| F60 | 4 | 1 | 1 | 0 | 3 | 0.40 |
| F70 | 4 | 2 | 2 | 0 | 2 | 0.67 |
| F80 | 5 | 2 | 2 | 0 | 3 | 0.57 |
| F90 | 5 | 2 | 2 | 0 | 3 | 0.57 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
