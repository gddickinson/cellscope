# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos30_GOF`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 5

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 4.8 | 1.5 | 0.0 | 0.86 | 0.86 |
| IoU≥0.5 | 4.8 | 1.5 | 0.0 | 0.86 | 0.86 |
| IoU≥0.7 | 4.8 | 1.5 | 0.0 | 0.86 | 0.86 |

- **Mean per-cell IoU (matched)**: 0.860
- **Median per-cell IoU (matched)**: 0.861
- **Out-of-scope predictions/frame**: 0.0

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **100.00%**
- **GT cells with perfect 1.0 consistency**:
  5 / 5

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 1 | 10 | 1 | 1.00 |
| 2 | 10 | 2 | 1.00 |
| 3 | 8 | 3 | 1.00 |
| 4 | 10 | 4 | 1.00 |
| 5 | 10 | 5 | 1.00 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 6 | 5 | 5 | 0 | 1 | 0.91 |
| F10 | 6 | 5 | 5 | 0 | 1 | 0.91 |
| F20 | 7 | 5 | 5 | 0 | 2 | 0.83 |
| F30 | 7 | 5 | 5 | 0 | 2 | 0.83 |
| F40 | 6 | 5 | 5 | 0 | 1 | 0.91 |
| F50 | 6 | 5 | 5 | 0 | 1 | 0.91 |
| F60 | 6 | 5 | 5 | 0 | 1 | 0.91 |
| F70 | 7 | 5 | 5 | 0 | 2 | 0.83 |
| F80 | 6 | 4 | 4 | 0 | 2 | 0.80 |
| F90 | 6 | 4 | 4 | 0 | 2 | 0.80 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
