# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos20_KO`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 9

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 7.1 | 0.4 | 1.1 | 0.90 | 0.97 |
| IoU≥0.5 | 7.1 | 0.4 | 1.1 | 0.90 | 0.97 |
| IoU≥0.7 | 6.7 | 0.8 | 1.5 | 0.85 | 0.91 |

- **Mean per-cell IoU (matched)**: 0.846
- **Median per-cell IoU (matched)**: 0.874
- **Out-of-scope predictions/frame**: 1.0

`F1_focused` excludes predictions with zero IoU vs *any* GT cell from
the FP count — they're real cells in the field the GT just didn't
annotate. Use it when GT only partially covers the field (e.g.
ignasi recordings have 1 GT cell per frame but the field shows 3).

## Tracking identity preservation

- **Mean ID consistency** (per GT cell, fraction of frames where it
  maps to the same pipeline ID): **91.17%**
- **GT cells with perfect 1.0 consistency**:
  6 / 9

| GT cell | matched in N frames | dominant pred | consistency |
|---:|---:|---:|---:|
| 2 | 2 | 2 | 1.00 |
| 5 | 10 | 4 | 1.00 |
| 6 | 10 | 1 | 1.00 |
| 7 | 10 | 5 | 1.00 |
| 8 | 7 | 7 | 1.00 |
| 10 | 5 | 11 | 1.00 |
| 9 | 10 | 8 | 0.90 |
| 3 | 8 | 2 | 0.75 |
| 1 | 9 | 9 | 0.56 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 9 | 8 | 7 | 1 | 2 | 0.82 |
| F10 | 8 | 9 | 8 | 1 | 0 | 0.94 |
| F20 | 7 | 8 | 7 | 1 | 0 | 0.93 |
| F30 | 7 | 8 | 7 | 1 | 0 | 0.93 |
| F40 | 7 | 8 | 7 | 1 | 0 | 0.93 |
| F50 | 8 | 10 | 8 | 2 | 0 | 0.89 |
| F60 | 8 | 9 | 8 | 1 | 0 | 0.94 |
| F70 | 7 | 8 | 7 | 1 | 0 | 0.93 |
| F80 | 7 | 7 | 6 | 1 | 1 | 0.86 |
| F90 | 7 | 7 | 6 | 1 | 1 | 0.86 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
