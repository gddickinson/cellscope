# Evaluation report

**Recording**: `/Users/george/claude_test/cellscope/data/ic295_gt_full/Pos31_GOF`
**GT frames evaluated**: 10
**GT cells (unique IDs)**: 5

## Detection accuracy

| Threshold | TP/frame | FN/frame | FP/frame | F1 | F1_focused |
|---:|---:|---:|---:|---:|---:|
| IoU≥0.3 | 3.2 | 0.5 | 1.0 | 0.79 | 0.90 |
| IoU≥0.5 | 3.0 | 0.7 | 1.2 | 0.74 | 0.84 |
| IoU≥0.7 | 2.5 | 1.2 | 1.7 | 0.63 | 0.71 |

- **Mean per-cell IoU (matched)**: 0.779
- **Median per-cell IoU (matched)**: 0.847
- **Out-of-scope predictions/frame**: 0.8

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
| 1 | 10 | 2 | 1.00 |
| 3 | 8 | 3 | 1.00 |
| 5 | 9 | 1 | 1.00 |
| 4 | 3 | 5 | 1.00 |
| 2 | 3 | 6 | 1.00 |

## Per-frame breakdown

| Frame | n_GT | n_pred | TP@.5 | FP | FN | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| F0 | 4 | 3 | 3 | 0 | 1 | 0.86 |
| F10 | 3 | 4 | 2 | 2 | 1 | 0.57 |
| F20 | 3 | 4 | 2 | 2 | 1 | 0.57 |
| F30 | 3 | 4 | 2 | 2 | 1 | 0.57 |
| F40 | 2 | 4 | 2 | 2 | 0 | 0.67 |
| F50 | 4 | 4 | 3 | 1 | 1 | 0.75 |
| F60 | 4 | 4 | 3 | 1 | 1 | 0.75 |
| F70 | 4 | 5 | 3 | 2 | 1 | 0.67 |
| F80 | 5 | 5 | 5 | 0 | 0 | 1.00 |
| F90 | 5 | 5 | 5 | 0 | 0 | 1.00 |

## Files

- `per_frame.csv` — TP/FP/FN at each IoU threshold per frame
- `per_cell.csv` — IoU + matched IDs for every cell pair
- `iou_distribution.png` — histogram of per-cell IoU
- `confusion.png` — GT-ID × pipeline-ID heat map
- `report.json` — machine-readable everything
