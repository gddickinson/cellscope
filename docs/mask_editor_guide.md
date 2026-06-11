# CellScope Mask Editor — User Guide

The mask editor is for **reviewing and correcting** per-cell segmentation
masks: removing artifact tracks, adding missed cells, trimming vignette
edges, and cleaning up mask defects, then saving the corrected
`masks.npz` back in place.

Launch with a recording + masks:

```bash
conda run -n cellpose4 python main_editor.py path/to/video.ome.tif path/to/masks.npz
```

## Typical review workflow

1. **Step through frames** with `←` / `→` and watch each track.
2. **Remove whole-track artifacts** — phantoms (high-velocity, short
   tracks), debris, duplicates. Press `X` (delete tool) and click the
   track, or use **Filter Cells…** (`Ctrl+Shift+F`) to remove many at
   once by criteria (speed, persistence, track length).
3. **Trim vignette / edge artifacts** with **Trim Edges…** (`Ctrl+T`) —
   removes a strip of mask along an image border across frames.
4. **Clean mask defects** — fill small holes and drop rogue specks — with
   **Clean (holes/specks)** (`Ctrl+K`). Keeps only the largest component
   per cell and fills interior holes.
5. **Add a missed cell** — press `N` (New Cell), then paint it with the
   brush (`B`).
6. **Fix a boundary** — `B` (brush) to add, `E` (eraser) to remove,
   `P` (polygon) for a region, `R` (relabel) to reassign a pixel region
   to another cell ID.
7. **Save** with `Ctrl+S` (in place). `Ctrl+Shift+S` saves and advances a
   frame for fluid frame-by-frame review.

Edits save straight back to the `masks.npz` you launched with, which is
what the analysis (`ic295_analyze_one.py`) reads — so after reviewing,
re-run analysis on the corrected recording.

## Tools

| Tool | Key | What it does |
|---|---|---|
| Brush | `B` | Paint pixels into the active cell |
| Eraser | `E` | Remove pixels from the active cell |
| Polygon | `P` | Outline a region to add to the active cell |
| Fill | `F` | Flood-fill a region into the active cell |
| Relabel | `R` | Reassign painted pixels to another cell ID |
| Delete | `X` / `Delete` | Click a track to delete it (whole-cell) |

Pick the **active cell** with the number keys (`1`–`9`, `0` = 10;
`Shift+1`–`Shift+0` = 11–20), or use **New Cell** (`N`) for the next free
ID. Toggle the on-image cell-ID labels with `I`.

## Cleanup helpers

- **Filter Cells…** (`Ctrl+Shift+F`) — bulk-remove whole tracks by
  velocity / persistence / track-length thresholds (the fastest way to
  clear detection false-positives).
- **Trim Edges…** (`Ctrl+T`) — strip a fixed-width band of mask along the
  chosen border(s) across a frame range (vignette / illumination edge).
- **Clean (holes/specks)** (`Ctrl+K`) — per cell, fill interior holes and
  keep only the largest connected component.

## Saving & undo

- `Ctrl+S` — save masks in place.
- `Ctrl+Shift+S` — save and advance one frame.
- `Ctrl+Z` / `Ctrl+Shift+Z` — undo / redo (per-frame).
- `Ctrl+G` / `Ctrl+Shift+G` — save GT for this frame / all GT frames
  (ground-truth labelling workflow).

See **Help ▸ Keyboard Shortcuts** in the editor for the full, always
up-to-date list.
