# IC295 ground-truth labeling

## What you have

- **12 recordings**, sampled randomly with `seed=0` so
  the selection is reproducible.
- **10 frames per recording** to annotate (every
  10th frame): [0, 10, 20, 30, 40, 50, 60, 70, 80, 90].
- **Total: 120 annotated frames** across all 6 conditions.

## Conditions covered
DMSO: 2, GOF: 2, KO: 2, OT: 2, WT: 2, Y1: 2

## Folder layout

```
data/ic295_gt_full/
├── manifest.csv            ← list of all 12 recordings
├── contact_sheet.png       ← thumbnail grid of frame 0 of each
├── LABELLING.md            ← this file
└── <Pos>_<Cond>/           ← one subfolder per recording
    ├── IC295__1_MMStack_<Pos>-<Cond>.ome.tif   (symlink)
    ├── IC295__1_MMStack_<Pos>-<Cond>_metadata.txt (symlink)
    ├── IC295__1_MMStack_<Pos>-<Cond>.ome.json     (sidecar with
    │                                               um_per_px etc.)
    ├── GT_FRAMES.txt       ← which frames to annotate
    └── gt_masks/           ← put your labelled masks here
```

## Workflow

1. Open the GUI:
   ```
   conda run -n cellpose python main_focused.py
   ```
2. **File → Open Recording**, pick one of the .ome.tif files (or drag
   it into the window).
3. When the channel chooser appears, pick **DIC = ch1, Fluo = ch0**.
4. Use the Frame slider to scrub through the recording and watch the
   cells move. This is the whole point of using full recordings — the
   temporal context makes it MUCH easier to identify which blobs are
   real cells vs debris.
5. For each frame listed in `GT_FRAMES.txt`, open the **Mask Editor**
   (Edit menu → Open Mask Editor) and label every visible cell.
6. **Save** masks as `mask_F<frame_idx>.png` (e.g. `mask_F30.png`)
   into the `gt_masks/` subfolder.

## Mask format

- Use **int32 PNGs** with one unique integer per cell (1, 2, 3, ...,
  background = 0). The mask editor saves this format by default.
- **Track identity matters.** Cell 1 in `mask_F0.png` should be the
  same physical cell as Cell 1 in `mask_F10.png` if it's still
  visible — that lets us evaluate tracking accuracy too, not just
  per-frame detection.
- If a cell first appears at F30, give it a fresh ID (e.g. cell 5)
  and use the same ID in F40, F50, etc. while it remains visible.

## Tips

- **Use the channel toggle** (DIC ⇄ Fluo) to confirm a faint DIC
  blob is actually a real cell with actin signal.
- **Use the playback** (Play button or arrow keys) to verify you're
  not labelling debris that doesn't move.
- **Don't worry about perfect boundaries** — what matters most for
  accuracy estimation is "did we detect this cell at all" and "did
  we keep the same ID across frames". Boundary IoU >0.5 is plenty.
- If a cell is undergoing mitosis / dying / detaching across the gap
  between two annotated frames (e.g. F30 → F40), that's fine — give
  the daughter cells fresh IDs starting at F40 and note it in
  `gt_masks/notes.txt`.

## When done

Once you've labelled at least one recording, ping me and I'll set up
an evaluation script that:
  - Compares the saved pipeline output against your masks
  - Computes per-frame detection precision/recall
  - Computes track-level identity preservation (TRA score)
  - Reports per-condition accuracy
