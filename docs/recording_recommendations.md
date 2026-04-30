# Recording-specific recommendations

Concrete settings for getting good results on different recording types.
All recommendations are based on benchmarks against held-out manual
ground truth — IoUs quoted are mean values on the relevant test set.

![CellScope on different recording types](figures/hero.png)
*The same pipeline (cpsam_dic + DeepSea + Hungarian tracking) handles
single-cell DIC, crowded multi-cell DIC, and (with default cpsam) phase-
contrast — the modality selector picks the right config automatically.*

---

## Quick reference

| Recording type | Modality | Mode | Model | Other |
|---|---|---|---|---|
| **Our 526² DIC** (Holt-style) | DIC | Single-Cell | cpsam_dic | DeepSea on, fallback on |
| **Jesse 1024² DIC** (multi-position) | DIC | Multi-Cell | cpsam_dic | min_area=500, expected_cells=Auto, tiling=3×3 if cells small |
| **VAMPIRE-style cropped DIC** | DIC | Single-Cell | cpsam_dic | DeepSea on |
| **Ignasi-style cropped phase-contrast** | Phase-contrast | Multi-Cell | default cpsam | DeepSea on, gap fill on, TTA optional |
| **Crowded multi-cell DIC** (≥3 cells/frame) | DIC | Multi-Cell | cpsam_dic | DeepSea on, gap fill on, expected_cells = typical N |
| **Noisy / low-light DIC** | DIC | Single- or Multi-Cell | cellpose_combined_robust | TTA on, lower min_area |

---

## DIC time-lapses

DIC has high local-variance backgrounds (texture relief). The default
**cpsam** picks up this texture as false positives, which is why we
fine-tuned **cpsam_dic** specifically.

### Recommended pipeline (any DIC recording)

```
modality      = DIC
mode          = Single-Cell  (or Multi-Cell if more than one cell)
DIC model     = Auto  (resolves to cpsam_dic when present)
DeepSea       = on
fallback      = on
TTA           = off  (turn on if you see missed cells, see below)
gap fill      = on   (multi-cell only)
min area      = 500
expected cells = Auto   (or set to N if you know the cell count)
```

Benchmarks for cpsam_dic (vs older cellpose_dic_v3):

| Test set | cellpose_dic_v3 | **cpsam_dic** |
|---|---:|---:|
| our-GT 526² in-domain | 0.740 | **0.795** |
| VAMPIRE held-out OOD crops | 0.279 | **0.697** |
| Detection rate (our-GT) | 95% | **100%** |

### When to use the older CP3 fine-tunes

- **cellpose_combined_robust** (CP3) — trained with heavy noise and
  brightness augmentation. Wins by 0.05+ IoU on perturbations
  (added noise, gamma shifts) where in-domain models drop. Use it
  when your imaging conditions are unstable across the time-lapse.
- **cellpose_dic_v3** (CP3) — fastest of the DIC fine-tunes. Use as
  a fallback if cpsam_dic OOMs (1.1 GB ViT) or if you can't download
  the cpsam_dic model.

### Common DIC failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| One cell tracked, another visibly missed | cpsam misses faint cell | TTA on; multi-cell + gap fill |
| Boundaries look "puffy" / too loose | cpsam tends to over-extend | DeepSea on (it tightens) |
| Filopodia rounded off | bbox padding rounds thin protrusions | Already best-effort; manual edit if critical |
| Lots of small false-positive blobs | cpsam catches debris | Raise min_area or set expected_cells |
| Cell missed when partly out of frame | Edge clipping | Set expected_cells; manual edit |

---

## Phase-contrast time-lapses

Phase-contrast has smooth backgrounds and bright edge halos. Default
cpsam works well out of the box. **Don't stack MedSAM** — it tightens
cpsam's already-good boundaries and rounds off filopodia (-0.148 IoU
on the Ignasi benchmark).

### Recommended pipeline (any phase-contrast recording)

```
modality      = Phase-contrast
mode          = Single-Cell (single-cell crop) or Multi-Cell (FoV)
DeepSea       = on  (auto-filters debris via fill_holes + largest_CC)
fallback      = on
TTA           = off  (on if cells missed)
gap fill      = on   (multi-cell only)
min area      = 500
```

Benchmark: cpsam + DeepSea = 0.932 IoU on Ignasi 65 GT (65/65 frames > 0.85).

### Multi-cell phase-contrast

The hybrid_cpsam_multi pipeline does:

1. cpsam at defaults → instance labels per frame
2. min_area filter (drops debris)
3. Per-cell DeepSea refinement (preserves identities)
4. Hungarian tracking + division detection
5. Gap fill via `cpsam(augment=True)` for tracks that disappear

100% gap fill rate on tested recordings (32/32 gaps on Pos3-WT,
9/9 on Pos2-WT).

---

## Cropped single-cell recordings (any modality)

If your recording is already cropped to a single cell of interest:

- Use **Single-Cell** mode.
- Skip Multi-Cell — it adds tracking overhead with no benefit.
- TTA usually unnecessary unless cell orientation varies a lot.
- Set Expected cells = 1 if you're seeing extra fragments tracked.

---

## Large multi-cell recordings (≥1024² with several cells)

For Jesse-style 1024² OME-TIFF recordings with 3-10 cells per frame:

- **Multi-Cell** mode. Always.
- **Tiling = 3×3 with overlap=64** if cpsam's default receptive field
  misses cells. The default `tile=True` in cellpose uses 224 px tiles
  with 0.1 overlap, which is fine for 526² but under-resolves on 1024²
  with small cells. Explicit 3×3 tiling roughly doubles detection rate
  (verified on Jesse pos0_wt 5/10 → 10/10, pos59_ko 8/10 → 10/10).
- **TTA** for stubbornly-missed cells.
- **Gap fill** is essential — large frames have more chances for any
  individual cell to dim out for a frame.

---

## Noisy / unstable recordings

If brightness drifts, focus changes mid-recording, or noise is heavy:

- Switch the DIC model dropdown to **cellpose_combined_robust** (CP3).
  This fine-tune was trained with elastic + motion-blur + defocus
  augmentation. It wins on 9 of 11 perturbation classes vs the
  in-domain models, especially gamma shifts (+0.05 IoU) and noise
  (+0.04 IoU).
- Enable **TTA**.
- Disable DeepSea if it's chasing noise — DeepSea works on
  texture-rich frames; on degraded ones it can hallucinate.

For perfectly clean DIC recordings, `cpsam_dic` still wins overall;
`cellpose_combined_robust` is the right choice when condition stability
isn't guaranteed.

---

## Recording-by-recording recommendations (project examples)

| Example file | Recommended preset |
|---|---|
| `data/examples/control/*.mp4` (our 526² DIC, single cell) | DIC, Single-Cell, cpsam_dic, defaults |
| `data/examples/cko/*.mp4` (our 526² DIC, single cell, filopodia-rich) | Same as above. Watch for filopodia under-segmentation. |
| `data/examples/jesse_wt/pos0_wt.ome.tif` (1024², 1 large cell) | DIC, Single-Cell, cpsam_dic, defaults |
| `data/examples/jesse_wt/pos17_wt.ome.tif` (1024², ~4 cells) | DIC, Multi-Cell, cpsam_dic, gap fill, expected_cells=Auto |
| `data/examples/jesse_ko/pos59_ko.ome.tif` (1024², 5+ cells) | DIC, Multi-Cell, cpsam_dic, tiling 3×3, gap fill |
| `data/examples/jesse_ko/pos65_ko.ome.tif` (1024², dim) | DIC, Multi-Cell, cellpose_combined_robust, TTA on |
| `data/ignasi/...Pos2-WT...` (cropped phase, 2 cells) | Phase-contrast, Multi-Cell, defaults, gap fill, TTA on |
| `data/ignasi/...Pos3-WT...` (cropped phase, 3 cells, division) | Phase-contrast, Multi-Cell, defaults, gap fill |

---

## When manual editing is faster than tweaking parameters

If 1–2 frames out of N have problematic detection and you've already
tried TTA + gap fill, just open the **Mask Editor** and fix them by
hand. It's faster than another full pipeline run. Edits load back
into the Detection GUI via "Send to GUI".

---

## Timing reference (M-series MPS, single cell)

| Pipeline step | ~time per 100-frame recording |
|---|---:|
| cpsam_dic detection (subprocess startup + per-frame) | 60–90 s |
| DeepSea refinement (per cell) | 30–60 s |
| Multi-cell tracking + gap fill | 5–15 s |
| Full single-cell pipeline | 100–150 s |
| Full multi-cell pipeline (3 cells) | 200–300 s |

CUDA on a recent NVIDIA GPU is 2–3× faster than M-series. CPU-only is
10–15× slower.
