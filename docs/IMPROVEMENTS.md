# CellScope — Improvement Roadmap

Comprehensive list of potential improvements, organized by priority
and category. Effort: S = hours, M = 1-2 days, L = 3-5 days, XL = 1+ weeks.

---

## Priority 0: Audit + optimize the 4-phase track gap-fill cascade

### Problem (raised 2026-05-31 during the IC295 batch)

The 4-phase track gap-fill cascade (`core/track_gap_fill.py`) is by
far the **dominant cost** of multi-cell detection on IC295 recordings.
Timing data from 7 completed detections in the live batch:

| Recording  | Max cells / frame | Detect (min) |
|---|---:|---:|
| Pos26-GOF  |  4 |  73 |
| Pos14-KO   |  5 |  84 |
| Pos1-WT    |  8 | 131 |
| Pos49-Y1   |  9 | 161 |
| Pos38-OT   | 10 | 210 |
| Pos2-WT    | 11 | 234 |
| Pos60-DMSO | 12 | 211 |

Linear fit: **~50 min base + ~17 min per cell**. The base cost
is cpsam + alignment + downsample (essentially fixed); the
per-cell cost is dominated by the cascade. Each cell with a gap can
trigger:

1. **Phase 1 — cpsam(augment=True)**: 4-rotation TTA at each gap.
2. **Phase 2 — CP3 + MedSAM + DeepSea**: subprocess in `cellpose` env
   per gap — pays ~5–10 s conda env warmup every time.
3. **Phase 3 — SAM2 video propagation**: per gap, full-res forward.
4. **Phase 4 — translation fill**: instant.

For a 12-cell recording, the per-cell cost is dominating the entire
detect: ~12 × ~17 min = ~200 min, vs ~50 min for everything else.

### Why it matters

A 50 % cost reduction on the cascade saves **1-2 days** off the
full IC295 batch (65 recordings × ~50 min saved each). A successful
audit might find that some phases are **redundant for most gaps**
already filled by earlier phases (CLAUDE.md notes "100 % fill rate
on tested recordings (41/41 gaps)" — if Phase 1 already gets the
bulk, Phases 2-3 may be over-engineered for typical IC295).

### Result (2026-06-05) — Phase 1 crop: 12.5× with no quality loss

Instrumentation pinned the cost on **Phase 1**: it ran
`cpsam.eval(full_frame, augment=True)` per gap on the whole 2048²
image (measured **~76 s/gap**) just to pick the one cell near an
already-interpolated centroid. Since Phase 1 only ever *accepts* a
cell within `search_radius` of that centroid, segmenting an adaptive
**crop** around it is provably loss-free for recall.

GT benchmark (`scripts/bench_gap_fill.py`, reviewed Pos7-WT masks as
truth, 20 synthetic gaps): crop **6.1 s/gap vs full 75.9 s/gap = 12.5×**,
**0** good fills (IoU≥0.5) dropped, shared-fill IoU within −0.002 of
full, mask agreement 0.967. Shipped as `DEFAULTS.gap_fill_crop=True`
(revert: `gap_fill_crop=False` / GUI toggle / `--no-gap-fill-crop`)
with a full-frame fallback on the rare crop-miss (recall ≥ full path).
Per-phase timing now logs + flows to `stats_out`.

**End-to-end caveat.** That 12.5× is at full **2048²**; the pipeline
auto-downsamples to **1024²**, where cpsam is NOT pixel-bound (the ViT
normalises to a target cell diameter), so the real gain is small —
Pos10-WT (4 cells, 118 gaps): gap-fill 46.7→39.8 min (1.2×), end-to-end
81.3→74.4 min (1.09×). Crop stays on (safe, and it *raised* Phase-1
fill rate 19→33, shifting work off slower SAM2), but gap-fill is still
~half of detect. **The real production lever is the number of Phase-1
`cpsam(augment=True)` calls (118 here) and the 4× `augment` cost — not
input pixels.** ✅ **Done (2026-06-05):** `DEFAULTS.gap_fill_augment
=False` — GT benchmark at 1024² shows crop+noaug vs crop+aug **2.1×**,
0 good fills dropped, IoU Δ −0.011, and Phase 1 now cascades no-augment
→ augment-on-miss → full-frame-on-miss (recall ≥ old path). crop+noaug
vs the original full+aug is ~18× at 1024². Still open: confident-track
short-circuit (cut the *number* of gaps that reach Phase 1) + Phase 2/3
ablation below.

### Plan (after the current IC295 batch finishes)

- [x] **S** — Instrument `track_gap_fill.fill_track_gaps` to log
      time + gaps-resolved per phase. `stats_out` dict + per-phase
      log line (done 2026-06-05).
- [x] **Phase-1 crop acceleration** — adaptive crop around the
      interpolated centroid; 12.5× faster, GT-validated (done 2026-06-05).
- [ ] **M** — Per-phase ablation study on 3-5 representative IC295
      recordings (sparse, medium, dense): run with each phase toggled
      off, compare F1 / IoU / ID consistency / division-catch vs the
      full cascade. Cheap-to-skip phases get a `DEFAULTS` toggle to
      off; expensive-but-essential phases stay.
- [ ] **S** — Subprocess overhead audit: Phase 2 calls `conda run -n
      cellpose` per gap; profile how much of Phase 2's wall time is
      env warmup vs actual work. If it's mostly warmup, batch all of
      a recording's Phase-2 gaps into a single subprocess call.
- [ ] **S** — Confident-detection short-circuit: if a track has no
      true gaps (every frame has a detection) AND no edge artifacts,
      skip the cascade entirely. Cheap eligibility check, big payoff.
- [ ] **M** — Resolution audit: Phase 3 (SAM2) runs at full
      resolution; cells at downsampled resolution may be enough for
      gap interpolation. Quick A/B test.
- [ ] **M** — Consider removing Phase 2 or Phase 3 outright if the
      ablation shows < 1 % gap-fill rate vs the cost. Document the
      decision with the F1 / IoU evidence.

### Success criteria

- Concrete per-phase timing + gap-fill yield in `RUN_METADATA.json`.
- A defensible decision (keep / kill / simplify) for each phase
  backed by the ablation numbers.
- Target: ≥ 30 % wall-time reduction on dense recordings, with no
  measurable F1 / IoU / ID consistency loss on the 13-recording GT
  corpus.

---

## Priority 0 (closed): Multi-Cell Detection Improvement (Active)

### Problem
When multiple cells touch or are close together:
1. **Touching cells merge** — cpsam treats two adjacent cells as one blob
2. **False positives** — spurious detection in the gap between two real cells
3. **Missed cells** — faint or partially occluded cells occasionally dropped

### Plan
**Step 1: Create multi-cell GT** (user, in progress)
- Label 15-20 frames from Pos2-WT (2 cells) and/or Pos3-WT (2-3 cells,
  includes division) using the mask editor with multi-cell labels (1-9 keys)
- Prioritize frames where cells are touching or close together
- Include frames spanning the full recording (early, mid, late) to capture
  morphological variation
- Save as `data/manual_gt/multicell/frame_NNNN_masks.png` (uint16, pixel
  value = cell ID)

**Step 2: Benchmark current pipeline** (M)
- [ ] Compute per-cell IoU on multi-cell GT for the current
  hybrid_cpsam_multi pipeline
- [ ] Identify systematic failure modes: which frames/configurations fail?
- [ ] Measure: merge rate, false positive rate, miss rate

**Step 3: Develop boundary separation** (L)
- [ ] Watershed from cell centroids using cpsam union mask as foreground
  — splits touching cells at the intensity valley between them
- [ ] Marker-controlled watershed seeded by cpsam instance labels
  — each cpsam-detected cell seeds a basin, watershed finds boundaries
- [ ] Train a thin boundary classifier (2-3 px band between touching cells)
  using the GT labels as supervision
- [ ] Evaluate each approach on held-out GT frames

**Step 4: Fine-tune cpsam on multi-cell data** (L)
- [ ] Crop multi-cell GT regions and augment (rotations, flips, noise)
- [ ] Fine-tune cpsam ViT with LoRA on the augmented crops
- [ ] 5-fold CV to measure improvement vs vanilla cpsam
- [ ] If successful, ship as a "multi-cell tuned" model option

**Step 5: Improve false positive rejection** (M)
- [ ] Analyze false positive characteristics (area, shape, intensity
  relative to real cells)
- [ ] Train a simple classifier (area + circularity + intensity) to
  reject inter-cell artifacts
- [ ] Alternatively: if expected_cells is set, keep only the N most
  cell-like detections per frame (ranked by area × circularity)

**Step 6: Improve missed cell recovery** (M)
- [ ] For frames where a tracked cell disappears but reappears later:
  use the previous frame's mask as a prompt for cpsam (bbox + mask hint)
- [ ] Lower cellprob_threshold locally around the expected position
- [ ] Template matching from the previous frame's cell crop

### Success Criteria
- Per-cell IoU > 0.85 on multi-cell GT (touching + non-touching)
- Zero false positive cells in GT frames
- Zero missed cells in GT frames where cells are visible
- Correct identity tracking through contact events

---

## Priority 1: High-Impact, Low-Effort

### UI/UX Polish
- [ ] **S** — Recent files list (File menu, persist across sessions)
- [ ] **S** — Remember last export directory per session
- [x] **S** — Show recording info (name, frames, scale) permanently in status bar, not just a dialog
- [x] **S** — Show elapsed time on pipeline stage buttons after completion
- [ ] **S** — Estimated time remaining in batch processing (based on per-recording average)
- [ ] **S** — Frame number tooltip on slider hover
- [x] **S** — Cancel button during detection (set a stop flag on the worker thread)
- [ ] **M** — Progress bar during first-run model download (cpsam 2.4 GB, MedSAM 375 MB)
- [x] **M** — Resizable graph panel (QSplitter between image and analysis, user can drag divider)

### Pipeline Quick Wins
- [x] **S** — TTA toggle (`augment=True`) exposed as a checkbox in detection params — already proven to recover missed frames
- [ ] **S** — Temporal mask smoothing (median filter across N frames) — reduces frame-to-frame jitter
- [x] **S** — Frame quality flagging — highlight frames where cell area changes >50% from neighbours
- [ ] **M** — Confidence score per frame — boundary gradient magnitude as a proxy for detection quality (already have `boundary_confidence` in core/evaluation.py, just surface in the nav bar)

### Export Improvements
- [x] **S** — Video export (MP4/AVI) — render contour overlay as a playable video, not just TIFF stack
- [ ] **M** — Cell division event CSV — explicit table of division time, parent track ID, daughter track IDs, area ratios
- [x] **M** — Per-cell CSV export — one row per cell per frame with all metrics (x, y, area, speed, perimeter...)

---

## Priority 2: Significant Features

### Analysis Enhancements
- [x] **M** — MSD diffusion model fitting — extract diffusion coefficient D and anomalous exponent α from MSD curve (linear fit in log-log)
- [x] **M** — Temporal trajectory smoothing — Kalman filter or Savitzky-Golay on centroids before speed computation
- [x] **M** — Bootstrap confidence intervals on all metrics — report 95% CI alongside mean ± SEM
- [x] **M** — Normality testing before parametric stats — Shapiro-Wilk on each group, auto-switch to non-parametric if violated
- [ ] **L** — Cell lineage tree visualization — diagram showing division events as branching tree (matplotlib or graphviz)
- [ ] **L** — Automated PDF/HTML report generation — compile all plots + metrics + statistical results into a publication-ready document

### Multi-Cell Improvements
- [ ] **M** — Contact detection — flag frames where two tracked cells' masks overlap or are within N px
- [ ] **M** — Per-cell color persistence — assign deterministic colors by track ID so the same cell always gets the same color across sessions
- [ ] **L** — Improved division detection — current heuristic (area ratio) misses many events; use temporal signature (sudden area halving + new centroid nearby)
- [ ] **L** — Cell-cell interaction metrics — pairwise distance, approach/separation velocity, contact duration

### GUI Features
- [ ] **M** — Side-by-side comparison view — split image viewer to show two recordings or two timepoints
- [ ] **M** — Batch mask editing — select frames and apply same operation (e.g., delete label 3 from frames 50-70)
- [ ] **M** — Annotation tools — add text labels, arrows, scale bars to frames for presentations
- [x] **M** — Project files — save/load complete analysis state (recording path + masks + results + settings) as a .cellscope file
- [ ] **L** — Measurement tools — ruler tool for point-to-point distance, angle tool for membrane curvature

---

## Priority 3: Architecture & Code Quality

### Testing
- [x] **M** — Unit tests for core modules (tracking, morphology, edge_dynamics, statistics) with known-answer inputs
- [ ] **S** — CI configuration (GitHub Actions) running unit tests on push
- [ ] **M** — Benchmark suite — standardized performance test on a reference recording (report detection time, IoU if GT available)

### Code Cleanup
- [ ] **M** — Split oversized core modules: pipeline.py (627 lines), refinement.py (550 lines), detection.py (539 lines) — each should be <500
- [ ] **M** — Remove unused presets from gui/options/presets.py — 33 presets, most for legacy pipelines. Keep only the 5-6 relevant to the focused GUI
- [ ] **M** — Clean config.py — remove constants for unused RF/cascade/snap pipelines
- [ ] **L** — Add type hints throughout core/ modules
- [ ] **S** — Consistent error handling — raise exceptions at system boundaries, return None/empty internally

### Distribution
- [ ] **M** — PyPI package (`pip install cellscope`) — setup.py/pyproject.toml with entry_points for console scripts
- [ ] **L** — Docker container — Dockerfile with CUDA support, pre-installed models
- [ ] **L** — Conda package — recipe for conda-forge
- [ ] **M** — Single-env solution — investigate whether cellpose 4.x can be patched to load CP3 models (would eliminate the dual-env complexity)
- [ ] **M** — Host models on Zenodo/HuggingFace with DOI — permanent URLs for setup wizard download

---

## Priority 4: Advanced Capabilities

### Detection Pipeline
- [ ] **L** — Multi-scale cpsam detection — run at multiple diameter hints, NMS-merge for recordings with cells spanning wide size range
- [ ] **L** — Active contour (Chan-Vese) post-refinement — proven +0.04 IoU on some recordings, could be a per-frame optional step
- [ ] **L** — Automatic parameter tuning — sample 5 frames, estimate cell size, set min_area and expected_cells automatically
- [ ] **XL** — Fine-tune cpsam on user data — LoRA adaptation of the ViT backbone on user-provided GT (like the MedSAM LoRA work but for cpsam itself)

### Tracking
- [ ] **L** — BTrack integration with tuned DIC config — Bayesian tracker handles mitosis events better than Hungarian; needs per-modality motion model tuning
- [ ] **L** — SAM2 video propagation — propagate a seed mask through the recording using SAM2's memory attention for temporally consistent segmentation
- [ ] **XL** — Graph neural network tracker — learn cell appearance embeddings for more robust identity assignment in crowded fields

### New Analysis Modalities
- [ ] **L** — 3D z-stack support — extend detection and tracking to volumetric recordings
- [ ] **L** — Fluorescence channel integration — co-register DIC + fluorescence, report per-cell fluorescence intensity
- [ ] **XL** — Machine learning phenotyping — classify cells by morphology (e.g., mesenchymal vs epithelial) using shape descriptors as features
- [ ] **XL** — Real-time analysis during acquisition — connect to microscope software, analyze frames as they arrive

### Platform
- [ ] **XL** — Web-based GUI (Flask/Dash) — browser-based interface for remote analysis servers
- [ ] **XL** — Napari plugin — integrate CellScope as a napari plugin for users who prefer that ecosystem
- [ ] **L** — Plugin system — allow users to register custom analysis modules that appear in the GUI

---

## Priority 5: Scientific Rigor

### Validation
- [ ] **M** — Tracking accuracy reporting — compute TRA/DET/SEG metrics (CTC standard) when GT is available
- [ ] **M** — Power analysis tool — given observed effect sizes and within-group variance, estimate required n per group
- [ ] **M** — Outlier detection in batch — flag recordings where metrics are >3 SD from group mean
- [ ] **L** — Cross-validation of analysis parameters — report sensitivity of results to min_area, search_radius choices

### Edge Dynamics
- [ ] **M** — Adaptive angular sectors — vary n_sectors based on cell perimeter (small cells get fewer sectors to avoid noise)
- [ ] **M** — Curvature-weighted edge velocity — weight protrusion/retraction by local membrane curvature
- [ ] **L** — Kymograph segmentation — automatically identify protrusion/retraction events as contiguous regions in the kymograph

---

## Not Recommended (low ROI or high risk)

- ~~Omnipose~~ — pretrained models give 0 IoU on DIC; needs domain fine-tuning (~1 week)
- ~~MC Dropout uncertainty~~ — cellpose has zero Dropout layers; requires source fork
- ~~CycleGAN domain adaptation~~ — data augmentation covers most of the benefit more cheaply
- ~~Mask2Former-VIS~~ — requires significant re-architecture, SAM2 video is simpler
- ~~GPU-accelerated RF~~ — RF pipeline is no longer used in the focused GUI
