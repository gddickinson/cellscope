# CellScope — Improvement Roadmap

Comprehensive list of potential improvements, organized by priority
and category. Effort: S = hours, M = 1-2 days, L = 3-5 days, XL = 1+ weeks.

---

## Priority 0: Multi-Cell Detection Improvement (Active)

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
