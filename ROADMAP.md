# CellScope — Roadmap

*Updated: 2026-05-01*

The project has matured through three stages: a single-cell DIC analyser
(reproducing Holt et al. 2021), a multi-cell tracking + GUI suite, and a
distribution-ready package. Stage 4 is using the tool for the actual
biology + releasing it as a citable artifact.

This roadmap is prioritised tiers, not a strict timeline. Tier 1 items
are gating before scaling up real use; Tiers 2–4 are polish, science,
and release.

---

## Tier 1 — Robustness for real use

### 1.1 Validate TTA on a held-out set
We threaded `use_tta` through DIC pipelines but never benched it. Run
`bench_cpsam_dic.py` with TTA on/off across the OOD + in-domain sets.
If improvement < 0.02 IoU, leave the GUI checkbox off by default and
document. If ≥ 0.02, document the cost/benefit and update
`recording_recommendations.md`.
**Effort:** 1 h compute + 15 min docs.

### 1.2 Fix gap-fill cross-env subprocess slowness
`track_gap_fill` falls back to the sibling conda env (CP3 ↔ CP4) for
gaps the in-env primary attempt couldn't fill. Each subprocess reloads
cellpose cold (~30 s × N gaps). Cheap fix: only fire the cross-env
fallback when the primary attempt actually fails. Already gated by a
fall-through, but verify and tighten.
**Effort:** ~2 h.

### 1.3 End-to-end run on full available dataset
Run the pipeline on every Jesse + Ignasi recording, save results +
quality dashboards. Surfaces detection drops, fragmentation, and
runtime issues that single-recording figure-runs miss.
**Effort:** ~2 h compute + 1 h review.

### 1.4 Per-recording quality dashboard
For each pipeline run, write a one-page `quality_report.html` with:
- Detection rate timeseries
- Per-track presence heatmap (rows=tracks, cols=frames)
- Speed distribution + outlier frame flagging
- 3 sample frames at 0/50/100% with overlay
- Suspicious frames (area Δ > 50%, centroid hop > 100 px) flagged

Lets a user spot a bad recording in 10 seconds.
**Effort:** ~1 day.

### 1.5 Pipeline regression test
Pin a 30-frame canonical recording + expected outputs (n_tracks, mean
mask area, IoU vs reference). `pytest` re-runs and compares. Catches
silent breakage when modules get refactored.
**Effort:** ~3 h.

---

## Tier 2 — UX & documentation polish

### 2.1 Interpolate short gaps in analysis graphs ✅
Done 2026-04-30. `core/gap_interp.py` adds `interpolate_short_gaps`
+ `plot_with_gaps`. Focused + tracking GUIs gain a "Gap fill" combo
(off / ≤1 / ≤2 / ≤3 / ≤5). Interpolated samples drawn dotted so
they're never confused with measured points. Demo + tests at
`scripts/test_gap_interp.py`; visual confirmation at
`results/gap_interp_demo/`.

### 2.2 First-analysis tutorial
Step-by-step `docs/tutorial.md`: load → detect → edit → analyse → export
on the bundled example recording. Screenshots + commentary.
**Effort:** ~3 h.

### 2.3 Example Jupyter notebooks
2–3 notebooks under `examples/`: scripted single-recording analysis,
batch comparison with statistics, custom-model detection.
**Effort:** ~4 h.

### 2.4 FAQ / troubleshooting tree
Short structured page covering specific failure modes.
**Effort:** ~2 h.

### 2.5 Track-quality indicator in tracking GUI ✅
Done 2026-04-30. `core/track_quality.py` computes a 0-1 composite from
frames-present + area stability + total path length. Tracking GUI gets
a "Quality" column with green/amber/red row coloring + tooltip showing
component breakdown. Tests at `scripts/test_track_quality.py`; demo
screenshot at `results/track_quality_demo/`.

---

## Tier 3 — Scientific validation & biology

### 3.1 Piezo1 comparison (the paper figure)
Run the full pipeline on Control vs Piezo1-cKO (vs GoF if available)
recordings, output:
- Per-genotype migration speed, persistence, MSD diffusion exponent
- Per-genotype shape mode distributions (VAMPIRE)
- Statistical comparison with effect sizes
- Comparison to Holt et al. 2021 published numbers

This is the original goal of the project.
**Effort:** ~1 week.

### 3.2 Cross-cell-type validation
Try the pipeline on a non-keratinocyte recording (fibroblasts, neurons,
MDA-MB-231). Either confirms generality or reveals limits.
**Effort:** 0.5–1 day.

### 3.3 Multi-cell GT on our recordings
Annotate 30 frames × 3 cells per genotype manually. Use to:
- Quantify multi-cell tracking accuracy on our domain
- Tune Hungarian tracker hyperparameters
- Provide citable per-domain numbers in the paper
**Effort:** ~1 day annotation + ~2 h evaluation.

### 3.4 ID-switch mitigation
Add appearance features to Hungarian cost, OR migrate to BTrack tuned
for slow-frame DIC.
**Effort:** ~3 days.

### 3.5 Brightness-augmented retrain — setup done; training pending
Setup done 2026-04-30. Workflow + scripts in
`docs/brightness_robustness.md`:

- `scripts/build_brightness_test.py` — generates an 8-perturbation
  test set (1,129 pairs × 8 = 9,032 images) under
  `data/training/dic_splits_v3/test_brightness/`
- `scripts/augment_brightness.py` — generates the v4 brightness-
  augmented training set (~21k pairs)
- `scripts/bench_brightness.py` — orchestrates the existing
  `bench_cpsam_dic.py` across all perturbations + computes IoU
  retention vs clean; supports `--compare` between two model labels
- `notebooks/train_cpsam_dic_v4_brightness_colab.ipynb` — Colab
  notebook configured for the v4 dataset (T4-friendly stratified
  subsample so each brightness variant is equally represented)

**Pending**: bench v2 baseline → train v4 on Colab (6 h T4) → re-bench
→ compare. Ship criterion: ≥0.8 IoU retention on every perturbation
without losing >0.02 on clean.

### 3.6 New Ignasi cohort (IC293) full pipeline run
Setup done 2026-05-01. 16 recordings × 1.6 GB × 2048×2048 phase
contrast across 5 conditions (WT, KO, GOF, Y1, DMSO). Found that:

- **`cpsam_base` (no fine-tune) wins by 2.5×** over the trained
  `cpsam_dic` on these out-of-distribution full-frame recordings
  (11.7 cells/frame vs 4.6) — see
  `results/ignasi_model_comparison/`. Fine-tune was trained on small
  448² crops at one cell density; it under-detects on 2048² sparse
  full-frame data.
- **Flat-field σ=80 preprocessing is essential** to remove the
  illumination vignette before cpsam.
- **Gap-fill + TTA cost 3.8 h/recording** with no real benefit on
  this dataset (cpsam_base hits 100% per-frame). Disabled by default
  in `scripts/run_ignasi_new_full.py`; opt-in via `--gap-fill`/`--tta`.

**Status**: full run in progress (started 2026-05-01); 8/16 recordings
done at time of writing, ~50 min/recording, ETA total ~13 h. Caching
per recording so resumable.

Outputs: `results/ignasi_new_full/<pos>_<cond>.{npz,html}` +
Fiji-ready overlay TIFFs + summary.csv + by_condition.csv +
RUN_METADATA.md.

### 3.7 Multi-channel (DIC + SiR-actin Cy5) pipeline — IC295
New 2026-05-01. 19 recordings × 3 channels (Cy5 SiR-actin, DIC 10x,
None) at `/Volumes/GeorgeDrive/ignasi/IC295/`. SiR-actin acts as a
ground-truth "is this a viable cell?" filter — DIC over-detects
(debris) and Cy5 cleanly separates cells from debris.

Implemented (all CPU-only, no GPU contention with the IC293 run):

- `core/multichannel.py` — N-channel loader, channel-aware
  preprocessing, Cy5 presence scoring (z-score vs local annulus,
  robust MAD denominator), AND-filter, per-cell features.
- `core/cy5_fallbacks.py` — three-tier fail-safe recovery:
  - **Tier 2**: Cy5+ regions without DIC mask → crop + cpsam(DIC,
    TTA optional) recovery.
  - **Tier 3**: track gap at frame N + Cy5 signal at interpolated
    centroid → fill via `cy5_cpsam` (default; runs cpsam on Cy5
    crop), `cy5_cleanup` (threshold + morph), or `cy5_threshold`
    (raw threshold). Temporal anchor + cpsam morphology defend
    against fluorescence artefacts.
- `core/hybrid_multichannel.py` — orchestrator wrapping
  `hybrid_cpsam_multi` (DIC) + Cy5 filter + optional Tier 2/3
  recovery. Returns kept tracks + dropped tracks (debris) + raw
  labels + n_cy5_recovered.
- `output/run_metadata.py` — reusable RUN_METADATA.md helper for
  the project's "always emit metadata" convention.
- `scripts/test_multichannel_unit.py` — 6 synthetic unit tests
  (bright/faint/debris/filopodia/edge/empty/missed-region). All pass.
- `scripts/test_multichannel_pilot.py` — real-data pilot with
  `--cy5-recovery` flag (queued for after IC293 finishes).
- `scripts/sample_gt_frames.py` — `--multichannel` mode → writes
  `<name>_dic.png` + `_cy5.png` + `_composite.png` per candidate.
- `scripts/bench_multichannel.py` — benchmark vs hand-labelled GT.
- 38 GT candidates sampled at `data/ic295_gt/` (19 recordings × 2
  frames × 3 PNGs = 114 files), pending hand-labelling.
- Channel inspection PDF + contact sheet at
  `data/ic295_inspection/` confirming SiR-actin pattern + sparse
  no-fully-un-stained-cell hypothesis.

Plan doc: `docs/multichannel_analysis_plan.md` covers all six phases
(detection → segmentation → tracking → quality flags → subcellular
features → mitosis lineage). Phases 2-3 (AND-fusion + Hungarian
extension) are next.

**Pending**:
1. Run pilot (`scripts/test_multichannel_pilot.py --cy5-recovery`)
   on Pos14-KO/Pos26-GOF/Pos0-WT once IC293 frees cellpose4 GPU.
2. Hand-label IC295 GT.
3. Quantitative benchmark (`bench_multichannel.py`): IoU DIC-only vs
   AND-fusion vs full 3-tier per condition.
4. Phases 4-6 (per-cell quality flag, subcellular features, mitosis).

---

## Tier 4 — Release & community

### 4.1 Methods paper draft
The methods paper has clean contributions: cpsam_dic fine-tune,
hybrid pipeline, multi-cell tracking with TRA 0.929, 5-GUI suite.
Target: Bioinformatics / Bioimage Informatics / Nature Methods Tools.
**Effort:** ~2 weeks.

### 4.2 Zenodo deposit + DOI
Deposit code + models + example data once a v1.0 tag is cut. For
citability.
**Effort:** half day.

### 4.3 GitHub release with v1.0 tag
SemVer release, README badges, issue templates, contributing guide.
**Effort:** ~1 day.

### 4.4 Public model cards
`models/MODEL_CARD.md` per shipped model: training data, intended use,
performance, limitations. Standard ML hygiene.
**Effort:** ~3 h.

### 4.5 ImageJ / Fiji bridge
Macro that reads `masks.npz` + the source recording into Fiji/ImageJ
as overlaid hyperstack. Lowers adoption friction for biology labs that
live in Fiji.
**Effort:** ~2 days.

---

## Tier 5 — Advanced features (deferred)

- 3D / Z-stack support
- Live-imaging "watch folder" mode
- OME-Zarr output (bioimage standard)
- SAM2 video propagation as alternative tracker
- Mixed-effects statistical models
- GPU-accelerated DeepSea per-cell

---

## Recommended sequence (next 4 weeks)

**This week — Ignasi cohort + multichannel:**
3.6 IC293 run finishes (in progress) → 3.7 pilot → IC295 GT
labelling → 3.7 quantitative benchmark.

**Week 2 — Publication prep:**
3.1 (Piezo1 comparison figure incorporating IC293 + IC295 results)
→ 3.3 multi-cell GT → 4.4 model cards.

**Week 3 — Robustness backfill:**
3.5 (brightness retrain — Colab training) → 1.5 (regression tests).

**Week 4 — Release & polish:**
4.1 paper draft outline → 4.5 Fiji bridge polish → 4.2/4.3 release.

In parallel: 2.2 / 2.3 / 2.4 (docs polish) — small chunks at a time.

---

## What's the single highest-leverage next thing?

(2026-05-01 update) — once the IC293 run completes, **the multichannel
pilot (3.7)**. It validates whether the AND-fusion strategy actually
removes the debris false positives we saw in DIC-only detection. If
it works, the IC295 dataset becomes the strongest cohort for the
paper figure since it has both genotype (WT/KO/GOF) AND
pharmacology (Y1/DMSO) AND a fluorescent ground-truth signal.
Original recommendation was 1.2+1.4+3.1 — those are still valid; the
multichannel work just leapfrogged the priority list when the IC295
data arrived.
