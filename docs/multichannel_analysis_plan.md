# Multi-channel (DIC + actin) analysis plan

**Source**: `/Volumes/GeorgeDrive/ignasi/IC295/` — 19 recordings × 3 channels
× 97 frames × 2048×2048 uint16, 0.6523 µm/px, 10-min interval.
Channels: `Cy5`, `DIC 10x`, `None` (placeholder). Cy5 is most likely
**SiR-actin** or similar far-red live-cell actin probe — bright cells
on a near-zero background.

This plan covers how to use actin labelling to improve the existing
DIC-only pipeline (`hybrid_cpsam_multi`) along three axes:
**detection accuracy**, **segmentation boundary quality**, and
**tracking robustness**.

---

## 1 — What I observed on a single recording

`IC295__1_MMStack_Pos1-WT.ome.tif`, frame 30, cpsam (cellpose 4.1.1,
default model) on each channel after appropriate preprocessing:

| Channel | Preprocessing | Cells detected | Visual quality |
|---|---|---:|---|
| Cy5 (actin) | percentile rescale | **5 cells** | All clearly real cells; clean instance borders; one bright cell + diffuse halo per detection |
| DIC | flat-field σ=80 | 15 cells | Real cells detected, **plus 8–10 false positives on dust/debris** that have no actin signal |

**Photobleaching is mild**: across 97 frames Cy5's p99.9 drops only
~20 % (193 → 152 counts); the max stays ~9800. No need for time-
adaptive normalisation.

**Cell density is sparse and uneven** across positions: 1–2 cells in
some fields, 6+ in others. Same as the IC293 dataset.

## 2 — The key insight — Cy5 = ground-truth cell filter

DIC labels everything that bends light: cells **and** debris, dust,
fibre fragments, air bubbles. Actin labels only viable cells with
intact cytoskeleton. Therefore:

> **DIC tells us *where* a cell-shaped object is.
> Cy5 tells us *whether* that object is a cell.**

Combining them gives:
* Higher precision (drops ~50 % of DIC false positives in the test)
* Higher recall than Cy5 alone (DIC sees cells with weak actin —
  e.g. cells in M phase, partially-stained cells)
* A free quality flag per track ("track has Cy5 throughout" vs
  "track is debris-suspicious")

## 3 — Concrete improvements per pipeline stage

### 3.1 — Detection

Three strategies, ranked by expected ROI:

#### 3.1a — **AND-fusion (recommended)** — *2-3 days*

```
mask_dic = cpsam(dic_uint8)          # accurate geometry, includes debris
mask_cy5 = cpsam(cy5_uint8)          # only real cells, less accurate edge
mask_real = filter_dic_by_cy5(
    mask_dic, cy5_uint8,
    min_cy5_overlap=0.15,            # 15% of DIC pixels must be bright Cy5
    cy5_threshold=cy5_p99 + 2*mad)
```

Implementation: for each DIC instance, compute Cy5 mean inside the
mask; reject if below threshold (= debris). Keep all DIC geometry
that survives.

* **Pros**: best of both modalities, removes most debris,
  retains DIC's accurate boundaries
* **Cons**: misses cells with very low actin (rare — typically
  mitotic cells, but those are visible in DIC and tracker can bridge)
* **Cost**: small — adds one Cy5 inference + per-cell stats per
  frame (~2 s extra per frame)

#### 3.1b — Cy5-first, DIC-augment

Detect on Cy5 (high precision); for each cpsam-detected DIC cell
not overlapping any Cy5 detection, keep it but flag as "no actin"
(may be debris OR mitotic).

* **Pros**: clean default output, actin flag is informative
* **Cons**: more code complexity, one mask source per cell

#### 3.1c — Multi-channel native

cellpose 4 doesn't natively combine channels (cellpose 3 did,
via `channels=[seg, bg]`). Could:
1. Stack Cy5 + DIC into RGB and feed to cpsam (lossy hack)
2. Train a tiny custom 2-channel cellpose head (1–2 days plus
   training data)

Not recommended over 3.1a unless 3.1a underperforms in practice.

### 3.2 — Segmentation refinement

DIC gives accurate cell *outlines* (you can see thin protrusions,
filopodia, lamellipodia). Cy5 emphasises cell *body* (cortex +
stress fibres) and is biased away from filopodia (low actin
density). So:

* Use DIC mask as canonical boundary
* Cy5 only as a presence/intensity feature

If we want stronger boundaries: **Chan-Vese refinement on the DIC
gradient, seeded with Cy5 mask**. This is a 30-line addition since
`core/alt_segmentation.py` already has Chan-Vese. Probably worth
testing only if DIC boundaries are insufficient — not usually the
case.

### 3.3 — Tracking

Current Hungarian cost (`core.multi_cell.track_all_cells`):
`α·distance + β·size_diff`.

With actin, extend to:

```
cost = α·distance
     + β·|area_t − area_t-1| / area_avg
     + γ·|cy5_mean_t − cy5_mean_t-1| / cy5_mean_avg
     + δ·feature_distance(cy5_pattern_t, cy5_pattern_t-1)
```

* `cy5_mean` is a per-cell stable identifier across frames
* `cy5_pattern` could be small embedding (8-16 floats from spatial
  histogram or hand-crafted features: stress-fibre presence,
  cortex thickness, polarity)
* Single biggest benefit: **resolving identity when two cells
  cross paths**

Effort: Hungarian refactor ~1 day; embedding ~1 day if hand-crafted,
2-3 days for a learned descriptor.

### 3.4 — Mitosis detection (cytokinesis ring)

During cell division, a contractile ring of actin forms at the
midzone. This shows in Cy5 as a **bright transverse band** across
an elongating cell, then resolves into two daughter cells with
inherited actin patterns.

Detection: per-cell, track shape descriptors:
* Major-axis elongation increasing → spindle elongation
* Bright ring along minor axis → constriction
* Track splits into two roughly-equal-sized children → division event

Outputs: per-track lineage tree (parent → daughter₁, daughter₂).
Useful for proliferation rate quantification.

Effort: ~3 days. Requires a heuristic + maybe a simple classifier
trained on a few hand-labelled events.

### 3.5 — Apoptosis / dead-cell detection

Dying cells show:
* Loss of polarised actin (uniform diffuse Cy5)
* Or eventual loss of all actin signal
* Cell rounds up (high circularity)

Filter: per-track, last-frame circularity high + Cy5 intensity
dropping → apoptosis flag. Track ends.

### 3.6 — Subcellular morphology features (advanced)

Actin distribution within each cell carries biological information:

* **Lamellipodia at leading edge**: bright actin band at one cell pole
  → migration polarity vector. Validates inferred trajectory.
* **Stress fibres**: directional actin striations → contractile state.
  Quantify via local anisotropy / orientation field.
* **Cortical actin thickness**: rim-to-centre ratio of Cy5
  intensity → cell stiffness proxy.

These become **new per-cell timeseries** alongside speed/area:
`actin_polarity`, `stress_fibre_density`, `cortex_ratio`.

For Piezo1 biology these are likely to differentiate WT/KO/GOF/Y1
better than speed alone — Piezo1 modulates actin organisation, so
this is closer to the molecular phenotype than migration is.

Effort: 1-2 days per feature.

## 4 — Recommended phased implementation

| Phase | What | Effort | Risk | Status (2026-05-01) |
|---|---|---:|---|---|
| **1** | Cy5-only baseline | 0.5 day | low | ad-hoc test done — cpsam(Cy5) gets 5 cells vs cpsam(DIC) 15 (debris); Cy5 is the cleaner instance source but DIC catches more cells |
| **2** | AND-fusion: cpsam(DIC) + Cy5 presence filter | 2 days | low | **DONE** — `core/multichannel.py` (filter + score); `core/hybrid_multichannel.py` (orchestrator). Pilot pending GPU. |
| **2b** | Tier 2 fail-safe: Cy5+ regions without DIC mask → crop + cpsam(DIC, TTA) | +1 day | low | **DONE** — `core/cy5_fallbacks.recover_missed_cells_via_dic_crop` |
| **2c** | Tier 3 fail-safe: temporal gap fill via Cy5 (cpsam on Cy5 crop) | +1 day | low | **DONE** — `core/cy5_fallbacks.cy5_gap_fill_for_track` |
| **3** | Per-track Cy5 mean + pattern in Hungarian cost (§3.3) | 1 day | low | next |
| **4** | Quality flag in `core/track_quality.py`: penalise tracks with low/no Cy5 | 0.5 day | low | next |
| **5** | Subcellular features (§3.6): polarity + cortex ratio | 2 days | med | queued |
| **6** | Mitosis lineage detection (§3.4) | 3 days | high | speculative |

Phase 1-2c done; Phase 3-4 are the next ROI items. Phase 5-6 require
hand-labelled ground truth + biology-driven validation.

## 5 — Code structure (as built)

```
core/
  multichannel.py          loaders + DIC/fluo preprocessing +
                             cy5_presence_score + filter_dic_labels_by_cy5
                             + per_cell_cy5_features  (210 lines)
  cy5_fallbacks.py         find_cy5_missed_regions +
                             recover_missed_cells_via_dic_crop +
                             cy5_gap_fill_for_track + 3 fill strategies
                             (348 lines)
  hybrid_multichannel.py   detect_hybrid_multichannel orchestrator —
                             wraps hybrid_cpsam_multi + Cy5 filter
                             + optional Tier 2/3 fail-safes (174 lines)
output/
  run_metadata.py          write_run_metadata helper (100 lines) —
                             auto-captures cellpose / torch / numpy /
                             tifffile / python / platform versions for
                             RUN_METADATA.md emitted by every script
scripts/
  test_multichannel_unit.py    6 synthetic tests, all passing
  test_multichannel_pilot.py   real-data pilot (queued for GPU)
  bench_multichannel.py        IoU benchmark vs hand-labelled GT
  sample_gt_frames.py          --multichannel mode added
  inspect_ic295_channels.py    DIC + Cy5 + composite contact sheet/PDF
docs/
  multichannel_analysis_plan.md  THIS FILE
data/
  ic295_inspection/        contact sheet + PDF + README (channel
                             inspection that informed the design)
  ic295_gt/                114 GT candidates (38 frames × 3 PNGs)
                             pending hand-labelling
```

`hybrid_cpsam_multi.py` itself stays unchanged — wrapped from outside
so the existing IC293 pipeline isn't disturbed.

## 6 — Resolved questions (2026-05-01)

1. **Is Cy5 always SiR-actin?** ✅ Yes (confirmed by data owner).
   Intensity is biologically meaningful — fluorogenic probe is only
   bright when bound to F-actin.
2. **Are some cells un-labelled?** ✅ User-confirmed: no fully
   un-stained cells; large intensity range with some cells faintly
   labelled. The adaptive z-score filter handles faint cells (relative
   to local background, not absolute threshold).
3. **GT for IC295?** ✅ Yes, sampled — `data/ic295_gt/` (38
   candidates), pending hand-labelling.
4. **4th channel coming?** Calcium signals planned for future
   experiments. User intends a separate analysis pipeline — don't
   build it into this one. Loader is N-channel-generic so it'll
   accommodate when the time comes.

## 7 — Defence against fluorescence artefacts

User concern: if we ever start using a binary Cy5 mask directly as
the cell mask, fluorescence artefacts could be classified as cells.
Design defends against this at every tier:

| Tier | What happens | Artefact defence |
|---|---|---|
| 1 — DIC primary | cpsam on DIC, Cy5 ignored | artefacts can't enter |
| 2 — recovery | crop + cpsam(DIC) on Cy5-flagged region | cpsam(DIC) finds nothing → recovery aborted |
| 3 — gap fill | requires temporal anchor in flanking frames | free-floating artefact never triggers; default `cy5_cpsam` strategy uses cpsam morphology |
| AND filter | drops DIC masks with low Cy5 | tightens precision (drops debris) |

The threshold-only fill strategies (`cy5_cleanup`, `cy5_threshold`)
are available but not default — only used as last resort when even
cpsam(Cy5) fails AND a track confirms the cell exists.

## 7 — Timing budget for IC295 full run (estimated)

If we adapt the current IC293 pipeline (cpsam on DIC) to add Cy5
filtering: ~50 min × 19 recordings = **~16 hours**, comparable to
the IC293 run. Adding Cy5 inference roughly doubles per-frame cost
to ~55 s/frame ≈ 1.5 hours per recording = **~28 hours total**.

Per-recording cache lets it resume; can run overnight + a day.

---

## TL;DR (2026-05-01)

* Actin (Cy5) channel gives **precision** — separates cells from DIC
  debris (~50 % FP drop on the test frame).
* **Phases 1–2c implemented + unit-tested**. Pipeline runs DIC
  primary → AND-filter → optional Tier 2/3 fail-safes for missed
  cells, with cpsam morphology verification at every tier so Cy5
  artefacts can't slip through.
* **Pilot pending GPU** (queued for after IC293 finishes). Run with
  `scripts/test_multichannel_pilot.py --cy5-recovery` for the full
  3-tier comparison.
* **GT pending hand-labelling** at `data/ic295_gt/candidates/`.
  After labelling, `scripts/bench_multichannel.py` quantifies
  IoU DIC-only vs AND-fusion vs full 3-tier per condition.
* Phases 3–4 (Hungarian Cy5 features + quality flag) are next.
* Phase 5–6 (subcellular features + mitosis) are speculative but
  likely the strongest biology signal for Piezo1.
