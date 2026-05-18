"""Unit tests for core.multichannel.

Covers:
* cy5_presence_score returns ~1 for bright cell, ~0 for debris, mid
  for faint cell
* filter_dic_labels_by_cy5 drops only the debris
* Robust to filopodia (long thin protrusions with low actin)
* Edge handling (cell at image border)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

from core.multichannel import (
    cy5_presence_score,
    filter_dic_labels_by_cy5,
    per_cell_cy5_features,
)


def make_synthetic_frame(H=200, W=200, seed=0):
    """Build a 200×200 uint8 'Cy5' frame with low-intensity background
    and 4 features:
      label 1: bright cell at (50, 50)        — score ~1
      label 2: faint cell at (50, 150)        — score mid
      label 3: debris at (150, 50)            — score ~0 (no Cy5 signal)
      label 4: cell with filopodia at (150,150) — score ~1 despite low
                                                  filopodia intensity
    """
    rng = np.random.default_rng(seed)
    cy5 = (rng.integers(20, 35, size=(H, W))).astype(np.uint8)
    labels = np.zeros((H, W), dtype=np.int32)

    # Bright cell — small disc with high Cy5
    yy, xx = np.mgrid[40:61, 40:61]
    disc = (yy - 50)**2 + (xx - 50)**2 <= 100
    labels[40:61, 40:61][disc] = 1
    cy5[40:61, 40:61][disc] = 220

    # Faint cell — half as bright (still well above bg)
    disc = (yy - 50)**2 + (xx - 50)**2 <= 100
    labels[40:61, 140:161][disc] = 2
    cy5[40:61, 140:161][disc] = 80

    # Debris — DIC mask but no Cy5 elevation
    labels[140:161, 40:61][disc] = 3
    # cy5 unchanged here — same as background

    # Cell with filopodia — bright body + thin low-intensity arms
    body_y, body_x = np.mgrid[140:161, 140:161]
    body = (body_y - 150)**2 + (body_x - 150)**2 <= 64
    labels[140:161, 140:161][body] = 4
    cy5[140:161, 140:161][body] = 200
    # Filopodia: 4 thin protrusions with low Cy5
    for arm_y, arm_x in [(slice(130, 140), slice(149, 152)),
                          (slice(161, 171), slice(149, 152)),
                          (slice(149, 152), slice(130, 140)),
                          (slice(149, 152), slice(161, 171))]:
        labels[arm_y, arm_x] = 4
        cy5[arm_y, arm_x] = 50  # only slightly above bg
    return labels, cy5


def test_bright_vs_debris():
    labels, cy5 = make_synthetic_frame()
    s1 = cy5_presence_score(labels == 1, cy5)
    s2 = cy5_presence_score(labels == 2, cy5)
    s3 = cy5_presence_score(labels == 3, cy5)
    s4 = cy5_presence_score(labels == 4, cy5)
    print(f"  bright cell (label 1)        score = {s1:.2f}  (expect ~1.0)")
    print(f"  faint cell (label 2)         score = {s2:.2f}  (expect mid)")
    print(f"  debris (label 3)             score = {s3:.2f}  (expect ~0)")
    print(f"  filopodia cell (label 4)     score = {s4:.2f}  (expect high)")
    assert s1 >= 0.9, f"bright cell under-scored: {s1}"
    assert s3 <= 0.15, f"debris over-scored: {s3}"
    assert s2 > s3, "faint cell should beat debris"
    assert s4 >= 0.6, "filopodia cell shouldn't drag score below 0.6"


def test_filter_drops_only_debris():
    labels, cy5 = make_synthetic_frame()
    filt, scores, kept = filter_dic_labels_by_cy5(
        labels, cy5, min_score=0.3)
    print(f"  scores: {scores}")
    print(f"  kept {kept}/4 labels")
    # Labels 1, 2, 4 should survive; label 3 (debris) should not
    surviving_old_ids = set(scores.keys()) - {3}
    surviving_old_ids = {k for k in surviving_old_ids if scores[k] >= 0.3}
    assert kept == len(surviving_old_ids), (
        f"kept count {kept} doesn't match surviving set {surviving_old_ids}")
    # Verify the filtered label frame has compacted IDs starting at 1
    new_ids = sorted(set(filt.flatten()) - {0})
    assert new_ids == list(range(1, kept + 1)), (
        f"label IDs not compacted: {new_ids}")


def test_features():
    labels, cy5 = make_synthetic_frame()
    feats = per_cell_cy5_features(labels == 1, cy5)
    print(f"  bright-cell features: {feats}")
    assert feats["mean"] > 150
    assert feats["p75"] >= feats["median"]
    assert feats["p95"] >= feats["p75"]


def test_edge_cell():
    """Cell touching image border: ring may be small, fallback should kick in."""
    H = W = 100
    cy5 = (np.random.default_rng(1).integers(20, 35, size=(H, W))
           .astype(np.uint8))
    labels = np.zeros((H, W), dtype=np.int32)
    # Cell in upper-left corner
    labels[0:15, 0:15] = 1
    cy5[0:15, 0:15] = 200
    s = cy5_presence_score(labels == 1, cy5)
    print(f"  edge cell score = {s:.2f}  (expect high)")
    assert s >= 0.5, "edge cell should still score"


def test_empty_mask():
    cy5 = np.zeros((100, 100), dtype=np.uint8)
    empty_mask = np.zeros((100, 100), dtype=bool)
    s = cy5_presence_score(empty_mask, cy5)
    assert s == 0.0


def test_find_cy5_missed_regions():
    """Build a frame with 2 DIC-detected cells + 1 Cy5-only bright
    region (= missed cell candidate). find_cy5_missed_regions should
    identify exactly the missed region."""
    from core.cy5_fallbacks import find_cy5_missed_regions
    H = W = 200
    rng = np.random.default_rng(42)
    cy5 = (rng.integers(20, 35, size=(H, W))).astype(np.uint8)
    labels = np.zeros((H, W), dtype=np.int32)

    # DIC-detected cell 1 — has Cy5
    yy, xx = np.mgrid[40:61, 40:61]
    disc = (yy - 50)**2 + (xx - 50)**2 <= 100
    labels[40:61, 40:61][disc] = 1
    cy5[40:61, 40:61][disc] = 200

    # DIC-detected cell 2 — has Cy5
    labels[40:61, 140:161][disc] = 2
    cy5[40:61, 140:161][disc] = 200

    # Cy5-only bright region — DIC missed this one
    by, bx = np.mgrid[140:161, 70:91]
    bdisc = (by - 150)**2 + (bx - 80)**2 <= 100
    cy5[140:161, 70:91][bdisc] = 220
    # No labels[...] = ... here — this is our "missed" cell

    cands = find_cy5_missed_regions(labels, cy5,
                                     k_mad=3.0, min_area_px=20)
    print(f"  found {len(cands)} candidate(s)")
    for c in cands:
        print(f"    centroid={c['centroid']} area={c['area_px']} "
              f"cy5_max={c['cy5_max']}")
    assert len(cands) == 1, (
        f"expected 1 missed region, got {len(cands)}")
    cy_y, cy_x = cands[0]["centroid"]
    assert 145 <= cy_y <= 155 and 75 <= cy_x <= 85, (
        f"missed region centroid ({cy_y:.0f}, {cy_x:.0f}) wrong")


if __name__ == "__main__":
    print("Test 1: bright vs faint vs debris vs filopodia")
    test_bright_vs_debris()
    print("\nTest 2: filter drops only debris")
    test_filter_drops_only_debris()
    print("\nTest 3: per-cell features")
    test_features()
    print("\nTest 4: edge cell")
    test_edge_cell()
    print("\nTest 5: empty mask")
    test_empty_mask()
    print("\nTest 6: find Cy5+ regions missed by DIC")
    test_find_cy5_missed_regions()
    print("\nAll multichannel unit tests passed ✓")
