"""Tests for core.track_quality + tracking GUI integration.

* Unit cases: short/long tracks, with/without analysis result, edge cases
* Smoke test: load real cache, build track table, verify colors
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

from core.track_quality import compute_track_quality, quality_color  # noqa


def _make_stack(n_total, frames_present, area_per_frame=200, jitter=0):
    """Build a (T, H, W) bool stack with given areas on the chosen frames."""
    H = W = 32
    stack = np.zeros((n_total, H, W), dtype=bool)
    rng = np.random.default_rng(0)
    for f in frames_present:
        a = max(1, int(area_per_frame + rng.integers(-jitter, jitter + 1)))
        # paint a square of approx that area
        side = max(1, int(np.sqrt(a)))
        stack[f, 0:side, 0:side] = True
    return stack


def test_full_track_no_analysis():
    """100% present, stable area → high frames_score, area_score; no path."""
    stack = _make_stack(20, list(range(20)), area_per_frame=200, jitter=0)
    q = compute_track_quality({"stack": stack}, 20)
    assert q["frames_score"] == 1.0
    assert q["area_score"] == 1.0  # zero variance
    assert q["path_score"] is None
    assert q["composite"] >= 0.9
    assert q["label"] == "good"
    print(f"  ✓ full stable track: {q['composite']:.2f} ({q['label']})")


def test_short_track():
    """5 of 20 frames with stable area → low frames_score but
    rescued by perfect area_score. Real short tracks usually have
    unstable area too (covered by test_unstable_area below)."""
    stack = _make_stack(20, [0, 1, 2, 3, 4], area_per_frame=200)
    q = compute_track_quality({"stack": stack}, 20)
    assert q["frames_score"] == 0.25
    # frames=0.25 weight 0.5 + area=1.0 weight 0.3 = ~0.53 → "ok"
    assert q["composite"] < 0.7
    print(f"  ✓ short track: {q['composite']:.2f} ({q['label']})")


def test_short_unstable_track():
    """5 of 20 frames, varying area → poor (no rescue)."""
    stack = _make_stack(20, [0, 1, 2, 3, 4],
                       area_per_frame=200, jitter=180)
    q = compute_track_quality({"stack": stack}, 20)
    assert q["composite"] < 0.5
    assert q["label"] in ("poor", "ok")
    print(f"  ✓ short unstable track: {q['composite']:.2f} "
          f"({q['label']})")


def test_unstable_area():
    """Half-present, wildly varying area → ok or poor."""
    stack = _make_stack(20, list(range(0, 20, 2)),
                        area_per_frame=200, jitter=180)
    q = compute_track_quality({"stack": stack}, 20)
    assert q["area_score"] is not None and q["area_score"] < 0.7
    print(f"  ✓ unstable area: {q['composite']:.2f} ({q['label']}), "
          f"area_score={q['area_score']:.2f}")


def test_with_analysis_result():
    """Analysis result with total_distance and area summary."""
    stack = _make_stack(20, list(range(20)), area_per_frame=200)
    ar = {
        "shape_summary": {"area_um2": {"mean": 200.0, "std": 10.0}},
        "total_distance": 60.0,
    }
    q = compute_track_quality({"stack": stack}, 20, analysis_result=ar)
    assert q["area_score"] is not None and q["area_score"] > 0.9
    assert q["path_score"] == 1.0  # 60 > 50 target → clipped
    assert q["composite"] > 0.9
    print(f"  ✓ with analysis: {q['composite']:.2f} ({q['label']})")


def test_no_path_short_movement():
    """Sessile cell: frames_present good but path 0."""
    stack = _make_stack(20, list(range(20)), area_per_frame=200)
    ar = {
        "shape_summary": {"area_um2": {"mean": 200.0, "std": 5.0}},
        "total_distance": 0.0,
    }
    q = compute_track_quality({"stack": stack}, 20, analysis_result=ar)
    # path_score weight is 0.2 → composite drops by ~0.2 from 1.0
    assert 0.7 <= q["composite"] <= 0.9
    print(f"  ✓ sessile cell: {q['composite']:.2f} ({q['label']})")


def test_color_mapping():
    assert quality_color("good")[1] == 240   # green channel
    assert quality_color("ok")[1] == 235     # amber-ish
    assert quality_color("poor")[1] == 200   # pale red (low green)
    print("  ✓ color mapping OK")


def test_on_real_cache():
    """Iterate every track in a real cache and report scores."""
    cache = "results/full_dataset/dic_pos59_ko.npz"
    if not os.path.exists(cache):
        print(f"  [skip] {cache} not present")
        return
    z = np.load(cache, allow_pickle=False)
    n_total = int(z["frames"].shape[0])
    track_keys = sorted(k for k in z.files
                        if k.startswith("track_") and k.endswith("_stack"))
    print(f"\n  Real cache {cache} ({n_total} frames, {len(track_keys)} tracks):")
    for k in track_keys:
        tid = k.split("_")[1]
        stack = z[k]
        q = compute_track_quality({"stack": stack}, n_total)
        print(f"    track {tid}: composite={q['composite']:.2f} ({q['label']}) "
              f"frames={q['frames_active']}/{q['frames_total']} "
              f"area={q['area_score']:.2f}"
              if q['area_score'] is not None
              else f"    track {tid}: composite={q['composite']:.2f}")


def test_gui_table_offscreen():
    """Build the tracking GUI offscreen and verify the table populates."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from gui_tracking.single_view import SingleTrackingView
    v = SingleTrackingView()
    cache = "results/full_dataset/dic_pos59_ko.npz"
    if not os.path.exists(cache):
        print("  [skip] GUI table test — no cache")
        return
    z = np.load(cache, allow_pickle=False)
    track_keys = sorted(k for k in z.files
                        if k.startswith("track_") and k.endswith("_stack"))
    v.masks = np.zeros(z["frames"].shape, dtype=np.int32)
    v.tracks = []
    for tid_str in [k.split("_")[1] for k in track_keys]:
        s = z[f"track_{tid_str}_stack"]
        v.tracks.append({"stack": s, "first_frame": 0, "parent_id": None})
    v._populate_track_table()
    n = v.track_table.rowCount()
    cols = v.track_table.columnCount()
    headers = [v.track_table.horizontalHeaderItem(c).text()
               for c in range(cols)]
    print(f"  ✓ table populated: {n} rows × {cols} cols, headers={headers}")
    # Check at least one row has a quality cell with non-default background
    q_item = v.track_table.item(0, 1)
    bg = q_item.background().color().getRgb()
    assert bg != (0, 0, 0, 0) and bg != (255, 255, 255, 255)
    print(f"  ✓ row 0 quality cell '{q_item.text()}' bg={bg}")


if __name__ == "__main__":
    print("Unit tests:")
    test_full_track_no_analysis()
    test_short_track()
    test_short_unstable_track()
    test_unstable_area()
    test_with_analysis_result()
    test_no_path_short_movement()
    test_color_mapping()

    test_on_real_cache()

    print("\nGUI integration:")
    test_gui_table_offscreen()
    print("\nAll track-quality tests passed")
