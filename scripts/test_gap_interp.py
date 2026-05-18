"""Tests for core.gap_interp + its use in analysis_plots.

Two checks:
  1. interpolate_short_gaps fills only short interior runs.
  2. plot_speed / plot_area render dotted segments and the result
     looks sane on a real cache (results/full_dataset/*.npz).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

from core.gap_interp import interpolate_short_gaps  # noqa: E402


def test_interpolate_short_gaps():
    # 1-frame gap: filled
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    f, w = interpolate_short_gaps(a, 3)
    assert f[2] == 3.0 and w[2]
    assert not w[0] and not w[1] and not w[3] and not w[4]

    # 2-frame gap with max=3: filled
    a = np.array([0.0, np.nan, np.nan, 6.0])
    f, w = interpolate_short_gaps(a, 3)
    assert f[1] == 2.0 and f[2] == 4.0
    assert w[1] and w[2]

    # 4-frame gap with max=3: NOT filled
    a = np.array([0.0, np.nan, np.nan, np.nan, np.nan, 5.0])
    f, w = interpolate_short_gaps(a, 3)
    assert np.isnan(f[1:5]).all()
    assert not w.any()

    # Edge NaN: not filled (no left neighbour)
    a = np.array([np.nan, np.nan, 3.0, 4.0])
    f, w = interpolate_short_gaps(a, 3)
    assert np.isnan(f[0]) and np.isnan(f[1])
    assert not w.any()

    # max_gap=0 disables
    a = np.array([1.0, np.nan, 3.0])
    f, w = interpolate_short_gaps(a, 0)
    assert np.isnan(f[1])
    assert not w.any()

    # No NaN: no-op
    a = np.array([1.0, 2.0, 3.0])
    f, w = interpolate_short_gaps(a, 5)
    assert (f == a).all()
    assert not w.any()
    print("✓ interpolate_short_gaps logic OK")


def test_render_on_cache():
    """Render plot_area twice (gap=0 vs gap=3) on a real cache and
    save side-by-side PNGs for visual confirmation."""
    cache = "results/full_dataset/dic_pos59_ko.npz"
    if not os.path.exists(cache):
        print(f"  [skip] {cache} not present — render test skipped")
        return
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from gui_focused.analysis_plots import plot_area, plot_speed

    z = np.load(cache, allow_pickle=False)
    # Build a result dict like the analysis worker does
    track_keys = [k for k in z.files if k.startswith("track_")
                  and k.endswith("_stack")]
    if not track_keys:
        print("  [skip] no tracks in cache")
        return
    stack = z[track_keys[0]]
    n = stack.shape[0]
    area_um2 = np.array([float(stack[i].sum()) * 0.65 ** 2
                         if stack[i].any() else np.nan
                         for i in range(n)])
    # Punch artificial gaps so we can see the dotted fill in action
    area_um2[5:7] = np.nan
    area_um2[20] = np.nan
    speed = np.diff(np.r_[0, np.cumsum(np.nan_to_num(area_um2)) / 1000])
    speed[5:7] = np.nan
    speed[20] = np.nan
    result = {
        "speed": speed,
        "shape_timeseries": {"area_um2": area_um2},
    }

    out_dir = "results/gap_interp_demo"
    os.makedirs(out_dir, exist_ok=True)
    for label, gap in [("off", 0), ("3frame", 3)]:
        fig = Figure(figsize=(10, 4))
        plot_area(fig, result, gap_interp_max=gap)
        path = os.path.join(out_dir, f"area_{label}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        print(f"  ✓ wrote {path}")

        fig = Figure(figsize=(10, 4))
        plot_speed(fig, result, gap_interp_max=gap)
        path = os.path.join(out_dir, f"speed_{label}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        print(f"  ✓ wrote {path}")


if __name__ == "__main__":
    test_interpolate_short_gaps()
    test_render_on_cache()
    print("\nAll gap-interp tests passed")
