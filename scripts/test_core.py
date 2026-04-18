"""Unit tests for core analysis modules."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_tracking():
    from core.tracking import (
        extract_centroids, instantaneous_speed, total_distance,
        net_displacement, persistence_ratio, mean_squared_displacement,
    )
    # Create a simple mask stack: cell moves right
    masks = np.zeros((10, 50, 50), dtype=bool)
    for i in range(10):
        masks[i, 20:30, 10 + i*2:20 + i*2] = True

    cents = extract_centroids(masks)
    assert cents.shape == (10, 2), f"centroids shape: {cents.shape}"
    assert not np.isnan(cents).any(), "NaN in centroids"

    speed = instantaneous_speed(cents, dt=1.0)
    assert len(speed) == 9
    assert all(s >= 0 for s in speed), "negative speed"

    dist = total_distance(cents)
    assert dist > 0, "zero distance"

    net = net_displacement(cents)
    assert net > 0, "zero net displacement"

    pers = persistence_ratio(cents)
    assert 0 <= pers <= 1, f"persistence {pers} out of range"

    lags, msd, sem = mean_squared_displacement(cents)
    assert len(lags) > 0
    assert all(m >= 0 for m in msd)
    print("  tracking: PASS")


def test_morphology():
    from core.morphology import shape_descriptors, shape_timeseries
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True  # 40x40 square

    desc = shape_descriptors(mask, um_per_px=1.0)
    assert "area_um2" in desc
    assert abs(desc["area_um2"] - 1600) < 10, f"area: {desc['area_um2']}"
    assert desc["circularity"] < 1.0
    assert desc["solidity"] > 0.9

    masks = np.stack([mask, mask, mask])
    ts = shape_timeseries(masks, um_per_px=1.0)
    assert len(ts["area_um2"]) == 3
    print("  morphology: PASS")


def test_statistics():
    from core.statistics import group_comparison, significance_marker

    # 2-group comparison
    g2 = group_comparison(
        {"A": [1, 2, 3, 4], "B": [10, 11, 12, 13]}, "test")
    assert g2["n_groups"] == 2
    assert any(t["test_name"] == "Welch's t-test" for t in g2["tests"])
    assert g2["tests"][0]["p_value"] < 0.001  # clearly different
    assert g2["pairwise"][0]["sig"] == "***"

    # 3-group comparison
    g3 = group_comparison(
        {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}, "test")
    assert g3["n_groups"] == 3
    assert any(t["test_name"] == "One-way ANOVA" for t in g3["tests"])
    assert len(g3["pairwise"]) == 3  # 3 pairs

    # Significance markers
    assert significance_marker(0.0001) == "***"
    assert significance_marker(0.005) == "**"
    assert significance_marker(0.03) == "*"
    assert significance_marker(0.1) == "ns"
    print("  statistics: PASS")


def test_advanced_analysis():
    from core.advanced_analysis import (
        fit_msd_diffusion, bootstrap_ci, smooth_trajectory,
        flag_quality_issues, check_normality,
    )

    # MSD fitting
    lags = np.arange(1, 20)
    msd = 4 * 0.5 * lags ** 1.0 + np.random.randn(19) * 0.01
    fit = fit_msd_diffusion(lags, msd, dt=1.0)
    assert abs(fit["alpha"] - 1.0) < 0.3, f"alpha: {fit['alpha']}"
    assert fit["r_squared"] > 0.9

    # Bootstrap
    ci = bootstrap_ci([1, 2, 3, 4, 5])
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]

    # Smoothing
    traj = np.column_stack([np.arange(10), np.arange(10)])
    traj_f = traj.astype(float)
    smoothed = smooth_trajectory(traj_f, sigma=1.0)
    assert smoothed.shape == traj.shape
    assert not np.isnan(smoothed).any()

    # Quality flagging
    masks = np.ones((10, 50, 50), dtype=bool)
    masks[5] = False  # missing frame
    flags = flag_quality_issues(masks)
    assert 5 in flags["suspect_frames"]

    # Normality
    norm = check_normality(np.random.randn(100))
    assert norm["is_normal"]  # normal data should pass
    print("  advanced_analysis: PASS")


def test_project():
    import tempfile
    from core.project import save_project, load_project

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.cellscope")
        rec = {"video_path": "/fake/path.tif", "name": "test",
               "um_per_px": 0.65, "time_interval_min": 5.0,
               "frames": np.zeros((5, 10, 10), dtype=np.uint8)}
        masks = np.ones((5, 10, 10), dtype=bool)
        det = {"masks": masks}
        result = {"mean_speed": 1.5, "persistence": 0.3}
        params = {"min_area_px": 500}

        save_project(path, rec, det, result, params, "single")
        assert os.path.exists(path)
        assert os.path.exists(path.replace(".cellscope", "_masks.npz"))

        proj = load_project(path)
        assert proj["mode"] == "single"
        assert proj["masks"] is not None
        assert proj["masks"].shape == (5, 10, 10)
        assert proj["analysis"]["mean_speed"] == 1.5
    print("  project: PASS")


def main():
    print("=== CellScope Core Unit Tests ===\n")
    tests = [
        ("tracking", test_tracking),
        ("morphology", test_morphology),
        ("statistics", test_statistics),
        ("advanced_analysis", test_advanced_analysis),
        ("project", test_project),
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {name}: FAIL — {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
