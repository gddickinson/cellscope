"""State-aware motility metrics: speed, MSD, persistence, computed
separately for each cell state (balled vs attached).

Centroid trajectory is the input. State labels per frame come from
`core.cell_state.classify_track_states`. We only count motion within
contiguous state segments — transitions between states (e.g. cell
spreading after division) are excluded from the motility numbers.
"""
from __future__ import annotations

import numpy as np


def _displacements(centroids, frame_indices, um_per_px):
    """Return frame-pair displacements (in µm) for the given indices.

    `centroids` is (T, 2) [row, col] in pixels with NaN for missing
    frames. `frame_indices` is a contiguous range we should consider.
    Returns dict with `displacements_um` (K-1, 2), `velocities_um_per_min`
    (K-1, 2), `valid_pairs` count.
    """
    out = {"displacements_um": np.zeros((0, 2)),
           "valid_pairs": 0}
    if len(frame_indices) < 2:
        return out
    cents = centroids[frame_indices]
    valid = ~np.isnan(cents[:, 0])
    if valid.sum() < 2:
        return out
    # Restrict to consecutive valid pairs
    cents_valid = cents[valid]
    d = np.diff(cents_valid, axis=0) * um_per_px
    out["displacements_um"] = d
    out["valid_pairs"] = len(d)
    return out


def state_speeds(centroids, states, target_state,
                  um_per_px, dt_min, speed_cap=15.0,
                  min_segment_len=3):
    """Per-step speeds (µm/min) during target_state.

    Only counts steps inside contiguous state segments of length
    ≥ `min_segment_len` to avoid noisy single-frame transitions.
    `speed_cap` drops outliers.

    Returns 1-D array of speeds.
    """
    from core.cell_state import state_segments
    segs = state_segments(states)
    speeds = []
    for state, s, e in segs:
        if state != target_state:
            continue
        if e - s + 1 < min_segment_len:
            continue
        idx = np.arange(s, e + 1)
        d = _displacements(centroids, idx, um_per_px)
        if d["valid_pairs"] == 0:
            continue
        sp = np.linalg.norm(d["displacements_um"], axis=1) / dt_min
        sp = sp[sp <= speed_cap]
        speeds.extend(sp.tolist())
    return np.array(speeds, dtype=np.float32)


def state_msd(centroids, states, target_state, um_per_px,
               max_lag=30, min_segment_len=10):
    """Mean-squared displacement vs lag time, computed only on
    pure-state segments of length ≥ `min_segment_len`.

    Returns dict with `lag` (1..L), `msd` (1..L) — average MSD
    across segments. `msd[k]` = mean(|r(t+k) - r(t)|²) in µm².
    Returns NaN array if no segment is long enough.
    """
    from core.cell_state import state_segments
    segs = [(s, e) for st, s, e in state_segments(states)
            if st == target_state and e - s + 1 >= min_segment_len]
    if not segs:
        return {"lag": np.arange(1, max_lag + 1),
                "msd": np.full(max_lag, np.nan),
                "n_segments": 0}

    msd_per_lag = [[] for _ in range(max_lag)]
    for s, e in segs:
        idx = np.arange(s, e + 1)
        cents = centroids[idx] * um_per_px
        valid = ~np.isnan(cents[:, 0])
        cents = cents[valid]
        T = len(cents)
        for lag in range(1, min(max_lag, T - 1) + 1):
            disp = cents[lag:] - cents[:-lag]
            sq = (disp ** 2).sum(axis=1)
            msd_per_lag[lag - 1].extend(sq.tolist())

    msd = np.array([float(np.mean(v)) if v else np.nan
                    for v in msd_per_lag])
    return {"lag": np.arange(1, max_lag + 1),
            "msd": msd,
            "n_segments": len(segs)}


def state_persistence(centroids, states, target_state, um_per_px,
                       min_segment_len=5, max_lag=10):
    """Directional persistence: lag-1 cosine similarity of velocity.

    Higher = more persistent (cell maintains direction).
    Range: -1 (random reverse) to +1 (perfectly straight).

    Computed only on pure-state segments. Also returns the
    autocorrelation curve up to `max_lag` for the segment-merged data.

    Returns dict with `lag1` (scalar — single-step persistence),
    `autocorr` (max_lag-length array), `n_segments`.
    """
    from core.cell_state import state_segments
    segs = [(s, e) for st, s, e in state_segments(states)
            if st == target_state and e - s + 1 >= min_segment_len]
    if not segs:
        return {"lag1": float("nan"),
                "autocorr": np.full(max_lag, np.nan),
                "n_segments": 0}

    cos_per_lag = [[] for _ in range(max_lag)]
    for s, e in segs:
        idx = np.arange(s, e + 1)
        cents = centroids[idx] * um_per_px
        valid = ~np.isnan(cents[:, 0])
        cents = cents[valid]
        if len(cents) < 3:
            continue
        v = np.diff(cents, axis=0)
        speeds = np.linalg.norm(v, axis=1)
        unit = v / np.maximum(speeds[:, None], 1e-9)
        for lag in range(1, min(max_lag, len(unit) - 1) + 1):
            cos = (unit[lag:] * unit[:-lag]).sum(axis=1)
            cos_per_lag[lag - 1].extend(cos.tolist())

    autocorr = np.array([float(np.mean(v)) if v else np.nan
                          for v in cos_per_lag])
    return {"lag1": autocorr[0] if not np.isnan(autocorr[0]) else 0.0,
            "autocorr": autocorr,
            "n_segments": len(segs)}


def state_total_displacement(centroids, states, target_state,
                              um_per_px, min_segment_len=5):
    """Sum of segment endpoint-to-endpoint distances during target_state.

    Distinguishes "directed motion" from "random walk" — together
    with state_speeds / state_msd, gives meaningful turnover rates.
    """
    from core.cell_state import state_segments
    segs = [(s, e) for st, s, e in state_segments(states)
            if st == target_state and e - s + 1 >= min_segment_len]
    total_disp = 0.0
    total_path = 0.0
    for s, e in segs:
        idx = np.arange(s, e + 1)
        cents = centroids[idx] * um_per_px
        valid = ~np.isnan(cents[:, 0])
        cents = cents[valid]
        if len(cents) < 2:
            continue
        endpoint = float(np.linalg.norm(cents[-1] - cents[0]))
        path = float(np.linalg.norm(np.diff(cents, axis=0),
                                       axis=1).sum())
        total_disp += endpoint
        total_path += path
    straightness = (total_disp / total_path) if total_path > 0 else 0.0
    return {"total_displacement_um": total_disp,
            "total_path_um": total_path,
            "straightness": straightness,
            "n_segments": len(segs)}
