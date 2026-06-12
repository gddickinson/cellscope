"""Flower (origin-centred track) plots for the IC295 batch, by condition.

Each cell's FULL trajectory (every tracked frame, both rounded and spread
states) is translated so its first position sits at the origin, then all
cells of a condition are overlaid — a "flower" / rose plot of migration.
One panel per condition, all on the SAME equal x/y axis so conditions are
directly comparable.

Three figures (compare/flower_plots/):
  flower_all.png            every cell (mixed + single-state)
  flower_rounded_only.png   cells that are ROUNDED for their whole track
  flower_spread_only.png    cells that are SPREAD for their whole track

A cell's single-state membership is judged over its CLASSIFIABLE frames
(rounded/spread; `unknown` + edge-truncated frames don't count). Tracks
themselves still use the full trajectory.

Usage:
  conda run -n cellpose4 python scripts/ic295_flower_plots.py
  conda run -n cellpose4 python scripts/ic295_flower_plots.py --um 0.4
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    RECORDINGS_ROOT, COMPARE_DIR, CONDITIONS, parse_condition)
import numpy as np  # noqa: E402

OUT_DIR = os.path.join(COMPARE_DIR, "flower_plots")
DEFAULT_UM = 0.6523                      # IC295 scope (single magnification)
_COND_COLOR = {"WT": "#1f77b4", "KO": "#d62728", "GOF": "#2ca02c",
               "Y1": "#9467bd", "OT": "#ff7f0e", "DMSO": "#7f7f7f"}


def collect(um):
    """Return {condition: {'all'|'rounded'|'spread': [ (T,2) µm tracks ]}}."""
    from core.cell_state import (classify_track_states, STATE_ROUNDED,
                                 STATE_SPREAD)
    from core.tracking import extract_centroids
    out = {c: {"all": [], "rounded": [], "spread": []} for c in CONDITIONS}
    paths = sorted(glob.glob(os.path.join(
        RECORDINGS_ROOT, "*", "*", "pipeline_results", "masks.npz")))
    for i, mp in enumerate(paths):
        label = os.path.basename(os.path.dirname(os.path.dirname(mp)))
        cond = os.path.basename(os.path.dirname(os.path.dirname(
            os.path.dirname(mp))))
        if cond not in CONDITIONS:
            cond = parse_condition(label)
        if cond not in out:
            continue
        try:
            labels = np.load(mp)["labels"]
        except Exception as e:
            print(f"  WARN {mp}: {e}"); continue
        for cid in [int(v) for v in np.unique(labels) if v > 0]:
            stack = labels == cid
            cents = extract_centroids(stack)          # (N,2) px, NaN absent
            valid = ~np.isnan(cents).any(axis=1)
            if valid.sum() < 2:
                continue
            cv = cents[valid]
            traj = (cv - cv[0]) * um                  # origin-centred, µm
            out[cond]["all"].append(traj)
            sd = classify_track_states(stack, um_per_px=um)
            st = np.asarray(sd["states"])
            cls = st[(st == STATE_ROUNDED) | (st == STATE_SPREAD)]
            if cls.size:
                if np.all(cls == STATE_ROUNDED):
                    out[cond]["rounded"].append(traj)
                elif np.all(cls == STATE_SPREAD):
                    out[cond]["spread"].append(traj)
        print(f"  [{i+1}/{len(paths)}] {label} ({cond})", flush=True)
    return out


def _axis_limit(data):
    """Symmetric square limit covering ~all tracks (99th pct of |coords|)."""
    pts = [t for c in CONDITIONS for t in data[c]["all"]]
    if not pts:
        return 50.0
    allc = np.concatenate(pts)
    lim = float(np.percentile(np.abs(allc), 99))
    return float(np.ceil(max(lim, 1.0) / 25.0) * 25.0)   # round up to 25 µm


def _plot_grid(data, key, title, out_path, lim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(13, 9))
    for ax, c in zip(axs.flat, CONDITIONS):
        tracks = data[c][key]
        col = _COND_COLOR.get(c, "#3a6ea5")
        for tr in tracks:
            ax.plot(tr[:, 0], tr[:, 1], lw=0.7, alpha=0.55, color=col)
            ax.plot(tr[-1, 0], tr[-1, 1], ".", ms=3.5, color=col, alpha=0.9)
        ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)
        ax.axvline(0, color="#cccccc", lw=0.6, zorder=0)
        ax.plot(0, 0, "+", color="k", ms=9, mew=1.4, zorder=4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{c}   (n={len(tracks)} cells)", fontsize=11)
        ax.grid(alpha=0.2)
    for ax in axs[:, 0]:
        ax.set_ylabel("Δy  (µm)")
    for ax in axs[1, :]:
        ax.set_xlabel("Δx  (µm)")
    fig.suptitle(f"{title}\n(origin-centred full tracks; axes ±{lim:.0f} µm, "
                 f"equal scale)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)


def main():
    um = DEFAULT_UM
    if "--um" in sys.argv:
        um = float(sys.argv[sys.argv.index("--um") + 1])
    print(f"Collecting tracks (um/px={um})…", flush=True)
    data = collect(um)
    lim = _axis_limit(data)
    os.makedirs(OUT_DIR, exist_ok=True)
    for key, title, fname in [
        ("all", "All cell tracks — full track (both states)",
         "flower_all.png"),
        ("rounded", "Cells ROUNDED for their entire track",
         "flower_rounded_only.png"),
        ("spread", "Cells SPREAD for their entire track",
         "flower_spread_only.png"),
    ]:
        n = sum(len(data[c][key]) for c in CONDITIONS)
        _plot_grid(data, key, title, os.path.join(OUT_DIR, fname), lim)
        print(f"  wrote {fname}  ({n} cells, axes ±{lim:.0f} µm)")
    print(f"\nWrote flower plots → {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
