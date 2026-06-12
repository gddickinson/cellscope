"""Flower (origin-centred track) plots + per-cell motility, by condition.

Each cell's FULL trajectory (every tracked frame, both rounded and spread
states) is translated so its first position sits at the origin. Three cell
groupings are formed per condition:
  all       — every cell
  rounded   — cells ROUNDED for their whole (classifiable) track
  spread    — cells SPREAD for their whole (classifiable) track
(single-state membership is judged over classifiable frames; `unknown` +
edge-truncated frames don't count — but the tracks/motility use the full
path.)

Outputs (compare/flower_plots/):
  flower_<all|rounded_only|spread_only>.png   origin-centred tracks, one
      panel per condition, shared equal x/y axis
  speed_<all|rounded|spread>.png              per-cell mean speed (µm/min)
  distance_<all|rounded|spread>.png           per-cell total path length (µm)
  netdisp_<all|rounded|spread>.png            per-cell net displacement (µm)
The motility plots are box + per-cell strip by condition (shared y-break).

Usage:
  conda run -n cellpose4 python scripts/ic295_flower_plots.py
  conda run -n cellpose4 python scripts/ic295_flower_plots.py --um 0.4 --dt 10
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    RECORDINGS_ROOT, COMPARE_DIR, CONDITIONS, parse_condition)
from scripts.ic295_plot_utils import apply_ybreak  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = os.path.join(COMPARE_DIR, "flower_plots")
DEFAULT_UM = 0.6523                      # IC295 scope (single magnification)
DEFAULT_DT = 10.0                        # min / frame
_SPEED_CAP = 15.0                        # µm/min — drop tracking glitches
_COND_COLOR = {"WT": "#1f77b4", "KO": "#d62728", "GOF": "#2ca02c",
               "Y1": "#9467bd", "OT": "#ff7f0e", "DMSO": "#7f7f7f"}
_GROUP_TITLE = {"all": "all cells",
                "rounded": "whole-track-rounded cells",
                "spread": "whole-track-spread cells"}


def collect(um, dt):
    """{condition: {'all'|'rounded'|'spread': [ {traj,speed,distance,netdisp} ]}}."""
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
            # per-step distances over CONSECUTIVE present frames (gaps→NaN)
            seg = np.linalg.norm(np.diff(cents, axis=0), axis=1) * um
            seg = seg[np.isfinite(seg)]
            spd = seg / dt
            spd = spd[spd <= _SPEED_CAP]
            rec = {
                "traj": (cv - cv[0]) * um,            # origin-centred µm
                "speed": float(np.mean(spd)) if spd.size else float("nan"),
                "distance": float(np.sum(seg)),       # path length µm
                "netdisp": float(np.linalg.norm(cv[-1] - cv[0]) * um),
            }
            out[cond]["all"].append(rec)
            sd = classify_track_states(stack, um_per_px=um)
            st = np.asarray(sd["states"])
            cls = st[(st == STATE_ROUNDED) | (st == STATE_SPREAD)]
            if cls.size:
                if np.all(cls == STATE_ROUNDED):
                    out[cond]["rounded"].append(rec)
                elif np.all(cls == STATE_SPREAD):
                    out[cond]["spread"].append(rec)
        print(f"  [{i+1}/{len(paths)}] {label} ({cond})", flush=True)
    return out


def _axis_limit(data):
    """Symmetric square limit covering ~all tracks (99th pct of |coords|)."""
    pts = [r["traj"] for c in CONDITIONS for r in data[c]["all"]]
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
        tracks = [r["traj"] for r in data[c][key]]
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


def _plot_by_cond(data, key, mkey, ylabel, title, out_path):
    """Box + per-cell strip of a per-cell metric by condition (y-break)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [[r[mkey] for r in data[c][key]
               if r[mkey] == r[mkey]] for c in CONDITIONS]   # drop NaN
    pos = list(range(1, len(CONDITIONS) + 1))
    rng = np.random.default_rng(0)
    jit = [rng.normal(p, 0.06, size=len(g)) for p, g in zip(pos, groups)]

    def draw(ax):
        bp = ax.boxplot(groups, positions=pos, widths=0.6, showfliers=False,
                        patch_artist=True)
        for patch in bp["boxes"]:
            patch.set(facecolor="#cce4ff", edgecolor="#446")
        for jx, g in zip(jit, groups):
            ax.scatter(jx, g, s=20, color="#234", alpha=0.7, zorder=3)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{c}\n(n={len(g)})"
                            for c, g in zip(CONDITIONS, groups)])

    fig = plt.figure(figsize=(7.5, 4.8))
    apply_ybreak(fig, draw, [v for g in groups for v in g],
                 ylabel=ylabel, xlabel="Condition", title=title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


# (metric key, y-axis label, file stem)
_MOTILITY = [
    ("speed",    "mean speed  (µm/min)",        "speed"),
    ("distance", "total path length  (µm)",     "distance"),
    ("netdisp",  "net displacement  (µm)",      "netdisp"),
]


def main():
    um, dt = DEFAULT_UM, DEFAULT_DT
    if "--um" in sys.argv:
        um = float(sys.argv[sys.argv.index("--um") + 1])
    if "--dt" in sys.argv:
        dt = float(sys.argv[sys.argv.index("--dt") + 1])
    print(f"Collecting tracks (um/px={um}, dt={dt} min)…", flush=True)
    data = collect(um, dt)
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
        print(f"  flower {fname}  ({n} cells, axes ±{lim:.0f} µm)")

    for key in ("all", "rounded", "spread"):
        for mkey, ylabel, stem in _MOTILITY:
            title = (f"{ylabel.split('  (')[0]} — {_GROUP_TITLE[key]} "
                     f"(full track)")
            _plot_by_cond(data, key, mkey, ylabel, title,
                          os.path.join(OUT_DIR, f"{stem}_{key}.png"))
        print(f"  motility plots ({key}): speed / distance / netdisp")

    print(f"\nWrote flower + motility plots → {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
