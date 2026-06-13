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
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    RECORDINGS_ROOT, COMPARE_DIR, CONDITIONS, parse_condition)
from scripts.ic295_plot_utils import apply_ybreak  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = os.path.join(COMPARE_DIR, "flower_plots")
_CACHE = os.path.join(OUT_DIR, "_track_cache.pkl")   # per-cell tracks (re-plot fast)
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
            sd = classify_track_states(stack, um_per_px=um)
            ar = np.asarray(sd["metrics"]["area"], dtype=float)   # px², per frame
            rec = {
                "traj": (cv - cv[0]) * um,            # origin-centred µm
                "cents": cents * um,                  # full (N,2) µm, NaN absent
                "speed": float(np.mean(spd)) if spd.size else float("nan"),
                "distance": float(np.sum(seg)),       # path length µm
                "netdisp": float(np.linalg.norm(cv[-1] - cv[0]) * um),
                "area": (float(np.nanmean(ar)) * um * um   # mean footprint µm²
                         if np.isfinite(ar).any() else float("nan")),
            }
            out[cond]["all"].append(rec)
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
    """Symmetric square limit covering ALL tracks (the longest one fully
    fits), shared by every panel so conditions are directly comparable."""
    pts = [r["traj"] for c in CONDITIONS for r in data[c]["all"]]
    if not pts:
        return 50.0
    lim = float(np.max(np.abs(np.concatenate(pts))))     # farthest point
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


def _area_speed_xy(data, key):
    xy = {c: [(r["area"], r["speed"]) for r in data[c][key]
              if r["area"] == r["area"] and r["speed"] == r["speed"]]
          for c in CONDITIONS}
    allx = [a for c in CONDITIONS for a, _ in xy[c]]
    ally = [s for c in CONDITIONS for _, s in xy[c]]
    xlim = float(np.percentile(allx, 99)) * 1.05 if allx else 1.0
    ylim = float(np.percentile(ally, 99)) * 1.05 if ally else 1.0
    return xy, xlim, ylim


def _plot_area_speed_facet(data, key, out_path):
    """area (µm²) vs mean speed (µm/min), one panel per treatment, equal axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy, xlim, ylim = _area_speed_xy(data, key)
    fig, axs = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for ax, c in zip(axs.flat, CONDITIONS):
        pts = xy[c]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=24, color=_COND_COLOR.get(c), alpha=0.75,
                       edgecolor="white", linewidth=0.3)
        ax.set_xlim(0, xlim); ax.set_ylim(0, ylim)
        ax.set_title(f"{c}   (n={len(pts)})", fontsize=11)
        ax.grid(alpha=0.3)
    for ax in axs[:, 0]:
        ax.set_ylabel("mean speed  (µm/min)")
    for ax in axs[1, :]:
        ax.set_xlabel("mean area  (µm²)")
    fig.suptitle("Cell area vs mean speed, by treatment "
                 "(full track, all cells; equal axes)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)


def _plot_area_speed_overlay(data, key, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy, xlim, ylim = _area_speed_xy(data, key)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in CONDITIONS:
        pts = xy[c]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=28, color=_COND_COLOR.get(c), alpha=0.7,
                   edgecolor="white", linewidth=0.3, label=f"{c} (n={len(pts)})")
    ax.set_xlim(0, xlim); ax.set_ylim(0, ylim)
    ax.set_xlabel("mean area  (µm²)"); ax.set_ylabel("mean speed  (µm/min)")
    ax.set_title("Cell area vs mean speed (full track, all cells)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)


_MSD_MAXLAG = 60          # frames (× dt = max lag time shown); more bins
_MSD_MIN_CELLS = 5        # need ≥ this many full-duration cells per condition

# Experimental arms (control listed first) — MSD is split this way so each
# panel compares only conditions that share a control (see ic295_compare_arms).
_MSD_ARMS = [
    ("genetic", "Genetic arm (WT, GOF, KO)", ["WT", "GOF", "KO"]),
    ("drug",    "Drug arm (DMSO, Y1, OT)",   ["DMSO", "Y1", "OT"]),
]


def _full_track(cents):
    """True if the cell is tracked the ENTIRE recording (present at the
    first + last frame and in ≥ 90% of frames)."""
    v = np.isfinite(cents).all(axis=1)
    return bool(v.size and v[0] and v[-1] and v.mean() >= 0.9)


def _cell_msd(cents, maxlag):
    """Per-cell MSD(τ) µm² for τ=0..maxlag from a full (N,2) track (NaN gaps)."""
    n = len(cents)
    msd = np.full(maxlag + 1, np.nan)
    for tau in range(1, min(maxlag, n - 1) + 1):
        dr = cents[tau:] - cents[:-tau]
        sq = np.sum(dr * dr, axis=1)                 # NaN where either absent
        v = np.isfinite(sq)
        if v.any():
            msd[tau] = float(np.mean(sq[v]))
    return msd


def _plot_msd(data, key, out_path, dt, maxlag=_MSD_MAXLAG, logscale=False,
              conds=CONDITIONS, arm_label=None):
    """Ensemble mean MSD ± SEM vs lag time, one curve per treatment.

    `conds` restricts which treatments are drawn (e.g. one experimental
    arm); `arm_label` is prepended to the title when set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lag = np.arange(maxlag + 1) * dt                 # min
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in conds:
        # only cells tracked the ENTIRE recording → fixed cohort, defined
        # at every lag, no jump-as-cells-drop-in/out artefact.
        recs = [r for r in data[c][key] if _full_track(r["cents"])]
        if not recs:
            continue
        arr = np.vstack([_cell_msd(r["cents"], maxlag) for r in recs])
        nc = arr.shape[0]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(nc) if nc else mean * np.nan
        sl = slice(1, maxlag + 1)
        ax.errorbar(lag[sl], mean[sl], yerr=sem[sl],
                    color=_COND_COLOR.get(c), lw=1.8, capsize=3, errorevery=2,
                    label=f"{c} (n={nc})")
    if logscale:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("lag time τ  (min)")
    ax.set_ylabel("mean MSD  (µm²)")
    if arm_label:
        ttl = f"{arm_label}\nmean MSD ± SEM (full-recording cells)"
    else:
        ttl = ("Ensemble mean MSD ± SEM, by treatment "
               "(cells tracked the full recording)")
    ax.set_title(ttl + ("  [log-log]" if logscale else ""), fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)


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
    os.makedirs(OUT_DIR, exist_ok=True)
    if "--from-cache" in sys.argv and os.path.exists(_CACHE):
        print(f"Loading cached tracks from {_CACHE}…", flush=True)
        with open(_CACHE, "rb") as f:
            data = pickle.load(f)
    else:
        print(f"Collecting tracks (um/px={um}, dt={dt} min)…", flush=True)
        data = collect(um, dt)
        with open(_CACHE, "wb") as f:
            pickle.dump(data, f)
        print(f"  cached tracks → {_CACHE}  (re-plot fast with --from-cache)")
    lim = _axis_limit(data)

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

    _plot_area_speed_facet(data, "all",
                           os.path.join(OUT_DIR, "area_vs_speed_by_treatment.png"))
    _plot_area_speed_overlay(data, "all",
                             os.path.join(OUT_DIR, "area_vs_speed_overlay.png"))
    print("  area-vs-speed plots (by treatment + overlay)")

    _plot_msd(data, "all", os.path.join(OUT_DIR, "msd_by_treatment.png"), dt)
    _plot_msd(data, "all", os.path.join(OUT_DIR, "msd_by_treatment_loglog.png"),
              dt, logscale=True)
    print("  MSD plots (linear + log-log)")

    for arm, label, conds in _MSD_ARMS:
        _plot_msd(data, "all", os.path.join(OUT_DIR, f"msd_{arm}.png"), dt,
                  conds=conds, arm_label=label)
        _plot_msd(data, "all", os.path.join(OUT_DIR, f"msd_{arm}_loglog.png"),
                  dt, logscale=True, conds=conds, arm_label=label)
        print(f"  MSD plots ({arm}: {', '.join(conds)}) — linear + log-log")

    print(f"\nWrote flower + motility + area-vs-speed + MSD plots → {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
