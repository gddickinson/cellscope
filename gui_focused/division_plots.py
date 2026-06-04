"""Cell-division / lineage plots for the focused GUI analysis tab.

Driven entirely by the per-cell analysis results' `track_info`
(`parent_id`, `division_frame`, `division_score`) + `shape_timeseries`,
so they render whenever divisions were detected — inline by the
detection pipeline (hybrid_cpsam_multi / hybrid_dic) or by
`core.division_annotator` during Analyze. Registered as multi-cell
graphs in `gui_focused.analysis_plots.GRAPH_REGISTRY`.

`parent_id` is the 0-based index into the track list; daughter labels
are 1-based cell IDs, so a daughter's parent cell ID is `parent_id + 1`.
"""
import numpy as np


def _colors():
    try:
        from gui.mask_editor_multicell import CELL_COLORS
        return [tuple(c / 255.0 for c in rgb) for rgb in CELL_COLORS]
    except Exception:
        return ["C%d" % i for i in range(10)]


_COLORS = _colors()


def _ccolor(cell_id):
    return _COLORS[(int(cell_id) - 1) % len(_COLORS)]


def _lifespan(r):
    """(first_frame, last_frame) a cell is present, from the area
    timeseries when available (accurate), else from track_info."""
    ti = r.get("track_info", {}) or {}
    area = (r.get("shape_timeseries") or {}).get("area_um2")
    if area is not None:
        a = np.asarray(area, dtype=float)
        fin = np.where(np.isfinite(a))[0]
        if fin.size:
            return int(fin[0]), int(fin[-1])
    first = ti.get("first_frame")
    if first is None:
        return None
    return int(first), int(first) + int(ti.get("frames_tracked") or 1) - 1


def _events(results):
    """[(division_frame, parent_cell_id, daughter_cell_id, score), …]."""
    ev = []
    for r in results:
        ti = r.get("track_info", {}) or {}
        pid = ti.get("parent_id")
        if pid is None:
            continue
        ev.append((ti.get("division_frame"), int(pid) + 1,
                   r.get("cell_id"), ti.get("division_score")))
    return ev


def _no_data(fig, msg):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=12)
    ax.axis("off")


def plot_lineage_tree(fig, results, **_):
    """Per-cell lifelines with parent→daughter division connectors."""
    fig.clear()
    if not results:
        _no_data(fig, "No cells to plot")
        return
    order = sorted(results, key=lambda r: (
        (r.get("track_info", {}) or {}).get("first_frame") or 0,
        r.get("cell_id") or 0))
    ypos = {r.get("cell_id"): i for i, r in enumerate(order)}
    ax = fig.add_subplot(111)
    for r in order:
        cid = r.get("cell_id")
        span = _lifespan(r)
        if span is None:
            continue
        y = ypos[cid]
        ax.plot([span[0], span[1]], [y, y], "-", color=_ccolor(cid),
                lw=3, solid_capstyle="round", zorder=2)
        ax.text(span[0] - 0.5, y, f"C{cid} ", ha="right", va="center",
                fontsize=8)
    n_div = 0
    for dframe, pcell, dcell, score in _events(results):
        if pcell not in ypos or dcell not in ypos or dframe is None:
            continue
        n_div += 1
        yp, yd = ypos[pcell], ypos[dcell]
        ax.plot([dframe, dframe], [yp, yd], "k-", lw=1.0, alpha=0.7,
                zorder=3)
        ax.plot(dframe, yd, "k>", markersize=7, zorder=4)
        if isinstance(score, (int, float)) and np.isfinite(score):
            ax.text(dframe, (yp + yd) / 2, f" {score:.2f}", fontsize=7,
                    va="center", color="k")
    ax.set_yticks([])
    ax.set_xlabel("Frame")
    ax.set_title(f"Cell lineage — {n_div} division event(s)")
    ax.grid(alpha=0.3, axis="x")
    ax.margins(y=0.1)
    fig.tight_layout()


def plot_division_timeline(fig, results, **_):
    """Division events over time (score per event) + cumulative count."""
    fig.clear()
    ev = [e for e in _events(results) if e[0] is not None]
    if not ev:
        _no_data(fig, "No divisions detected in this recording")
        return
    ev.sort(key=lambda e: e[0])
    frames = np.array([e[0] for e in ev], dtype=float)
    scores = np.array([e[3] if isinstance(e[3], (int, float)) else np.nan
                       for e in ev], dtype=float)
    have_scores = np.isfinite(scores).any()
    stems = np.where(np.isfinite(scores), scores, 1.0)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.vlines(frames, 0, stems, color="#8e44ad", lw=1.5, zorder=1)
    ax1.scatter(frames, stems, s=40, color="#8e44ad",
                edgecolor="black", linewidth=0.6, zorder=2)
    for (dframe, pcell, dcell, _score), yv in zip(ev, stems):
        ax1.annotate(f"C{pcell}→C{dcell}", (dframe, yv),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=7)
    ax1.set_ylabel("Division score" if have_scores else "event")
    ax1.set_ylim(0, max(1.05, float(np.nanmax(stems)) * 1.25))
    ax1.set_title(f"Division events (n={len(ev)})")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    xs = np.sort(frames)
    ax2.step(np.concatenate([[xs[0]], xs]),
             np.arange(0, len(xs) + 1), where="post",
             color="#2980b9", lw=2)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Cumulative divisions")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
