"""Shared plotting helper: automatic broken y-axis for the IC295 plots.

`apply_ybreak(fig, draw, values, ...)` renders grouped data via a
`draw(ax)` callback. When `values` contain high outliers that would squish
the bulk of the data near zero, it splits the y-axis into a broken layout —
a small top panel for the outlier range and a large bottom panel for the
main range, with diagonal break marks. When there are no such outliers it
falls back to a single axes (identical to the un-broken plot), so only the
outlier-heavy plots change.

`fig` may be a Figure or a SubFigure (so multi-panel plots can break each
panel independently). `draw(ax)` must plot the data artists + set the
x-ticks; it should NOT set ylabel/xlabel/title (the helper does, on the
correct panel). `headroom` reserves space above the bulk for annotations
(e.g. significance brackets) in the broken bottom panel.
"""
from __future__ import annotations

import numpy as np


def _decide_break(v):
    """Return (bulk_top, base, hi_bottom, vmax) if a break looks warranted."""
    v = v[np.isfinite(v)]
    if v.size < 5:
        return None
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return None
    fence = q3 + 1.5 * iqr
    out = v[v > fence]
    inl = v[v <= fence]
    if out.size == 0 or inl.size == 0:
        return None
    if out.size > 0.25 * v.size:          # too many "outliers" → no clean break
        return None
    bulk_top = float(inl.max())
    base = min(0.0, float(v.min()))
    vmax = float(v.max())
    # Only break when the outliers genuinely SQUISH the bulk: the inlier
    # range must occupy < half the full range (else the data isn't
    # squished near zero and a break just adds clutter).
    if bulk_top - base >= 0.5 * (vmax - base):
        return None
    if float(out.min()) <= bulk_top + 0.15 * (bulk_top - base):   # no real gap
        return None
    hi_bot = float(out.min()) - 0.12 * (float(out.min()) - bulk_top)
    return bulk_top, base, hi_bot, vmax


def _break_marks(ax_hi, ax_lo):
    d = 0.5
    kw = dict(marker=[(-1, -d), (1, d)], markersize=9, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False)
    ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **kw)
    ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **kw)


def _finish_single(ax, draw, ylabel, xlabel, title, grid):
    draw(ax)
    if grid:
        ax.grid(axis="y", alpha=0.3)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=10)
    return [ax]


def apply_ybreak(fig, draw, values, *, ylabel=None, xlabel=None, title=None,
                 grid=True, height_ratio=3.2, headroom=0.10):
    """Draw via `draw(ax)`, breaking the y-axis when `values` have high
    outliers. Returns the axes used (bottom panel first)."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    brk = _decide_break(v) if v.size else None
    if brk is None:
        return _finish_single(fig.add_subplot(111), draw, ylabel, xlabel,
                              title, grid)
    bulk_top, base, hi_bot, vmax = brk
    low_top = bulk_top + max(headroom, 0.08) * (bulk_top - base)
    if hi_bot <= low_top:                 # gap closed by the headroom → no break
        return _finish_single(fig.add_subplot(111), draw, ylabel, xlabel,
                              title, grid)

    gs = fig.add_gridspec(2, 1, height_ratios=[1, height_ratio], hspace=0.07)
    ax_hi = fig.add_subplot(gs[0, 0])
    ax_lo = fig.add_subplot(gs[1, 0], sharex=ax_hi)
    draw(ax_hi)
    draw(ax_lo)
    ax_lo.set_ylim(base, low_top)
    ax_hi.set_ylim(hi_bot, vmax * 1.05)
    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(labelbottom=False, bottom=False)
    if ax_hi.get_legend():
        ax_hi.get_legend().remove()
    if grid:
        ax_hi.grid(axis="y", alpha=0.3)
        ax_lo.grid(axis="y", alpha=0.3)
    _break_marks(ax_hi, ax_lo)
    if ylabel:
        ax_lo.set_ylabel(ylabel)
        ax_lo.yaxis.set_label_coords(-0.10, 0.62)
    if xlabel:
        ax_lo.set_xlabel(xlabel)
    if title:
        ax_hi.set_title(title, fontsize=10)
    return [ax_lo, ax_hi]
