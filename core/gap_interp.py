"""Linear interpolation of short gaps in analysis timeseries.

When a cell briefly disappears from a track (≤ a few frames) the
speed/area/etc. arrays have NaN runs that visually break the line
plot. This module supplies two helpers:

* ``interpolate_short_gaps(arr, max_gap)`` — fill NaN runs of length
  ≤ ``max_gap`` in a 1-D array; long gaps and edge NaNs are kept as
  NaN so we never invent data at the start/end of a track.
* ``plot_with_gaps(ax, y, was_interpolated, ...)`` — draw the real
  samples solid and the interpolated samples dotted in the same
  colour, so users can always tell which points are measured.

We keep these two pieces separate so the same fill logic can be used
by exporters/CSV writers in future without dragging matplotlib in.
"""
from __future__ import annotations

import numpy as np


def interpolate_short_gaps(arr, max_gap=0):
    """Linearly interpolate NaN runs of length ≤ ``max_gap`` in a 1D array.

    ``max_gap=0`` disables interpolation. Edge NaNs (no valid neighbour
    on one side) and runs longer than ``max_gap`` are left untouched.

    Returns ``(filled, was_interpolated)``:
      * ``filled`` — copy of the input with short gaps filled.
      * ``was_interpolated`` — bool array, True where the value was
        synthesised by this function (i.e. should be displayed dotted).
    """
    arr = np.asarray(arr, dtype=float)
    interpolated = np.zeros(arr.shape, dtype=bool)
    if max_gap <= 0 or arr.ndim != 1 or len(arr) < 3:
        return arr.copy(), interpolated

    is_nan = np.isnan(arr)
    if not is_nan.any():
        return arr.copy(), interpolated

    out = arr.copy()
    n = len(arr)
    i = 0
    while i < n:
        if not is_nan[i]:
            i += 1
            continue
        j = i
        while j < n and is_nan[j]:
            j += 1
        gap_len = j - i
        if gap_len <= max_gap and i > 0 and j < n:
            x0, x1 = i - 1, j
            y0, y1 = arr[x0], arr[x1]
            for k in range(i, j):
                t = (k - x0) / (x1 - x0)
                out[k] = y0 + t * (y1 - y0)
                interpolated[k] = True
        i = j
    return out, interpolated


def plot_with_gaps(ax, y, was_interpolated=None, x=None, **kwargs):
    """Plot ``y`` on ``ax``; draw interpolated samples dotted.

    ``kwargs`` are passed to ``ax.plot`` for the solid (real) trace.
    The dotted trace inherits its colour and uses linestyle ``":"``
    so the user always sees which points are measured vs synthesised.

    If ``was_interpolated`` is None or all-False this collapses to a
    single ``ax.plot(x, y, **kwargs)`` call.
    """
    y = np.asarray(y, dtype=float)
    if x is None:
        x = np.arange(len(y))

    if was_interpolated is None or not np.any(was_interpolated):
        return ax.plot(x, y, **kwargs)

    real = ~was_interpolated & ~np.isnan(y)
    real_y = np.where(real, y, np.nan)
    line = ax.plot(x, real_y, **kwargs)
    color = line[0].get_color()

    # Extend the dashed mask by one sample on each side so the dotted
    # segment visibly connects to the solid trace at the endpoints.
    extended = was_interpolated.copy()
    idx = np.where(was_interpolated)[0]
    for i in idx:
        if i - 1 >= 0:
            extended[i - 1] = True
        if i + 1 < len(extended):
            extended[i + 1] = True
    dashed_y = np.where(extended, y, np.nan)

    interp_kwargs = dict(kwargs)
    interp_kwargs.pop("label", None)
    interp_kwargs["linestyle"] = ":"
    interp_kwargs["color"] = color
    if "lw" in interp_kwargs:
        interp_kwargs["lw"] = max(0.5, interp_kwargs["lw"] * 0.9)
    ax.plot(x, dashed_y, **interp_kwargs)
    return line
