"""Plot functions for the focused GUI analysis tab.

Each function takes a matplotlib Figure + result dict (or list for
multi-cell), clears the figure, and renders one plot type.

Timeseries plots accept an optional ``gap_interp_max`` kwarg
(default 0 = off). When > 0, NaN runs of that length or shorter
are linearly interpolated and shown dotted so the line doesn't
visually break for brief detection misses. See ``core/gap_interp.py``.
Plot functions that don't render timeseries accept and ignore the
kwarg via ``**_`` so the dispatcher can pass it unconditionally.
"""
import numpy as np
from matplotlib.figure import Figure

from core.gap_interp import interpolate_short_gaps, plot_with_gaps

try:
    from gui.mask_editor_multicell import CELL_COLORS
    _COLORS = [tuple(c / 255.0 for c in rgb) for rgb in CELL_COLORS]
except ImportError:
    _COLORS = ["C0", "C1", "C2", "C3", "C4", "C5"]


def _cell_color(idx):
    return _COLORS[idx % len(_COLORS)]


def plot_trajectory(fig: Figure, result: dict, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    traj = result.get("trajectory")
    if traj is None:
        ax.text(0.5, 0.5, "No trajectory data", ha="center", va="center",
                transform=ax.transAxes)
        return
    valid = ~np.isnan(traj[:, 0])
    t = np.arange(len(traj))
    sc = ax.scatter(traj[valid, 1], traj[valid, 0], c=t[valid],
                    cmap="viridis", s=8, zorder=2)
    ax.plot(traj[valid, 1], traj[valid, 0], "k-", alpha=0.3, lw=0.5)
    fig.colorbar(sc, ax=ax, label="Frame")
    ax.set_xlabel("X (um)"); ax.set_ylabel("Y (um)")
    ax.set_title("Cell Trajectory")
    ax.set_aspect("equal")
    fig.tight_layout()


def plot_speed(fig: Figure, result: dict, gap_interp_max=0, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    speed = result.get("speed")
    if speed is None:
        ax.text(0.5, 0.5, "No speed data", ha="center", va="center",
                transform=ax.transAxes)
        return
    speed = np.asarray(speed, dtype=float)
    filled, was_interp = interpolate_short_gaps(speed, gap_interp_max)
    plot_with_gaps(ax, filled, was_interp,
                   color="steelblue", lw=0.8)
    valid = speed[~np.isnan(speed)]
    if len(valid):
        ax.axhline(valid.mean(), color="orange", ls="--",
                   label=f"mean {valid.mean():.2f}")
        if was_interp.any():
            ax.plot([], [], color="steelblue", linestyle=":", lw=0.8,
                    label=f"interp ≤{gap_interp_max} frame")
        ax.legend()
    ax.set_xlabel("Frame"); ax.set_ylabel("Speed (um/min)")
    ax.set_title("Instantaneous Speed")
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_msd(fig: Figure, result: dict, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    traj = result.get("trajectory")
    if traj is None:
        ax.text(0.5, 0.5, "No MSD data", ha="center", va="center",
                transform=ax.transAxes)
        return
    from core.tracking import mean_squared_displacement
    um = result.get("um_per_px", 1.0)
    dt = result.get("time_interval_min", 1.0)
    cents = result.get("centroids_px")
    if cents is None:
        return
    cents_um = cents * um
    lags, msd, sem = mean_squared_displacement(cents_um)
    if len(lags) == 0:
        return
    t_lags = lags * dt
    ax.errorbar(t_lags, msd, yerr=sem, fmt="o-", ms=3, capsize=2)
    ax.set_xlabel("Lag (min)"); ax.set_ylabel("MSD (um^2)")
    ax.set_title("Mean Squared Displacement")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()


def plot_direction_autocorrelation(fig: Figure, result: dict, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    cents = result.get("centroids_px")
    if cents is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return
    from core.tracking import direction_autocorrelation
    um = result.get("um_per_px", 1.0)
    dt = result.get("time_interval_min", 1.0)
    lags, ac, sem = direction_autocorrelation(cents * um)
    if len(lags) == 0:
        return
    ax.errorbar(lags * dt, ac, yerr=sem, fmt="o-", ms=3, capsize=2)
    ax.axhline(0, color="gray", ls="--")
    ax.set_xlabel("Lag (min)"); ax.set_ylabel("Direction autocorrelation")
    ax.set_title("Direction Persistence")
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_area(fig: Figure, result: dict, gap_interp_max=0, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    ts = result.get("shape_timeseries", {})
    area = ts.get("area_um2")
    if area is None:
        ax.text(0.5, 0.5, "No area data", ha="center", va="center",
                transform=ax.transAxes)
        return
    area = np.asarray(area, dtype=float)
    filled, was_interp = interpolate_short_gaps(area, gap_interp_max)
    plot_with_gaps(ax, filled, was_interp,
                   color="steelblue", lw=0.8)
    valid = area[~np.isnan(area)]
    if len(valid):
        m, s = valid.mean(), valid.std()
        ax.axhline(m, color="orange", ls="--", label=f"mean {m:.0f}")
        ax.fill_between(range(len(area)),
                        np.full(len(area), m - s),
                        np.full(len(area), m + s),
                        alpha=0.15, color="orange")
        if was_interp.any():
            ax.plot([], [], color="steelblue", linestyle=":", lw=0.8,
                    label=f"interp ≤{gap_interp_max} frame")
        ax.legend()
    ax.set_xlabel("Frame"); ax.set_ylabel("Area (um^2)")
    ax.set_title("Cell Area Over Time")
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_shape_panel(fig: Figure, result: dict, gap_interp_max=0, **_):
    fig.clear()
    ts = result.get("shape_timeseries", {})
    keys = ["area_um2", "perimeter_um", "circularity",
            "solidity", "aspect_ratio", "eccentricity"]
    labels = ["Area (um^2)", "Perimeter (um)", "Circularity",
              "Solidity", "Aspect Ratio", "Eccentricity"]
    any_interp = False
    for i, (k, lab) in enumerate(zip(keys, labels)):
        ax = fig.add_subplot(2, 3, i + 1)
        arr = ts.get(k)
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            filled, was_interp = interpolate_short_gaps(
                arr, gap_interp_max)
            plot_with_gaps(ax, filled, was_interp, lw=0.7)
            any_interp = any_interp or bool(was_interp.any())
        ax.set_title(lab, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    if any_interp:
        fig.suptitle(f"(dotted = interp ≤{gap_interp_max} frame)",
                     fontsize=8, y=1.02)
    fig.tight_layout()


def plot_edge_kymograph(fig: Figure, result: dict, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    vel = result.get("edge_velocity")
    angles = result.get("edge_angles")
    if vel is None or angles is None:
        ax.text(0.5, 0.5, "No edge data", ha="center", va="center",
                transform=ax.transAxes)
        return
    vmax = max(abs(vel.min()), abs(vel.max()), 0.1)
    im = ax.imshow(vel.T, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=[0, vel.shape[0], 360, 0])
    fig.colorbar(im, ax=ax, label="Velocity (um/min)")
    ax.set_xlabel("Frame"); ax.set_ylabel("Angle (deg)")
    ax.set_title("Edge Velocity Kymograph")
    fig.tight_layout()


def plot_edge_summary_bar(fig: Figure, result: dict, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    es = result.get("edge_summary", {})
    if not es:
        ax.text(0.5, 0.5, "No edge summary", ha="center", va="center",
                transform=ax.transAxes)
        return
    keys = ["mean_protrusion_velocity", "mean_retraction_velocity",
            "net_velocity", "max_protrusion", "max_retraction"]
    labels = ["Protrusion", "Retraction", "Net", "Max prot.", "Max retr."]
    vals = [es.get(k, 0) for k in keys]
    colors = ["#2196F3", "#F44336", "#4CAF50", "#03A9F4", "#E91E63"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("um/min")
    ax.set_title("Edge Velocity Summary")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()


def plot_boundary_confidence(fig: Figure, result: dict,
                             gap_interp_max=0, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    bc = result.get("boundary_confidence_per_frame")
    if bc is None:
        mean_bc = result.get("mean_boundary_confidence")
        if mean_bc is not None:
            ax.text(0.5, 0.5, f"Mean boundary confidence: {mean_bc:.3f}",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
        return
    bc = np.asarray(bc, dtype=float)
    filled, was_interp = interpolate_short_gaps(bc, gap_interp_max)
    plot_with_gaps(ax, filled, was_interp, color="steelblue", lw=0.8)
    ax.axhline(np.nanmean(bc), color="orange", ls="--",
               label=f"mean {np.nanmean(bc):.3f}")
    if was_interp.any():
        ax.plot([], [], color="steelblue", linestyle=":", lw=0.8,
                label=f"interp ≤{gap_interp_max} frame")
    ax.legend()
    ax.set_xlabel("Frame"); ax.set_ylabel("Confidence")
    ax.set_title("Boundary Confidence per Frame")
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_consecutive_iou(fig: Figure, result: dict,
                         gap_interp_max=0, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    stab = result.get("area_stability", {})
    ciou = stab.get("mean_consec_iou")
    masks = result.get("masks")
    if masks is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return
    from core.evaluation import compute_iou
    ious = np.array([compute_iou(masks[i], masks[i+1])
                     for i in range(len(masks)-1)], dtype=float)
    # Empty masks → IoU 0; treat those as missing for interp purposes
    if gap_interp_max > 0:
        empty_pair = np.array([
            (not masks[i].any()) or (not masks[i+1].any())
            for i in range(len(masks)-1)])
        ious_masked = np.where(empty_pair, np.nan, ious)
    else:
        ious_masked = ious
    filled, was_interp = interpolate_short_gaps(
        ious_masked, gap_interp_max)
    plot_with_gaps(ax, filled, was_interp, color="steelblue", lw=0.8)
    if len(ious):
        ax.axhline(np.nanmean(ious), color="orange", ls="--",
                   label=f"mean {np.nanmean(ious):.3f}")
        if was_interp.any():
            ax.plot([], [], color="steelblue", linestyle=":", lw=0.8,
                    label=f"interp ≤{gap_interp_max} frame")
        ax.legend()
    ax.set_xlabel("Frame pair"); ax.set_ylabel("IoU")
    ax.set_title("Consecutive Frame IoU")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()


# --- Multi-cell comparison plots ---

def plot_speed_comparison(fig: Figure, results: list,
                          gap_interp_max=0, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    any_interp = False
    for i, r in enumerate(results):
        speed = r.get("speed")
        if speed is not None:
            speed = np.asarray(speed, dtype=float)
            filled, was_interp = interpolate_short_gaps(
                speed, gap_interp_max)
            plot_with_gaps(ax, filled, was_interp,
                           color=_cell_color(i), lw=0.8, alpha=0.8,
                           label=f"Cell {r.get('cell_id', i+1)}")
            any_interp = any_interp or bool(was_interp.any())
    ax.set_xlabel("Frame"); ax.set_ylabel("Speed (um/min)")
    title = "Speed Comparison (all cells)"
    if any_interp:
        title += f"  (dotted = interp ≤{gap_interp_max} frame)"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_area_comparison(fig: Figure, results: list,
                         gap_interp_max=0, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    any_interp = False
    for i, r in enumerate(results):
        ts = r.get("shape_timeseries", {})
        area = ts.get("area_um2")
        if area is not None:
            area = np.asarray(area, dtype=float)
            filled, was_interp = interpolate_short_gaps(
                area, gap_interp_max)
            plot_with_gaps(ax, filled, was_interp,
                           color=_cell_color(i), lw=0.8, alpha=0.8,
                           label=f"Cell {r.get('cell_id', i+1)}")
            any_interp = any_interp or bool(was_interp.any())
    ax.set_xlabel("Frame"); ax.set_ylabel("Area (um^2)")
    title = "Area Comparison (all cells)"
    if any_interp:
        title += f"  (dotted = interp ≤{gap_interp_max} frame)"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_trajectory_comparison(fig: Figure, results: list, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    for i, r in enumerate(results):
        traj = r.get("trajectory")
        if traj is None:
            continue
        valid = ~np.isnan(traj[:, 0])
        ax.plot(traj[valid, 1], traj[valid, 0],
                color=_cell_color(i), lw=1.0, alpha=0.8,
                label=f"Cell {r.get('cell_id', i+1)}")
        if valid.any():
            ax.plot(traj[valid, 1][0], traj[valid, 0][0],
                    "*", color=_cell_color(i), ms=8)
    ax.set_xlabel("X (um)"); ax.set_ylabel("Y (um)")
    ax.set_title("Trajectory Comparison")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_cell_summary_table(fig: Figure, results: list, **_):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axis("off")
    headers = ["Cell", "Speed\n(um/min)", "Area\n(um^2)",
               "Persist.", "Frames", "Parent"]
    rows = []
    for r in results:
        cid = r.get("cell_id", "?")
        spd = f"{r.get('mean_speed', 0):.2f}"
        ss = r.get("shape_summary", {})
        area_d = ss.get("area_um2", {})
        area = f"{area_d.get('mean', 0):.0f}"
        pers = f"{r.get('persistence', 0):.3f}"
        ti = r.get("track_info", {})
        frames = str(ti.get("frames_tracked", "?"))
        parent = str(ti.get("parent_id") or "-")
        rows.append([str(cid), spd, area, pers, frames, parent])
    if rows:
        colors = [[_cell_color(i)] * len(headers) for i in range(len(rows))]
        table = ax.table(cellText=rows, colLabels=headers,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
        for i in range(len(rows)):
            for j in range(len(headers)):
                cell = table[i + 1, j]
                r, g, b = _cell_color(i)
                cell.set_facecolor((*_cell_color(i), 0.15))
    ax.set_title("Per-Cell Summary", fontsize=12, pad=20)
    fig.tight_layout()


def plot_msd_with_fit(fig: Figure, result: dict, **_):
    """MSD with diffusion model fit (D, alpha)."""
    fig.clear()
    ax = fig.add_subplot(111)
    cents = result.get("centroids_px")
    if cents is None:
        ax.text(0.5, 0.5, "No MSD data", ha="center", va="center",
                transform=ax.transAxes)
        return
    from core.tracking import mean_squared_displacement
    um = result.get("um_per_px", 1.0)
    dt = result.get("time_interval_min", 1.0)
    lags, msd, sem = mean_squared_displacement(cents * um)
    if len(lags) == 0:
        return
    t_lags = lags * dt
    ax.errorbar(t_lags, msd, yerr=sem, fmt="o", ms=3, capsize=2,
                label="Data")
    from core.advanced_analysis import fit_msd_diffusion
    fit = fit_msd_diffusion(lags, msd, dt)
    if fit["fit_lags"]:
        ax.plot(fit["fit_lags"], fit["fit_msd"], "r--", lw=1.5,
                label=f"Fit: D={fit['D']:.3f}, "
                      f"\u03b1={fit['alpha']:.2f} "
                      f"(R\u00b2={fit['r_squared']:.3f})")
    ax.set_xlabel("Lag (min)"); ax.set_ylabel("MSD (um\u00b2)")
    ax.set_title("MSD with Diffusion Fit")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()


def plot_quality_flags(fig: Figure, result: dict, **_):
    """Frame quality flagging — highlight suspect frames."""
    fig.clear()
    ax = fig.add_subplot(111)
    masks = result.get("masks")
    if masks is None:
        ax.text(0.5, 0.5, "No mask data", ha="center", va="center",
                transform=ax.transAxes)
        return
    from core.advanced_analysis import flag_quality_issues
    flags = flag_quality_issues(masks)
    changes = flags["area_changes"]
    ax.plot(changes, "steelblue", lw=0.8)
    ax.axhline(0.5, color="orange", ls="--", label="50% threshold")
    suspect = flags["suspect_frames"]
    if suspect:
        ax.scatter(np.array(suspect) - 1,
                   changes[np.array(suspect) - 1],
                   color="red", s=30, zorder=3,
                   label=f"{len(suspect)} suspect")
    ax.set_xlabel("Frame pair")
    ax.set_ylabel("Relative area change")
    ax.set_title(f"Frame Quality ({len(suspect)} suspect frames)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()


from gui_focused.vampire_plots import (  # noqa: E402
    plot_vampire_modes, plot_vampire_distribution,
    plot_vampire_timeseries, plot_vampire_eigenshapes,
)


GRAPH_REGISTRY = {
    "Trajectory": (plot_trajectory, False),
    "Speed vs Time": (plot_speed, False),
    "MSD": (plot_msd, False),
    "MSD with Diffusion Fit": (plot_msd_with_fit, False),
    "Direction Autocorrelation": (plot_direction_autocorrelation, False),
    "Area vs Time": (plot_area, False),
    "Shape Panel (6 metrics)": (plot_shape_panel, False),
    "Edge Kymograph": (plot_edge_kymograph, False),
    "Edge Summary Bar": (plot_edge_summary_bar, False),
    "Boundary Confidence": (plot_boundary_confidence, False),
    "Consecutive IoU": (plot_consecutive_iou, False),
    "Frame Quality": (plot_quality_flags, False),
    "Speed Comparison (all cells)": (plot_speed_comparison, True),
    "Area Comparison (all cells)": (plot_area_comparison, True),
    "Trajectory Comparison (all cells)": (plot_trajectory_comparison, True),
    "Cell Summary Table": (plot_cell_summary_table, True),
    "VAMPIRE Shape Modes": (plot_vampire_modes, False),
    "VAMPIRE Mode Distribution": (plot_vampire_distribution, False),
    "VAMPIRE Mode Over Time": (plot_vampire_timeseries, False),
    "VAMPIRE Eigenshapes": (plot_vampire_eigenshapes, False),
}
