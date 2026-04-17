"""Per-recording output: NPZ + JSON metrics + figures."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.contour import get_contour
from config import FIGURE_DPI


def write_recording_results(result, out_dir):
    """Write all per-recording outputs to `out_dir`.

    Files:
        masks.npz                — boolean masks
        metrics.json             — scalar + small array metrics
        boundary_overlay.png     — sample frames with contour overlays
        trajectory.png           — cell migration path
        edge_kymograph.png       — edge velocity heatmap
        shape_timeseries.png     — area, circularity, etc. over time
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Masks (compressed)
    np.savez_compressed(
        os.path.join(out_dir, "masks.npz"),
        masks=result["masks"].astype(np.uint8),
        centroids_px=result["centroids_px"],
    )

    # 2. Metrics JSON
    metrics = _build_metrics_dict(result)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 3. Figures
    _save_boundary_overlay(result, os.path.join(out_dir, "boundary_overlay.png"))
    _save_trajectory(result, os.path.join(out_dir, "trajectory.png"))
    _save_edge_kymograph(result, os.path.join(out_dir, "edge_kymograph.png"))
    _save_shape_timeseries(result, os.path.join(out_dir, "shape_timeseries.png"))

    return out_dir


def _build_metrics_dict(result):
    """Convert result dict to JSON-serializable metrics."""
    es = result["edge_summary"]
    stab = result["area_stability"]
    cfg = result.get("refinement_config", {})
    return {
        "name": result["name"],
        "video_path": result.get("video_path", ""),
        "n_frames": result["n_frames"],
        "um_per_px": result["um_per_px"],
        "time_interval_min": result["time_interval_min"],

        "mean_speed_um_min": result["mean_speed"],
        "total_distance_um": result["total_distance"],
        "net_displacement_um": result["net_displacement"],
        "persistence": result["persistence"],

        "mean_area_um2": stab["mean_area_um2"],
        "area_cv": stab["area_cv"],
        "max_min_area_ratio": stab["max_min_ratio"],
        "n_large_jumps": stab["n_large_jumps"],
        "mean_consec_iou": stab["mean_consec_iou"],

        "mean_boundary_confidence": result["mean_boundary_confidence"],

        "mean_protrusion_velocity_um_min": es["mean_protrusion_velocity"],
        "mean_retraction_velocity_um_min": es["mean_retraction_velocity"],
        "protrusion_fraction": es["protrusion_fraction"],
        "net_edge_velocity_um_min": es["net_velocity"],
        "max_protrusion_um_min": es["max_protrusion"],
        "max_retraction_um_min": es["max_retraction"],

        "shape_summary": result["shape_summary"],
        "refinement_config": cfg,
        "refinement_score_log": result.get("refinement_score_log", []),
        "mean_flow_quality": result.get("mean_flow_quality", float("nan")),
        "detection_elapsed_s": result.get("detection_elapsed_s", 0),
        "refinement_elapsed_s": result.get("refinement_elapsed_s", 0),
    }


def _save_boundary_overlay(result, path, n_panels=6):
    """Show contour overlay on representative frames."""
    frames = result["frames"]
    masks = result["masks"]
    n = len(frames)
    indices = np.linspace(0, n - 1, n_panels, dtype=int)
    cols = 3
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4),
                              dpi=FIGURE_DPI)
    axes = np.atleast_1d(axes).flatten()
    for ax, fi in zip(axes, indices):
        ax.imshow(frames[fi], cmap="gray")
        if masks[fi].any():
            c = get_contour(masks[fi])
            if c is not None:
                ax.plot(c[:, 1], c[:, 0], "-", color="#00ff00",
                        linewidth=1.5)
        t_min = fi * result["time_interval_min"]
        ax.set_title(f"frame {fi} (t={t_min:.0f} min)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(indices):]:
        ax.axis("off")
    fig.suptitle(f"{result['name']} — refined boundary", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_trajectory(result, path):
    """Trajectory + speed time series."""
    traj = result["trajectory"]
    speed = result["speed"]
    dt = result["time_interval_min"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=FIGURE_DPI)

    valid = ~np.isnan(traj[:, 0])
    axes[0].plot(traj[valid, 0], traj[valid, 1], "-o", color="#1f77b4",
                  markersize=2, linewidth=1)
    axes[0].plot(0, 0, "k*", markersize=12)
    axes[0].set_xlabel("X (μm)")
    axes[0].set_ylabel("Y (μm)")
    axes[0].set_title("Trajectory")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    t = np.arange(len(speed)) * dt
    axes[1].plot(t, speed, color="#1f77b4")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("Speed (μm/min)")
    axes[1].set_title(
        f"Speed (mean={result['mean_speed']:.3f} μm/min, "
        f"persistence={result['persistence']:.2f})"
    )
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(result["name"], fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_edge_kymograph(result, path):
    """Edge velocity heatmap."""
    angles = result["edge_angles"]
    vel = result["edge_velocity"]
    dt = result["time_interval_min"]
    if vel.size == 0 or np.all(np.isnan(vel)):
        return

    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIGURE_DPI)
    angle_deg = np.degrees(angles)
    t = np.arange(vel.shape[0]) * dt
    vmax = np.nanpercentile(np.abs(vel), 95)
    im = ax.imshow(
        vel.T, aspect="auto", origin="lower",
        extent=[t[0], t[-1], angle_deg[0], angle_deg[-1]],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Angular sector (°)")
    es = result["edge_summary"]
    ax.set_title(
        f"{result['name']} — edge velocity kymograph "
        f"(prot={es['mean_protrusion_velocity']:.2f}, "
        f"retr={es['mean_retraction_velocity']:.2f} μm/min)"
    )
    fig.colorbar(im, ax=ax, label="Radial vel (μm/min)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_shape_timeseries(result, path):
    """Per-frame shape descriptor traces."""
    ts = result["shape_timeseries"]
    dt = result["time_interval_min"]
    keys = ["area_um2", "perimeter_um", "circularity",
            "solidity", "aspect_ratio", "eccentricity"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=FIGURE_DPI)
    axes = axes.flatten()
    for ax, k in zip(axes, keys):
        v = ts[k]
        t = np.arange(len(v)) * dt
        ax.plot(t, v, color="#1f77b4")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(k)
        ax.set_title(k)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{result['name']} — shape descriptors", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
