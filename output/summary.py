"""Batch summary CSVs: per-recording rows + per-group aggregation."""
import csv
import os
import math
import numpy as np


SCALAR_COLUMNS = [
    "group", "name", "n_frames",
    "mean_speed_um_min", "total_distance_um", "net_displacement_um",
    "persistence",
    "mean_area_um2", "area_cv", "n_large_jumps", "mean_consec_iou",
    "mean_boundary_confidence",
    "mean_protrusion_velocity_um_min", "mean_retraction_velocity_um_min",
    "protrusion_fraction", "net_edge_velocity_um_min",
    "max_protrusion_um_min", "max_retraction_um_min",
    "refinement_chosen",
]


def _row_from_metrics(group, metrics):
    cfg = metrics.get("refinement_config", {}) or {}
    chosen = cfg.get("name", "?")
    row = {
        "group": group,
        "name": metrics["name"],
        "n_frames": metrics["n_frames"],
        "mean_speed_um_min": metrics["mean_speed_um_min"],
        "total_distance_um": metrics["total_distance_um"],
        "net_displacement_um": metrics["net_displacement_um"],
        "persistence": metrics["persistence"],
        "mean_area_um2": metrics["mean_area_um2"],
        "area_cv": metrics["area_cv"],
        "n_large_jumps": metrics["n_large_jumps"],
        "mean_consec_iou": metrics["mean_consec_iou"],
        "mean_boundary_confidence": metrics["mean_boundary_confidence"],
        "mean_protrusion_velocity_um_min":
            metrics["mean_protrusion_velocity_um_min"],
        "mean_retraction_velocity_um_min":
            metrics["mean_retraction_velocity_um_min"],
        "protrusion_fraction": metrics["protrusion_fraction"],
        "net_edge_velocity_um_min": metrics["net_edge_velocity_um_min"],
        "max_protrusion_um_min": metrics["max_protrusion_um_min"],
        "max_retraction_um_min": metrics["max_retraction_um_min"],
        "refinement_chosen": chosen,
    }
    return row


def write_batch_summary(rows, out_path):
    """Write a per-recording CSV with one row per recording.

    Args:
        rows: list of (group_name, metrics_dict) tuples
        out_path: output CSV path
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCALAR_COLUMNS)
        w.writeheader()
        for group, metrics in rows:
            w.writerow(_row_from_metrics(group, metrics))
    return out_path


def write_group_summary(rows, out_path):
    """Write per-group aggregated CSV (mean, std, sem, n).

    Args:
        rows: list of (group_name, metrics_dict)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    by_group = {}
    for group, metrics in rows:
        by_group.setdefault(group, []).append(_row_from_metrics(group, metrics))

    numeric_cols = [c for c in SCALAR_COLUMNS
                    if c not in ("group", "name", "refinement_chosen", "n_frames")]
    fieldnames = ["group", "n_recordings"]
    for col in numeric_cols:
        fieldnames += [f"{col}_mean", f"{col}_std", f"{col}_sem"]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for group, group_rows in sorted(by_group.items()):
            n = len(group_rows)
            row = {"group": group, "n_recordings": n}
            for col in numeric_cols:
                vals = [r[col] for r in group_rows
                        if r[col] is not None and not (
                            isinstance(r[col], float) and math.isnan(r[col]))]
                if vals:
                    arr = np.array(vals, dtype=float)
                    row[f"{col}_mean"] = float(np.mean(arr))
                    row[f"{col}_std"] = float(np.std(arr, ddof=1)) if n > 1 else 0.0
                    row[f"{col}_sem"] = float(np.std(arr, ddof=1) / np.sqrt(n)) \
                        if n > 1 else 0.0
                else:
                    row[f"{col}_mean"] = ""
                    row[f"{col}_std"] = ""
                    row[f"{col}_sem"] = ""
            w.writerow(row)
    return out_path


def write_all_summaries(group_metrics, results_dir):
    """Write all summary files for a batch run.

    Args:
        group_metrics: list of (group_name, metrics_dict) tuples
        results_dir: top-level results directory

    Returns:
        dict of {summary_name: path}
    """
    out = {}
    out["batch"] = write_batch_summary(
        group_metrics, os.path.join(results_dir, "_batch_summary.csv"),
    )
    out["group"] = write_group_summary(
        group_metrics, os.path.join(results_dir, "_group_summary.csv"),
    )

    # Also write per-group summaries inside each group folder
    by_group = {}
    for grp, m in group_metrics:
        by_group.setdefault(grp, []).append((grp, m))
    for grp, gms in by_group.items():
        write_batch_summary(
            gms, os.path.join(results_dir, grp, "_group_summary.csv"),
        )

    return out
