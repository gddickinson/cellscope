"""Project file save/load — persist complete analysis state.

A .cellscope project file is a JSON document containing:
  - recording path
  - pipeline mode and parameters
  - analysis results (scalar metrics)
  - mask file reference
  - ROI definition
"""
import os
import json
import numpy as np


PROJECT_VERSION = 1


def save_project(path, recording, detect_result, analysis_result,
                 params, mode, roi_mask=None):
    """Save analysis state to a .cellscope project file.

    Args:
        path: output .cellscope file path
        recording: recording dict (only metadata saved, not frames)
        detect_result: detection result dict
        analysis_result: analysis result dict or list
        params: dict from ParamsPanel.get_detect_params()
        mode: "single" or "multi"
        roi_mask: (H, W) bool array or None
    """
    proj = {
        "version": PROJECT_VERSION,
        "recording": {
            "video_path": recording.get("video_path", ""),
            "name": recording.get("name", ""),
            "um_per_px": recording.get("um_per_px", 1.0),
            "time_interval_min": recording.get("time_interval_min", 1.0),
            "n_frames": len(recording.get("frames", [])),
        },
        "mode": mode,
        "params": params,
    }

    # Save masks alongside the project file
    masks_path = path.replace(".cellscope", "_masks.npz")
    if detect_result is not None:
        save_dict = {"masks": detect_result.get("masks", np.array([]))}
        labels = detect_result.get("labels")
        if labels is not None:
            save_dict["labels"] = labels
        np.savez_compressed(masks_path, **save_dict)
        proj["masks_file"] = os.path.basename(masks_path)

    # Scalar metrics from analysis
    if analysis_result is not None:
        if isinstance(analysis_result, list):
            proj["analysis"] = [_extract_scalars(r) for r in analysis_result]
        else:
            proj["analysis"] = _extract_scalars(analysis_result)

    if roi_mask is not None:
        roi_path = path.replace(".cellscope", "_roi.npz")
        np.savez_compressed(roi_path, roi=roi_mask)
        proj["roi_file"] = os.path.basename(roi_path)

    with open(path, "w") as f:
        json.dump(proj, f, indent=2, default=str)

    return path


def load_project(path):
    """Load a .cellscope project file.

    Returns:
        dict with keys: recording_info, mode, params, masks, labels,
        analysis, roi_mask
    """
    with open(path) as f:
        proj = json.load(f)

    result = {
        "recording_info": proj.get("recording", {}),
        "mode": proj.get("mode", "single"),
        "params": proj.get("params", {}),
        "analysis": proj.get("analysis"),
        "masks": None,
        "labels": None,
        "roi_mask": None,
    }

    proj_dir = os.path.dirname(os.path.abspath(path))

    masks_file = proj.get("masks_file")
    if masks_file:
        masks_path = os.path.join(proj_dir, masks_file)
        if os.path.exists(masks_path):
            data = np.load(masks_path)
            result["masks"] = data.get("masks")
            result["labels"] = data.get("labels")

    roi_file = proj.get("roi_file")
    if roi_file:
        roi_path = os.path.join(proj_dir, roi_file)
        if os.path.exists(roi_path):
            result["roi_mask"] = np.load(roi_path)["roi"]

    return result


def _extract_scalars(result):
    """Extract JSON-serializable scalar metrics from a result dict."""
    out = {}
    for key in ["name", "n_frames", "um_per_px", "time_interval_min",
                "mean_speed", "total_distance", "net_displacement",
                "persistence", "mean_boundary_confidence", "cell_id"]:
        if key in result:
            v = result[key]
            out[key] = float(v) if isinstance(v, (np.floating,)) else v
    for key in ["shape_summary", "edge_summary", "area_stability",
                "track_info"]:
        if key in result:
            out[key] = result[key]
    return out
