"""Save / load intermediate detection + refinement results.

Lets the user run detection once and experiment with many refinement
configurations without re-running cellpose (which dominates wall time
for most recordings). Stored as a single `.npz` with metadata.

The file format is versioned so future extensions can add fields
without breaking older checkpoints.
"""
import os
import json
import time
import numpy as np

CHECKPOINT_VERSION = 1


def save_detection(recording, det_result, out_path, params=None,
                   extra=None):
    """Write a detection checkpoint to disk.

    Args:
        recording: the recording dict from core.io.load_recording (only
            `name`, `video_path`, `um_per_px`, `time_interval_min`
            are persisted — not the frames).
        det_result: the dict returned by pipeline.detect() (masks,
            provenance, stats, elapsed, ...).
        out_path: where to write. Directory is created if needed.
        params: optional RunParams.to_dict() for provenance.
        extra: optional dict of other JSON-serializable fields.

    Returns:
        out_path
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    meta = {
        "version": CHECKPOINT_VERSION,
        "kind": "detection",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "recording": {
            "name": recording.get("name"),
            "video_path": recording.get("video_path"),
            "um_per_px": recording.get("um_per_px"),
            "time_interval_min": recording.get("time_interval_min"),
            "n_frames": int(len(recording.get("frames", []))),
        },
        "det_elapsed_s": float(det_result.get("elapsed", 0.0)),
    }
    if "cascade_stats" in det_result:
        meta["cascade_stats"] = det_result["cascade_stats"]
    if "retry_stats" in det_result:
        meta["retry_stats"] = det_result["retry_stats"]
    if params is not None:
        meta["params"] = params
    if extra is not None:
        meta["extra"] = extra

    payload = {
        "masks": det_result["masks"].astype(bool),
        "meta_json": np.array(
            json.dumps(meta, default=str), dtype="<U65536"
        ),
    }
    # Provenance from cascade / threshold_retry
    prov = det_result.get("provenance")
    if prov is not None:
        payload["provenance"] = np.array(prov, dtype="<U32")
    # Flow quality + magnitudes can be large; keep optional.
    if ("flow_quality" in det_result
            and det_result["flow_quality"] is not None):
        fq = np.asarray(det_result["flow_quality"])
        if fq.size:
            payload["flow_quality"] = fq
    np.savez_compressed(out_path, **payload)
    return out_path


def load_detection(path):
    """Load a detection checkpoint.

    Returns:
        det_dict: compatible with the `det` output of pipeline.detect()
            (masks, provenance, flow_quality, elapsed, plus the saved
            cascade_stats / retry_stats / meta).
        meta: full meta dict from the checkpoint.
    """
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    det = {"masks": data["masks"].astype(bool),
           "elapsed": meta.get("det_elapsed_s", 0.0)}
    if "provenance" in data.files:
        det["provenance"] = list(data["provenance"])
    if "flow_quality" in data.files:
        det["flow_quality"] = data["flow_quality"]
    else:
        det["flow_quality"] = np.zeros(len(det["masks"]))
    # Zero-sized flow magnitudes placeholder (compatible shape)
    det["flow_magnitudes"] = np.zeros(det["masks"].shape, dtype=np.float32)
    if "cascade_stats" in meta:
        det["cascade_stats"] = meta["cascade_stats"]
    if "retry_stats" in meta:
        det["retry_stats"] = meta["retry_stats"]
    return det, meta


def default_checkpoint_path(results_root: str, recording_name: str) -> str:
    """Conventional location for a detection checkpoint."""
    return os.path.join(results_root, recording_name, "detection.npz")
