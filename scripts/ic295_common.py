"""Shared utilities for the IC295 batch analysis.

Recording inventory, condition parsing, path conventions, atomic
progress.json updates, recording-folder setup, .cellscope project
writing. Used by ic295_detect_one, ic295_analyze_one, ic295_batch,
ic295_status, ic295_compare.
"""
import os
import json
import glob
import shutil
import tempfile

DRIVE_SOURCES = {
    "IC295":        "/Volumes/GeorgeDrive/ignasi/IC295",
    "IC295_batch2": "/Volumes/GeorgeDrive/ignasi/IC295_batch2",
}
CONDITIONS = ("WT", "KO", "GOF", "Y1", "OT", "DMSO")

PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_ROOT    = os.path.join(PROJECT_ROOT, "ic295_analysis")
RECORDINGS_ROOT  = os.path.join(ANALYSIS_ROOT, "by_condition")
RUNS_DIR         = os.path.join(ANALYSIS_ROOT, "_runs")
LOGS_DIR         = os.path.join(RUNS_DIR, "logs")
PROGRESS_FILE    = os.path.join(RUNS_DIR, "progress.json")
LOCK_FILE        = os.path.join(RUNS_DIR, "lock.txt")
COMPARE_DIR      = os.path.join(ANALYSIS_ROOT, "compare")
CACHE_DIR        = os.path.join(ANALYSIS_ROOT, "_cache")


def best_video_path(info):
    """Return the cached local .ome.tif when a complete cache entry
    exists (`.ome.tif` + `.ome.json` adjacent), otherwise the drive
    path. Used by `ic295_detect_one` at LOAD time so detection reads
    from local SSD when the prefetcher has staged the file. The
    `by_condition/<cond>/<label>/` symlinks always point to the drive
    (durable across cache eviction)."""
    tif_name = os.path.basename(info["video_path"])
    cached_tif = os.path.join(CACHE_DIR, tif_name)
    if not os.path.exists(cached_tif):
        return info["video_path"]
    # .ome.json sidecar is essential for um/px + n_frames; require it
    # adjacent to the cached TIF so the loader sees the same metadata
    # it would on the drive. (`_metadata.txt` is now optional — the
    # tifffile.series.axes fix means channel detection works without it.)
    json_name = os.path.basename(info["json_sidecar"])
    if not os.path.exists(os.path.join(CACHE_DIR, json_name)):
        return info["video_path"]
    return cached_tif


def parse_condition(label):
    """`Pos21-KO` -> `KO`. Returns '?' if no recognizable suffix."""
    if "-" not in label:
        return "?"
    return label.rsplit("-", 1)[1]


def inventory_drive():
    """Map label -> info dict for every IC295 recording on the drive.

    Prefers IC295_batch2 on collision (mini's IoU+area run).
    """
    out = {}
    for src_name, src_dir in DRIVE_SOURCES.items():
        if not os.path.isdir(src_dir):
            continue
        for tif in sorted(glob.glob(
                os.path.join(src_dir, "IC295__1_MMStack_*.ome.tif"))):
            base = os.path.basename(tif)
            label = base.replace("IC295__1_MMStack_", "")\
                        .replace(".ome.tif", "")
            cond = parse_condition(label)
            if cond not in CONDITIONS:
                continue
            info = {
                "label":         label,
                "condition":     cond,
                "source_dir":    src_dir,
                "source_name":   src_name,
                "video_path":    tif,
                "json_sidecar":  tif.replace(".ome.tif", ".ome.json"),
                "metadata_txt":  tif.replace(".ome.tif", "_metadata.txt"),
                "drive_masks":   os.path.join(
                    src_dir, "processed", label,
                    "pipeline_results", "masks.npz"),
            }
            # IC295_batch2 wins ties (e.g. Pos51-Y1 in both)
            if label in out and src_name == "IC295":
                continue
            out[label] = info
    return out


def has_drive_detection(info):
    return os.path.exists(info["drive_masks"])


def recording_dir(label, condition):
    return os.path.join(RECORDINGS_ROOT, condition, label)


# ───────── atomic JSON write (crash-safe progress / metadata) ─────────
def atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json",
                                dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _json_default(obj):
    """JSON serializer for numpy scalars / arrays."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
    except Exception:
        return {}
    # Schema migration: earlier init_entry used 'detection'/'analysis'
    # keys while the driver + CLI use 'detect'/'analyze'. Merge so all
    # readers see one consistent key. If both forms exist for a phase,
    # the short-form (driver-written) wins.
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        for old, new in (("detection", "detect"),
                         ("analysis", "analyze")):
            if old in entry:
                if new not in entry:
                    entry[new] = entry.pop(old)
                else:
                    entry.pop(old)
    return data


def save_progress(progress):
    atomic_write_json(PROGRESS_FILE, progress)


# ───────── priority queue (existing dets first, then balance n) ─────────
def detection_priority_order(inv=None):
    """Order labels for the detect-phase driver.

    1. Recordings with existing drive detections (fast — analysis only).
    2. Undetected, round-robined across conditions so n grows balanced.
    """
    if inv is None:
        inv = inventory_drive()
    by_cond = {c: [] for c in CONDITIONS}
    for label, info in inv.items():
        by_cond[info["condition"]].append(label)
    for c in CONDITIONS:
        by_cond[c].sort()

    detected = sorted(
        (l for l, i in inv.items() if has_drive_detection(i)),
        key=lambda l: (inv[l]["condition"], l))
    undetected = {c: [l for l in by_cond[c] if l not in set(detected)]
                  for c in CONDITIONS}

    queue = list(detected)
    while any(undetected.values()):
        for c in CONDITIONS:
            if undetected[c]:
                queue.append(undetected[c].pop(0))
    return queue


# ───────── recording-folder setup (symlinks + sidecar + cellscope) ─────────
def _link_abs(target, linkpath):
    target = os.path.abspath(target)
    if os.path.lexists(linkpath):
        os.remove(linkpath)
    os.symlink(target, linkpath)


def setup_recording_folder(info):
    """Build by_condition/<cond>/<label>/ as a self-contained project
    folder: recording symlink + metadata.txt symlink + sidecar copy +
    pipeline_results/ dir. Idempotent. Returns the folder path."""
    rec_dir = recording_dir(info["label"], info["condition"])
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(os.path.join(rec_dir, "pipeline_results"), exist_ok=True)
    # recording symlink (large .ome.tif — never copy)
    tif_name = os.path.basename(info["video_path"])
    _link_abs(info["video_path"], os.path.join(rec_dir, tif_name))
    # metadata.txt symlink (the focused GUI's detect_channels needs this
    # for multichannel channel-axis detection on Micro-Manager TIFFs)
    if os.path.exists(info["metadata_txt"]):
        md_name = os.path.basename(info["metadata_txt"])
        _link_abs(info["metadata_txt"], os.path.join(rec_dir, md_name))
    # .ome.json sidecar — copy so the recording is fully self-contained
    if os.path.exists(info["json_sidecar"]):
        js_name = os.path.basename(info["json_sidecar"])
        shutil.copy2(info["json_sidecar"], os.path.join(rec_dir, js_name))
    return rec_dir


def write_cellscope_project(rec_dir, info, recording_meta=None):
    """Save <label>.cellscope alongside the recording so the user can
    open it in the focused GUI for review / mask editing."""
    label  = info["label"]
    tif    = os.path.basename(info["video_path"])
    rmeta  = recording_meta or {}
    proj = {
        "version": 1,
        "recording": {
            # absolute drive path = portable regardless of project root
            "video_path":        info["video_path"],
            "name":              label,
            "um_per_px":         float(rmeta.get("um_per_px", 0.6523)),
            "time_interval_min": float(rmeta.get("time_interval_min", 10.0)),
            "n_frames":          int(rmeta.get("n_frames", 97)),
        },
        "mode": "multi",
        "params": {"mode": "multi"},
        "masks_file": "pipeline_results/masks.npz",
    }
    atomic_write_json(os.path.join(rec_dir, f"{label}.cellscope"), proj)


# ───────── progress.json entry helpers ─────────
def init_entry(progress, label, condition):
    if label not in progress:
        progress[label] = {
            "condition": condition,
            "detect":  {"state": "pending"},
            "analyze": {"state": "pending"},
        }
    return progress[label]


def set_phase(progress, label, phase, **kwargs):
    """phase: 'detect' | 'analyze' (matches CLI + driver args.phase)."""
    entry = progress.setdefault(label, {"condition": parse_condition(label),
                                          "detect": {}, "analyze": {}})
    entry.setdefault(phase, {}).update(kwargs)
    return entry
