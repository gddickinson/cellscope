"""Shared utilities for the IC293 single-cell-crop analysis.

IC293 differs from IC295 in three ways that this module encodes:

  1. SOURCE is the LOCAL staged cache (`ic293_analysis/_cache/`), not a
     drive. Crops were converted to the IC295 on-disk format by
     `scripts/ic293_stage_crops.py` (single-channel DIC, .ome.tif +
     .ome.json + _metadata.txt). There is NO drive-adopt path.

  2. Each "recording" is ONE hand-cropped cell. Several crops share an
     original field of view (`Pos{N}`). The POSITION is therefore the
     correct cluster for pseudoreplication — `parse_position()` extracts
     it and the stats layer reduces to one value per position (the exact
     analog of IC295's "recording is the unit").

  3. DIC only — no Cy5/fluorescence. Recordings load single-channel
     (`load_recording(path)` with no channel args) and detection runs
     with `cy5_frames=None` (all Cy5 logic is guarded off downstream).

Same six conditions + same two-arm design as IC295 (genetic WT→GOF/KO,
drug DMSO→Y1/OT), so the arm-structured stats machinery is reused
verbatim from the ic295_* scripts.
"""
import os
import re
import glob
import json
import shutil

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

# Generic, path-taking helpers are identical to IC295 — reuse them so the
# atomic-write / progress-entry behaviour can't drift between datasets.
from scripts.ic295_common import (  # noqa: E402
    atomic_write_json, _json_default, parse_condition, init_entry, set_phase,
)

CONDITIONS = ("WT", "KO", "GOF", "Y1", "OT", "DMSO")

# Recordings kept on disk but EXCLUDED from the statistical analysis after
# manual review (documented in ic293_analysis/CURATION_DECISIONS.md). Files
# are never deleted; the stats consumers (ic293_track_data, ic293_compare)
# skip these labels. Pos36_div-GOF: a genuine dividing cell (violates the
# single-cell / non-dividing criterion); its pre-division window (13 frames)
# is below the 30-frame minimum + the 24-frame MSD window — too short to
# contribute comparable metrics.
ANALYSIS_EXCLUDE = frozenset({"Pos36_div-GOF"})

# IC293 optics / cadence (same scope as IC295; embedded in every sidecar
# by the staging script, these are only the fallback when a sidecar is
# missing or partial).
DEFAULT_UM_PER_PX = 0.6523
DEFAULT_DT_MIN = 10.0

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_ROOT   = os.path.join(PROJECT_ROOT, "ic293_analysis")
CACHE_DIR       = os.path.join(ANALYSIS_ROOT, "_cache")
RECORDINGS_ROOT = os.path.join(ANALYSIS_ROOT, "by_condition")
RUNS_DIR        = os.path.join(ANALYSIS_ROOT, "_runs")
LOGS_DIR        = os.path.join(RUNS_DIR, "logs")
PROGRESS_FILE   = os.path.join(RUNS_DIR, "progress.json")
LOCK_FILE       = os.path.join(RUNS_DIR, "lock.txt")
COMPARE_DIR     = os.path.join(ANALYSIS_ROOT, "compare")

# Staged cache files keep the Micro-Manager stem (mirrors IC295's
# `IC295__1_MMStack_<label>.ome.tif`); the human label strips it.
CACHE_PREFIX = "IC293__1_MMStack_"
_POS_RE = re.compile(r"^(Pos\d+)")


def parse_position(label):
    """`Pos13_cell2-WT` -> `Pos13`. The position is the field of view; it
    is the cluster/biological-replicate unit for the IC293 stats (several
    single-cell crops can come from one position). Returns the whole label
    if it doesn't start with PosN (defensive — shouldn't happen)."""
    m = _POS_RE.match(label)
    return m.group(1) if m else label


def inventory_cache():
    """Map label -> info dict for every staged IC293 crop in the cache.

    Looks only at the LOCAL `_cache/` (the staging step already copied +
    converted everything; Ignasi's originals on the share are never
    touched). Each info carries the local .ome.tif, its .ome.json
    sidecar, the _metadata.txt, the parsed condition, and the parsed
    position."""
    out = {}
    for tif in sorted(glob.glob(os.path.join(CACHE_DIR, "*.ome.tif"))):
        label = os.path.basename(tif).replace(CACHE_PREFIX, "")\
                                     .replace(".ome.tif", "")
        cond = parse_condition(label)
        if cond not in CONDITIONS:
            continue
        out[label] = {
            "label":        label,
            "condition":    cond,
            "position":     parse_position(label),
            "video_path":   tif,
            "json_sidecar": tif.replace(".ome.tif", ".ome.json"),
            "metadata_txt": tif.replace(".ome.tif", "_metadata.txt"),
        }
    return out


def read_sidecar_meta(info):
    """Return {um_per_px, time_interval_min, n_frames, name} from the crop's
    .ome.json sidecar, falling back to IC293 scope defaults."""
    out = {"um_per_px": DEFAULT_UM_PER_PX,
           "time_interval_min": DEFAULT_DT_MIN,
           "n_frames": None,
           "name": info["label"]}
    sc = info.get("json_sidecar")
    if sc and os.path.exists(sc):
        try:
            with open(sc) as f:
                d = json.load(f)
            out["um_per_px"] = float(d.get("um_per_px") or out["um_per_px"])
            out["time_interval_min"] = float(
                d.get("time_interval_min") or out["time_interval_min"])
            if d.get("n_frames"):
                out["n_frames"] = int(d["n_frames"])
            out["name"] = d.get("name") or out["name"]
        except Exception:
            pass
    return out


def select_primary_cell(labels):
    """Pick the ONE real cell in a single-cell crop; the rest are artifacts.

    Ground truth (from the experimenter): each crop is hand-cropped to one
    target cell that is **present in every frame** and centred at the start
    (before it migrates); anything else cellpose detects (debris/fragments)
    is transient and/or peripheral. So presence is the PRIMARY signal:

      1. gate to the full-presence object(s) — those tracked in (near) all
         T frames (within a 5% tolerance for brief detection gaps that the
         track gap-fill didn't recover);
      2. among those, prefer the most central-at-start, then the largest.

    Returns (best_id, candidates, ambiguous). Each candidate carries
    frames / presence / total_area / start_dist_norm / centrality / score.
    `ambiguous` is True only when ≥2 full-presence objects have comparable
    centrality×size (a genuine "which is the target?" — flag for review).
    best_id is None if no objects were detected."""
    import numpy as np
    ids = [int(v) for v in np.unique(labels) if v > 0]
    if not ids:
        return None, [], False
    T, H, W = labels.shape
    cy0, cx0 = (H - 1) / 2.0, (W - 1) / 2.0
    half_diag = (((H ** 2 + W ** 2) ** 0.5) / 2.0) or 1.0
    cands = []
    for cid in ids:
        mask = labels == cid
        present = mask.any(axis=(1, 2))
        frames = int(present.sum())
        total_area = int(mask.sum())
        f0 = int(np.argmax(present))
        ys, xs = np.nonzero(mask[f0])
        if len(ys):
            dist = ((((ys.mean() - cy0) ** 2 + (xs.mean() - cx0) ** 2) ** 0.5)
                    / half_diag)
        else:
            dist = 1.0
        cands.append({"id": cid, "frames": frames, "total_area": total_area,
                      "presence": frames / max(T, 1),
                      "start_dist_norm": float(min(dist, 1.0))})
    max_area = max(c["total_area"] for c in cands) or 1
    for c in cands:
        c["centrality"] = 1.0 - c["start_dist_norm"]
        c["score"] = c["presence"] * (c["total_area"] / max_area) * c["centrality"]

    # (1) Presence gate: the target is present in (near) every frame.
    max_frames = max(c["frames"] for c in cands)
    tol = max(1, int(round(0.05 * T)))
    leaders = [c for c in cands if c["frames"] >= max_frames - tol]

    # (2) Among full-presence objects: most central, then largest.
    def _lead_key(c):
        return (c["centrality"], c["total_area"] / max_area)
    leaders.sort(key=_lead_key, reverse=True)
    best = leaders[0]
    ambiguous = bool(
        len(leaders) > 1
        and _lead_key(leaders[1])[0] * (leaders[1]["total_area"] / max_area)
            > 0.5 * (_lead_key(leaders[0])[0] * (leaders[0]["total_area"] / max_area)))

    cands.sort(key=lambda c: c["score"], reverse=True)
    cands = [best] + [c for c in cands if c["id"] != best["id"]]
    return best["id"], cands, ambiguous


def recording_dir(label, condition):
    return os.path.join(RECORDINGS_ROOT, condition, label)


# ───────── recording-folder setup (mirrors ic295, local-cache source) ─────────
def _link_abs(target, linkpath):
    target = os.path.abspath(target)
    if os.path.lexists(linkpath):
        os.remove(linkpath)
    os.symlink(target, linkpath)


def setup_recording_folder(info):
    """Build by_condition/<cond>/<label>/ as a self-contained project
    folder: recording symlink (to the local cache .ome.tif) + metadata.txt
    symlink + sidecar copy + pipeline_results/ dir. Idempotent."""
    rec_dir = recording_dir(info["label"], info["condition"])
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(os.path.join(rec_dir, "pipeline_results"), exist_ok=True)
    tif_name = os.path.basename(info["video_path"])
    _link_abs(info["video_path"], os.path.join(rec_dir, tif_name))
    if os.path.exists(info["metadata_txt"]):
        md_name = os.path.basename(info["metadata_txt"])
        _link_abs(info["metadata_txt"], os.path.join(rec_dir, md_name))
    if os.path.exists(info["json_sidecar"]):
        js_name = os.path.basename(info["json_sidecar"])
        shutil.copy2(info["json_sidecar"], os.path.join(rec_dir, js_name))
    return rec_dir


def write_cellscope_project(rec_dir, info, recording_meta=None):
    """Save <label>.cellscope alongside the recording so the user can open
    it in the focused GUI for review / mask editing. Mode is 'single' —
    these are DIC-only single-cell crops."""
    label = info["label"]
    rmeta = recording_meta or {}
    proj = {
        "version": 1,
        "recording": {
            "video_path":        info["video_path"],
            "name":              label,
            "um_per_px":         float(rmeta.get("um_per_px", DEFAULT_UM_PER_PX)),
            "time_interval_min": float(rmeta.get("time_interval_min",
                                                 DEFAULT_DT_MIN)),
            "n_frames":          int(rmeta.get("n_frames", 0)),
        },
        "mode": "single",
        "params": {"mode": "single"},
        "masks_file": "pipeline_results/masks.npz",
    }
    atomic_write_json(os.path.join(rec_dir, f"{label}.cellscope"), proj)


# ───────── progress.json (IC293 paths; same schema as IC295) ─────────
def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_progress(progress):
    atomic_write_json(PROGRESS_FILE, progress)


def detection_order(inv=None):
    """Order labels for the driver: round-robin across conditions so n
    grows balanced if the run is interrupted partway."""
    if inv is None:
        inv = inventory_cache()
    by_cond = {c: [] for c in CONDITIONS}
    for label, info in inv.items():
        by_cond[info["condition"]].append(label)
    for c in CONDITIONS:
        by_cond[c].sort()
    queue = []
    while any(by_cond.values()):
        for c in CONDITIONS:
            if by_cond[c]:
                queue.append(by_cond[c].pop(0))
    return queue


__all__ = [
    "CONDITIONS", "ANALYSIS_EXCLUDE", "DEFAULT_UM_PER_PX", "DEFAULT_DT_MIN",
    "ANALYSIS_ROOT", "CACHE_DIR", "RECORDINGS_ROOT", "RUNS_DIR", "LOGS_DIR",
    "PROGRESS_FILE", "LOCK_FILE", "COMPARE_DIR",
    "parse_condition", "parse_position", "inventory_cache",
    "select_primary_cell",
    "read_sidecar_meta", "recording_dir", "setup_recording_folder",
    "write_cellscope_project", "load_progress", "save_progress",
    "init_entry", "set_phase", "detection_order",
    "atomic_write_json", "_json_default",
]
