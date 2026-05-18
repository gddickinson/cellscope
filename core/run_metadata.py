"""Write a RUN_METADATA.{md,json} alongside every results dump.

Every analysis run — GUI export, batch worker, programmatic
script — MUST call `write_run_metadata(...)` so we can reconstruct
how each set of results was generated.

The MD file is human-readable for quick inspection. The JSON file
is the machine-readable canonical record (full param dump + diff
from defaults + env versions + runtime).

Required fields:
  - source_recording: input file path + checksum
  - mode: "single" / "multi"
  - pipeline_function: "hybrid_dic_multi" / "hybrid_cpsam_multi" / ...
  - params_used: full dict from get_detect_params() (or equivalent)
  - params_non_default: only the keys whose value differs from
    core.pipeline_defaults.DEFAULTS
  - models_used: e.g. {"dic": "data/models/cpsam_dic", ...}
  - env: {"conda_env", "python_version", "cellpose_version",
          "platform"}
  - timestamp_started / timestamp_finished
  - runtime_seconds
  - n_frames_processed
  - n_tracks
  - rerun_command: shell command that would reproduce the run
  - any cellscope git hash if the repo is git-tracked

Schema version is in METADATA_SCHEMA_VERSION; readers should check it.
"""
from __future__ import annotations

import os
import sys
import json
import socket
import hashlib
import logging
import platform
import subprocess
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

METADATA_SCHEMA_VERSION = "1.0.0"


def _file_checksum(path, blocksize=2 ** 20):
    """SHA-256 of the first ~64 MB of a file; full hash on smaller
    files. Good enough to detect "is this the same recording" without
    paying the cost of hashing multi-GB TIFFs."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            chunk = f.read(64 * blocksize)
            h.update(chunk)
        return f"sha256-64mb:{h.hexdigest()[:16]}"
    except Exception as e:
        return f"unhashable:{type(e).__name__}"


def _env_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "<none>"),
        "hostname": socket.gethostname(),
    }
    for name in ("cellpose", "numpy", "scipy", "tifffile",
                 "skimage", "torch"):
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "?")
        except Exception:
            info[name] = "(not installed)"
    return info


def _git_hash(project_root):
    try:
        out = subprocess.run(
            ["git", "-C", project_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()[:12]
    except Exception:
        pass
    return None


def write_run_metadata(out_dir,
                        recording,
                        params,
                        detect_result,
                        analysis_results=None,
                        pipeline_function: str = "unknown",
                        mode: str = "multi",
                        runtime_seconds: Optional[float] = None,
                        started_at: Optional[str] = None,
                        rerun_command: Optional[str] = None,
                        extra: Optional[dict] = None,
                        project_root: Optional[str] = None):
    """Write RUN_METADATA.md + RUN_METADATA.json into `out_dir`.

    Args:
      out_dir: where the masks/metrics live; metadata is written here
      recording: the recording dict (must have `video_path` or `name`)
      params: dict of all detect parameters that were used
      detect_result: detection result dict (for counts)
      analysis_results: per-cell list (single → 1-element)
      pipeline_function: name of the detect function called
      mode: 'single' or 'multi'
      runtime_seconds: total wall-clock runtime
      started_at: ISO timestamp (defaults to now)
      rerun_command: shell snippet to reproduce
      extra: arbitrary dict merged into metadata for caller-specific
        fields (e.g. fusion stats)
      project_root: cellscope root for git hash detection
    """
    from core.pipeline_defaults import diff_from_defaults, DEFAULTS

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    video = recording.get("video_path") or recording.get("name", "")
    has_cy5 = recording.get("cy5_frames") is not None
    deviations = diff_from_defaults(params or {}, has_cy5=has_cy5)

    n_frames = int(len(recording.get("frames", [])))
    n_tracks = len(detect_result.get("tracks", []) or [])
    n_cy5_fusion_added = detect_result.get("n_cy5_fusion_added", 0)
    fusion_src_summary = None
    if detect_result.get("tracks"):
        from collections import Counter
        fusion_src_summary = dict(Counter(
            t.get("fusion_source", "dic_only")
            for t in detect_result["tracks"]))

    started_at = started_at or datetime.now().isoformat(timespec="seconds")
    finished_at = datetime.now().isoformat(timespec="seconds")

    meta = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": runtime_seconds,
        "pipeline_function": pipeline_function,
        "mode": mode,
        "source_recording": {
            "name": recording.get("name", ""),
            "video_path": video,
            "n_frames": n_frames,
            "has_cy5_channel": has_cy5,
            "um_per_px": recording.get("um_per_px"),
            "time_interval_min": recording.get("time_interval_min"),
            "checksum": _file_checksum(video) if video and os.path.exists(video) else None,
        },
        "params_used": params or {},
        "params_non_default": deviations,
        "defaults_reference": DEFAULTS.as_dict(),
        "results_summary": {
            "n_frames_processed": n_frames,
            "n_tracks": n_tracks,
            "n_cy5_fusion_added": n_cy5_fusion_added,
            "fusion_source_breakdown": fusion_src_summary,
            "n_analysis_cells": (len(analysis_results)
                                  if analysis_results else 0),
        },
        "env": _env_info(),
        "git_commit": _git_hash(project_root) if project_root else None,
        "rerun_command": rerun_command,
        "out_dir": out_dir,
    }
    if extra:
        meta["extra"] = extra

    # JSON canonical record
    with open(os.path.join(out_dir, "RUN_METADATA.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Markdown summary
    md = _format_metadata_md(meta)
    with open(os.path.join(out_dir, "RUN_METADATA.md"), "w") as f:
        f.write(md)

    log.info("Wrote RUN_METADATA.{md,json} to %s", out_dir)
    return meta


def _format_metadata_md(meta):
    src = meta["source_recording"]
    res = meta["results_summary"]
    env = meta["env"]
    rt = meta.get("runtime_seconds")
    runtime = f"{rt:.0f} s" if rt is not None else "unknown"
    nondef = meta["params_non_default"]
    nondef_lines = []
    for k, v in sorted(nondef.items()):
        if isinstance(v, dict) and "used" in v and "default" in v:
            nondef_lines.append(f"  - **{k}**: `{v['used']}` "
                                f"(default `{v['default']}`)")
        else:
            nondef_lines.append(f"  - **{k}**: `{v}` (not in defaults)")
    if not nondef_lines:
        nondef_block = "  _(none — all defaults used)_"
    else:
        nondef_block = "\n".join(nondef_lines)

    fusion_breakdown = res.get("fusion_source_breakdown")
    if fusion_breakdown:
        fb_line = "  - fusion source breakdown: " + ", ".join(
            f"{k} = {v}" for k, v in sorted(fusion_breakdown.items()))
    else:
        fb_line = ""

    rerun = meta.get("rerun_command") or ""
    rerun_block = (f"```bash\n{rerun}\n```"
                   if rerun else "  _(not recorded)_")

    git = meta.get("git_commit") or "(not a git repo or git unavailable)"
    return f"""# RUN_METADATA — {src.get("name", "unnamed")}

**Schema version**: {meta["schema_version"]}
**Started**: {meta["started_at"]}
**Finished**: {meta["finished_at"]}
**Runtime**: {runtime}
**Pipeline**: `{meta["pipeline_function"]}` (mode = {meta["mode"]})

## Source recording

- **video_path**: `{src["video_path"]}`
- **n_frames**: {src["n_frames"]}
- **has Cy5 channel**: {src["has_cy5_channel"]}
- **um_per_px**: {src["um_per_px"]}
- **time_interval_min**: {src["time_interval_min"]}
- **checksum**: `{src.get("checksum")}`

## Results

- **n_tracks**: {res["n_tracks"]}
- **n_cy5_fusion_added** (pre-tracking): {res["n_cy5_fusion_added"]}
- **n_analysis_cells**: {res["n_analysis_cells"]}
{fb_line}

## Parameters deviating from defaults

{nondef_block}

## Environment

- **conda env**: `{env["conda_env"]}`
- **python**: `{env["python"]}`
- **platform**: `{env["platform"]}`
- **cellpose**: `{env["cellpose"]}`
- **numpy**: `{env["numpy"]}`
- **tifffile**: `{env["tifffile"]}`
- **scipy**: `{env["scipy"]}`
- **skimage**: `{env["skimage"]}`
- **torch**: `{env["torch"]}`

## Reproducibility

- **git commit**: `{git}`
- **rerun command**:

{rerun_block}

## Output directory

`{meta["out_dir"]}`

---
*Generated automatically. If this file is missing alongside results,
the run did not write metadata — see core/run_metadata.py for the
expected API. This is a requirement: every analysis path (GUI export,
batch worker, programmatic script) MUST call `write_run_metadata`.*
"""
