"""Helper for writing RUN_METADATA.md alongside analysis results.

Per project convention (memory: results_metadata.md, 2026-05-01),
every results-producing script should write a RUN_METADATA.md
documenting source data, preprocessing, pipeline config, environment
versions, output structure, exact rerun CLI, and wall-clock timing.

Usage:
    from output.run_metadata import write_run_metadata
    write_run_metadata(
        out_path="results/my_run/RUN_METADATA.md",
        title="My analysis run",
        sections={
            "Source data": "...",
            "Pipeline config": "...",
            ...
        },
        rerun_cli="conda run -n cellpose4 python scripts/...",
    )

Captures the cellpose / python / platform versions automatically.
"""
from __future__ import annotations

import datetime as dt
import os
import platform
import sys
import textwrap


def _capture_env():
    """Capture the running environment versions for the metadata file."""
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ("cellpose", "torch", "numpy", "scipy", "tifffile"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__",
                               getattr(mod, "version", "?"))
        except ImportError:
            out[pkg] = "(not installed)"
    return out


def _format_section(title, body):
    return f"## {title}\n\n{body.rstrip()}\n"


def write_run_metadata(out_path, title, sections,
                        rerun_cli=None, extra_notes=None,
                        timing_seconds=None):
    """Write a structured RUN_METADATA.md.

    Args:
      out_path: file path to write
      title: top-level heading
      sections: dict of {section_title: body_markdown} — order preserved
      rerun_cli: shell command to reproduce the run
      extra_notes: optional footer markdown
      timing_seconds: dict of {stage: seconds} for wall-clock summary
    """
    env = _capture_env()
    now = dt.datetime.now().isoformat(timespec="seconds")

    parts = [f"# {title}\n",
             f"**Run written:** {now}\n",
             f"**Cwd:** `{os.getcwd()}`\n"]

    for sec_title, body in sections.items():
        parts.append(_format_section(sec_title, body))

    env_block = "\n".join(f"* {k}: `{v}`" for k, v in env.items())
    parts.append(_format_section("Software environment",
                                 env_block))

    if rerun_cli:
        parts.append(_format_section(
            "How to re-run",
            f"```bash\n{rerun_cli.strip()}\n```"))

    if timing_seconds:
        rows = ["| Stage | Time (s) |", "|---|---:|"]
        total = 0.0
        for stage, secs in timing_seconds.items():
            rows.append(f"| {stage} | {secs:.1f} |")
            total += float(secs)
        rows.append(f"| **Total** | **{total:.1f}** |")
        parts.append(_format_section("Wall-clock timing",
                                     "\n".join(rows)))

    if extra_notes:
        parts.append(_format_section("Notes", extra_notes))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    return out_path
