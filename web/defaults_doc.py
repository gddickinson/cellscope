"""The pipeline defaults as data for the web Parameters page, read from the source itself.

Values come from the live objects (``core.pipeline_defaults.DEFAULTS``,
``core.cell_state.DEFAULT_THRESHOLDS``, ``core.detection_presets``); the explanation
shown next to each value is the code comment written above it (or after it on the same
line) in the source file, and the group is the nearest ``# --- Group ---`` header. No
text is written here by hand, so the page cannot drift from the code.
"""
from __future__ import annotations

import ast
import inspect
import re

from core import cell_state, detection_presets, pipeline_defaults, runtime_estimate

_HEADER = re.compile(r"^#\s*-{2,}\s*(.+?)\s*-{2,}\s*$")


def _comments(src_lines, start, end):
    """Plain comment lines in [start, end) (1-based line numbers), header lines excluded."""
    out = []
    for ln in range(start, end):
        s = src_lines[ln - 1].strip()
        if s.startswith("#") and not _HEADER.match(s) and not set(s) <= {"#", "-", " "}:
            out.append(s.lstrip("#").strip())
        elif not s.startswith("#"):
            out = []                       # a code line breaks the comment block
    return " ".join(out)


def _trailing(line):
    m = re.search(r"#\s*(.+)$", line)
    return m.group(1).strip() if m and "#" not in line.split("=")[0] else ""


def _fields_with_docs(node, src_lines, values):
    """[{name, value, group, doc}] for every annotated or plain assignment in a class / dict."""
    rows, group, prev_end = [], "", node.lineno + 1
    for stmt in node.body if isinstance(node, ast.ClassDef) else []:
        for ln in range(prev_end, stmt.lineno):
            m = _HEADER.match(src_lines[ln - 1].strip())
            if m:
                group = m.group(1)
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            doc = _comments(src_lines, prev_end, stmt.lineno)
            tail = _trailing(src_lines[stmt.lineno - 1])
            rows.append({"name": name, "value": values.get(name), "group": group,
                         "doc": (doc + (" " if doc and tail else "") + tail).strip()})
        prev_end = stmt.end_lineno + 1
    return rows


def pipeline_defaults_doc():
    src = inspect.getsource(pipeline_defaults)
    lines = src.splitlines()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PipelineDefaults")
    return _fields_with_docs(cls, lines, pipeline_defaults.DEFAULTS.as_dict())


def thresholds_doc():
    """DEFAULT_THRESHOLDS with the comments above each key."""
    src = inspect.getsource(cell_state)
    lines = src.splitlines()
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "DEFAULT_THRESHOLDS")
    rows, prev = [], node.lineno + 1
    for k, v in zip(node.value.keys, node.value.values):
        rows.append({"name": k.value, "value": cell_state.DEFAULT_THRESHOLDS[k.value],
                     "doc": _comments(lines, prev, k.lineno)})
        prev = v.end_lineno + 1
    return rows


def module_constants():
    """Planner constants straight from core.pipeline_defaults / core.runtime_estimate."""
    pd = pipeline_defaults
    return {k: getattr(pd, k) for k in (
        "DOWNSAMPLE_SMALL_PX", "DOWNSAMPLE_LARGE_PX", "MIRROR_PAD_PX", "MIRROR_PAD_MIN_DIM_PX",
        "MULTICELL_THRESHOLD", "N_SAMPLE_FRAMES_FOR_AUTO", "PROBE_AGGREGATOR_PERCENTILE")} | {
        "DENSITY_MULTIPLIERS": [list(m) for m in runtime_estimate.DENSITY_MULTIPLIERS]}


def presets_doc():
    """Detection presets: order, the knobs they set, resolved values and the comment above each."""
    src = inspect.getsource(detection_presets)
    lines = src.splitlines()
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "DETECTION_PRESETS")
    docs, prev = {}, node.lineno + 1
    for k, v in zip(node.value.keys, node.value.values):
        docs[k.value] = _comments(lines, prev, k.lineno)
        prev = v.end_lineno + 1
    return {"order": detection_presets.PRESET_ORDER, "params": detection_presets.PRESET_PARAMS,
            "presets": [{"name": n, "doc": docs.get(n, ""), "deltas": detection_presets.DETECTION_PRESETS[n],
                         "values": detection_presets.preset_values(n)} for n in detection_presets.PRESET_ORDER]}
