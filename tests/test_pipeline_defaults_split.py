"""Splitting core/pipeline_defaults.py into core/pipeline_policy.py + core/auto_select.py changed no value
and no import: the DEFAULTS dict, the Cy5 variant, every public name and constant match the snapshot taken
from the file before the split (tests/fixtures/pipeline_defaults_snapshot.json)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core.pipeline_defaults as pd  # noqa: E402

SNAP = json.loads((ROOT / "tests" / "fixtures" / "pipeline_defaults_snapshot.json").read_text())
NEW_FIELDS = {"tracker_hop_mode"}          # added after the split (scale-aware tracker hop), see test_tracker_hop.py


def test_defaults_identical():
    now = pd.DEFAULTS.as_dict()
    assert {k: v for k, v in now.items() if k not in NEW_FIELDS} == SNAP["defaults"]
    assert {k: v for k, v in pd.get_defaults(True).as_dict().items() if k not in NEW_FIELDS} == SNAP["with_cy5"]


def test_names_and_constants_still_importable():
    for name in SNAP["names"]:
        assert hasattr(pd, name), name
    for name, value in SNAP["constants"].items():
        if name == "CY5_FILTER_MODES":
            continue                        # extended on purpose (persistence_guard), see test_cy5_filter_modes.py
        v = getattr(pd, name)
        assert (list(v) if isinstance(v, tuple) else v) == value, name
    assert pd._PROBE_SCRIPT == SNAP["probe_script"]


def test_files_within_budget():
    for f in ("pipeline_defaults.py", "pipeline_policy.py", "auto_select.py"):
        assert len((ROOT / "core" / f).read_text().splitlines()) < 500, f
