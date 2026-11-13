"""Web interface tests: every exported endpoint answers, the static build is complete, the JS engines match the
Python fixture (node), the fixture is current, and the API reflects the Python modules it reads.

Run: python -m pytest -q tests/test_web.py   (needs only requirements-web.txt)
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)
FIXTURE = ROOT / "tests" / "fixtures" / "web_engine"


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    site = tmp_path / "site"
    assert (site / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (site / "api" / (path.strip("/") + ".json")).exists()
    demo = json.loads((site / "api" / "demo.json").read_text())
    for rel in demo["files"].values():
        assert (site / rel).exists(), rel
    assert not list((site / "static").glob("*.test.mjs"))


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "style_app.css", "main.js", "icon.svg"):
        assert f"static/{name}" in html and (STATIC / name).exists()
    main = (STATIC / "main.js").read_text()
    for page in ("home", "track", "state", "measure", "planner", "params", "check", "about"):
        assert f"'{page}.js'" in main and (STATIC / f"{page}.js").exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    assert tests
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


def _close(a, b, path=""):
    if isinstance(b, dict):
        assert set(a) == set(b), path
        for k in b:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(b, float) and not isinstance(b, bool):
        assert a is not None and np.isclose(a, b, rtol=1e-6, atol=1e-9), (path, a, b)
    else:
        assert a == b, (path, a, b)


@pytest.mark.skipif(bool(os.environ.get("CI")), reason="tracking + state are discrete choices from float maths; "
                    "the node test still checks the JS against the committed fixture")
def test_fixture_is_current():
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_web_fixture
    arrays, expected = make_web_fixture.build()
    stored = np.load(FIXTURE / "masks.npz")
    for k, a in arrays.items():
        assert np.array_equal(stored[k], a), k
    _close(json.loads(json.dumps(expected)), json.loads((FIXTURE / "expected.json").read_text()))


def test_demo_is_synthetic_and_labelled():
    d = client.get("/api/demo").json()
    assert d["synthetic"] and "SYNTHETIC" in d["note"]
    assert d["truth"]["synthetic"] and len(d["truth"]["frames"]) == 40
    assert client.get("/" + d["files"]["detections"]).status_code == 200
    side = json.loads(client.get("/" + d["files"]["sidecar"]).text)
    assert side["synthetic"] and side["um_per_px"] == d["um_per_px"]


def test_defaults_endpoint_mirrors_python():
    from core.cell_state import DEFAULT_THRESHOLDS
    from core.detection_presets import PRESET_ORDER
    from core.pipeline_defaults import DEFAULTS, DOWNSAMPLE_SMALL_PX, MULTICELL_THRESHOLD
    d = client.get("/api/defaults").json()
    assert {r["name"]: r["value"] for r in d["fields"]} == json.loads(json.dumps(DEFAULTS.as_dict()))
    assert {r["name"]: r["value"] for r in d["thresholds"]} == DEFAULT_THRESHOLDS
    assert d["constants"]["DOWNSAMPLE_SMALL_PX"] == DOWNSAMPLE_SMALL_PX
    assert d["constants"]["MULTICELL_THRESHOLD"] == MULTICELL_THRESHOLD
    assert d["presets"]["order"] == PRESET_ORDER
    docs = {r["name"]: r["doc"] for r in d["fields"]}
    assert "Hungarian cost-matrix weights" in docs["track_w_dist"]   # comments are read from the source
    assert docs["max_gap_frames"].startswith("Max consecutive frames")


def test_reference_is_the_tracking_gui_path():
    from core.multi_cell import track_all_cells
    from web.api_data import demo_paths, gui_track_params
    from web.demo_data import DT_MIN, UM_PER_PX
    ref = client.get("/api/demo/reference").json()
    labels = np.load(demo_paths()["detections"])["labels"]
    tracks = track_all_cells(labels, **gui_track_params(UM_PER_PX, DT_MIN))
    assert len(ref["tracks"]) == len(tracks)
    assert ref["params"]["max_hop_px"] == 115 and ref["params"]["min_area_px"] == 50   # 15 µm/min·10 min / 1.3 µm/px


def test_info_counts():
    i = client.get("/api/info").json()
    assert i["n_apps"] == len(list(ROOT.glob("main_*.py")))
    assert i["demo"]["n_cells"] == 9
