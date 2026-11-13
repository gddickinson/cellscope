"""FastAPI web app for CellScope: serves the front end in static/, the synthetic demo files under
/demo, and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8181, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site (and copies the demo files), so keep that list in step with the
GET routes below. Tracking, shape/state metrics and the planner run in the browser (static/*_engine.js, JS ports of
the Python functions); the API gives the defaults read from the Python modules and the Python reference results
on the demo. Detection (cellpose / cpsam / SAM2) is not part of the web interface.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import api_data

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="CellScope")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
api_data.demo_paths()                                      # writes web_demo/ on first start
app.mount("/demo", StaticFiles(directory=api_data.DEMO_DIR), name="demo")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary for the home page (counts from the code and the demo)."""
    return api_data.info()


@app.get("/api/defaults")
def defaults():
    """DEFAULTS, DEFAULT_THRESHOLDS, planner constants and detection presets, with their source comments."""
    return api_data.defaults()


@app.get("/api/demo")
def demo():
    """The synthetic demo: file URLs, scale and the simulation's ground truth."""
    return api_data.demo_info()


@app.get("/api/demo/reference")
def demo_reference():
    """Python tracking + colour-by metrics on the demo, as the desktop Tracking GUI computes them."""
    return api_data.demo_reference()


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/defaults", "/demo", "/demo/reference"]
