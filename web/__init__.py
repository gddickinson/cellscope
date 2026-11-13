"""CellScope web interface (local FastAPI app + static GitHub Pages site).

Contents:
    app.py           FastAPI routes under /api, STATIC_ENDPOINTS for the static export
    api_data.py      data behind the API (counts, defaults, demo files, Python reference results)
    defaults_doc.py  DEFAULTS / DEFAULT_THRESHOLDS / presets with their source-code comments
    demo_data.py     the SYNTHETIC demo recording (simulated cells, detections, ground truth)
    static/          the front end: ES modules, no build step (see INTERFACE.md)
"""
