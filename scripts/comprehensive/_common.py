"""Shared helpers + paths for the comprehensive GUI tests."""
import os, sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("cellpose").setLevel(logging.ERROR)

OUT_DIR = "results/comprehensive_gui_tests"
SHOTS = os.path.join(OUT_DIR, "screenshots")
os.makedirs(SHOTS, exist_ok=True)

SINGLE_CELL = (
    "/Users/george/claude_test/piezo1_analysis/data/ignasi/"
    "C1-IC293__1_MMStack_Pos0-WT.ome-1cropped.tif")
MULTI_CELL = (
    "/Users/george/claude_test/piezo1_analysis/data/ignasi/"
    "IC293__1_MMStack_Pos19-KO.ome-cropped.tif")

RESULTS = []   # populated by each phase: (phase, name, status, detail)


def check(phase, name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((phase, name, status, detail))
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] [{phase}] {name}" + (f" -- {detail}" if detail else ""))


def shot(widget, name):
    pix = widget.grab()
    path = os.path.join(SHOTS, name + ".png")
    pix.save(path)
    return path


def trim(frames, n=20):
    """Trim frame stack for runtime sanity."""
    return frames[:n] if len(frames) > n else frames
