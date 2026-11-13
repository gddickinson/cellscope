"""SYNTHETIC demo recording for the web interface (no real microscopy data).

Nine simulated cells (irregular, wobbling outlines) move by persistent random walks in a
200 x 200 px field for 40 frames, with the events a tracker has to handle written into the
script: one cell rounds up for a few frames, one rounds up and divides into two daughters,
one migrates in from the left edge, one leaves through the bottom, one is missed by the
"detector" for three frames, two pass close to each other, and small debris specks appear.

What is written is what a detector hands to the tracker: a per-frame label stack whose IDs
are shuffled every frame (no identity across frames), plus a phase-contrast-like image made
from the same masks. The simulation's own ground truth (which true cell each label is, the
rounded frames, the division) goes to a separate truth JSON so the web pages can score the
tracker and the rounded/spread rule against it. Every value comes from a fixed seed.

Files (``ensure_demo(out)``, rewritten when ``DEMO_VERSION`` changes):
    <out>/synthetic_demo.ome.tif        (T, H, W) uint8, OME metadata with the pixel size
    <out>/synthetic_demo.ome.json       CellScope sidecar: um_per_px, time_interval_min
    <out>/synthetic_demo_detections.npz key ``labels``: (T, H, W) int32 per-frame detections
    <out>/synthetic_demo_truth.json     simulation ground truth
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import ndimage as ndi

DEMO_VERSION = 2
NAME = "synthetic_demo"
T, H, W = 40, 200, 200
UM_PER_PX = 1.3          # synthetic scale, chosen so spread cells are > 960 µm² and rounded ones below
DT_MIN = 10.0
SEED = 20261112
SOURCE = "web/demo_data.py (synthetic, seed %d)" % SEED
NOTE = ("SYNTHETIC demo: simulated cells and a simulated phase-contrast-like image made by "
        "web/demo_data.py. Not a microscopy recording.")


def _cell_mask(cy, cx, radius, elong, phi, wob):
    """Boolean mask of a star-shaped cell: r(θ) = R(1 + e·cos 2(θ−φ) + Σ wobble harmonics)."""
    yy, xx = np.mgrid[:H, :W]
    dy, dx = yy - cy, xx - cx
    th = np.arctan2(dy, dx)
    r = radius * (1.0 + elong * np.cos(2 * (th - phi))
                  + sum(a * np.cos(k * th + p) for k, a, p in wob))
    return dy * dy + dx * dx <= r * r


def _cells(rng):
    """The scripted cell list: start/end frames, start position, heading, speed and events."""
    def cell(cid, y, x, theta, v, R, **kw):
        c = {"id": cid, "y": y, "x": x, "theta": theta, "v": v, "R": R, "elong": rng.uniform(0.15, 0.32),
             "wob": [(k, rng.uniform(0.02, 0.06), rng.uniform(0, 2 * np.pi)) for k in (3, 4, 5)],
             "start": 0, "end": T, "round": [], "missing": [], "parent": None, "turn": 0.25, "grow": None}
        c.update(kw)
        return c
    return [
        cell(1, 45, 50, 0.3, 1.6, 19, round=list(range(14, 20)), note="rounds up for six frames"),
        cell(2, 100, 100, 2.2, 1.2, 20, round=list(range(15, 21)), end=21, note="rounds up, then divides"),
        cell(3, 140, -12, 0.05, 2.4, 17, start=8, turn=0.12, note="migrates in from the left edge"),
        cell(4, 160, 150, -2.0, 1.4, 18, missing=[24, 25, 26], note="missed by the detector for three frames"),
        cell(5, 30, 125, 0.0, 2.2, 21, turn=0.0, note="passes close to cell 6"),
        cell(6, 30 + 21 + 14 + 3, 170, np.pi, 2.2, 14, turn=0.0, elong=0.1, note="passes close to cell 5"),
        cell(7, 0, 0, 0, 1.6, 17, start=21, parent=2, grow=(0.6, 8), note="daughter of cell 2"),
        cell(8, 0, 0, 0, 1.6, 17, start=21, parent=2, grow=(0.6, 8), note="daughter of cell 2"),
        cell(9, 140, 60, 1.35, 2.8, 16, turn=0.1, note="leaves through the bottom edge"),
    ]


def simulate():
    """(image (T,H,W) uint8, labels (T,H,W) int32, truth dict) for the demo recording."""
    rng = np.random.default_rng(SEED)
    cells = _cells(rng)
    by_id = {c["id"]: c for c in cells}
    labels = np.zeros((T, H, W), np.int32)
    image = np.zeros((T, H, W), np.uint8)
    truth_frames, rounded = [], {c["id"]: [] for c in cells}
    for t in range(T):
        for c in cells:
            if c["parent"] and t == c["start"]:            # daughters start beside the parent, splitting apart
                p, sgn = by_id[c["parent"]], (1 if c["id"] % 2 else -1)
                c["theta"] = 0.9 + (0 if sgn > 0 else np.pi)
                c["y"] = p["y"] + sgn * 8 * np.sin(0.9)
                c["x"] = p["x"] + sgn * 8 * np.cos(0.9)
            elif t > c["start"]:
                c["theta"] += rng.normal(0, c["turn"])
                for o in cells:                            # steer away from close neighbours
                    if o is c or not o["start"] <= t < o["end"]:
                        continue
                    dy, dx = c["y"] - o["y"], c["x"] - o["x"]
                    if np.hypot(dy, dx) < c["R"] + o["R"] + 2 and {c["id"], o["id"]} != {5, 6}:
                        c["theta"] = np.arctan2(dy, dx)
                c["y"] += c["v"] * np.sin(c["theta"])
                c["x"] += c["v"] * np.cos(c["theta"])
                if c["id"] not in (3, 9):                  # bounce off the field edge (3 enters, 9 leaves)
                    if not 22 < c["y"] < H - 22:
                        c["theta"] = -c["theta"]; c["y"] = min(max(c["y"], 22), H - 22)
                    if not 22 < c["x"] < W - 22:
                        c["theta"] = np.pi - c["theta"]; c["x"] = min(max(c["x"], 22), W - 22)
        frame_cells, masks = [], []
        for c in cells:
            if not c["start"] <= t < c["end"]:
                continue
            R, elong, is_round = c["R"], c["elong"], t in c["round"]
            if is_round:
                R, elong = R * 0.62, 0.02
            if c["grow"]:
                f0, n = c["grow"]
                R *= f0 + (1 - f0) * min(1.0, (t - c["start"]) / n)
            m = _cell_mask(c["y"], c["x"], R, elong, c["theta"], c["wob"])
            if not m.any():
                continue
            masks.append((c["id"], m, is_round))
        n_speck = int(rng.integers(0, 3))
        for _ in range(n_speck):                           # debris: below any min-area filter
            yy, xx = np.mgrid[:H, :W]
            sy, sx, r = rng.uniform(5, H - 5), rng.uniform(5, W - 5), rng.uniform(1.5, 3.0)
            masks.append((0, (yy - sy) ** 2 + (xx - sx) ** 2 <= r * r, False))
        order = rng.permutation(len(masks)) + 1            # detector IDs: shuffled every frame
        img = _background(rng)
        for (cid, m, is_round), lab in zip(masks, order):
            if cid and t in by_id[cid]["missing"]:
                _paint_cell(img, m, is_round, rng)         # visible in the image, missed by the detector
                continue
            labels[t][m] = lab
            _paint_cell(img, m, is_round, rng)
            frame_cells.append({"label": int(lab), "cell": int(cid)})
            if cid and is_round:
                rounded[cid].append(t)
        image[t] = np.clip(ndi.gaussian_filter(img, 0.8), 0, 255).astype(np.uint8)
        truth_frames.append(sorted(frame_cells, key=lambda d: d["label"]))
    truth = {
        "synthetic": True, "source": SOURCE, "note": NOTE, "um_per_px": UM_PER_PX, "dt_min": DT_MIN,
        "cells": [{"id": c["id"], "start": c["start"], "end": c["end"] - 1, "parent": c["parent"],
                   "rounded_frames": rounded[c["id"]], "missing_frames": c["missing"], "note": c["note"]}
                  for c in cells],
        "frames": truth_frames,
    }
    return image, labels, truth


def _background(rng):
    return 118 + ndi.gaussian_filter(rng.normal(0, 14, (H, W)), 1.2)


def _paint_cell(img, m, is_round, rng):
    """Phase-contrast-like look: dark textured body (bright when rounded), bright halo outside."""
    halo = ndi.binary_dilation(m, iterations=2) & ~m
    rim = m & ~ndi.binary_erosion(m, iterations=1)
    tex = ndi.gaussian_filter(rng.normal(0, 30, (H, W)), 1.5)
    img[halo] = 178
    img[m] = (165 if is_round else 92) + tex[m]
    img[rim] = 66


def ensure_demo(out: str) -> dict:
    """Write the demo files under ``out`` once per DEMO_VERSION; return {kind: absolute path}."""
    paths = {"image": os.path.join(out, NAME + ".ome.tif"), "sidecar": os.path.join(out, NAME + ".ome.json"),
             "detections": os.path.join(out, NAME + "_detections.npz"),
             "truth": os.path.join(out, NAME + "_truth.json")}
    stamp = os.path.join(out, "DEMO_VERSION.json")
    try:
        with open(stamp) as f:
            if json.load(f).get("version") == DEMO_VERSION and all(map(os.path.exists, paths.values())):
                return paths
    except (OSError, ValueError):
        pass
    import tifffile
    os.makedirs(out, exist_ok=True)
    image, labels, truth = simulate()
    tifffile.imwrite(paths["image"], image, ome=True, compression="zlib",
                     metadata={"axes": "TYX", "Name": "SYNTHETIC demo", "PhysicalSizeX": UM_PER_PX,
                               "PhysicalSizeY": UM_PER_PX, "TimeIncrement": DT_MIN, "TimeIncrementUnit": "min"})
    with open(paths["sidecar"], "w") as f:
        json.dump({"name": NAME, "um_per_px": UM_PER_PX, "time_interval_min": DT_MIN,
                   "synthetic": True, "source": SOURCE}, f, indent=1)
    np.savez_compressed(paths["detections"], labels=labels)
    with open(paths["truth"], "w") as f:
        json.dump(truth, f, separators=(",", ":"))
    with open(stamp, "w") as f:
        json.dump({"version": DEMO_VERSION, "synthetic": True, "source": SOURCE}, f)
    return paths
