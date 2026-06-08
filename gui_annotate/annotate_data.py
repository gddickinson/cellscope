"""Data layer for the annotation GUI.

`LabelStore` — load / edit / save a cell-frame label CSV (the IC295
state_labels schema: idx, label, condition, label_dir, cid, frame,
rel_area, + shape features). `CropRenderer` — load recordings (LRU
cached, so memory stays bounded) and produce a fixed-size DIC window +
mask contour for one cell-frame.

Kept GUI-agnostic so the window code stays thin and this is unit-testable.
"""
from __future__ import annotations

import os
import csv
import threading
from collections import Counter

import numpy as np


# Class set (extensible). Single-key code → human label.
CLASS_LABELS = {"r": "rounded", "s": "spread", "u": "unsure"}


def fixed_window(img, cy, cx, size):
    """Centre a size×size window on (cy,cx); zero-pad past edges.
    Returns (window, valid_mask) — valid is False over the padding."""
    half = size // 2
    H, W = img.shape[:2]
    out = np.zeros((size, size), dtype=img.dtype)
    valid = np.zeros((size, size), dtype=bool)
    y0, x0 = cy - half, cx - half
    sy0, sy1 = max(0, y0), min(H, y0 + size)
    sx0, sx1 = max(0, x0), min(W, x0 + size)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    valid[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = True
    return out, valid


class LabelStore:
    """A cell-frame label table backed by a CSV."""

    def __init__(self, csv_path=None):
        self.path = None
        self.rows = []
        self.fieldnames = []
        if csv_path:
            self.load(csv_path)

    def load(self, path):
        with open(path) as f:
            rd = csv.DictReader(f)
            self.rows = list(rd)
            self.fieldnames = list(rd.fieldnames or [])
        if "label" not in self.fieldnames:
            self.fieldnames.insert(1, "label")
        for r in self.rows:
            r.setdefault("label", "")
        self.path = path

    def save(self, path=None):
        p = path or self.path
        if not p:
            raise ValueError("no path to save to")
        tmp = p + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames,
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, p)
        self.path = p

    def set_label(self, i, code):
        self.rows[i]["label"] = code

    def counts(self):
        return Counter((r.get("label") or "—") for r in self.rows)

    def recordings_root(self):
        """`…/by_condition` guessed from the CSV location
        (…/ic295_analysis/state_labels/labels.csv)."""
        if not self.path:
            return None
        analysis = os.path.dirname(os.path.dirname(os.path.abspath(self.path)))
        cand = os.path.join(analysis, "by_condition")
        return cand if os.path.isdir(cand) else None


class CropRenderer:
    """Render a fixed-size DIC window + contour for a cell-frame row.

    Recordings are loaded lazily and LRU-evicted (`max_cache`) so a big
    label set spanning many recordings doesn't exhaust memory.
    """

    def __init__(self, recordings_root, margin=18, max_cache=4):
        self.root = recordings_root
        self.margin = margin
        self.max_cache = max_cache
        self._cache = {}     # mp -> (frames, labels)
        self._order = []     # LRU, no duplicates
        self._lock = threading.Lock()
        self._loading = set()

    def _masks_path(self, row):
        return os.path.join(self.root, row["condition"], row["label_dir"],
                            "pipeline_results", "masks.npz")

    def _read(self, mp):
        from core.io import load_recording
        d = os.path.dirname(os.path.dirname(mp))
        tif = next((os.path.join(d, f) for f in os.listdir(d)
                    if f.endswith(".ome.tif")), None)
        frames = (load_recording(tif, dic_channel=1, fluo_channel=0)["frames"]
                  if tif else None)
        return frames, np.load(mp)["labels"]

    def _load(self, mp):
        """Thread-safe LRU load. The slow file read happens OUTSIDE the
        lock so a background prefetch never blocks the UI thread."""
        with self._lock:
            if mp in self._cache:
                self._order.remove(mp); self._order.append(mp)
                return self._cache[mp]
        data = self._read(mp)
        with self._lock:
            if mp not in self._cache:
                self._cache[mp] = data
                self._order.append(mp)
                while len(self._order) > self.max_cache:
                    self._cache.pop(self._order.pop(0), None)
            self._loading.discard(mp)
            return self._cache.get(mp, data)

    def prefetch(self, row):
        """Warm the cache for `row`'s recording on a background thread."""
        if row is None:
            return
        mp = self._masks_path(row)
        with self._lock:
            if mp in self._cache or mp in self._loading:
                return
            self._loading.add(mp)
        threading.Thread(target=self._load, args=(mp,), daemon=True).start()

    def fixed_size(self, rows):
        """One uniform window size big enough for the largest cell,
        estimated from recorded px areas (no mask load)."""
        areas = [float(r["area"]) for r in rows
                 if r.get("area") not in (None, "", "nan")]
        if not areas:
            return 256
        return int(np.clip(2.4 * np.sqrt(max(areas)) + 2 * self.margin,
                           96, 420))

    def render(self, row, size):
        """Return (window_img, valid_mask, contour_mask, edge_touch) for one
        row, or (None, None, None, False) if it can't be drawn. `edge_touch`
        is measured on the FULL-FRAME mask (before the window crop), so it
        reflects the IMAGE border, not the crop border."""
        try:
            frames, labels = self._load(self._masks_path(row))
            fi = int(row["frame"])
            m = labels[fi] == int(row["cid"])
            rr = np.any(m, axis=1)
            cc = np.any(m, axis=0)
            if not rr.any():
                return None, None, None, False
            from core.edge_filter import mask_touches_edge
            from core.cell_state import DEFAULT_THRESHOLDS
            edge = mask_touches_edge(m, DEFAULT_THRESHOLDS["edge_margin_px"])
            r0, r1 = np.where(rr)[0][[0, -1]]
            c0, c1 = np.where(cc)[0][[0, -1]]
            cy, cx = (r0 + r1) // 2, (c0 + c1) // 2
            base = frames[fi] if frames is not None else m.astype(np.uint16)
            imgw, valid = fixed_window(base, cy, cx, size)
            mw, _ = fixed_window(m.astype(np.uint8), cy, cx, size)
            return imgw, valid, mw, edge
        except Exception:
            return None, None, None, False
