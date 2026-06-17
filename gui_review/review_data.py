"""Data layer for the single-cell review GUI.

`discover_recordings` — find every recording under by_condition/.
`ReviewRenderer` — load recordings (single-channel DIC, LRU cached) and
hand back (frames, labels). `trajectory_bbox` — the cell's whole-track
bounding box (+margin) for the default zoom. `FlagStore` — persist which
recordings the reviewer flags as problematic, to a CSV.

Kept GUI-agnostic so the window stays thin and this is testable.
"""
from __future__ import annotations

import os
import csv
import glob
import threading

import numpy as np


def discover_recordings(root, exclude=frozenset()):
    """List recordings under `root` (= …/by_condition). Each:
    {label, condition, masks_path, video_path, excluded}."""
    recs = []
    for mp in sorted(glob.glob(os.path.join(
            root, "*", "*", "pipeline_results", "masks.npz"))):
        rec_dir = os.path.dirname(os.path.dirname(mp))
        label = os.path.basename(rec_dir)
        cond = os.path.basename(os.path.dirname(rec_dir))
        tif = next((os.path.join(rec_dir, f) for f in os.listdir(rec_dir)
                    if f.endswith(".ome.tif")), None)
        recs.append({"label": label, "condition": cond, "masks_path": mp,
                     "video_path": tif, "excluded": label in exclude})
    recs.sort(key=lambda r: (r["condition"], r["label"]))
    return recs


def trajectory_bbox(labels, margin):
    """(r0, r1, c0, c1) covering the cell across ALL frames, +margin,
    clamped to the image. Used as the default zoom so the cell stays in
    view the whole recording without per-frame re-centring jitter."""
    m = labels > 0
    H, W = labels.shape[1:]
    rr = np.any(m, axis=(0, 2))
    cc = np.any(m, axis=(0, 1))
    if not rr.any():
        return 0, H, 0, W
    r0, r1 = np.where(rr)[0][[0, -1]]
    c0, c1 = np.where(cc)[0][[0, -1]]
    return (max(0, r0 - margin), min(H, r1 + margin + 1),
            max(0, c0 - margin), min(W, c1 + margin + 1))


class ReviewRenderer:
    """LRU cache of (frames, labels) per recording. Loads SINGLE-CHANNEL
    (no channel args) — IC293 crops are DIC-only. The slow read happens
    outside the lock so a background prefetch never blocks the UI."""

    def __init__(self, max_cache=4):
        self._cache = {}
        self._order = []
        self._loading = set()
        self._lock = threading.Lock()
        self.max_cache = max_cache

    def _read(self, rec):
        from core.io import load_recording
        frames = (load_recording(rec["video_path"])["frames"]
                  if rec.get("video_path") else None)
        return frames, np.load(rec["masks_path"])["labels"]

    def get(self, rec):
        mp = rec["masks_path"]
        with self._lock:
            if mp in self._cache:
                self._order.remove(mp); self._order.append(mp)
                return self._cache[mp]
        data = self._read(rec)
        with self._lock:
            if mp not in self._cache:
                self._cache[mp] = data
                self._order.append(mp)
                while len(self._order) > self.max_cache:
                    self._cache.pop(self._order.pop(0), None)
            self._loading.discard(mp)
            return self._cache.get(mp, data)

    def prefetch(self, rec):
        if not rec:
            return
        mp = rec["masks_path"]
        with self._lock:
            if mp in self._cache or mp in self._loading:
                return
            self._loading.add(mp)
        threading.Thread(target=self.get, args=(rec,), daemon=True).start()


class FlagStore:
    """Which recordings the reviewer flagged as problematic, backed by CSV."""
    FIELDS = ["label", "condition", "flagged", "frame", "note", "timestamp"]

    def __init__(self, path):
        self.path = path
        self.flags = {}                      # label -> row dict
        if os.path.exists(path):
            self._load()

    def _load(self):
        with open(self.path) as f:
            for r in csv.DictReader(f):
                self.flags[r["label"]] = r

    def is_flagged(self, label):
        v = str(self.flags.get(label, {}).get("flagged", "")).lower()
        return v in ("1", "true", "yes")

    def toggle(self, label, condition, frame, timestamp, note=""):
        if self.is_flagged(label):
            self.flags.pop(label, None)
        else:
            self.flags[label] = {"label": label, "condition": condition,
                                 "flagged": "1", "frame": str(frame),
                                 "note": note, "timestamp": timestamp}
        self.save()
        return self.is_flagged(label)

    def count(self):
        return sum(1 for l in self.flags if self.is_flagged(l))

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore")
            w.writeheader()
            for lab in sorted(self.flags):
                w.writerow(self.flags[lab])
        os.replace(tmp, self.path)
