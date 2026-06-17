"""Programmatic QC sweep over every IC293 recording's masks.

Enumerates machine-detectable problems so manual review is reserved for
what genuinely can't be fixed in code. For each recording's selected
primary cell it reports:

  * missing frames (leading / internal / trailing) — cell absent
  * of those, how many are MERGE-recoverable (another label is present
    there = a fragmented track / ID-switch, fixable by merging — no
    re-detection needed) vs GENUINE misses (no label at all → need SAM2
    propagation or sensitive re-detect)
  * artifacts (extra objects) + whether selection was ambiguous
  * abrupt area jumps (>2.5× frame-to-frame = likely bad segmentation)
  * edge-truncation fraction (cell at the crop border → shape unreliable)

Writes compare/qc_diagnostic.json + prints a categorized summary.
Usage: conda run -n cellpose4 python scripts/ic293_qc_diagnostic.py
"""
import os
import sys
import glob
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

import numpy as np  # noqa: E402
from scripts.ic293_common import (  # noqa: E402
    RECORDINGS_ROOT, COMPARE_DIR, select_primary_cell, atomic_write_json)


def _present_frames(labels, cid):
    return np.where((labels == cid).any(axis=(1, 2)))[0]


def _runs(frames_missing, T):
    """Classify missing frame indices into leading/internal/trailing."""
    s = set(int(f) for f in frames_missing)
    lead = [f for f in s if all((g in s) for g in range(0, f + 1))]
    trail = [f for f in s if all((g in s) for g in range(f, T))]
    internal = sorted(s - set(lead) - set(trail))
    return sorted(lead), internal, sorted(trail)


def _touches_edge(mask2d, margin=0):
    if not mask2d.any():
        return False
    ys, xs = np.where(mask2d)
    H, W = mask2d.shape
    return bool(ys.min() <= margin or xs.min() <= margin
                or ys.max() >= H - 1 - margin or xs.max() >= W - 1 - margin)


def diagnose(mp):
    label = os.path.basename(os.path.dirname(os.path.dirname(mp)))
    cond = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(mp))))
    npz = np.load(mp)
    if "labels" not in npz.files:
        return {"label": label, "cond": cond, "error": "no labels array"}
    labels = npz["labels"]
    T = labels.shape[0]
    pid, cands, ambiguous = select_primary_cell(labels)
    n_objects = len(cands)
    if pid is None:
        return {"label": label, "cond": cond, "T": T, "n_objects": 0,
                "error": "no objects detected"}

    present = set(int(f) for f in _present_frames(labels, pid))
    missing = sorted(set(range(T)) - present)
    lead, internal, trail = _runs(missing, T)

    # Merge-recoverable: missing frames where ANOTHER label exists (a
    # fragmented track / ID switch — merging fixes it with no re-detect).
    other_ids = [c["id"] for c in cands if c["id"] != pid]
    merge_cover = set()
    for f in missing:
        if any((labels[f] == oid).any() for oid in other_ids):
            merge_cover.add(f)
    genuine = sorted(set(missing) - merge_cover)

    # Abrupt area jumps for the primary cell (segmentation glitches).
    areas = np.array([(labels[f] == pid).sum() for f in sorted(present)],
                     dtype=float)
    jumps = 0
    for i in range(1, len(areas)):
        if areas[i - 1] > 0 and areas[i] > 0:
            r = areas[i] / areas[i - 1]
            if r > 2.5 or r < 1 / 2.5:
                jumps += 1
    med_area = float(np.median(areas)) if areas.size else 0.0

    # Edge truncation among present frames.
    edge = sum(1 for f in present if _touches_edge(labels[f] == pid))

    return {"label": label, "cond": cond, "T": T, "n_objects": n_objects,
            "primary_id": int(pid), "ambiguous": bool(ambiguous),
            "present": len(present), "presence": round(len(present) / T, 3),
            "n_missing": len(missing), "lead": lead, "internal": internal,
            "trail": trail, "n_merge_recoverable": len(merge_cover),
            "n_genuine_missing": len(genuine), "genuine_frames": genuine,
            "n_artifacts": n_objects - 1, "area_jumps": jumps,
            "med_area_px": med_area,
            "edge_frames": edge, "edge_frac": round(edge / max(len(present), 1), 2)}


def main():
    paths = sorted(glob.glob(os.path.join(
        RECORDINGS_ROOT, "*", "*", "pipeline_results", "masks.npz")))
    rows = [diagnose(mp) for mp in paths]
    os.makedirs(COMPARE_DIR, exist_ok=True)
    atomic_write_json(os.path.join(COMPARE_DIR, "qc_diagnostic.json"), rows)

    ok = [r for r in rows if not r.get("error")]
    print(f"\n=== IC293 QC sweep: {len(rows)} recordings ===\n")

    def cat(pred):
        return sorted([r["label"] for r in ok if pred(r)])

    miss = [r for r in ok if r.get("n_missing", 0) > 0]
    merge = [r for r in ok if r.get("n_merge_recoverable", 0) > 0]
    genuine = [r for r in ok if r.get("n_genuine_missing", 0) > 0]
    arts = [r for r in ok if r.get("n_artifacts", 0) > 0]
    jumpy = [r for r in ok if r.get("area_jumps", 0) >= 2]
    edgey = [r for r in ok if r.get("edge_frac", 0) >= 0.3]
    amb = [r for r in ok if r.get("ambiguous")]
    clean = [r for r in ok if r.get("n_missing", 0) == 0
             and r.get("n_artifacts", 0) == 0 and r.get("area_jumps", 0) < 2
             and r.get("edge_frac", 0) < 0.3]

    print(f"CLEAN (no issues):                 {len(clean)}")
    print(f"missing-frame recordings:          {len(miss)}")
    print(f"  └ merge-recoverable (frag/ID):   {len(merge)}")
    print(f"  └ genuine misses (SAM2/redetect):{len(genuine)}")
    print(f"artifacts present (extra objects): {len(arts)}")
    print(f"  └ ambiguous selection:           {len(amb)}")
    print(f"abrupt area jumps (≥2):            {len(jumpy)}")
    print(f"heavy edge-truncation (≥30%):      {len(edgey)}")

    def show(title, rs, fields):
        if not rs:
            return
        print(f"\n── {title} ──")
        for r in sorted(rs, key=lambda x: (-x.get("n_missing", 0), x["label"])):
            print("  " + r["label"].ljust(18)
                  + "  ".join(f"{k}={r.get(k)}" for k in fields))

    show("MISSING FRAMES", miss,
         ["presence", "lead", "internal", "trail",
          "n_merge_recoverable", "n_genuine_missing"])
    show("ARTIFACTS", arts, ["n_artifacts", "ambiguous"])
    show("AREA JUMPS", jumpy, ["area_jumps", "med_area_px"])
    show("EDGE-TRUNCATED", edgey, ["edge_frac", "edge_frames", "T"])
    print(f"\nWrote {os.path.join(COMPARE_DIR, 'qc_diagnostic.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
