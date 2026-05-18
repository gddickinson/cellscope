"""Post-hoc division annotator.

Scans existing tracked output (`pipeline_results/masks.npz`) for
candidate cell-division events using two signals:

  1. **Area pattern in an existing track** — for each tracked cell,
     find frames where the area approximately HALVES from frame F-Δ
     to frame F. (The Hungarian tracker assigns one daughter to the
     parent's ID at division; the parent's mask therefore appears to
     shrink suddenly.)

  2. **New track appears nearby** — at the same frame F (±2), a new
     track ID first appears within R pixels of the shrunk track's
     centroid. The new track is the OTHER daughter.

For each candidate we also record:

  - cell state (balled / attached / transitional) in the frames
    BEFORE the split, computed from circularity + solidity via
    core.cell_state.classify_state. Division typically follows
    mitotic rounding, so pre-split balled state is a strong prior.
  - mass conservation: total area of (parent_post + daughter) at F
    relative to parent at F-Δ. ~1.0 = clean split.
  - score = combination of (1/distance) × balled_prior × mass_score.

Outputs (per recording, under `results/divisions/<recording>/`):

  - divisions.json — list of {frame, parent_track, daughter_track,
    distance, area_before, area_parent_after, area_daughter, score,
    pre_state}
  - per-candidate 5-frame strip PNG (F-2 .. F+2)
  - summary.md tabulating candidates

Usage:

  conda run -n cellpose4 python scripts/annotate_divisions.py \\
      data/ic295_gt_full/Pos7_WT \\
      data/ic295_gt_full/Pos20_KO \\
      data/ic295_gt_full/Pos30_GOF \\
      data/ic295_gt_full/Pos39_OT
"""
import os
import sys
import json
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cell_state import (
    shape_metrics_for_mask, classify_state,
    STATE_BALLED, STATE_TRANSITIONAL,
)

OUT_ROOT = "results/divisions"

# Tunable thresholds
AREA_RATIO_HALF = 0.65      # area[F] / area[F-Δ] ≤ this → "halved"
AREA_RATIO_MAX = 1.4        # parent total mass must not change wildly
LOOKBACK_DELTA = 2          # check F-1 and F-2
MAX_PAIR_DISTANCE_UM = 50   # daughter centroid must be within this of
                             # parent centroid (60 px at 0.65 µm/px ~ 39 µm)
DAUGHTER_FRAME_WINDOW = 2   # new track must appear within ±this frames
MIN_TRACK_LENGTH = 3        # ignore tiny tracks both as parent + as
                             # daughter (avoids spurious matches)
PRE_STATE_LOOKBACK = 3      # how many frames before split to inspect state

# Per-cell palette for visualisations
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
    "#9a6324", "#e6beff", "#aaffc3", "#808000", "#ffd8b1",
]


# ---------------------------------------------------------------------
# Tiff-page reader robust to numpy 2.0 + big-endian quirks
# ---------------------------------------------------------------------
def _read_page_raw(page):
    fh = page.parent.filehandle
    dtype = page.dtype
    h, w = page.shape[-2:]
    offsets = page.dataoffsets
    bytecounts = page.databytecounts
    if not offsets:
        return page.asarray()
    buf = bytearray()
    for off, n in zip(offsets, bytecounts):
        fh.seek(off)
        buf.extend(fh.read(n))
    base = np.dtype(dtype.kind + str(dtype.itemsize))
    arr = np.frombuffer(bytes(buf), dtype=base).reshape(h, w)
    if dtype.byteorder == ">":
        arr = arr.byteswap()
    return arr.copy()


def _load_frame(tif_path, frame_idx, channel=None):
    with tifffile.TiffFile(tif_path) as t:
        series = t.series[0]
        shape = series.shape
        if len(shape) == 4:
            n_ch = shape[1]
            page_idx = frame_idx * n_ch + (channel or 0)
        elif len(shape) == 3:
            page_idx = frame_idx
        else:
            page_idx = 0
        page = t.pages[page_idx]
        try:
            return page.asarray()
        except AttributeError:
            return _read_page_raw(page)


# ---------------------------------------------------------------------
# Per-track scan
# ---------------------------------------------------------------------
def _track_present_frames(labels, track_id):
    return [f for f in range(len(labels))
            if (labels[f] == track_id).any()]


def _centroid(label_frame, track_id):
    mask = label_frame == track_id
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def _track_table(labels):
    """Per-track summary: first_frame, last_frame, frames_present, area
    per frame, centroid per frame, state per frame."""
    n_frames = len(labels)
    track_ids = np.unique(labels)
    track_ids = track_ids[track_ids > 0]
    out = {}
    for tid in track_ids:
        present = []
        areas = {}
        centroids = {}
        states = {}
        for f in range(n_frames):
            m = labels[f] == tid
            if not m.any():
                continue
            present.append(f)
            ys, xs = np.where(m)
            areas[f] = int(m.sum())
            centroids[f] = (float(xs.mean()), float(ys.mean()))
            metrics = shape_metrics_for_mask(m)
            states[f] = classify_state(metrics)
        if not present:
            continue
        out[int(tid)] = {
            "first_frame": present[0],
            "last_frame": present[-1],
            "frames_present": present,
            "area": areas,
            "centroid": centroids,
            "state": states,
        }
    return out


# ---------------------------------------------------------------------
# Candidate finding
# ---------------------------------------------------------------------
def find_candidates(labels, um_per_px=1.0):
    tracks = _track_table(labels)
    if not tracks:
        return [], tracks

    max_pair_px = MAX_PAIR_DISTANCE_UM / max(um_per_px, 0.01)
    candidates = []

    # Build index: which tracks first-appear at each frame?
    first_at_frame = {}
    for tid, t in tracks.items():
        first_at_frame.setdefault(t["first_frame"], []).append(tid)

    for tid, t in tracks.items():
        if len(t["frames_present"]) < MIN_TRACK_LENGTH:
            continue
        present = t["frames_present"]
        for i in range(LOOKBACK_DELTA, len(present)):
            f = present[i]
            for delta in range(1, LOOKBACK_DELTA + 1):
                if i - delta < 0:
                    break
                f_prev = present[i - delta]
                a_now = t["area"][f]
                a_prev = t["area"][f_prev]
                if a_prev <= 0:
                    continue
                ratio = a_now / a_prev
                if ratio > AREA_RATIO_HALF:
                    continue
                # Possible split — look for a new track appearing near
                # the parent's centroid within the window [f-W, f+W]
                cx, cy = t["centroid"][f]
                for f_new in range(max(0, f - DAUGHTER_FRAME_WINDOW),
                                    f + DAUGHTER_FRAME_WINDOW + 1):
                    for new_tid in first_at_frame.get(f_new, []):
                        if new_tid == tid:
                            continue
                        nt = tracks[new_tid]
                        if len(nt["frames_present"]) < MIN_TRACK_LENGTH:
                            continue
                        if f_new not in nt["centroid"]:
                            continue
                        nx, ny = nt["centroid"][f_new]
                        dist_px = float(np.hypot(nx - cx, ny - cy))
                        if dist_px > max_pair_px:
                            continue
                        # Mass-conservation check
                        a_daughter = nt["area"][f_new]
                        mass_ratio = (a_now + a_daughter) / a_prev
                        # Pre-split state — look at last few frames of
                        # parent BEFORE the split
                        pre_states = []
                        for f_pre in present[max(0, i - PRE_STATE_LOOKBACK):i]:
                            pre_states.append(t["state"][f_pre])
                        balled = sum(1 for s in pre_states
                                      if s in (STATE_BALLED,
                                                STATE_TRANSITIONAL))
                        pre_balled = (balled / max(len(pre_states), 1))
                        # Score (higher = better)
                        prox = 1.0 / (1.0 + dist_px / max_pair_px)
                        mass_score = (
                            1.0 if 0.7 <= mass_ratio <= 1.3 else
                            max(0.0, 1.0 - abs(mass_ratio - 1.0)))
                        score = prox * (0.5 + 0.5 * pre_balled) * mass_score
                        # Daughter persistence — count consecutive
                        # frames after first appearance. A 1-frame
                        # daughter is likely a mask flicker, not a
                        # real division.
                        d_frames = nt["frames_present"]
                        persist = 1
                        for j in range(1, len(d_frames)):
                            if d_frames[j] == d_frames[j - 1] + 1:
                                persist += 1
                            else:
                                break
                        # Downweight ephemeral daughters
                        persist_score = min(1.0, persist / 5.0)
                        score *= (0.3 + 0.7 * persist_score)
                        candidates.append({
                            "frame": int(f),
                            "frame_prev": int(f_prev),
                            "parent_track": int(tid),
                            "daughter_track": int(new_tid),
                            "daughter_first_frame": int(f_new),
                            "daughter_persistence_frames": int(persist),
                            "distance_px": dist_px,
                            "area_before": int(a_prev),
                            "area_parent_after": int(a_now),
                            "area_daughter": int(a_daughter),
                            "mass_ratio": float(mass_ratio),
                            "area_ratio_half": float(ratio),
                            "pre_split_states": pre_states,
                            "pre_split_balled_frac": float(pre_balled),
                            "score": float(score),
                            "parent_centroid": (cx, cy),
                            "daughter_centroid": (nx, ny),
                        })

    # Deduplicate: keep highest-score entry per (parent, daughter)
    seen = {}
    for c in candidates:
        key = (c["parent_track"], c["daughter_track"])
        if key not in seen or c["score"] > seen[key]["score"]:
            seen[key] = c
    deduped = sorted(seen.values(), key=lambda c: -c["score"])
    return deduped, tracks


# ---------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------
def render_candidate_strip(rec_dir, candidate, tracks, tif_path,
                            out_path, dic_channel=1):
    f = candidate["frame"]
    parent = candidate["parent_track"]
    daughter = candidate["daughter_track"]
    cx, cy = candidate["parent_centroid"]
    nx, ny = candidate["daughter_centroid"]
    cx0 = (cx + nx) / 2
    cy0 = (cy + ny) / 2

    masks = np.load(os.path.join(rec_dir, "pipeline_results",
                                  "masks.npz"))["labels"]
    n_frames = len(masks)
    H, W = masks.shape[1:]
    win = 200
    y0 = max(0, int(cy0 - win))
    y1 = min(H, int(cy0 + win))
    x0 = max(0, int(cx0 - win))
    x1 = min(W, int(cx0 + win))

    # Show 7 frames: F-3..F+3 (or slide to fit boundaries)
    n_show = 7
    half = n_show // 2
    f_start = max(0, f - half)
    f_end = min(n_frames - 1, f_start + n_show - 1)
    f_start = max(0, f_end - n_show + 1)
    show_frames = list(range(f_start, f_end + 1))

    fig, axes = plt.subplots(1, len(show_frames),
                              figsize=(2.4 * len(show_frames), 3.2),
                              dpi=130)
    if len(show_frames) == 1:
        axes = [axes]
    for ax, fi in zip(axes, show_frames):
        try:
            img = _load_frame(tif_path, fi, channel=dic_channel)
        except Exception:
            img = np.zeros((H, W), dtype=np.uint8)
        crop = img[y0:y1, x0:x1]
        ax.imshow(crop, cmap="gray",
                   vmin=np.percentile(crop, 1),
                   vmax=np.percentile(crop, 99))
        labels_crop = masks[fi][y0:y1, x0:x1]
        for tid, col in [(parent, "#e6194b"), (daughter, "#46f0f0")]:
            mask = labels_crop == tid
            if not mask.any():
                continue
            for cnt in measure.find_contours(mask.astype(float), 0.5):
                ax.plot(cnt[:, 1], cnt[:, 0], color=col, lw=1.8)
        # State + area annotation
        state_p = tracks.get(parent, {}).get("state", {}).get(fi, "?")
        area_p = tracks.get(parent, {}).get("area", {}).get(fi, 0)
        state_d = tracks.get(daughter, {}).get("state", {}).get(fi, "-")
        area_d = tracks.get(daughter, {}).get("area", {}).get(fi, 0)
        if area_d > 0:
            label_txt = (f"F{fi}  P:{area_p}({state_p[:3]})\n"
                         f"          D:{area_d}({state_d[:3]})")
        else:
            label_txt = f"F{fi}  P:{area_p}({state_p[:3]})"
        ax.set_title(label_txt, fontsize=9, loc="left")
        ax.set_axis_off()
        if fi == f:
            ax.text(0.5, 1.02, "▼ DIVISION ▼", color="white",
                    transform=ax.transAxes, fontsize=10,
                    fontweight="bold", va="bottom", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#d62728", alpha=0.9,
                              edgecolor="none"))

    fig.suptitle(
        f"{candidate.get('_recording', '')} · "
        f"parent T{parent} (red) → daughter T{daughter} (cyan) at F{f} "
        f"  ·  dist={candidate['distance_px']:.0f} px, "
        f"mass={candidate['mass_ratio']:.2f}, "
        f"pre-balled={candidate['pre_split_balled_frac']:.0%}, "
        f"score={candidate['score']:.2f}",
        fontsize=10, y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------
# Per-recording driver
# ---------------------------------------------------------------------
def render_track_area_timeseries(name, tracks, candidates, out_path):
    """Plot area over time for every track. Sudden drops are
    candidate division events. Detected candidates are marked with
    arrows so the reviewer can spot misses by eye."""
    if not tracks:
        return
    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    n_frames = max(t["last_frame"] for t in tracks.values()) + 1
    sorted_tids = sorted(tracks.keys())
    for tid in sorted_tids:
        col = PALETTE[(tid - 1) % len(PALETTE)]
        t = tracks[tid]
        frames = sorted(t["area"].keys())
        areas = [t["area"][f] for f in frames]
        ax.plot(frames, areas, color=col, lw=1.4, alpha=0.85,
                label=f"T{tid}")
        # Mark balled frames with a dot
        balled_frames = [f for f in frames
                          if t["state"].get(f) == STATE_BALLED]
        balled_areas = [t["area"][f] for f in balled_frames]
        ax.scatter(balled_frames, balled_areas, color=col, s=20,
                    edgecolor="black", linewidth=0.4, zorder=5)
    # Mark candidates
    for c in candidates:
        ax.axvline(c["frame"], color="red", ls="--", alpha=0.4, lw=0.8)
        ax.annotate(
            f"T{c['parent_track']}→T{c['daughter_track']}\nscore={c['score']:.2f}",
            xy=(c["frame"], c["area_before"]),
            xytext=(c["frame"] + 1, c["area_before"] * 1.1),
            fontsize=8, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Track area (px)")
    ax.set_title(
        f"{name} — per-track area over time\n"
        "filled dots = balled state · red dashed = division candidate",
        fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=8, ncol=1)
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, n_frames)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def process_recording(rec_dir):
    name = os.path.basename(rec_dir.rstrip("/"))
    masks_path = os.path.join(rec_dir, "pipeline_results", "masks.npz")
    if not os.path.exists(masks_path):
        print(f"  [{name}] no masks.npz — skipping")
        return None

    labels = np.load(masks_path)["labels"]
    # Find the recording TIFF (symlink target)
    tif_candidates = [os.path.join(rec_dir, f)
                       for f in os.listdir(rec_dir)
                       if f.endswith(".ome.tif")]
    tif = tif_candidates[0] if tif_candidates else None

    # Pixel size — try the .ome.json sidecar
    um_per_px = 1.0
    for f in os.listdir(rec_dir):
        if f.endswith(".ome.json"):
            try:
                with open(os.path.join(rec_dir, f)) as jf:
                    meta = json.load(jf)
                um_per_px = float(meta.get("um_per_px", 1.0))
            except Exception:
                pass
            break

    print(f"\n[{name}] labels={labels.shape} n_tracks="
          f"{int(labels.max())} um_per_px={um_per_px}")
    candidates, tracks = find_candidates(labels, um_per_px=um_per_px)
    print(f"  → {len(candidates)} division candidate(s)")

    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "divisions.json"), "w") as jf:
        json.dump({
            "recording": name,
            "n_candidates": len(candidates),
            "thresholds": {
                "area_ratio_half": AREA_RATIO_HALF,
                "max_pair_distance_um": MAX_PAIR_DISTANCE_UM,
                "lookback_delta": LOOKBACK_DELTA,
                "daughter_frame_window": DAUGHTER_FRAME_WINDOW,
            },
            "candidates": candidates,
        }, jf, indent=2)

    # Per-recording track-area-over-time diagnostic (so the user can
    # eyeball whether any divisions were missed)
    render_track_area_timeseries(
        name, tracks, candidates,
        os.path.join(out_dir, "track_areas.png"))

    # Render visualisations for top candidates (max 12)
    if tif and candidates:
        for i, cand in enumerate(candidates[:12]):
            cand["_recording"] = name
            png = os.path.join(out_dir, f"candidate_{i:02d}.png")
            try:
                render_candidate_strip(rec_dir, cand, tracks, tif, png)
                print(f"    rendered candidate_{i:02d}: T{cand['parent_track']}→T{cand['daughter_track']} F{cand['frame']} score={cand['score']:.2f}")
            except Exception as e:
                print(f"    skipped candidate_{i}: {e}")

    return {"name": name, "n_candidates": len(candidates),
            "candidates": candidates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    summary = []
    for rec in args.recordings:
        result = process_recording(rec)
        if result is not None:
            summary.append(result)

    # Write summary index
    lines = [
        "# Division candidate scan",
        "",
        f"_Generated on {os.popen('date').read().strip()}_",
        "",
        "## Recordings",
        "",
        "| Recording | Candidates | Top-score |",
        "|---|---:|---:|",
    ]
    for rec in summary:
        top = max((c["score"] for c in rec["candidates"]), default=0)
        lines.append(
            f"| {rec['name']} | {rec['n_candidates']} | {top:.2f} |")
    lines.append("")
    lines.append(
        "## Thresholds")
    lines.append(
        f"- Area ratio for 'halved': ≤ **{AREA_RATIO_HALF}** of prev frame")
    lines.append(
        f"- Max pair distance: **{MAX_PAIR_DISTANCE_UM} µm** "
        f"(parent ↔ daughter centroid)")
    lines.append(
        f"- Daughter must first appear within ±{DAUGHTER_FRAME_WINDOW} "
        "frames of parent's area-halving")
    lines.append(
        f"- Pre-split state checked over last {PRE_STATE_LOOKBACK} frames "
        "(balled / transitional → division prior)")
    lines.append("")
    lines.append("## Per-candidate review")
    lines.append("")
    for rec in summary:
        lines.append(f"### {rec['name']}")
        lines.append("")
        for i, c in enumerate(rec["candidates"][:8]):
            lines.append(
                f"**Candidate {i}** — parent T{c['parent_track']} → "
                f"daughter T{c['daughter_track']} at F{c['frame']}, "
                f"score={c['score']:.2f}, "
                f"distance={c['distance_px']:.0f}px, "
                f"mass-ratio={c['mass_ratio']:.2f}, "
                f"pre-balled={c['pre_split_balled_frac']:.0%}")
            png_rel = f"{rec['name']}/candidate_{i:02d}.png"
            png_abs = os.path.join(OUT_ROOT, png_rel)
            if os.path.exists(png_abs):
                lines.append(f"")
                lines.append(f"![{png_rel}]({png_rel})")
            lines.append("")
    with open(os.path.join(OUT_ROOT, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {OUT_ROOT}/summary.md")
    print(f"  {sum(r['n_candidates'] for r in summary)} candidates "
          f"across {len(summary)} recording(s)")


if __name__ == "__main__":
    main()
