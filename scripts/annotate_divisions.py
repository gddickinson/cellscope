"""Post-hoc cell-division annotator — runner script.

Thin wrapper around `core.division_annotator.find_candidates`. For
each recording it:

  1. Loads pipeline_results/masks.npz (labels stack)
  2. Calls core.division_annotator.find_candidates
  3. Writes divisions.json + summary
  4. Renders per-candidate 9-frame strips + track-area timeseries

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

from core.division_annotator import find_candidates
from core.cell_state import STATE_BALLED

OUT_ROOT = "results/divisions"

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
    "#9a6324", "#e6beff", "#aaffc3", "#808000", "#ffd8b1",
]


# ---------------------------------------------------------------------
# TIFF page reader (robust to tifffile + numpy 2.0 quirks)
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
# Visualisation
# ---------------------------------------------------------------------
def render_candidate_strip(rec_dir, candidate, tracks, tif_path,
                            out_path, dic_channel=1):
    f = candidate["frame"]
    peak_frame = candidate.get("peak_frame", f - 1)
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

    # Window: must span pre-mitotic context → split → daughter's
    # actual appearance + a few frames where both cells are clearly
    # separated. Daughters often stay in contact for several frames
    # after the parent's mask halves (the tracker sees them as one
    # blob), so we MUST extend past `daughter_first_frame + 3` to
    # show two distinct cells.
    daughter_first = candidate.get("daughter_first_frame", f)
    f_start = max(0, f - 2)
    # End at max(split + 4, daughter_first + 3) — whichever is later
    f_end = max(f + 4, daughter_first + 3)
    f_end = min(n_frames - 1, f_end)
    show_frames = list(range(f_start, f_end + 1))
    # Cap at 12 panels — drop pre-split frames first
    if len(show_frames) > 12:
        show_frames = show_frames[-12:]

    fig, axes = plt.subplots(1, len(show_frames),
                              figsize=(2.4 * len(show_frames), 3.4),
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
            ax.text(0.5, 1.02, "▼ SPLIT ▼", color="white",
                    transform=ax.transAxes, fontsize=10,
                    fontweight="bold", va="bottom", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#d62728", alpha=0.9,
                              edgecolor="none"))
        elif fi == peak_frame:
            ax.text(0.5, 1.02, "● PEAK ●", color="white",
                    transform=ax.transAxes, fontsize=9,
                    fontweight="bold", va="bottom", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#ff7f0e", alpha=0.9,
                              edgecolor="none"))

    fig.suptitle(
        f"{candidate.get('_recording', '')}  ·  "
        f"T{parent} (red) → T{daughter} (cyan)  at F{f}  "
        f"(peak F{peak_frame}, score {candidate['score']:.2f})\n"
        f"swell {candidate.get('swelling_ratio', 1):.2f}× · "
        f"mass {candidate['mass_ratio']:.2f} · "
        f"daughter persists {candidate['daughter_persistence_frames']} fr · "
        f"pre-balled {candidate['pre_split_balled_frac']:.0%} · "
        f"dist {candidate['distance_px']:.0f}px",
        fontsize=10, y=1.06)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def render_track_area_timeseries(name, tracks, candidates, out_path):
    """Per-track area-over-time, with balled-state dots and candidate
    markers. Lets reviewer eyeball missed divisions."""
    if not tracks:
        return
    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    n_frames = max(t["last_frame"] for t in tracks.values()) + 1
    for tid in sorted(tracks.keys()):
        col = PALETTE[(tid - 1) % len(PALETTE)]
        t = tracks[tid]
        frames = sorted(t["area"].keys())
        areas = [t["area"][f] for f in frames]
        ax.plot(frames, areas, color=col, lw=1.4, alpha=0.85,
                label=f"T{tid}")
        balled_frames = [f for f in frames
                          if t["state"].get(f) == STATE_BALLED]
        balled_areas = [t["area"][f] for f in balled_frames]
        ax.scatter(balled_frames, balled_areas, color=col, s=20,
                    edgecolor="black", linewidth=0.4, zorder=5)
    for c in candidates:
        ax.axvline(c["frame"], color="red", ls="--", alpha=0.4, lw=0.8)
        ax.annotate(
            f"T{c['parent_track']}→T{c['daughter_track']}\nscore "
            f"{c['score']:.2f}",
            xy=(c["frame"], c.get("area_peak", c.get("area_parent_at_split", 0))),
            xytext=(c["frame"] + 1,
                    c.get("area_peak", 1) * 1.05),
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


# ---------------------------------------------------------------------
# Per-recording driver
# ---------------------------------------------------------------------
def process_recording(rec_dir):
    name = os.path.basename(rec_dir.rstrip("/"))
    masks_path = os.path.join(rec_dir, "pipeline_results", "masks.npz")
    if not os.path.exists(masks_path):
        print(f"  [{name}] no masks.npz — skipping")
        return None

    labels = np.load(masks_path)["labels"]
    tif_candidates = [os.path.join(rec_dir, f)
                       for f in os.listdir(rec_dir)
                       if f.endswith(".ome.tif")
                          or f.endswith(".tif")]
    tif = tif_candidates[0] if tif_candidates else None

    um_per_px = 1.0
    for f in os.listdir(rec_dir):
        if f.endswith(".ome.json") or f.endswith(".json"):
            if f.endswith(".cellscope"):
                continue
            try:
                with open(os.path.join(rec_dir, f)) as jf:
                    meta = json.load(jf)
                um_per_px = float(meta.get("um_per_px", 1.0))
                break
            except Exception:
                pass

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
            "candidates": candidates,
        }, jf, indent=2)

    render_track_area_timeseries(
        name, tracks, candidates,
        os.path.join(out_dir, "track_areas.png"))

    if tif and candidates:
        for i, cand in enumerate(candidates[:12]):
            cand["_recording"] = name
            png = os.path.join(out_dir, f"candidate_{i:02d}.png")
            try:
                render_candidate_strip(rec_dir, cand, tracks, tif, png)
                print(
                    f"    rendered {i:02d}: T{cand['parent_track']}→"
                    f"T{cand['daughter_track']} F{cand['frame']} "
                    f"score={cand['score']:.2f}")
            except Exception as e:
                print(f"    skipped {i}: {e}")

    return {"name": name, "n_candidates": len(candidates),
            "candidates": candidates}


def write_summary(summary):
    lines = [
        "# Division candidate scan (v2)",
        "",
        f"_Generated on {os.popen('date').read().strip()}_",
        "",
        "## Recordings",
        "",
        "| Recording | Candidates | Top score |",
        "|---|---:|---:|",
    ]
    for rec in summary:
        top = max((c["score"] for c in rec["candidates"]), default=0)
        lines.append(
            f"| {rec['name']} | {rec['n_candidates']} | {top:.2f} |")
    lines.append("")
    lines.append("## Per-candidate review")
    lines.append("")
    for rec in summary:
        lines.append(f"### {rec['name']}")
        lines.append("")
        for i, c in enumerate(rec["candidates"][:12]):
            lines.append(
                f"**Candidate {i}** — T{c['parent_track']} → "
                f"T{c['daughter_track']} at F{c['frame']} "
                f"(peak F{c.get('peak_frame', '?')}) · "
                f"score {c['score']:.2f} · "
                f"swell {c.get('swelling_ratio', 1):.2f}× · "
                f"mass {c['mass_ratio']:.2f} · "
                f"persists {c['daughter_persistence_frames']} fr · "
                f"pre-balled {c['pre_split_balled_frac']:.0%} · "
                f"dist {c['distance_px']:.0f}px")
            png_rel = f"{rec['name']}/candidate_{i:02d}.png"
            png_abs = os.path.join(OUT_ROOT, png_rel)
            if os.path.exists(png_abs):
                lines.append("")
                lines.append(f"![{png_rel}]({png_rel})")
            lines.append("")
    with open(os.path.join(OUT_ROOT, "summary.md"), "w") as f:
        f.write("\n".join(lines))


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

    write_summary(summary)
    total = sum(r["n_candidates"] for r in summary)
    print(f"\nWrote {OUT_ROOT}/summary.md")
    print(f"  {total} candidate(s) across {len(summary)} recording(s)")


if __name__ == "__main__":
    main()
