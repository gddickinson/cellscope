"""Per-recording quality dashboard.

Given a pipeline result (frames + masks/labels + tracks), write a
single self-contained HTML page summarising:

  - Detection rate timeseries
  - Per-track presence heatmap
  - Speed distribution + outlier frames
  - 3 sample frames at 0/50/100% with overlay
  - Suspicious frames flagged by simple heuristics

Use directly:
    from output.quality_report import write_quality_report
    write_quality_report(out_html_path, frames=frames,
                          masks=masks, tracks=tracks)

Or as a one-shot from a cache .npz:
    python -m output.quality_report cache.npz report.html
"""
import base64
import io
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _png64(fig):
    """Render a matplotlib figure to a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.read()).decode("ascii"))


def _imnorm(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    return np.clip((img - p1) / max(p99 - p1, 1e-6), 0, 1)


def _detection_rate_chart(masks):
    """Line chart: per-frame mask presence."""
    n = len(masks)
    rate = np.array([1.0 if m.any() else 0.0 for m in masks])
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.fill_between(np.arange(n), rate, color="#5e9ed6", alpha=0.5,
                    step="mid")
    ax.plot(np.arange(n), rate, color="#3b6cad", linewidth=1, drawstyle="steps-mid")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, n - 1)
    ax.set_xlabel("Frame")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["empty", "detected"])
    ax.set_title(f"Detection: {int(rate.sum())}/{n} frames "
                 f"({rate.mean() * 100:.0f}%)")
    ax.grid(alpha=0.3, axis="x")
    return _png64(fig)


def _track_presence_heatmap(tracks, n_frames):
    """Heatmap: rows=tracks, cols=frames, colored by mask presence."""
    if not tracks:
        return None
    grid = np.zeros((len(tracks), n_frames), dtype=float)
    for ti, t in enumerate(tracks):
        s = t.get("stack")
        if s is None or not s.any():
            continue
        grid[ti] = s.any(axis=(1, 2)).astype(float)
    fig_h = max(2.0, 0.35 * len(tracks) + 1.0)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.imshow(grid, aspect="auto", cmap="Greens",
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(tracks)))
    ax.set_yticklabels([f"Cell {i + 1}" for i in range(len(tracks))])
    ax.set_xlabel("Frame")
    ax.set_title("Per-track presence (green = mask present)")
    return _png64(fig)


def _speed_distribution(tracks, um_per_px, dt_min):
    """Speed histogram + per-track timeseries panel."""
    if not tracks:
        return None
    from core.tracking import extract_centroids
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(tracks), 10)))
    all_speeds = []
    for ti, t in enumerate(tracks):
        s = t.get("stack")
        if s is None or not s.any():
            continue
        cents = extract_centroids(s)
        valid = ~np.isnan(cents[:, 0])
        if valid.sum() < 2:
            continue
        d = np.diff(cents, axis=0) * um_per_px
        speeds = np.full(len(cents), np.nan)
        for i in range(1, len(cents)):
            if valid[i] and valid[i - 1]:
                speeds[i] = float(np.linalg.norm(d[i - 1]) / dt_min)
        ts = np.arange(len(cents)) * dt_min
        axes[0].plot(ts, speeds, "-", color=cmap[ti % 10],
                     linewidth=1.2, alpha=0.9, label=f"Cell {ti + 1}")
        all_speeds.extend(speeds[~np.isnan(speeds)].tolist())
    axes[0].set_xlabel("Time (min)"); axes[0].set_ylabel("Speed (µm/min)")
    axes[0].set_title("Speed timeseries")
    axes[0].grid(alpha=0.3)
    if len(tracks) <= 8:
        axes[0].legend(fontsize=8, ncol=2, loc="upper right",
                       framealpha=0.85)

    if all_speeds:
        axes[1].hist(all_speeds, bins=30, color="#3b6cad",
                     edgecolor="white", alpha=0.85)
        axes[1].set_xlabel("Speed (µm/min)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Speed distribution (all tracks)")
        axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _png64(fig)


def _sample_frames(frames, masks_or_labels, tracks=None, n=3):
    """3-panel: 0/50/100% frames with overlay."""
    n_frames = len(frames)
    indices = [0, n_frames // 2, n_frames - 1]
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    for ax, fi in zip(axes, indices):
        ax.imshow(_imnorm(frames[fi]), cmap="gray", vmin=0, vmax=1)
        if tracks:
            cmap = plt.cm.tab10(np.linspace(0, 1, max(len(tracks), 10)))
            for i, t in enumerate(tracks):
                s = t.get("stack")
                if s is None or not s[fi].any():
                    continue
                ax.contour(s[fi].astype(np.uint8), levels=[0.5],
                           colors=[cmap[i % 10]], linewidths=1.6)
        elif masks_or_labels is not None:
            m = masks_or_labels[fi]
            if m.any():
                ax.contour((m > 0).astype(np.uint8), levels=[0.5],
                           colors=["#ff5555"], linewidths=1.8)
        ax.set_title(f"Frame {fi}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return _png64(fig)


def _flag_suspicious(tracks, n_frames, area_change_thresh=0.5,
                     centroid_hop_thresh_px=100):
    """Heuristics flagging frames worth review."""
    flags = []
    from core.tracking import extract_centroids
    for ti, t in enumerate(tracks):
        s = t.get("stack")
        if s is None or not s.any():
            continue
        present = s.any(axis=(1, 2))
        cents = extract_centroids(s)
        for i in range(1, n_frames):
            if not (present[i] and present[i - 1]):
                continue
            a_prev = int(s[i - 1].sum()); a_cur = int(s[i].sum())
            if a_prev > 0:
                rel = abs(a_cur - a_prev) / a_prev
                if rel > area_change_thresh:
                    flags.append((ti, i,
                                  f"area Δ {rel * 100:.0f}% "
                                  f"({a_prev} → {a_cur} px)"))
            d = np.linalg.norm(cents[i] - cents[i - 1])
            if d > centroid_hop_thresh_px:
                flags.append((ti, i,
                              f"centroid hop {d:.0f} px"))
    return flags


def write_quality_report(html_path,
                         *,
                         frames,
                         masks=None,
                         labels=None,
                         tracks=None,
                         recording_name="recording",
                         um_per_px=0.65, dt_min=5.0):
    """Write a single self-contained HTML report.

    Args:
        html_path: where to write the .html file
        frames: (N, H, W) image stack (uint8)
        masks: (N, H, W) bool stack (single-cell pipeline) — pass either
            this or `labels`
        labels: (N, H, W) int32 stack (multi-cell pipeline) — pass
            either this or `masks`
        tracks: list of track dicts (multi-cell only). If provided,
            per-track stats appear in the report.
        recording_name: shown in the report header
        um_per_px, dt_min: physical units for speed
    """
    n = len(frames)
    masks_or_labels = masks if masks is not None else labels

    sections = []
    sections.append(_detection_rate_chart(
        [m.astype(bool) for m in (masks if masks is not None else labels)]))
    if tracks:
        sections.append(_track_presence_heatmap(tracks, n))
        sections.append(_speed_distribution(tracks, um_per_px, dt_min))
    sections.append(_sample_frames(frames, masks_or_labels, tracks))

    flags = []
    if tracks:
        flags = _flag_suspicious(tracks, n)

    n_tracks = len(tracks) if tracks else 0
    n_detected = (sum(1 for m in masks if m.any()) if masks is not None
                  else sum(1 for L in labels if L.any())
                  if labels is not None else 0)

    html = []
    html.append("<!doctype html><html><head>")
    html.append(f"<title>Quality report — {recording_name}</title>")
    html.append("""<style>
body{font-family:-apple-system,Helvetica,Arial,sans-serif;
     margin:30px;color:#222;max-width:1100px;line-height:1.5;}
h1{margin-bottom:0}h2{margin-top:35px;border-bottom:1px solid #ddd;
   padding-bottom:5px;font-size:1.15em}
.meta{color:#888;font-size:0.9em;margin-bottom:25px}
.kpi{display:inline-block;border:1px solid #ddd;border-radius:6px;
     padding:10px 18px;margin:0 8px 8px 0;background:#f8f8f8;
     min-width:120px}
.kpi b{display:block;font-size:1.6em;color:#3b6cad}
.kpi span{font-size:0.85em;color:#666}
img{max-width:100%;border:1px solid #ddd;border-radius:4px;
    margin:8px 0}
table{border-collapse:collapse;font-size:0.92em;margin:10px 0}
th,td{border:1px solid #ddd;padding:6px 10px;text-align:left}
th{background:#f3f3f3}
.flag-warn{background:#fff7e6}
.flag-bad{background:#ffe6e6}
</style>""")
    html.append("</head><body>")
    html.append(f"<h1>Quality report — {recording_name}</h1>")
    html.append(f"<div class='meta'>Generated "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M')} · "
                f"{n} frames · {n_detected} with cells · "
                f"{n_tracks} tracks</div>")

    # KPI tiles
    html.append("<div>")
    html.append(
        f"<div class='kpi'><b>{n_detected}/{n}</b>"
        f"<span>Frames with cells</span></div>")
    html.append(
        f"<div class='kpi'><b>{int(n_detected / max(n, 1) * 100)}%</b>"
        f"<span>Detection rate</span></div>")
    html.append(
        f"<div class='kpi'><b>{n_tracks}</b><span>Tracks</span></div>")
    html.append(
        f"<div class='kpi'><b>{len(flags)}</b>"
        f"<span>Flagged frames</span></div>")
    html.append("</div>")

    html.append("<h2>Detection over time</h2>")
    html.append(f"<img src='{sections[0]}'>")

    if tracks:
        idx = 1
        html.append("<h2>Per-track presence</h2>")
        html.append(f"<img src='{sections[idx]}'>"); idx += 1
        html.append("<h2>Migration speed</h2>")
        html.append(f"<img src='{sections[idx]}'>"); idx += 1
        html.append("<h2>Sample frames</h2>")
        html.append(f"<img src='{sections[idx]}'>")
    else:
        html.append("<h2>Sample frames</h2>")
        html.append(f"<img src='{sections[1]}'>")

    if flags:
        html.append("<h2>Flagged frames "
                    f"(<span style='color:#c66'>{len(flags)}</span>)</h2>")
        html.append("<table><tr><th>Track</th><th>Frame</th>"
                    "<th>Reason</th></tr>")
        for tid, fi, reason in flags[:80]:
            html.append(f"<tr class='flag-warn'><td>Cell {tid + 1}</td>"
                        f"<td>{fi}</td><td>{reason}</td></tr>")
        if len(flags) > 80:
            html.append(
                f"<tr><td colspan='3'>… {len(flags) - 80} more</td></tr>")
        html.append("</table>")

    if tracks:
        html.append("<h2>Per-track summary</h2>")
        html.append("<table><tr><th>Cell</th><th>First</th>"
                    "<th>Last</th><th>Frames present</th>"
                    "<th>Mean area (px)</th></tr>")
        for ti, t in enumerate(tracks):
            s = t.get("stack")
            if s is None or not s.any():
                continue
            present = s.any(axis=(1, 2))
            first = int(np.argmax(present))
            last = len(present) - 1 - int(np.argmax(present[::-1]))
            n_present = int(present.sum())
            areas = s[present].sum(axis=(1, 2))
            mean_area = float(areas.mean()) if len(areas) else 0
            html.append(f"<tr><td>Cell {ti + 1}</td>"
                        f"<td>{first}</td><td>{last}</td>"
                        f"<td>{n_present}/{n}</td>"
                        f"<td>{mean_area:.0f}</td></tr>")
        html.append("</table>")

    html.append("</body></html>")

    os.makedirs(os.path.dirname(os.path.abspath(html_path)) or ".",
                exist_ok=True)
    with open(html_path, "w") as f:
        f.write("\n".join(html))
    return html_path


# ──────────────────────────────────────────────────────────────────────
# CLI: dump from a cached .npz (works for the cache files written by
# scripts/cache_pipelines.py and friends).
# ──────────────────────────────────────────────────────────────────────
def _from_cache_cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("out_html")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=False)
    name = args.name or os.path.basename(args.cache).split(".")[0]

    if "labels" in z.files and "tracks_n" in z.files:
        # multi-cell cache
        tracks = []
        for i in range(50):
            if f"track_{i}_stack" not in z.files:
                continue
            tracks.append({
                "id": i,
                "stack": z[f"track_{i}_stack"],
            })
        out = write_quality_report(
            args.out_html,
            frames=z["frames"],
            labels=z["labels"],
            tracks=tracks,
            recording_name=name,
        )
    else:
        # single-cell cache (frames + masks)
        out = write_quality_report(
            args.out_html,
            frames=z["frames"],
            masks=z["masks"],
            recording_name=name,
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    _from_cache_cli()
