"""Stage IC293 single-cell crops into the IC295 on-disk format.

Ignasi hand-cropped single cells from the IC293_ECmigration dataset
(genetic WT/KO/GOF + drug DMSO/Y1/OT arms) onto the Pathak lab share
as ImageJ-cropped single-channel DIC TIFFs. This script copies them
LOCALLY and converts each crop into the same on-disk format the IC295
batch pipeline ingests, so the existing CellScope detection / analysis
stack runs on them unchanged.

Verified facts (2026-06-16, by inspecting the share read-only):
  * Source: /Volumes/pathaklab/.../IC293_CroppedSelectedCells
  * Same microscope as IC295: 0.6523 um/px, Interval_ms=600000 (10 min).
  * Crops are SINGLE-CHANNEL DIC. The original recording was 2-channel
    ['DIC 10x', 'None']; Ignasi kept only DIC. There is NO Cy5/SirActin
    channel in IC293 -> n_channels=1, dic_channel=0, fluo_channel=None.
  * 78 crops. Multi-cell positions yield `-cellK-` variants; one
    `-dividing`; two `-croped` (typo) files. Frame counts vary (74-97);
    one file stores frames as ImageJ Z-slices (normalised to T here).

Output (per crop) under ic293_analysis/_cache/:
  <label>.ome.tif        clean OME-TIFF (T,H,W) uint16, axes=TYX,
                         physical pixel size + frame interval embedded
  <label>.ome.json       sidecar (um_per_px, interval, n_channels=1,
                         dic_channel=0, fluo_channel=null, provenance)
  <label>_metadata.txt   Micro-Manager-style Summary with Channels=1
                         (so core.io.detect_channels resolves correctly)
plus _source_metadata/<orig>_metadata.txt   untouched original (kept for
                         provenance; named so the loader never reads it).

label = Pos{N}-{COND}        for a position's primary crop
        Pos{N}_cell{K}-{COND}  for the K-th cropped cell
        Pos{N}_div-{COND}      for the `-dividing` crop
(condition is always the final '-'-delimited token so
 ic295_common.parse_condition keeps working.)

SAFETY: every read of the lab share is strictly read-only (tifffile
read / open 'rb'). Nothing is ever written back to /Volumes/pathaklab.
Local writes are atomic (.tmp + os.replace) and idempotent.

Usage:
    conda run -n cellpose4 python scripts/ic293_stage_crops.py --dry-run
    conda run -n cellpose4 python scripts/ic293_stage_crops.py
    conda run -n cellpose4 python scripts/ic293_stage_crops.py --label Pos0-WT
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import tifffile

SRC_DEFAULT = (
    "/Volumes/pathaklab/Lab/Ignasi/IC293_ECmigration/IC293_CroppedSelectedCells"
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_ROOT = os.path.join(PROJECT_ROOT, "ic293_analysis")
CACHE_DIR = os.path.join(ANALYSIS_ROOT, "_cache")
SRC_META_DIR = os.path.join(CACHE_DIR, "_source_metadata")

CONDITIONS = ("WT", "KO", "GOF", "Y1", "OT", "DMSO")
UM_PER_PX_FALLBACK = 0.6523
INTERVAL_MS_FALLBACK = 600000  # 10 min

# IC293__1_MMStack_Pos11-WT.ome-cell1-cropped.tif (and -croped typo,
# -cropped-dividing, plain -cropped).  `crop+ed` matches both spellings.
NAME_RE = re.compile(
    r"^IC293__1_MMStack_Pos(\d+)-([A-Za-z0-9]+)\.ome"
    r"-(?:(cell\d+)-)?crop+ed(?:-(dividing))?\.tif$"
)


# ───────── inventory ─────────
def parse_crop_name(fn):
    """Parse a source crop filename -> dict, or None if it doesn't match."""
    m = NAME_RE.match(fn)
    if not m:
        return None
    pos, cond, cell, dividing = m.groups()
    if cond not in CONDITIONS:
        return None
    if cell:                       # 'cell1' -> '_cell1'
        suffix = "_" + cell
    elif dividing:
        suffix = "_div"
    else:
        suffix = ""
    label = f"Pos{pos}{suffix}-{cond}"
    return {
        "src_name": fn,
        "pos": int(pos),
        "condition": cond,
        "cell_tag": cell or ("dividing" if dividing else "primary"),
        "label": label,
        "meta_name": f"IC293__1_MMStack_Pos{pos}-{cond}_metadata.txt",
    }


def build_inventory(src):
    """Enumerate + parse every crop on the share. Errors on label clash."""
    entries, seen = [], {}
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".tif"):
            continue
        e = parse_crop_name(fn)
        if e is None:
            print(f"  [skip] unrecognised name: {fn}", flush=True)
            continue
        if e["label"] in seen:
            raise ValueError(
                f"label collision {e['label']!r}: "
                f"{seen[e['label']]} vs {fn}")
        seen[e["label"]] = fn
        e["src_path"] = os.path.join(src, fn)
        entries.append(e)
    return entries


# ───────── Micro-Manager metadata ─────────
def load_summary(src, meta_name, cache):
    """Return the original recording's MM Summary dict (cached per file)."""
    if meta_name in cache:
        return cache[meta_name]
    p = os.path.join(src, meta_name)
    summ = {}
    if os.path.exists(p):
        try:
            with open(p) as f:
                summ = json.load(f).get("Summary", {})
        except Exception as exc:        # noqa: BLE001
            print(f"  [warn] could not parse {meta_name}: {exc!r}",
                  flush=True)
    else:
        print(f"  [warn] no metadata.txt for {meta_name}; using "
              f"fallback scale/interval", flush=True)
    cache[meta_name] = summ
    return summ


def physical_params(summ):
    """(um_per_px, interval_ms) from the Summary, with constants as
    fallback. IC293 was acquired on the same scope as IC295, so these
    are the same 0.6523 / 600000 across the experiment."""
    um = summ.get("PixelSize_um") or summ.get("PixelSizeUm")
    iv = summ.get("Interval_ms")
    return (float(um) if um else UM_PER_PX_FALLBACK,
            float(iv) if iv else float(INTERVAL_MS_FALLBACK))


def corrected_metadata(summ, t, h, w):
    """A Micro-Manager-style Summary corrected for the single-channel
    crop: Channels=1, ChNames=['DIC 10x'], crop T/H/W. Built from the
    real original Summary so acquisition provenance survives, minus the
    per-frame FrameKey entries. core.io.detect_channels reads
    `"Channels": 1` from here, so the crop loads as single-channel."""
    s = dict(summ)
    s["Channels"] = 1
    s["ChNames"] = ["DIC 10x"]
    s["ChColors"] = s.get("ChColors", [])[:1]
    s["Frames"] = int(t)
    s["Slices"] = 1
    s["Width"] = int(w)
    s["Height"] = int(h)
    return {
        "Summary": s,
        "_cellscope_note": (
            "Single-channel DIC crop staged from the IC293 share by "
            "scripts/ic293_stage_crops.py. Channels/ChNames/Frames/"
            "Width/Height corrected for the crop; the untouched original "
            "metadata is preserved under _cache/_source_metadata/."
        ),
    }


# ───────── array normalisation ─────────
def probe_shape(s):
    """(t, h, w, n_channels) from a tifffile series, via its axes label.
    Frames = product of every non-spatial, non-channel axis (handles T,
    Z-as-time, Q); channel axis is dropped. Falls back to last-two-dims
    when Y/X aren't labelled."""
    ax, shp = list(s.axes), list(s.shape)
    nch = shp[ax.index("C")] if "C" in ax else 1
    if "Y" in ax and "X" in ax:
        h, w = shp[ax.index("Y")], shp[ax.index("X")]
        t = 1
        for a, d in zip(ax, shp):
            if a not in ("Y", "X", "C"):
                t *= d
    else:                              # unlabelled: assume (...,H,W)
        h, w = shp[-2], shp[-1]
        t = int(np.prod(shp[:-2])) // max(nch, 1)
    return int(t), int(h), int(w), int(nch)


def _dic_channel_index(arr, cax):
    """DIC is the channel with real structure; the IC293 'None' channel
    is near-blank. Pick the highest-variance channel (matches the
    ChNames order, where DIC is first)."""
    nch = arr.shape[cax]
    if nch == 1:
        return 0
    stds = []
    for c in range(nch):
        sl = [slice(None)] * arr.ndim
        sl[cax] = c
        stds.append(float(arr[tuple(sl)].std()))
    return int(np.argmax(stds))


def read_crop_array(src_path):
    """Read a source crop as (T, H, W) uint16 DIC. Handles single-channel
    crops (TYX / ImageJ Z-as-time) and the occasional 2-channel TCYX crop
    (keeps the DIC channel, drops the blank 'None')."""
    with tifffile.TiffFile(src_path) as tf:
        s = tf.series[0]
        ax = list(s.axes)
        arr = s.asarray()
    if "C" in ax:
        cax = ax.index("C")
        arr = np.take(arr, _dic_channel_index(arr, cax), axis=cax)
        ax.pop(cax)
    arr = np.squeeze(arr)
    if arr.ndim == 2:                  # single frame
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"unexpected crop shape {arr.shape} in "
                         f"{os.path.basename(src_path)} (need T,H,W "
                         f"after channel selection)")
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return np.ascontiguousarray(arr)


# ───────── atomic OME-TIFF write ─────────
def write_ome_tif(dst, arr, um_per_px, interval_ms):
    tmp = dst + ".tmp"
    tifffile.imwrite(
        tmp, arr, ome=True, photometric="minisblack",
        metadata={
            "axes": "TYX",
            "PhysicalSizeX": um_per_px, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": um_per_px, "PhysicalSizeYUnit": "µm",
            "TimeIncrement": interval_ms / 1000.0, "TimeIncrementUnit": "s",
        },
    )
    os.replace(tmp, dst)


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def dst_valid(dst_tif, t):
    """True if dst already exists with the right page/frame count."""
    if not os.path.exists(dst_tif):
        return False
    try:
        with tifffile.TiffFile(dst_tif) as tf:
            return len(tf.pages) == int(t)
    except Exception:                  # noqa: BLE001
        return False


# ───────── convert one crop ─────────
def convert_one(e, src, summ_cache, dry_run):
    name = f"IC293__1_MMStack_{e['label']}"
    dst_tif = os.path.join(CACHE_DIR, f"{name}.ome.tif")
    dst_json = os.path.join(CACHE_DIR, f"{name}.ome.json")
    dst_meta = os.path.join(CACHE_DIR, f"{name}_metadata.txt")

    # probe source header (cheap) for shape + idempotency
    with tifffile.TiffFile(e["src_path"]) as tf:
        t, h, w, nch = probe_shape(tf.series[0])
    e.update({"t": int(t), "h": int(h), "w": int(w),
              "src_channels": int(nch),
              "src_bytes": os.path.getsize(e["src_path"])})

    summ = load_summary(src, e["meta_name"], summ_cache)
    um_per_px, interval_ms = physical_params(summ)
    e["um_per_px"], e["interval_ms"] = um_per_px, interval_ms

    if (dst_valid(dst_tif, t) and os.path.exists(dst_json)
            and os.path.exists(dst_meta)):
        e["outcome"] = "skipped"
        return e
    if dry_run:
        e["outcome"] = "would-convert"
        return e

    arr = read_crop_array(e["src_path"])
    t, h, w = arr.shape
    e.update({"t": int(t), "h": int(h), "w": int(w)})
    write_ome_tif(dst_tif, arr, um_per_px, interval_ms)

    # verify round-trip shape + single-channel detection
    with tifffile.TiffFile(dst_tif) as tf:
        if len(tf.pages) != t:
            raise IOError(f"{name}: wrote {len(tf.pages)} pages, expected {t}")

    write_json(dst_json, {
        "um_per_px": um_per_px,
        "time_interval_min": interval_ms / 60000.0,
        "dic_channel": 0,
        "fluo_channel": None,
        "n_channels": 1,
        "channel_names": ["DIC 10x"],
        "name": e["label"],
        "metadata_source": f"{name}_metadata.txt",
        "n_frames": int(t),
        "source_crop": e["src_name"],
        "source_dir": src,
        "staged_by": "scripts/ic293_stage_crops.py",
    })
    write_json(dst_meta, corrected_metadata(summ, t, h, w))

    # preserve the untouched original metadata for provenance (named so
    # core.io.detect_channels' `_metadata.txt` lookup never matches it)
    orig_meta = os.path.join(src, e["meta_name"])
    if os.path.exists(orig_meta):
        prov = os.path.join(SRC_META_DIR, e["meta_name"])
        if not os.path.exists(prov):
            with open(orig_meta, "rb") as fi, open(prov + ".tmp", "wb") as fo:
                while True:
                    b = fi.read(8 * 1024 * 1024)
                    if not b:
                        break
                    fo.write(b)
            os.replace(prov + ".tmp", prov)
    e["outcome"] = "converted"
    return e


# ───────── provenance ─────────
def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                  # noqa: BLE001
        return None


def write_provenance(entries, src, started, finished):
    by_cond = {}
    for e in entries:
        by_cond.setdefault(e["condition"], []).append(e)
    meta = {
        "dataset": "IC293_ECmigration (cropped single cells)",
        "source_dir": src,
        "staged_to": CACHE_DIR,
        "n_crops": len(entries),
        "by_condition": {c: len(v) for c, v in sorted(by_cond.items())},
        "format": {
            "ome_tif": "(T,H,W) uint16, axes=TYX, single-channel DIC",
            "n_channels": 1, "dic_channel": 0, "fluo_channel": None,
            "channel_names": ["DIC 10x"],
            "um_per_px_default": UM_PER_PX_FALLBACK,
            "interval_ms_default": INTERVAL_MS_FALLBACK,
        },
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__, "tifffile": tifffile.__version__,
        },
        "git_commit": git_commit(),
        "timestamp_started": started,
        "timestamp_finished": finished,
        "crops": [
            {k: e.get(k) for k in (
                "label", "condition", "cell_tag", "src_name", "src_bytes",
                "src_channels", "t", "h", "w", "um_per_px", "interval_ms",
                "outcome")}
            for e in sorted(entries, key=lambda e: (e["condition"], e["pos"]))
        ],
    }
    write_json(os.path.join(CACHE_DIR, "CONVERSION_METADATA.json"), meta)

    lines = [
        "# IC293 cropped single cells — staging provenance", "",
        f"Source (read-only): `{src}`  ",
        f"Staged to: `ic293_analysis/_cache/`  ",
        f"Staged: {finished}  •  git `{(meta['git_commit'] or '?')[:10]}`  ",
        f"tifffile {tifffile.__version__}, numpy {np.__version__}", "",
        "## What this is", "",
        "Ignasi's hand-cropped single cells from the **IC293_ECmigration** "
        "dataset, converted to the IC295 on-disk format so the existing "
        "CellScope detection/analysis pipeline runs on them unchanged.", "",
        "**Single-channel DIC** — the IC293 originals were 2-channel "
        "`['DIC 10x','None']` and only the DIC channel was cropped. There "
        "is no Cy5/fluorescence channel (no SirActin in IC293), so the "
        "Cy5-dependent pipeline steps (alignment, persistence-guard) "
        "auto-skip; detection runs on DIC as usual.", "",
        f"Same microscope as IC295: **{UM_PER_PX_FALLBACK} µm/px**, "
        f"**{INTERVAL_MS_FALLBACK//60000} min** frame interval.", "",
        "## Per-crop files (in `_cache/`)", "",
        "- `<label>.ome.tif` — clean OME-TIFF `(T,H,W)` uint16, axes=TYX, "
        "physical pixel size + frame interval embedded.",
        "- `<label>.ome.json` — sidecar: `n_channels=1`, `dic_channel=0`, "
        "`fluo_channel=null`, scale/interval, source provenance.",
        "- `<label>_metadata.txt` — MM-style Summary with `Channels=1` "
        "(so `core.io.detect_channels` resolves single-channel).",
        "- `_source_metadata/<orig>_metadata.txt` — untouched original "
        "MM metadata, kept for provenance.", "",
        "## Label scheme", "",
        "`Pos{N}-{COND}` (primary crop), `Pos{N}_cell{K}-{COND}` "
        "(K-th cell), `Pos{N}_div-{COND}` (dividing crop). Condition is "
        "always the final `-` token, so `parse_condition` works.", "",
        "## Inventory", "",
    ]
    for cond in CONDITIONS:
        evs = sorted(by_cond.get(cond, []), key=lambda e: e["pos"])
        if not evs:
            continue
        lines.append(f"### {cond} ({len(evs)})")
        lines.append("")
        lines.append("| label | source file | T×H×W |")
        lines.append("|---|---|---|")
        for e in evs:
            lines.append(f"| `{e['label']}` | `{e['src_name']}` | "
                         f"{e.get('t','?')}×{e.get('h','?')}×{e.get('w','?')} |")
        lines.append("")
    with open(os.path.join(ANALYSIS_ROOT, "PROVENANCE.md"), "w") as f:
        f.write("\n".join(lines))


# ───────── main ─────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SRC_DEFAULT)
    ap.add_argument("--label", default=None, help="convert a single label")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.source):
        sys.exit(f"ERROR: source not mounted: {args.source}\n"
                 f"  Finder > Go > Connect to Server > smb://pathaklab")

    entries = build_inventory(args.source)
    if args.label:
        entries = [e for e in entries if e["label"] == args.label]
        if not entries:
            sys.exit(f"no crop with label {args.label!r}")

    by_cond = {}
    for e in entries:
        by_cond.setdefault(e["condition"], 0)
        by_cond[e["condition"]] += 1
    print(f"\n=== IC293 staging plan ===", flush=True)
    print(f"  source : {args.source}", flush=True)
    print(f"  dest   : {CACHE_DIR}", flush=True)
    print(f"  crops  : {len(entries)}  "
          f"({', '.join(f'{c}={n}' for c, n in sorted(by_cond.items()))})",
          flush=True)

    if not args.dry_run:
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(SRC_META_DIR, exist_ok=True)
        nidx = os.path.join(CACHE_DIR, ".metadata_never_index")
        if not os.path.exists(nidx):
            open(nidx, "w").close()

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()
    summ_cache = {}
    stats = {"converted": 0, "skipped": 0, "would-convert": 0, "failed": 0}
    for i, e in enumerate(entries, 1):
        try:
            convert_one(e, args.source, summ_cache, args.dry_run)
            stats[e["outcome"]] += 1
            print(f"  [{i:2d}/{len(entries)}] {e['label']:20s} "
                  f"{e.get('t','?')}×{e.get('h','?')}×{e.get('w','?')}  "
                  f"{e['outcome']}", flush=True)
        except Exception as exc:        # noqa: BLE001
            e["outcome"] = f"FAILED: {exc!r}"
            stats["failed"] += 1
            print(f"  [{i:2d}/{len(entries)}] {e['label']:20s} "
                  f"✗ {exc!r}", flush=True)

    finished = time.strftime("%Y-%m-%dT%H:%M:%S")
    if not args.dry_run:
        write_provenance(entries, args.source, started, finished)

    print(f"\n=== summary ({time.time()-t0:.0f}s) ===", flush=True)
    for k, v in stats.items():
        print(f"  {k:14s}: {v}", flush=True)
    if not args.dry_run:
        print(f"  provenance    : ic293_analysis/PROVENANCE.md + "
              f"_cache/CONVERSION_METADATA.json", flush=True)
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
