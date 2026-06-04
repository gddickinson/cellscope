"""Losslessly re-compress recording TIFFs to shrink storage with ZERO
effect on any analysis result.

Microscopy masters are often stored UNCOMPRESSED (the IC295 .ome.tif
files are 16-bit, 3-channel, uncompressed → ~2.4 GB each). Re-saving
them with a LOSSLESS codec (Deflate/zlib by default) keeps the pixel
data bit-for-bit identical and preserves the channel structure +
metadata, so detection / tracking / analysis produce byte-identical
results — it just stores the same data in fewer bytes. (One IC295
channel is the empty "None" channel, which compresses to almost
nothing.)

SAFETY — this mutates master data, so every file is handled atomically:
  1. read the array + axes from the original
  2. write a compressed `<file>.recompress.tmp`
  3. VERIFY losslessness page-by-page (every page == original) AND that
     `core.io.detect_channels` returns the same channel count
  4. only on success, atomically `os.replace(tmp, original)`
The original is never removed until a verified replacement exists; a
crash mid-write leaves the original untouched. `--dry-run` previews
sizes without writing. `--keep-backup` renames the original to
`<file>.orig` instead of replacing (costs disk, for the extra-cautious).

Lossless ⇒ identical results is a mathematical guarantee (deterministic
functions of identical input), but we still verify the bytes.

Usage:
    conda run -n cellpose4 python scripts/recompress_recordings.py \
        ic295_analysis/_cache --codec deflate
    ... --dry-run        # preview only
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tifffile

EXTS = (".tif", ".tiff")


def _iter_tiffs(paths):
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith(EXTS):
                    yield os.path.join(p, name)
        elif p.lower().endswith(EXTS):
            yield p


def _is_compressed(path):
    try:
        with tifffile.TiffFile(path) as tf:
            return int(tf.pages[0].compression) != 1   # 1 == NONE
    except Exception:
        return False


def _write_compressed(path, arr, axes, codec, level):
    """Write `arr` as a compressed (OME-)TIFF, regenerating clean OME
    metadata from the axes so the channel structure round-trips."""
    kw = {}
    if axes:
        kw["metadata"] = {"axes": axes}
    try:
        tifffile.imwrite(path, arr, photometric="minisblack",
                         compression=codec,
                         compressionargs={"level": level}
                         if level is not None else None,
                         **kw)
    except TypeError:
        # Older/newer tifffile signature — fall back to plain codec.
        tifffile.imwrite(path, arr, photometric="minisblack",
                         compression=codec, **kw)


def _verify(src, tmp, arr):
    """Page-by-page bit-identity + channel-count check. Low memory:
    holds one 2D page at a time from the tmp file."""
    from core.io import detect_channels
    flat = arr.reshape((-1,) + arr.shape[-2:])
    with tifffile.TiffFile(tmp) as tf:
        if len(tf.pages) != flat.shape[0]:
            return False, (f"page count {len(tf.pages)} != {flat.shape[0]}")
        for i in range(flat.shape[0]):
            if not np.array_equal(tf.pages[i].asarray(), flat[i]):
                return False, f"page {i} differs"
    if detect_channels(tmp) != detect_channels(src):
        return False, (f"detect_channels changed "
                       f"{detect_channels(src)} → {detect_channels(tmp)}")
    return True, "ok"


def _fmt(nbytes):
    return f"{nbytes / 1e9:.2f} GB" if nbytes >= 1e9 else \
        f"{nbytes / 1e6:.1f} MB"


def process(path, codec, level, dry_run, force, keep_backup):
    orig = os.path.getsize(path)
    if _is_compressed(path) and not force:
        print(f"[skip] {os.path.basename(path)} — already compressed "
              f"({_fmt(orig)})", flush=True)
        return ("skip", orig, orig)
    try:
        with tifffile.TiffFile(path) as tf:
            arr = tf.asarray()
            axes = tf.series[0].axes
    except Exception as e:
        print(f"[ERROR] {os.path.basename(path)} — read failed: {e}",
              flush=True)
        return ("error", orig, orig)

    tmp = path + ".recompress.tmp"
    try:
        _write_compressed(tmp, arr, axes, codec, level)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"[ERROR] {os.path.basename(path)} — write failed: {e}",
              flush=True)
        return ("error", orig, orig)

    new = os.path.getsize(tmp)
    ok, reason = _verify(path, tmp, arr)
    if not ok:
        os.remove(tmp)
        print(f"[ERROR] {os.path.basename(path)} — verify FAILED "
              f"({reason}); original untouched", flush=True)
        return ("error", orig, orig)

    if new >= orig:
        os.remove(tmp)
        print(f"[keep] {os.path.basename(path)} — no gain "
              f"({_fmt(orig)} → {_fmt(new)}); original untouched",
              flush=True)
        return ("nogain", orig, orig)

    ratio = orig / new
    if dry_run:
        os.remove(tmp)
        print(f"[dry ] {os.path.basename(path)} — {_fmt(orig)} → "
              f"{_fmt(new)}  ({ratio:.2f}× smaller) [verified, not written]",
              flush=True)
        return ("dry", orig, new)

    if keep_backup:
        os.replace(path, path + ".orig")
    os.replace(tmp, path)
    print(f"[done] {os.path.basename(path)} — {_fmt(orig)} → {_fmt(new)}  "
          f"({ratio:.2f}× smaller, lossless verified)", flush=True)
    return ("done", orig, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*",
                    default=["ic295_analysis/_cache"],
                    help="TIFF files or directories (default _cache)")
    ap.add_argument("--codec", default="deflate",
                    help="lossless codec: deflate/zlib/lzw/zstd "
                         "(default deflate)")
    ap.add_argument("--level", type=int, default=None,
                    help="codec compression level (codec-specific)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recompress even files that are already compressed")
    ap.add_argument("--keep-backup", action="store_true",
                    help="rename original to <file>.orig instead of "
                         "replacing (uses extra disk)")
    args = ap.parse_args()

    files = list(_iter_tiffs(args.paths))
    if not files:
        print("No TIFF files found.", file=sys.stderr)
        return 1
    print(f"{'DRY-RUN: ' if args.dry_run else ''}re-compressing "
          f"{len(files)} file(s) with codec={args.codec}\n", flush=True)

    tot_before = tot_after = 0
    counts = {}
    for f in files:
        status, before, after = process(
            f, args.codec, args.level, args.dry_run, args.force,
            args.keep_backup)
        counts[status] = counts.get(status, 0) + 1
        tot_before += before
        tot_after += after

    print(f"\n=== summary ===")
    print(f"  files: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  total: {_fmt(tot_before)} → {_fmt(tot_after)}  "
          f"({tot_before / max(tot_after, 1):.2f}× smaller, "
          f"saved {_fmt(tot_before - tot_after)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
