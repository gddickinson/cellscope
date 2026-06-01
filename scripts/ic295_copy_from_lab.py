"""Copy IC295 source TIFFs from the Pathak lab share to local SSD.

When the original GeorgeDrive failed mid-batch (2026-05-31), 24 of the
29 already-detected recordings still had their .ome.tif as a symlink
into the dead /Volumes/GeorgeDrive mount, blocking re-analysis,
overlay generation, and mask review.

This script copies the lab's master copies from
    /Volumes/pathaklab/Lab/Ignasi/IC295_ECmigrationwithSirActin/IC295__1
into the local ic295_analysis/_cache/, synthesizes the .ome.json
sidecars (lab source doesn't have them), and repoints the
by_condition/<cond>/<label>/*.ome.tif symlinks so everything resolves
locally without the dead drive.

Default selection: every label whose detect.state == 'done' in
progress.json — i.e. the n>=4 minimum-per-condition subset already
in flight. Use --include-undetected to also stage TIFFs for labels
queued for future detect runs.

Usage:
    # dry-run preview
    python scripts/ic295_copy_from_lab.py --dry-run

    # copy all already-detected labels (the default ~59GB job)
    python scripts/ic295_copy_from_lab.py

    # copy a single label or a single condition
    python scripts/ic295_copy_from_lab.py --label Pos68-DMSO
    python scripts/ic295_copy_from_lab.py --condition WT

    # also stage TIFFs for labels not yet detected
    python scripts/ic295_copy_from_lab.py --include-undetected

The script is idempotent and resume-friendly — re-run any time.
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time

# Make ic295_common importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ic295_common import (  # noqa: E402
    CACHE_DIR, RECORDINGS_ROOT, parse_condition,
    load_progress,
)

LAB_SOURCE_DEFAULT = (
    "/Volumes/pathaklab/Lab/Ignasi/IC295_ECmigrationwithSirActin/IC295__1"
)
# Template sidecar (every IC295 recording has identical microscope
# metadata — only `name` + `metadata_source` differ per label).
SIDECAR_TEMPLATE = {
    "um_per_px": 0.6523,
    "time_interval_min": 10.0,
    "dic_channel": 1,
    "fluo_channel": 0,
    "channel_names": ["Cy5", "DIC 10x", "None"],
}

_STOP = {"flag": False}


def _install_signals():
    def _handler(signum, _frame):
        _STOP["flag"] = True
        print(f"\n[signal] caught {signum}; finishing current file then "
              f"stopping.", flush=True)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ─────── label selection ───────
def select_labels(progress, mode, filter_label, filter_cond):
    """Pick which labels to copy."""
    if filter_label:
        return [filter_label]
    if mode == "detected":
        labels = [l for l, r in progress.items()
                  if r.get("detect", {}).get("state") == "done"]
    elif mode == "include-undetected":
        labels = list(progress.keys())
    else:
        raise ValueError(f"unknown mode: {mode}")
    if filter_cond:
        labels = [l for l in labels
                  if parse_condition(l) == filter_cond]
    return sorted(labels)


# ─────── source / destination paths ───────
def _src_paths(src_root, label):
    """Return (tif_path, metadata_path) for a label on the lab share."""
    name = f"IC295__1_MMStack_{label}"
    tif = os.path.join(src_root, f"{name}.ome.tif")
    meta = os.path.join(src_root, f"{name}_metadata.txt")
    return tif, meta


def _dst_paths(label):
    """Return (tif_path, metadata_path, sidecar_path) under CACHE_DIR."""
    name = f"IC295__1_MMStack_{label}"
    return (
        os.path.join(CACHE_DIR, f"{name}.ome.tif"),
        os.path.join(CACHE_DIR, f"{name}_metadata.txt"),
        os.path.join(CACHE_DIR, f"{name}.ome.json"),
    )


# ─────── space pre-check ───────
def _bytes_needed(plan):
    """Sum of source sizes for files that don't already exist locally
    at the matching size."""
    total = 0
    for entry in plan:
        for src, dst in ((entry["tif_src"], entry["tif_dst"]),
                         (entry["meta_src"], entry["meta_dst"])):
            if not os.path.exists(src):
                continue
            try:
                src_size = os.path.getsize(src)
            except OSError:
                continue
            if (os.path.exists(dst)
                    and os.path.getsize(dst) == src_size):
                continue
            total += src_size
    return total


def _free_bytes(path):
    s = shutil.disk_usage(path)
    return s.free


# ─────── atomic copy with verify + retry ───────
def _hexdigest(path, blk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blk), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_with_progress(src, dst_tmp, src_size, label, kind):
    """Stream-copy src -> dst_tmp with periodic progress prints. Raises
    on any I/O error (caller catches + retries)."""
    blk = 8 * 1024 * 1024
    copied = 0
    last_report = time.time()
    with open(src, "rb") as fi, open(dst_tmp, "wb") as fo:
        while True:
            if _STOP["flag"]:
                raise InterruptedError("stop requested mid-copy")
            buf = fi.read(blk)
            if not buf:
                break
            fo.write(buf)
            copied += len(buf)
            now = time.time()
            if now - last_report >= 5.0:
                pct = 100.0 * copied / src_size if src_size else 100.0
                mb_s = copied / (now - last_report + 1e-9) / 1e6
                print(f"    {label} {kind}: {pct:5.1f}%  "
                      f"({copied/1e9:.2f}/{src_size/1e9:.2f} GB)",
                      flush=True)
                last_report = now
        fo.flush()
        os.fsync(fo.fileno())


def copy_atomic(src, dst, label, kind, retries=3,
                verify_sha256=False, dry_run=False):
    """Copy src -> dst atomically (.tmp + rename). Idempotent: skips
    when dst already exists with matching size.

    Returns one of: "copied", "skipped", "missing-src".
    """
    if not os.path.exists(src):
        return "missing-src"
    src_size = os.path.getsize(src)
    if os.path.exists(dst) and os.path.getsize(dst) == src_size:
        if verify_sha256:
            if _hexdigest(src) == _hexdigest(dst):
                return "skipped"
            print(f"    {label} {kind}: SHA mismatch on existing dst — "
                  f"will re-copy", flush=True)
        else:
            return "skipped"
    if dry_run:
        return "would-copy"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    dst_tmp = dst + ".tmp"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            print(f"  [{label}] {kind}: copying "
                  f"{src_size/1e9:.2f} GB (attempt {attempt}/{retries})",
                  flush=True)
            if os.path.exists(dst_tmp):
                os.unlink(dst_tmp)
            _copy_with_progress(src, dst_tmp, src_size, label, kind)
            # Preserve mtime where we can (network mount may say no)
            try:
                st = os.stat(src)
                os.utime(dst_tmp, (st.st_atime, st.st_mtime))
            except OSError:
                pass
            # Size verify (cheap; catches truncations)
            if os.path.getsize(dst_tmp) != src_size:
                raise IOError(
                    f"size mismatch after copy: "
                    f"{os.path.getsize(dst_tmp)} != {src_size}")
            if verify_sha256:
                if _hexdigest(src) != _hexdigest(dst_tmp):
                    raise IOError("SHA256 mismatch after copy")
            os.replace(dst_tmp, dst)
            return "copied"
        except InterruptedError:
            if os.path.exists(dst_tmp):
                os.unlink(dst_tmp)
            raise
        except (IOError, OSError) as e:
            last_err = e
            print(f"    error: {e!r}", flush=True)
            if os.path.exists(dst_tmp):
                try:
                    os.unlink(dst_tmp)
                except OSError:
                    pass
            if attempt < retries:
                wait = 2 ** attempt
                print(f"    waiting {wait}s before retry...", flush=True)
                time.sleep(wait)
    raise IOError(f"copy failed after {retries} attempts: {last_err!r}")


# ─────── sidecar synthesis ───────
def write_sidecar(label, sidecar_path, dry_run=False):
    """Write a per-label .ome.json into the cache. Idempotent."""
    if os.path.exists(sidecar_path):
        return "skipped"
    if dry_run:
        return "would-write"
    data = dict(SIDECAR_TEMPLATE)
    data["name"] = label
    data["metadata_source"] = f"IC295__1_MMStack_{label}_metadata.txt"
    tmp = sidecar_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, sidecar_path)
    return "written"


# ─────── symlink update under by_condition/ ───────
def update_local_symlinks(label, cond, cache_tif, cache_meta,
                          cache_sidecar, dry_run=False):
    """Repoint by_condition/<cond>/<label>/ symlinks at the cache.

    Replaces dead /Volumes/GeorgeDrive symlinks. Creates the recording
    folder if it doesn't exist yet (for undetected labels we're
    staging ahead of time).
    """
    rec_dir = os.path.join(RECORDINGS_ROOT, cond, label)
    if not os.path.isdir(rec_dir):
        if dry_run:
            return "would-create-dir"
        os.makedirs(rec_dir, exist_ok=True)
    name = f"IC295__1_MMStack_{label}"
    targets = [
        (cache_tif,     os.path.join(rec_dir, f"{name}.ome.tif")),
        (cache_meta,    os.path.join(rec_dir, f"{name}_metadata.txt")),
    ]
    actions = []
    for src, link in targets:
        if not os.path.exists(src):
            continue
        # Read current state without following dead symlinks
        if os.path.islink(link):
            try:
                current = os.readlink(link)
            except OSError:
                current = None
            if current == src:
                actions.append(f"{os.path.basename(link)}=ok")
                continue
        elif os.path.exists(link):
            # A real file at this path means setup_recording_folder
            # copied it instead of symlinked — leave it alone.
            actions.append(f"{os.path.basename(link)}=real-file-kept")
            continue
        if dry_run:
            actions.append(f"{os.path.basename(link)}=would-relink")
            continue
        try:
            if os.path.islink(link) or os.path.exists(link):
                os.unlink(link)
        except OSError:
            pass
        os.symlink(src, link)
        actions.append(f"{os.path.basename(link)}=relinked")
    # The .ome.json is a regular file (not a symlink) in by_condition;
    # copy our cache sidecar over if missing.
    sidecar_in_rec = os.path.join(rec_dir, f"{name}.ome.json")
    if (not os.path.exists(sidecar_in_rec)
            and os.path.exists(cache_sidecar)):
        if dry_run:
            actions.append(f"{name}.ome.json=would-copy")
        else:
            shutil.copy2(cache_sidecar, sidecar_in_rec)
            actions.append(f"{name}.ome.json=copied")
    return ",".join(actions) if actions else "no-op"


# ─────── plan builder ───────
def build_plan(labels, src_root):
    """For each label, resolve src/dst paths and source sizes."""
    plan = []
    for label in labels:
        tif_src, meta_src = _src_paths(src_root, label)
        tif_dst, meta_dst, sidecar_dst = _dst_paths(label)
        tif_size = (os.path.getsize(tif_src)
                    if os.path.exists(tif_src) else None)
        plan.append({
            "label":       label,
            "condition":   parse_condition(label),
            "tif_src":     tif_src,
            "meta_src":    meta_src,
            "tif_dst":     tif_dst,
            "meta_dst":    meta_dst,
            "sidecar_dst": sidecar_dst,
            "tif_size":    tif_size,
        })
    return plan


# ─────── main ───────
def main():
    ap = argparse.ArgumentParser(
        description="Copy IC295 source TIFFs from lab share to local cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--source", default=LAB_SOURCE_DEFAULT,
                    help="lab source directory (default: %(default)s)")
    ap.add_argument("--label", default=None,
                    help="copy a single label (e.g. Pos68-DMSO)")
    ap.add_argument("--condition", default=None,
                    choices=("WT", "KO", "GOF", "Y1", "OT", "DMSO"),
                    help="restrict to a single condition")
    ap.add_argument("--include-undetected", action="store_true",
                    help="also copy labels that haven't been detected "
                    "yet (default: only detect.state == 'done')")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be copied; do not write")
    ap.add_argument("--no-update-symlinks", action="store_true",
                    help="skip repointing by_condition symlinks")
    ap.add_argument("--verify-sha256", action="store_true",
                    help="stronger integrity check (slower)")
    ap.add_argument("--retries", type=int, default=3,
                    help="per-file retry count on I/O error")
    args = ap.parse_args()

    if not os.path.isdir(args.source):
        sys.exit(f"ERROR: source not mounted: {args.source}\n"
                 f"In Finder: Go > Connect to Server > "
                 f"smb://pathaklab")

    progress = load_progress()
    mode = ("include-undetected" if args.include_undetected
            else "detected")
    labels = select_labels(progress, mode, args.label, args.condition)
    if not labels:
        sys.exit("no labels match the selection criteria")

    plan = build_plan(labels, args.source)

    missing = [p for p in plan if p["tif_size"] is None]
    if missing:
        print(f"WARNING: {len(missing)} labels have no .ome.tif at "
              f"the lab source:", flush=True)
        for p in missing[:8]:
            print(f"  {p['label']}", flush=True)

    present = [p for p in plan if p["tif_size"] is not None]
    needed = _bytes_needed(present)
    free = _free_bytes(CACHE_DIR
                       if os.path.isdir(CACHE_DIR)
                       else os.path.dirname(CACHE_DIR.rstrip("/")))
    print(f"\n=== copy plan ===", flush=True)
    print(f"  source : {args.source}", flush=True)
    print(f"  dest   : {CACHE_DIR}", flush=True)
    print(f"  labels : {len(plan)} total ({len(present)} found, "
          f"{len(missing)} missing at source)", flush=True)
    print(f"  needed : {needed/1e9:.2f} GB (excludes files already "
          f"present at matching size)", flush=True)
    print(f"  free   : {free/1e9:.2f} GB local", flush=True)
    if needed > free * 0.9:
        sys.exit(f"ERROR: would use >90%% of free space — bail")

    if args.dry_run:
        print(f"\n=== dry-run: per-label outcome ===", flush=True)
        for p in plan:
            if p["tif_size"] is None:
                print(f"  {p['label']:14s}  MISSING at source")
                continue
            tif_state = "SKIP" if (
                os.path.exists(p["tif_dst"])
                and os.path.getsize(p["tif_dst"]) == p["tif_size"]
            ) else "COPY"
            sidecar_state = ("SKIP" if os.path.exists(p["sidecar_dst"])
                             else "WRITE")
            print(f"  {p['label']:14s}  cond={p['condition']:4s}  "
                  f"tif={tif_state}  sidecar={sidecar_state}  "
                  f"({p['tif_size']/1e9:.2f} GB)")
        return 0

    _install_signals()
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Tell Spotlight to ignore the cache so indexing doesn't burn CPU
    no_index = os.path.join(CACHE_DIR, ".metadata_never_index")
    if not os.path.exists(no_index):
        open(no_index, "w").close()

    t0 = time.time()
    stats = {"copied": 0, "skipped": 0, "missing": 0, "failed": 0,
             "symlinks": 0}
    for i, p in enumerate(plan, start=1):
        if _STOP["flag"]:
            print("\n[stop] honouring signal — exiting cleanly",
                  flush=True)
            break
        if p["tif_size"] is None:
            stats["missing"] += 1
            continue
        print(f"\n[{i}/{len(plan)}] {p['label']} "
              f"(condition={p['condition']})", flush=True)
        try:
            tif_outcome = copy_atomic(
                p["tif_src"], p["tif_dst"], p["label"], "tif",
                retries=args.retries,
                verify_sha256=args.verify_sha256)
            meta_outcome = copy_atomic(
                p["meta_src"], p["meta_dst"], p["label"], "meta",
                retries=args.retries,
                verify_sha256=args.verify_sha256)
        except Exception as e:
            print(f"  ✗ FAILED: {e!r}", flush=True)
            stats["failed"] += 1
            continue
        if tif_outcome == "copied":
            stats["copied"] += 1
        elif tif_outcome == "skipped":
            stats["skipped"] += 1
        write_sidecar(p["label"], p["sidecar_dst"])
        if not args.no_update_symlinks:
            update_local_symlinks(
                p["label"], p["condition"],
                p["tif_dst"], p["meta_dst"], p["sidecar_dst"])
            stats["symlinks"] += 1
        print(f"  ✓ {p['label']}: tif={tif_outcome}, "
              f"meta={meta_outcome}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== summary ===", flush=True)
    print(f"  elapsed     : {elapsed:.0f}s ({elapsed/60:.1f} min)",
          flush=True)
    print(f"  copied      : {stats['copied']}", flush=True)
    print(f"  skipped     : {stats['skipped']}", flush=True)
    print(f"  missing-src : {stats['missing']}", flush=True)
    print(f"  failed      : {stats['failed']}", flush=True)
    print(f"  symlinks    : {stats['symlinks']}", flush=True)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
