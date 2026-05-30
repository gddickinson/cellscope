"""Local-SSD prefetch daemon for the IC295 detect phase.

Keeps the next `--lookahead N=5` undetected recordings copied to
`ic295_analysis/_cache/` so the detect driver reads them from local
SSD instead of `/Volumes/GeorgeDrive`. Evicts files whose detection
is `done` (no longer needed) — never evicts a recording currently
being read, since eviction is keyed on the explicit `done` state.

Safe to run alongside `ic295_batch.py --phase=detect`:
- Reads `progress.json` (atomic writes from the driver are safe to read).
- `ic295_detect_one.best_video_path()` picks the cache file when
  present; otherwise reads from drive — graceful fallback.
- Eviction guarded by `state == 'done'`, so it never deletes the
  currently-running recording's TIF mid-read.

Disk-pressure safeguards:
- Won't START if free space < `--disk-refuse` (default 30 GB).
- Won't COPY a recording that would push free space below
  `--disk-floor` (default 50 GB). Logs and skips.

Usage:
  nohup bash -lc 'conda run -n cellpose4 python scripts/ic295_prefetch.py' \\
    > ic295_analysis/_runs/prefetch.log 2>&1 &
  disown
"""
import os
import sys
import time
import shutil
import signal
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    inventory_drive, detection_priority_order, load_progress,
    CACHE_DIR, RUNS_DIR, ANALYSIS_ROOT,
)

LOCK = os.path.join(RUNS_DIR, "prefetch.lock")
DEFAULT_LOOKAHEAD  = 5
DEFAULT_POLL_SEC   = 120
DEFAULT_DISK_FLOOR_GB  = 50
DEFAULT_DISK_REFUSE_GB = 30

_STOP = {"flag": False}


def acquire_lock():
    if os.path.exists(LOCK):
        with open(LOCK) as f:
            owner = f.read().strip()
        sys.exit(f"prefetcher already running (lock: {owner}).\n"
                 f"if you're sure none is, delete {LOCK}.")
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    with open(LOCK, "w") as f:
        f.write(f"{os.getpid()} {datetime.utcnow().isoformat()}Z\n")


def release_lock():
    try:
        os.remove(LOCK)
    except FileNotFoundError:
        pass


def install_signals():
    def _stop(signum, frame):
        _STOP["flag"] = True
        print(f"\n[prefetch] caught signal {signum}; exiting after "
              f"current copy.", flush=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)


def free_gb(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def cached_labels(inv):
    out = set()
    if not os.path.isdir(CACHE_DIR):
        return out
    files = set(os.listdir(CACHE_DIR))
    for label, info in inv.items():
        if os.path.basename(info["video_path"]) in files:
            out.add(label)
    return out


def target_labels(inv, progress, n_lookahead):
    """First N priority-queue labels whose detection is not 'done'."""
    pri = detection_priority_order(inv)
    target = []
    for label in pri:
        if len(target) >= n_lookahead:
            break
        st = progress.get(label, {}).get("detect", {}).get("state")
        if st == "done":
            continue
        target.append(label)
    return set(target)


def copy_recording(info, disk_floor):
    """Copy .ome.tif + .ome.json (+ _metadata.txt if present) into cache.
    Refuses if free space would drop below `disk_floor` after the copy."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tif = info["video_path"]
    name = os.path.basename(tif)
    cache_tif = os.path.join(CACHE_DIR, name)
    if not os.path.exists(tif):
        print(f"[prefetch]   drive .ome.tif missing: {tif}",
              flush=True)
        return False
    if os.path.exists(cache_tif):
        # Already there — just top up adjacent files
        _copy_adjacent(info)
        return True
    sz_gb = os.path.getsize(tif) / (1024 ** 3)
    fg = free_gb(CACHE_DIR)
    if fg - sz_gb < disk_floor:
        print(f"[prefetch]   skip {info['label']}: would leave "
              f"{fg - sz_gb:.1f} GB free (< floor {disk_floor:.0f})",
              flush=True)
        return False
    print(f"[prefetch]   copy {info['label']} ({sz_gb:.1f} GB)…",
          flush=True)
    t0 = time.time()
    shutil.copy2(tif, cache_tif)
    dt = time.time() - t0
    rate = sz_gb * 1024 / max(dt, 0.001)
    print(f"[prefetch]   ✓ {info['label']} cached in "
          f"{dt:.0f}s (~{rate:.0f} MB/s); free now "
          f"{free_gb(CACHE_DIR):.1f} GB", flush=True)
    _copy_adjacent(info)
    return True


def _copy_adjacent(info):
    for src in (info.get("json_sidecar"), info.get("metadata_txt")):
        if src and os.path.exists(src):
            dst = os.path.join(CACHE_DIR, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)


def evict_label(info):
    """Delete .ome.tif + adjacent cached sidecars for one label."""
    n = 0
    for kind in ("video_path", "json_sidecar", "metadata_txt"):
        src = info.get(kind)
        if not src:
            continue
        path = os.path.join(CACHE_DIR, os.path.basename(src))
        if os.path.exists(path):
            try:
                os.remove(path)
                n += 1
            except OSError as e:
                print(f"[prefetch]   evict warn: {path}: {e}",
                      flush=True)
    if n:
        print(f"[prefetch]   evicted {info['label']} ({n} files); "
              f"free now {free_gb(CACHE_DIR):.1f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD,
                    help=f"keep N recordings ahead in cache "
                    f"(default {DEFAULT_LOOKAHEAD})")
    ap.add_argument("--interval", type=int, default=DEFAULT_POLL_SEC,
                    help=f"seconds between checks "
                    f"(default {DEFAULT_POLL_SEC})")
    ap.add_argument("--disk-floor", type=float,
                    default=DEFAULT_DISK_FLOOR_GB,
                    help="don't copy if free space would drop below "
                    "this many GB (default 50)")
    ap.add_argument("--disk-refuse", type=float,
                    default=DEFAULT_DISK_REFUSE_GB,
                    help="don't start at all if free space is below "
                    "this many GB (default 30)")
    ap.add_argument("--once", action="store_true",
                    help="single pass and exit")
    args = ap.parse_args()

    fg0 = free_gb(ANALYSIS_ROOT)
    if fg0 < args.disk_refuse:
        sys.exit(f"prefetch refused: only {fg0:.1f} GB free, need "
                 f">= {args.disk_refuse:.0f} GB to start safely")

    acquire_lock()
    install_signals()
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[prefetch] start pid={os.getpid()} "
          f"lookahead={args.lookahead} interval={args.interval}s "
          f"floor={args.disk_floor}GB refuse={args.disk_refuse}GB",
          flush=True)
    print(f"[prefetch] free disk at start: {fg0:.1f} GB", flush=True)
    print(f"[prefetch] cache dir: {CACHE_DIR}", flush=True)
    try:
        while not _STOP["flag"]:
            inv = inventory_drive()
            progress = load_progress()
            target = target_labels(inv, progress, args.lookahead)
            cached = cached_labels(inv)

            # Evict cached recordings whose detection is done.
            for label in sorted(cached):
                st = progress.get(label, {}).get("detect", {})\
                                            .get("state")
                if st == "done":
                    evict_label(inv[label])
            cached = cached_labels(inv)

            # Add missing (sorted so logs are predictable).
            for label in sorted(target - cached):
                if _STOP["flag"]:
                    break
                copy_recording(inv[label], args.disk_floor)

            missing = sorted(target - cached_labels(inv))
            print(f"[prefetch] cache={sorted(cached_labels(inv))}, "
                  f"target={sorted(target)}, "
                  f"free={free_gb(CACHE_DIR):.1f}GB"
                  f"{'  missing=' + str(missing) if missing else ''}",
                  flush=True)

            if args.once:
                break
            time.sleep(args.interval)
    finally:
        release_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
