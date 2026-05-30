"""Long-running Phase-2 watcher.

Polls every `--interval` seconds for recordings with
`pipeline_results/masks.npz` but no `analysis.json` and runs
`ic295_analyze_one.py` on each one **sequentially** (analysis uses
~7 GB on a 2048² stack — running multiple in parallel risks memory
pressure).

Designed to run concurrently with `ic295_batch.py --phase=detect`:
uses a separate lock file (`_runs/analyze.lock`) so it doesn't fight
the detect driver for the main `_runs/lock.txt`.

Usage:
  python scripts/ic295_analyze_watch.py                # loop forever
  python scripts/ic295_analyze_watch.py --once         # single pass
  python scripts/ic295_analyze_watch.py --interval 30  # poll every 30s
"""
import os
import sys
import time
import glob
import signal
import argparse
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    RUNS_DIR, LOGS_DIR, RECORDINGS_ROOT, PROJECT_ROOT,
    load_progress, save_progress, set_phase,
)

ANALYZE_LOCK = os.path.join(RUNS_DIR, "analyze.lock")
# Default: 10 min. Analysis is ~2 min/recording and detections take
# hours, so checking more often than this just wastes wakeups.
POLL_INTERVAL = 600
_STOP = {"flag": False}


def acquire_lock():
    if os.path.exists(ANALYZE_LOCK):
        with open(ANALYZE_LOCK) as f:
            owner = f.read().strip()
        sys.exit(f"another analyze watcher appears to be running "
                 f"(lock: {owner}).\n"
                 f"if you're sure none is, delete {ANALYZE_LOCK}.")
    os.makedirs(os.path.dirname(ANALYZE_LOCK), exist_ok=True)
    with open(ANALYZE_LOCK, "w") as f:
        f.write(f"{os.getpid()} {datetime.utcnow().isoformat()}Z\n")


def release_lock():
    try:
        os.remove(ANALYZE_LOCK)
    except FileNotFoundError:
        pass


def install_signals():
    def _stop(signum, frame):
        _STOP["flag"] = True
        print(f"\n[watch] caught signal {signum}; finishing current "
              f"recording then exiting.", flush=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)


def find_pending():
    """Recordings with masks.npz but no analysis.json. Returns sorted
    list of (label, condition, rec_dir) tuples."""
    out = []
    for masks_path in sorted(glob.glob(
            os.path.join(RECORDINGS_ROOT, "*", "*",
                          "pipeline_results", "masks.npz"))):
        rec_dir = os.path.dirname(os.path.dirname(masks_path))
        label   = os.path.basename(rec_dir)
        cond    = os.path.basename(os.path.dirname(rec_dir))
        if os.path.exists(os.path.join(rec_dir, "analysis.json")):
            continue
        out.append((label, cond, rec_dir))
    return out


def run_one(label, cond):
    log_path = os.path.join(LOGS_DIR, f"{label}.analyze.log")
    os.makedirs(LOGS_DIR, exist_ok=True)
    cmd = [sys.executable, "scripts/ic295_analyze_one.py", label]

    progress = load_progress()
    set_phase(progress, label, "analysis",
              state="running", started=time.time())
    save_progress(progress)

    print(f"[watch] ▶ analyze {label} ({cond})", flush=True)
    t0 = time.time()
    with open(log_path, "a") as logf:
        logf.write(f"\n=== {datetime.utcnow().isoformat()}Z START "
                   f"{' '.join(cmd)} ===\n")
        logf.flush()
        try:
            rc = subprocess.run(
                cmd, cwd=PROJECT_ROOT,
                stdout=logf, stderr=subprocess.STDOUT,
            ).returncode
        except Exception as e:
            logf.write(f"\n=== watcher exception: {e!r} ===\n")
            rc = 1
        logf.write(f"=== {datetime.utcnow().isoformat()}Z "
                   f"END rc={rc} ===\n")
    dur = time.time() - t0

    progress = load_progress()
    if rc == 0:
        set_phase(progress, label, "analysis",
                  state="done", duration_s=dur,
                  finished=time.time(), error=None)
        print(f"[watch]   ✓ {label} done in {dur:.0f}s", flush=True)
    else:
        set_phase(progress, label, "analysis",
                  state="failed", duration_s=dur,
                  finished=time.time(),
                  error=f"rc={rc}; see {log_path}")
        print(f"[watch]   ✗ {label} FAILED rc={rc} (see {log_path})",
              flush=True)
    save_progress(progress)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="single pass; exit when no pending recordings")
    ap.add_argument("--interval", type=int, default=POLL_INTERVAL,
                    help=f"seconds between polls (default {POLL_INTERVAL})")
    args = ap.parse_args()

    acquire_lock()
    install_signals()
    print(f"[watch] starting analyze watcher pid={os.getpid()}, "
          f"poll={args.interval}s", flush=True)
    try:
        empty_streak = 0
        while not _STOP["flag"]:
            pending = find_pending()
            if not pending:
                if args.once:
                    print("[watch] no pending; exiting (--once)",
                          flush=True)
                    break
                empty_streak += 1
                if empty_streak == 1 or empty_streak % 20 == 0:
                    print(f"[watch] no pending; sleeping "
                          f"{args.interval}s (streak={empty_streak})",
                          flush=True)
                time.sleep(args.interval)
                continue
            empty_streak = 0
            print(f"[watch] {len(pending)} recording(s) pending analysis",
                  flush=True)
            for label, cond, _ in pending:
                if _STOP["flag"]:
                    break
                run_one(label, cond)
    finally:
        release_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
