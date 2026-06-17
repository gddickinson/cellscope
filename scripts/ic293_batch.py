"""Orchestrator for the IC293 single-cell-crop batch.

Mirrors ic295_batch: per-recording work runs in a subprocess (a cellpose
segfault / OOM won't kill the driver), state lives in atomic progress.json,
a lock file prevents concurrent drivers, SIGTERM finishes the current crop
then exits cleanly. Crops are small single cells so each detect is minutes,
not the IC295 multi-cell hours.

Usage (run from cellpose4 so cpsam_dic loads):
  conda run -n cellpose4 python scripts/ic293_batch.py --phase detect
  conda run -n cellpose4 python scripts/ic293_batch.py --phase analyze
  conda run -n cellpose4 python scripts/ic293_batch.py --phase both
  conda run -n cellpose4 python scripts/ic293_batch.py --label Pos0-WT --phase detect
  conda run -n cellpose4 python scripts/ic293_batch.py --retry-failed --phase detect
"""
import os
import sys
import time
import signal
import argparse
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic293_common import (  # noqa: E402
    inventory_cache, detection_order, load_progress, save_progress,
    init_entry, set_phase, LOCK_FILE, LOGS_DIR)

PHASES = ("detect", "analyze", "both")
PHASE_CMDS = {"detect": "scripts/ic293_detect_one.py",
              "analyze": "scripts/ic293_analyze_one.py"}
_STOP = {"flag": False}


def acquire_lock():
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE) as f:
            owner = f.read().strip()
        sys.exit(f"another IC293 driver appears to be running (lock: {owner}).\n"
                 f"if none is, delete {LOCK_FILE} and retry.")
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        f.write(f"{os.getpid()} {datetime.utcnow().isoformat()}Z\n")


def release_lock():
    try:
        os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass


def install_signal_handlers():
    def _stop(signum, frame):
        _STOP["flag"] = True
        print(f"\n[driver] caught signal {signum}; exit after current crop.",
              flush=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)


def select_pending(progress, queue, phase, retry_failed, force=False):
    out = []
    for label in queue:
        st = progress.get(label, {}).get(phase, {}).get("state")
        if st == "done" and not force:
            continue
        if st == "failed" and not retry_failed:
            continue
        out.append(label)
    return out


def _run_subprocess(script_rel, label, log_path, force=False):
    cmd = [sys.executable, script_rel, label] + (["--force"] if force else [])
    err_buf = []
    with open(log_path, "a") as logf:
        logf.write(f"\n=== {datetime.utcnow().isoformat()}Z START "
                   f"{' '.join(cmd)} ===\n")
        logf.flush()
        try:
            proc = subprocess.Popen(
                cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in proc.stdout:
                logf.write(line); logf.flush()
                err_buf.append(line)
                if len(err_buf) > 200:
                    err_buf.pop(0)
            rc = proc.wait()
        except Exception as e:
            logf.write(f"\n=== driver exception: {e!r} ===\n")
            return 1, str(e)
        logf.write(f"=== {datetime.utcnow().isoformat()}Z END rc={rc} ===\n")
    return rc, "".join(err_buf)


def run_phase(phase, inv, queue, args):
    progress = load_progress()
    for label in queue:
        init_entry(progress, label, inv[label]["condition"])
    save_progress(progress)

    pending = select_pending(progress, queue, phase, args.retry_failed, args.force)
    if args.label:
        pending = [args.label] if args.label in queue else []
    if args.limit:
        pending = pending[:args.limit]
    print(f"[driver] phase={phase}: {len(pending)} pending (of {len(queue)})",
          flush=True)

    cmd_script = PHASE_CMDS[phase]
    for i, label in enumerate(pending, start=1):
        if _STOP["flag"]:
            break
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_path = os.path.join(LOGS_DIR, f"{label}.{phase}.log")
        progress = load_progress()
        set_phase(progress, label, phase, state="running", started=time.time())
        save_progress(progress)
        print(f"\n[{i}/{len(pending)}] {phase} {label} "
              f"(cond={inv[label]['condition']})", flush=True)
        t0 = time.time()
        rc, err = _run_subprocess(cmd_script, label, log_path, args.force)
        dur = time.time() - t0
        progress = load_progress()
        if rc == 0:
            set_phase(progress, label, phase, state="done", duration_s=dur,
                      finished=time.time(), error=None)
            print(f"   ✓ done in {dur:.0f}s", flush=True)
        else:
            tail = err.strip().splitlines()[-1] if err.strip() else ""
            set_phase(progress, label, phase, state="failed", duration_s=dur,
                      finished=time.time(), error=err[-2000:])
            print(f"   ✗ FAILED rc={rc} in {dur:.0f}s (tail: {tail})", flush=True)
        save_progress(progress)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=PHASES, default="detect")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run even recordings already marked done "
                    "(forwards --force to the per-recording worker)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    inv = inventory_cache()
    if not inv:
        sys.exit("No IC293 crops found in ic293_analysis/_cache/ — run "
                 "scripts/ic293_stage_crops.py first.")
    queue = detection_order(inv)

    acquire_lock()
    install_signal_handlers()
    try:
        if args.phase in ("detect", "both"):
            run_phase("detect", inv, queue, args)
        if args.phase in ("analyze", "both") and not _STOP["flag"]:
            run_phase("analyze", inv, queue, args)
    finally:
        release_lock()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[driver] fatal: {e!r}")
        release_lock()
        sys.exit(1)
