"""Long-running orchestrator for the IC295 batch analysis.

Usage:
  python scripts/ic295_batch.py --phase detect          # Phase 1
  python scripts/ic295_batch.py --phase analyze         # Phase 2
  python scripts/ic295_batch.py --phase both            # 1 then 2
  python scripts/ic295_batch.py --status                # see ic295_status.py
  python scripts/ic295_batch.py --retry-failed --phase detect
  python scripts/ic295_batch.py --label Pos21-KO --phase analyze

Per-recording work runs in a subprocess so a hard crash (segfault /
OOM / cellpose blowup) doesn't take down the driver. State lives in
ic295_analysis/_runs/progress.json (atomic writes), with per-recording
stderr/stdout in _runs/logs/<label>.log. A lock file prevents two
drivers from running simultaneously. SIGTERM finishes the current
recording, marks it, and exits cleanly.
"""
import os
import sys
import time
import json
import signal
import argparse
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    inventory_drive, detection_priority_order, has_drive_detection,
    recording_dir, load_progress, save_progress, init_entry, set_phase,
    parse_condition, LOCK_FILE, LOGS_DIR, RUNS_DIR, ANALYSIS_ROOT,
    CONDITIONS,
)

PHASES = ("detect", "analyze", "both")
_STOP = {"flag": False}
PHASE_CMDS = {
    "detect":  "scripts/ic295_detect_one.py",
    "analyze": "scripts/ic295_analyze_one.py",
}


# ─────── lock / signals ───────
def acquire_lock():
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE) as f:
            owner = f.read().strip()
        sys.exit(f"another driver appears to be running "
                 f"(lock owner: {owner}).\n"
                 f"if you're sure none is, delete {LOCK_FILE} and retry.")
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
        print(f"\n[driver] caught signal {signum}; will exit after "
              f"current recording.", flush=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)


# ─────── pending-recording selection ───────
def _is_done(progress, label, phase):
    entry = progress.get(label, {})
    return entry.get(phase, {}).get("state") == "done"


def _is_failed(progress, label, phase):
    entry = progress.get(label, {})
    return entry.get(phase, {}).get("state") == "failed"


def select_pending(progress, queue, phase, retry_failed):
    out = []
    for label in queue:
        st = progress.get(label, {}).get(phase, {}).get("state")
        if st == "done":
            continue
        if st == "failed" and not retry_failed:
            continue
        out.append(label)
    return out


# ─────── run a single phase ───────
def run_phase(phase, inv, queue, args):
    progress = load_progress()
    # Seed entries so the queue + progress always have all recordings
    for label in queue:
        init_entry(progress, label, inv[label]["condition"])
    save_progress(progress)

    pending = select_pending(progress, queue, phase, args.retry_failed)
    if args.label:
        if args.label not in queue:
            print(f"[driver] {args.label} not in queue; skipping", flush=True)
            return
        pending = [args.label]
    if args.limit:
        pending = pending[: args.limit]

    print(f"[driver] phase={phase}: {len(pending)} pending "
          f"(of {len(queue)} total)", flush=True)

    cmd_script = PHASE_CMDS[phase]
    for i, label in enumerate(pending, start=1):
        if _STOP["flag"]:
            break
        info = inv[label]
        log_path = os.path.join(LOGS_DIR, f"{label}.{phase}.log")
        os.makedirs(LOGS_DIR, exist_ok=True)

        progress = load_progress()
        set_phase(progress, label, phase,
                  state="running", started=time.time())
        save_progress(progress)

        print(f"\n[{i}/{len(pending)}] {phase} {label} "
              f"(condition={info['condition']})", flush=True)

        t0 = time.time()
        rc, err_tail = _run_subprocess(cmd_script, label, log_path)
        dur = time.time() - t0

        progress = load_progress()
        if rc == 0:
            set_phase(progress, label, phase,
                      state="done", duration_s=dur,
                      finished=time.time(), error=None)
            print(f"   ✓ done in {dur:.0f}s", flush=True)
        else:
            set_phase(progress, label, phase,
                      state="failed", duration_s=dur,
                      finished=time.time(), error=err_tail[-2000:])
            print(f"   ✗ FAILED rc={rc} in {dur:.0f}s (tail: "
                  f"{err_tail.strip().splitlines()[-1] if err_tail.strip() else ''})",
                  flush=True)
        save_progress(progress)


def _run_subprocess(script_rel, label, log_path):
    """Run scripts/<script_rel> <label>, tee output to log_path,
    return (returncode, stderr-tail)."""
    cmd = [sys.executable, script_rel, label]
    err_buf = []
    with open(log_path, "a") as logf:
        logf.write(f"\n=== {datetime.utcnow().isoformat()}Z START "
                   f"{' '.join(cmd)} ===\n")
        logf.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                err_buf.append(line)
                if len(err_buf) > 200:
                    err_buf.pop(0)
            rc = proc.wait()
        except Exception as e:
            logf.write(f"\n=== driver exception: {e!r} ===\n")
            return 1, str(e)
        logf.write(f"=== {datetime.utcnow().isoformat()}Z END "
                   f"rc={rc} ===\n")
    return rc, "".join(err_buf)


# ─────── entry point ───────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=PHASES, default="detect")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-run recordings previously marked failed")
    ap.add_argument("--label", default=None,
                    help="restrict to a single recording")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap how many pending recordings to process")
    ap.add_argument("--analyze-only-detected", action="store_true",
                    help="(analyze phase) only include recordings whose "
                    "detection state is 'done'")
    args = ap.parse_args()

    inv = inventory_drive()
    if not inv:
        sys.exit("No IC295 recordings found on the drive — is "
                 "/Volumes/GeorgeDrive mounted?")
    queue = detection_priority_order(inv)

    if args.analyze_only_detected:
        progress = load_progress()
        queue = [l for l in queue
                 if progress.get(l, {}).get("detection", {})
                                       .get("state") == "done"]

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
