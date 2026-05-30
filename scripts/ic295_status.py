"""Status reporter for the IC295 batch. Safe to run while batch is in
progress (read-only — no lock acquired)."""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    inventory_drive, load_progress, parse_condition,
    CONDITIONS, LOCK_FILE, LOGS_DIR,
)

STATES = ("pending", "running", "done", "failed")


def _hms(s):
    if s is None:
        return "—"
    s = int(s)
    h, m = divmod(s, 3600); m, s = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _state_counts(progress, queue, phase):
    counts = {s: 0 for s in STATES}
    for label in queue:
        st = (progress.get(label, {}).get(phase, {}).get("state")
              or "pending")
        counts[st] = counts.get(st, 0) + 1
    return counts


def _mean_duration(progress, phase):
    durs = [e[phase]["duration_s"] for e in progress.values()
            if e.get(phase, {}).get("state") == "done"
               and isinstance(e[phase].get("duration_s"), (int, float))]
    return sum(durs) / len(durs) if durs else None


def _by_condition(progress, queue, inv, phase):
    """Return dict cond -> {state: count} restricted to known queue."""
    out = {c: {s: 0 for s in STATES} for c in CONDITIONS}
    for label in queue:
        cond = inv[label]["condition"]
        st = (progress.get(label, {}).get(phase, {}).get("state")
              or "pending")
        out[cond][st] = out[cond].get(st, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failed", action="store_true",
                    help="show failed recordings + error tail")
    ap.add_argument("--phase", choices=("detect", "analyze", "both"),
                    default="both")
    args = ap.parse_args()

    inv = inventory_drive()
    if not inv:
        print("No IC295 recordings on the drive — is the drive mounted?")
        return 1
    progress = load_progress()
    queue = sorted(inv.keys(),
                    key=lambda l: (inv[l]["condition"], l))

    print(f"=== IC295 batch status ===  ({len(queue)} recordings)")
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE) as f:
            print(f"  driver: RUNNING  (lock: {f.read().strip()})")
    else:
        print(f"  driver: idle")
    print()

    phases = (args.phase,) if args.phase != "both" else ("detect", "analyze")

    for phase in phases:
        counts = _state_counts(progress, queue, phase)
        avg = _mean_duration(progress, phase)
        remaining = counts["pending"] + counts["running"]
        eta = remaining * avg if avg else None
        print(f"── PHASE: {phase} ──")
        print(f"  done={counts['done']:>3}  running={counts['running']:>2}  "
              f"failed={counts['failed']:>2}  pending={counts['pending']:>3}"
              f"   (avg/rec: {_hms(avg)}, ETA: {_hms(eta)})")
        bc = _by_condition(progress, queue, inv, phase)
        print(f"  {'cond':>5s} {'done':>5} {'run':>4} {'fail':>4} "
              f"{'pend':>5} {'total':>6}")
        for c in CONDITIONS:
            r = bc[c]
            tot = sum(r.values())
            print(f"  {c:>5s} {r['done']:>5} {r['running']:>4} "
                  f"{r['failed']:>4} {r['pending']:>5} {tot:>6}")
        print()

    # Currently running
    running_now = [(l, e) for l, e in progress.items()
                   if e.get("detection", {}).get("state") == "running"
                   or e.get("analysis", {}).get("state") == "running"]
    if running_now:
        print("── currently running ──")
        for l, e in running_now:
            for ph in ("detection", "analysis"):
                if e.get(ph, {}).get("state") == "running":
                    started = e[ph].get("started")
                    age = (time.time() - started) if started else None
                    print(f"  {ph:>9s}  {l}  ({_hms(age)})")
        print()

    if args.failed:
        fails = [(l, e) for l, e in progress.items()
                 if e.get("detection", {}).get("state") == "failed"
                 or e.get("analysis", {}).get("state") == "failed"]
        if fails:
            print("── FAILED recordings ──")
            for l, e in fails:
                for ph in ("detection", "analysis"):
                    if e.get(ph, {}).get("state") == "failed":
                        err = (e[ph].get("error") or "").strip()
                        tail = err.splitlines()[-1] if err else ""
                        print(f"  {ph:>9s}  {l}  → {tail[:140]}")
            print(f"\n(full logs in {LOGS_DIR}/<label>.<phase>.log)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
