"""Status reporter for the IC293 batch. Read-only (no lock)."""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic293_common import (  # noqa: E402
    inventory_cache, load_progress, CONDITIONS, LOCK_FILE, LOGS_DIR)

STATES = ("pending", "running", "done", "failed")


def _hms(s):
    if s is None:
        return "—"
    s = int(s)
    h, m = divmod(s, 3600)
    m, s = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _counts(progress, queue, phase):
    c = {s: 0 for s in STATES}
    for label in queue:
        st = progress.get(label, {}).get(phase, {}).get("state") or "pending"
        c[st] = c.get(st, 0) + 1
    return c


def _mean_dur(progress, phase):
    durs = [e[phase]["duration_s"] for e in progress.values()
            if e.get(phase, {}).get("state") == "done"
            and isinstance(e[phase].get("duration_s"), (int, float))]
    return sum(durs) / len(durs) if durs else None


def _by_condition(progress, queue, inv, phase):
    out = {c: {s: 0 for s in STATES} for c in CONDITIONS}
    for label in queue:
        cond = inv[label]["condition"]
        st = progress.get(label, {}).get(phase, {}).get("state") or "pending"
        out[cond][st] = out[cond].get(st, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failed", action="store_true")
    ap.add_argument("--phase", choices=("detect", "analyze", "both"),
                    default="both")
    args = ap.parse_args()

    inv = inventory_cache()
    if not inv:
        print("No IC293 crops in ic293_analysis/_cache/.")
        return 1
    progress = load_progress()
    queue = sorted(inv.keys(), key=lambda l: (inv[l]["condition"], l))

    print(f"=== IC293 batch status ===  ({len(queue)} crops)")
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE) as f:
            print(f"  driver: RUNNING  (lock: {f.read().strip()})")
    else:
        print("  driver: idle")
    print()

    phases = (args.phase,) if args.phase != "both" else ("detect", "analyze")
    for phase in phases:
        c = _counts(progress, queue, phase)
        avg = _mean_dur(progress, phase)
        remaining = c["pending"] + c["running"]
        eta = remaining * avg if avg else None
        print(f"── PHASE: {phase} ──")
        print(f"  done={c['done']:>3}  running={c['running']:>2}  "
              f"failed={c['failed']:>2}  pending={c['pending']:>3}"
              f"   (avg/crop: {_hms(avg)}, ETA: {_hms(eta)})")
        bc = _by_condition(progress, queue, inv, phase)
        print(f"  {'cond':>5s} {'done':>5} {'run':>4} {'fail':>4} "
              f"{'pend':>5} {'total':>6}")
        for cond in CONDITIONS:
            r = bc[cond]
            print(f"  {cond:>5s} {r['done']:>5} {r['running']:>4} "
                  f"{r['failed']:>4} {r['pending']:>5} {sum(r.values()):>6}")
        print()

    if args.failed:
        fails = [(l, e) for l, e in progress.items()
                 for ph in ("detect", "analyze")
                 if e.get(ph, {}).get("state") == "failed"]
        seen = set()
        if fails:
            print("── FAILED ──")
            for l, e in fails:
                if l in seen:
                    continue
                seen.add(l)
                for ph in ("detect", "analyze"):
                    if e.get(ph, {}).get("state") == "failed":
                        err = (e[ph].get("error") or "").strip()
                        tail = err.splitlines()[-1] if err else ""
                        print(f"  {ph:>8s}  {l}  → {tail[:140]}")
            print(f"\n(full logs in {LOGS_DIR}/<label>.<phase>.log)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
