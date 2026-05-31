"""Guided manual mask review for the IC295 batch.

Backs up each recording's original masks before review, launches the
focused GUI on the recording's `.cellscope` project file (via the
remote-control RPC), waits for the user to close the window, then
computes a before/after diff and tracks the result.

Subcommands:
  status                 — review queue + per-condition counts
  next                   — open the next pending recording in the GUI
  open <label>           — open a specific recording in the GUI
  diff <label>           — print diff between masks.npz and masks_original.npz
  mark <label> --status  — manually mark accepted/edited/skipped/pending
  reanalyze-pending      — re-run Phase 2 (--force) on all edited recordings

Safety:
  • Refuses to open a recording whose Phase-2 analyze is currently
    running (avoids the watcher-read-during-save race).
  • Uses a separate review.lock so two reviewers can't open the same
    recording concurrently.
  • Backups (`pipeline_results/masks_original.npz`) are written once
    per recording and never overwritten — your "ground zero" is safe.

To compare original vs current inside the GUI, drag
`pipeline_results/masks_original.npz` onto the window to load the
backup, then drag `masks.npz` back to return to the current.
"""
import os
import sys
import json
import time
import shutil
import hashlib
import argparse
import subprocess
import urllib.request
import urllib.error
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    inventory_drive, recording_dir, load_progress,
    RECORDINGS_ROOT, RUNS_DIR, PROJECT_ROOT, CONDITIONS,
    atomic_write_json,
)

REVIEW_FILE      = os.path.join(RUNS_DIR, "review_state.json")
REVIEW_AUDIT     = os.path.join(RUNS_DIR, "review_audit.log")
REVIEW_LOCK      = os.path.join(RUNS_DIR, "review.lock")
REMOTE_PORT      = 8765
GUI_READY_TIMEOUT_S = 90


# ─────── state ───────
def load_review():
    if not os.path.exists(REVIEW_FILE):
        return {}
    try:
        with open(REVIEW_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_review(data):
    atomic_write_json(REVIEW_FILE, data)


def update_entry(label, condition, **fields):
    g = load_review()
    g.setdefault(condition, {}).setdefault(label, {}).update(fields)
    save_review(g)


def append_audit(line):
    os.makedirs(os.path.dirname(REVIEW_AUDIT), exist_ok=True)
    with open(REVIEW_AUDIT, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()}Z  {line}\n")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────── lock (prevent two reviewers on the same machine) ───────
def acquire_lock():
    if os.path.exists(REVIEW_LOCK):
        with open(REVIEW_LOCK) as f:
            owner = f.read().strip()
        sys.exit(f"review tool already has a session open (lock: {owner}).\n"
                 f"close the focused GUI window from that session, or "
                 f"remove {REVIEW_LOCK} if it's stale.")
    os.makedirs(os.path.dirname(REVIEW_LOCK), exist_ok=True)
    with open(REVIEW_LOCK, "w") as f:
        f.write(f"{os.getpid()} {datetime.utcnow().isoformat()}Z\n")


def release_lock():
    try:
        os.remove(REVIEW_LOCK)
    except FileNotFoundError:
        pass


# ─────── safety checks ───────
def is_ready_for_review(label, progress):
    entry = progress.get(label, {})
    det = entry.get("detect", {})
    if det.get("state") != "done":
        return False, f"detect state = {det.get('state', 'pending')}"
    ana = entry.get("analyze", {})
    if ana.get("state") == "running":
        return False, "analyze watcher is currently running on this recording"
    return True, None


def completed_recordings():
    """Sorted [(label, condition, rec_dir)] for all detect-done recordings."""
    inv = inventory_drive()
    progress = load_progress()
    out = []
    for label, info in inv.items():
        ok, _ = is_ready_for_review(label, progress)
        if not ok:
            continue
        rec_dir = recording_dir(label, info["condition"])
        if os.path.exists(os.path.join(rec_dir, "pipeline_results",
                                         "masks.npz")):
            out.append((label, info["condition"], rec_dir))
    out.sort(key=lambda r: (r[1], r[0]))
    return out


def entry_status(label, cond):
    return load_review().get(cond, {}).get(label, {}).get("status", "pending")


# ─────── diff ───────
def compute_diff(orig_path, edited_path):
    import numpy as np
    o = np.load(orig_path)
    e = np.load(edited_path)
    ol = o["labels"] if "labels" in o.files else o["masks"]
    el = e["labels"] if "labels" in e.files else e["masks"]
    if ol.shape != el.shape:
        return {"error": f"shape mismatch {ol.shape} vs {el.shape}"}

    # Foreground XOR per frame
    per_frame = []
    for i in range(ol.shape[0]):
        op = (ol[i] > 0)
        ep = (el[i] > 0)
        per_frame.append(int((op != ep).sum()))
    frames_touched = [i for i, c in enumerate(per_frame) if c > 0]

    # Per-cell area change (matched by ID — assumes IDs preserved)
    o_ids = set(np.unique(ol).tolist()) - {0}
    e_ids = set(np.unique(el).tolist()) - {0}
    per_cell = {}
    for cid in sorted(o_ids | e_ids):
        oa = int((ol == cid).sum())
        ea = int((el == cid).sum())
        if oa == 0 and ea == 0:
            continue
        if oa == 0:
            per_cell[str(int(cid))] = {"action": "added",
                                          "new_area": ea}
        elif ea == 0:
            per_cell[str(int(cid))] = {"action": "removed",
                                          "old_area": oa}
        elif oa != ea:
            pct = 100.0 * (ea - oa) / oa
            per_cell[str(int(cid))] = {
                "action": "resized",
                "old_area": oa, "new_area": ea,
                "change_pct": round(pct, 2),
            }
    return {
        "n_cells_original":     len(o_ids),
        "n_cells_edited":       len(e_ids),
        "n_frames_touched":     len(frames_touched),
        "frames_touched":       frames_touched[:30] + (
            ["..."] if len(frames_touched) > 30 else []),
        "total_pixels_changed": int(sum(per_frame)),
        "per_cell_change":      per_cell,
    }


def print_diff(diff):
    if "error" in diff:
        print(f"  ERROR: {diff['error']}")
        return
    print(f"  cells: {diff['n_cells_original']} → {diff['n_cells_edited']}")
    print(f"  frames touched: {diff['n_frames_touched']}")
    print(f"  total pixels changed: {diff['total_pixels_changed']:,}")
    if diff["per_cell_change"]:
        print("  per-cell changes:")
        for cid, ch in sorted(diff["per_cell_change"].items(),
                                key=lambda kv: int(kv[0])):
            a = ch["action"]
            if a == "added":
                print(f"    cell {cid}: ADDED ({ch['new_area']:,} px)")
            elif a == "removed":
                print(f"    cell {cid}: REMOVED ({ch['old_area']:,} px)")
            else:
                print(f"    cell {cid}: {ch['change_pct']:+.1f}% "
                      f"({ch['old_area']:,} → {ch['new_area']:,} px)")


# ─────── GUI launch ───────
def _wait_ready(timeout_s):
    url = f"http://127.0.0.1:{REMOTE_PORT}/status"
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def _load_project_via_rpc(cellscope_path):
    url = f"http://127.0.0.1:{REMOTE_PORT}/load_project"
    body = json.dumps({"path": cellscope_path}).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=30).read()
        return True
    except Exception as e:
        print(f"[review] WARN: load_project RPC failed: {e}",
              file=sys.stderr)
        return False


def open_in_gui(label):
    inv = inventory_drive()
    if label not in inv:
        print(f"unknown label: {label}", file=sys.stderr); return 2
    info = inv[label]
    cond = info["condition"]
    rec_dir = recording_dir(label, cond)
    cellscope_path = os.path.join(rec_dir, f"{label}.cellscope")
    if not os.path.exists(cellscope_path):
        print(f"missing .cellscope at {cellscope_path}", file=sys.stderr)
        return 2
    masks_path = os.path.join(rec_dir, "pipeline_results", "masks.npz")
    orig_path  = os.path.join(rec_dir, "pipeline_results",
                                "masks_original.npz")
    if not os.path.exists(masks_path):
        print(f"missing masks.npz at {masks_path}", file=sys.stderr)
        return 2

    progress = load_progress()
    ok, reason = is_ready_for_review(label, progress)
    if not ok:
        print(f"refusing to open {label}: {reason}", file=sys.stderr)
        return 3

    # One-time backup of the original masks (idempotent, never overwritten)
    if not os.path.exists(orig_path):
        shutil.copy2(masks_path, orig_path)
        print(f"[review] backed up original → {orig_path}")
    else:
        print(f"[review] original already backed up → {orig_path}")

    pre_hash = md5(masks_path)
    update_entry(label, cond,
                 status="in_review",
                 opened_at=datetime.utcnow().isoformat() + "Z")
    append_audit(f"OPEN {label} ({cond})")

    # Launch the focused GUI as a subprocess with CELLSCOPE_REMOTE=8765
    env = os.environ.copy()
    env["CELLSCOPE_REMOTE"] = str(REMOTE_PORT)
    cmd = ["conda", "run", "--no-capture-output", "-n", "cellpose4",
           "python", "main_focused.py"]
    print(f"[review] launching focused GUI on {label} (cond={cond})…")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)

    if _wait_ready(GUI_READY_TIMEOUT_S):
        _load_project_via_rpc(cellscope_path)
        print(f"[review] {label}.cellscope loaded — review, edit, save, close")
        print(f"  COMPARE: drag this file into the window to view original:")
        print(f"    {orig_path}")
        print(f"  then drag {os.path.basename(masks_path)} back to see current")
    else:
        print("[review] GUI didn't come up in time; project may need manual "
              "load (File → Open Project)", file=sys.stderr)

    proc.wait()

    # ───────── after close: capture edits ─────────
    post_hash = md5(masks_path)
    if pre_hash == post_hash:
        update_entry(label, cond,
                     status="accepted",
                     closed_at=datetime.utcnow().isoformat() + "Z")
        append_audit(f"ACCEPT {label} (no changes)")
        print(f"\n[review] {label}: no edits → marked accepted")
    else:
        diff = compute_diff(orig_path, masks_path)
        update_entry(label, cond,
                     status="edited",
                     closed_at=datetime.utcnow().isoformat() + "Z",
                     edits=diff,
                     needs_reanalysis=True)
        append_audit(f"EDIT {label}  cells={diff.get('n_cells_original')}"
                     f"→{diff.get('n_cells_edited')}  "
                     f"frames={diff.get('n_frames_touched')}  "
                     f"px_changed={diff.get('total_pixels_changed')}")
        print(f"\n[review] {label}: EDITED")
        print_diff(diff)
        print(f"\nTo re-analyze with edits (so Phase-2 outputs update):")
        print(f"  conda run -n cellpose4 python scripts/ic295_review.py "
              f"reanalyze-pending")
        print(f"  # or just {label} only:")
        print(f"  conda run -n cellpose4 python scripts/ic295_analyze_one.py "
              f"{label} --force")
    return 0


# ─────── subcommands ───────
def cmd_status(args):
    completed = completed_recordings()
    g = load_review()
    by_cond = {c: {"pending": 0, "in_review": 0,
                    "accepted": 0, "edited": 0, "skipped": 0}
               for c in CONDITIONS}
    detail = []
    for label, cond, _ in completed:
        st = g.get(cond, {}).get(label, {}).get("status", "pending")
        by_cond.setdefault(cond, {}).setdefault(st, 0)
        by_cond[cond][st] = by_cond[cond].get(st, 0) + 1
        detail.append((cond, label, st, g.get(cond, {}).get(label, {})))
    print(f"=== IC295 review status ===  ({len(completed)} ready)")
    print(f"  {'cond':>5} {'pending':>8} {'in_rev':>7} {'accepted':>9} "
          f"{'edited':>7} {'skipped':>8}")
    for c in CONDITIONS:
        r = by_cond[c]
        print(f"  {c:>5} {r.get('pending',0):>8} {r.get('in_review',0):>7} "
              f"{r.get('accepted',0):>9} {r.get('edited',0):>7} "
              f"{r.get('skipped',0):>8}")
    pending_reana = [(c, l) for c, recs in g.items()
                     for l, info in recs.items()
                     if info.get("needs_reanalysis")]
    if pending_reana:
        print(f"\n  {len(pending_reana)} recording(s) need re-analysis "
              f"after edits:")
        for cond, label in pending_reana[:10]:
            print(f"    {cond}/{label}")
    if args.verbose:
        print()
        for cond, label, st, _ in detail:
            print(f"  {cond:>5}  {label:11s}  [{st}]")
    return 0


def cmd_next(args):
    completed = completed_recordings()
    g = load_review()
    for label, cond, _ in completed:
        st = g.get(cond, {}).get(label, {}).get("status", "pending")
        if st == "pending":
            return open_in_gui(label)
    print("no pending recordings — all reviewed.")
    return 0


def cmd_open(args):
    return open_in_gui(args.label)


def cmd_diff(args):
    inv = inventory_drive()
    if args.label not in inv:
        print(f"unknown label: {args.label}", file=sys.stderr); return 2
    rec_dir = recording_dir(args.label, inv[args.label]["condition"])
    masks_path = os.path.join(rec_dir, "pipeline_results", "masks.npz")
    orig_path  = os.path.join(rec_dir, "pipeline_results",
                                "masks_original.npz")
    if not os.path.exists(orig_path):
        print(f"no backup at {orig_path}; was this recording ever "
              f"reviewed?")
        return 1
    print(f"=== diff: {args.label} ===")
    print_diff(compute_diff(orig_path, masks_path))
    return 0


def cmd_mark(args):
    inv = inventory_drive()
    if args.label not in inv:
        print(f"unknown label: {args.label}", file=sys.stderr); return 2
    cond = inv[args.label]["condition"]
    update_entry(args.label, cond,
                 status=args.status,
                 marked_at=datetime.utcnow().isoformat() + "Z")
    append_audit(f"MARK {args.label} {args.status}")
    print(f"marked {args.label} → {args.status}")
    return 0


def cmd_reanalyze_pending(args):
    g = load_review()
    pending = [(c, l) for c, recs in g.items()
               for l, info in recs.items()
               if info.get("needs_reanalysis")]
    if not pending:
        print("no recordings flagged for reanalysis"); return 0
    print(f"re-analyzing {len(pending)} edited recording(s):")
    for cond, label in pending:
        print(f"\n=== {cond}/{label} ===")
        rc = subprocess.run(
            ["conda", "run", "-n", "cellpose4", "python",
             "scripts/ic295_analyze_one.py", label, "--force"],
            cwd=PROJECT_ROOT,
        ).returncode
        if rc == 0:
            update_entry(label, cond,
                         needs_reanalysis=False,
                         reanalyzed_at=datetime.utcnow().isoformat() + "Z")
            append_audit(f"REANALYZE {label} (rc=0)")
        else:
            append_audit(f"REANALYZE {label} (rc={rc} — FAILED)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("status"); p.add_argument("-v", "--verbose",
                                                  action="store_true")
    p.set_defaults(func=cmd_status, needs_lock=False)
    sp.add_parser("next").set_defaults(func=cmd_next, needs_lock=True)
    p = sp.add_parser("open"); p.add_argument("label")
    p.set_defaults(func=cmd_open, needs_lock=True)
    p = sp.add_parser("diff"); p.add_argument("label")
    p.set_defaults(func=cmd_diff, needs_lock=False)
    p = sp.add_parser("mark"); p.add_argument("label")
    p.add_argument("--status", required=True,
                   choices=["pending", "accepted", "edited", "skipped"])
    p.set_defaults(func=cmd_mark, needs_lock=False)
    sp.add_parser("reanalyze-pending")\
      .set_defaults(func=cmd_reanalyze_pending, needs_lock=True)

    args = ap.parse_args()
    if getattr(args, "needs_lock", False):
        acquire_lock()
        try:
            return args.func(args)
        finally:
            release_lock()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
