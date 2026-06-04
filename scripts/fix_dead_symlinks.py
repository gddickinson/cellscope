"""Fix or remove every dead symlink in the project.

The GeorgeDrive external disk failed; many symlinks point into
`/Volumes/GeorgeDrive/...` (and a couple into a missing piezo1_analysis
path) and are now dangling. This tool, run from the repo root:

  1. REPOINT source recordings / metadata to `ic295_analysis/_cache/`
     when the same-named file is cached locally.
  2. REPOINT `gt_review/<Pos>_<COND>/pipeline_results/masks.npz` to the
     reviewed `ic295_analysis/by_condition/<COND>/<Pos>-<COND>/.../masks.npz`.
  3. MATERIALIZE `results` (a dead drive symlink) as a real local dir.
  4. REMOVE the remaining dead symlinks that have no local copy.

It only ever touches symlinks that are CURRENTLY dangling — never a
real file or directory, and never a `gt_masks/*.png`. Every removed
link's original target is recorded in DEAD_SYMLINK_RECOVERY.md so it
can be restored if the drive is ever recovered.

Use --dry-run to preview without changing anything.
"""
import os
import sys
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CACHE = os.path.join(ROOT, "ic295_analysis", "_cache")
BYCOND = os.path.join(ROOT, "ic295_analysis", "by_condition")
CONDS = {"WT", "KO", "GOF", "Y1", "OT", "DMSO"}

DRY = "--dry-run" in sys.argv


def find_dead():
    dead = []
    for dp, dns, fns in os.walk(ROOT):
        if os.sep + ".git" in dp:
            dns[:] = [d for d in dns if d != ".git"]
        for name in dns + fns:
            p = os.path.join(dp, name)
            if os.path.islink(p) and not os.path.exists(p):
                dead.append(p)
    return sorted(dead)


def bycond_masks(label_dir):
    """gt_review label 'Pos7_WT' -> by_condition/WT/Pos7-WT/.../masks.npz."""
    if "_" not in label_dir:
        return None
    pos, cond = label_dir.rsplit("_", 1)
    if cond not in CONDS:
        return None
    cand = os.path.join(BYCOND, cond, f"{pos}-{cond}",
                        "pipeline_results", "masks.npz")
    return cand if os.path.exists(cand) else None


def main():
    cache_names = set(os.listdir(CACHE)) if os.path.isdir(CACHE) else set()
    repointed, materialized, removed = [], [], []

    for p in find_dead():
        rel = os.path.relpath(p, ROOT)
        tgt = os.readlink(p)
        base = os.path.basename(p)

        # 1. results dir → materialize
        if rel == "results":
            if not DRY:
                os.remove(p)
                os.makedirs(p, exist_ok=True)
            materialized.append((rel, tgt))
            continue

        # 2. source / metadata cached locally → repoint to _cache
        if base in cache_names:
            newt = os.path.join(CACHE, base)
            if not DRY:
                os.remove(p)
                os.symlink(newt, p)
            repointed.append((rel, tgt, os.path.relpath(newt, ROOT)))
            continue

        # 3. gt_review pipeline_results/masks.npz → reviewed by_condition
        if (rel.startswith("gt_review" + os.sep)
                and rel.endswith(os.path.join("pipeline_results",
                                              "masks.npz"))):
            label_dir = rel.split(os.sep)[1]
            m = bycond_masks(label_dir)
            if m:
                if not DRY:
                    os.remove(p)
                    os.symlink(m, p)
                repointed.append((rel, tgt, os.path.relpath(m, ROOT)))
                continue

        # 4. no local copy → remove (provenance recorded below)
        if not DRY:
            os.remove(p)
        removed.append((rel, tgt))

    _report(repointed, materialized, removed)
    if not DRY and removed:
        _write_manifest(removed, repointed, materialized)


def _report(repointed, materialized, removed):
    tag = "[DRY-RUN] " if DRY else ""
    print(f"{tag}repointed: {len(repointed)}  "
          f"materialized: {len(materialized)}  removed: {len(removed)}\n")
    if materialized:
        print("MATERIALIZED (dead symlink → real dir):")
        for rel, tgt in materialized:
            print(f"  {rel}   (was → {tgt})")
        print()
    if removed:
        print("REMOVED (no local copy — target recorded in manifest):")
        for rel, tgt in removed:
            print(f"  {rel}\n      was → {tgt}")
        print()
    print(f"repointed {len(repointed)} source/metadata/masks links "
          f"to local copies (_cache / by_condition).")


def _write_manifest(removed, repointed, materialized):
    path = os.path.join(ROOT, "DEAD_SYMLINK_RECOVERY.md")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Dead-symlink recovery record",
        "",
        f"Generated {now} by `scripts/fix_dead_symlinks.py` after the "
        "GeorgeDrive external disk failed.",
        "",
        "If the drive is ever recovered, the **Removed** links below can "
        "be recreated with `ln -s <original target> <path>`. No real files "
        "or `gt_masks/*.png` were deleted — only dangling symlinks.",
        "",
        "## Removed (no local copy existed)",
        "",
        "| path | original target |",
        "|---|---|",
    ]
    for rel, tgt in removed:
        lines.append(f"| `{rel}` | `{tgt}` |")
    lines += ["", "## Repointed to local copies", "",
              "| path | old target | new target |", "|---|---|---|"]
    for rel, old, new in repointed:
        lines.append(f"| `{rel}` | `{old}` | `{new}` |")
    if materialized:
        lines += ["", "## Materialized as real directories", ""]
        for rel, tgt in materialized:
            lines.append(f"- `{rel}` (was → `{tgt}`)")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {os.path.relpath(path, ROOT)} "
          f"({len(removed)} removed, {len(repointed)} repointed)")


if __name__ == "__main__":
    main()
