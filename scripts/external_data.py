"""Check or (re)create the symlinks to data kept on the external drive.

The large untracked data (raw IC29x recordings in `*_analysis/_cache/`
and the cpsam/SAM2 model weights) live on FANTOM, listed in
`external_data.json`. Each repo path is a symlink to `<root>/<path>`,
so code and the ~500 absolute per-file symlinks into `_cache/` keep
working unchanged — as long as the drive is mounted.

  python scripts/external_data.py           # status of every item
  python scripts/external_data.py --link    # create missing links

`--link` replaces a local file or directory only when the external copy
has the same file count and total bytes (verify checksums first when
moving data that exists nowhere else).
"""
import os
import shutil
import sys

from _paths import PROJECT_ROOT, external_data


def tree_stats(path):
    """(n_files, total_bytes) of a file, or of every regular file under a dir."""
    if os.path.isfile(path):
        return 1, os.path.getsize(path)
    n = size = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            p = os.path.join(dp, f)
            if os.path.isfile(p) and not os.path.islink(p):
                n += 1
                size += os.path.getsize(p)
    return n, size


def status(item, root):
    local = PROJECT_ROOT / item
    target = root / item
    if os.path.islink(local):
        if os.readlink(local) != str(target):
            return "LINK-ELSEWHERE", f"-> {os.readlink(local)}"
        return ("OK", "") if target.exists() else (
            "DRIVE-MISSING", f"{target} not found (is the drive mounted?)")
    if local.exists():
        return "LOCAL", "real file/dir (not moved yet)"
    return "ABSENT", ""


def link(item, root):
    local = PROJECT_ROOT / item
    target = root / item
    if not target.exists():
        print(f"  skip {item}: {target} missing")
        return False
    if local.exists() and not os.path.islink(local):
        if tree_stats(local) != tree_stats(target):
            print(f"  REFUSE {item}: local and external copies differ")
            return False
        if local.is_dir():
            shutil.rmtree(local)
        else:
            os.remove(local)
    elif os.path.islink(local):
        os.unlink(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, local)
    print(f"  linked {item} -> {target}")
    return True


def main():
    root, items = external_data()
    print(f"External data root: {root}"
          f"{'' if root.exists() else '  (NOT MOUNTED)'}")
    if "--link" in sys.argv:
        for item in items:
            if status(item, root)[0] != "OK":
                link(item, root)
    bad = 0
    for item in items:
        st, msg = status(item, root)
        bad += st != "OK"
        print(f"  {st:<15} {item}  {msg}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
