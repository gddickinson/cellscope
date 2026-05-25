"""Aggregate per-recording outputs from an IC295 batch run into one
corpus-wide CSV.

Reads each recording's `per_cell.csv` from a summaries directory
(typically `~/iCloud/cellscope_mini_run/results_summary/<label>/`),
tags every row with the recording label + parsed genotype, and writes
a single `corpus_per_cell.csv` for downstream stats.

Genotype is parsed from the recording label suffix:
  Pos*-WT   → WT
  Pos*-KO   → KO
  Pos*-GOF  → GOF
  Pos*-OT   → OT          (Y-27632 off-target / unknown treatment)
  Pos*-Y1   → Y1          (Y-27632 1 µM)
  Pos*-DMSO → DMSO        (vehicle control)
  anything else            → "unknown"

Usage:
  python scripts/aggregate_ic295_batch.py \\
      --summaries ~/iCloud/.../results_summary \\
      --out ~/iCloud/.../aggregated
"""
import os, sys, csv, json, glob, argparse, re


GENOTYPE_PATTERNS = [
    ("WT",   re.compile(r"-WT$",   re.IGNORECASE)),
    ("KO",   re.compile(r"-KO$",   re.IGNORECASE)),
    ("GOF",  re.compile(r"-GOF$",  re.IGNORECASE)),
    ("OT",   re.compile(r"-OT$",   re.IGNORECASE)),
    ("Y1",   re.compile(r"-Y1$",   re.IGNORECASE)),
    ("DMSO", re.compile(r"-DMSO$", re.IGNORECASE)),
]


def genotype_of(label):
    for name, pat in GENOTYPE_PATTERNS:
        if pat.search(label):
            return name
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", required=True,
                    help="Dir containing one subfolder per recording, "
                    "each with per_cell.csv")
    ap.add_argument("--out", required=True,
                    help="Output dir for corpus_per_cell.csv + "
                    "summary.json")
    args = ap.parse_args()

    sub_dirs = sorted([d for d in glob.glob(
        os.path.join(args.summaries, "*")) if os.path.isdir(d)])
    if not sub_dirs:
        print(f"No subfolders found under {args.summaries}")
        sys.exit(2)

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "corpus_per_cell.csv")
    out_summary = os.path.join(args.out, "summary.json")

    rows = []
    per_recording = []
    for d in sub_dirs:
        label = os.path.basename(d)
        genotype = genotype_of(label)
        cells_csv = os.path.join(d, "per_cell.csv")
        analysis_summary = os.path.join(d, "analysis_summary.json")
        n_cells_in_rec = 0
        if os.path.exists(cells_csv):
            with open(cells_csv) as f:
                reader = csv.DictReader(f)
                for r in reader:
                    r["recording"] = label
                    r["genotype"] = genotype
                    rows.append(r)
                    n_cells_in_rec += 1
        rec_info = {"recording": label, "genotype": genotype,
                    "n_cells": n_cells_in_rec}
        if os.path.exists(analysis_summary):
            with open(analysis_summary) as f:
                rec_info.update(json.load(f))
        per_recording.append(rec_info)

    if not rows:
        print(f"No per_cell.csv rows found across {len(sub_dirs)} "
              f"recordings — did the analytics step run?")
        sys.exit(2)

    fieldnames = ["recording", "genotype"] + [k for k in rows[0].keys()
        if k not in ("recording", "genotype")]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    summary = {
        "n_recordings": len(per_recording),
        "n_cells_total": len(rows),
        "per_genotype_recordings": {},
        "per_genotype_cells": {},
        "per_recording": per_recording,
    }
    for r in per_recording:
        g = r["genotype"]
        summary["per_genotype_recordings"][g] = (
            summary["per_genotype_recordings"].get(g, 0) + 1)
    for r in rows:
        g = r["genotype"]
        summary["per_genotype_cells"][g] = (
            summary["per_genotype_cells"].get(g, 0) + 1)
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {out_csv} ({len(rows)} cell rows across "
          f"{len(sub_dirs)} recordings)")
    print(f"Wrote {out_summary}")
    print(f"Recordings per genotype: "
          f"{summary['per_genotype_recordings']}")
    print(f"Cells per genotype: {summary['per_genotype_cells']}")


if __name__ == "__main__":
    main()
