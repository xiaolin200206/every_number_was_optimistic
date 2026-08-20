#!/usr/bin/env python3
"""Repair the unquoted 'cores' field in the Platform B logs.

The 'cores' column holds a comma-separated core list (e.g. 0,1,2,3) that was
written without quoting, so affected rows carry more fields than the header.
A naive pandas.read_csv() therefore misaligns every column to the right of
'l2d_cache' -- silently, with no error raised. This script rewrites the
affected files with the field properly quoted (RFC 4180), so that

    pandas.read_csv(path)

returns correct values with no custom reader.

Run from the repository root:
    python fix_cores_field.py            # report only
    python fix_cores_field.py --write    # rewrite in place
"""
import argparse
import csv
import glob
import os
import sys

# Columns that follow 'cores' in the header, in order.
TRAILING = ["ort_threads", "ort_version"]


def repair_row(header, fields):
    """Re-join the over-split 'cores' field. Returns a list of len(header)."""
    n_extra = len(fields) - len(header)
    if n_extra <= 0:
        return fields
    i = header.index("cores")
    cores = ",".join(fields[i:i + 1 + n_extra])
    return fields[:i] + [cores] + fields[i + 1 + n_extra:]


def process(path, write=False):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    if "cores" not in header:
        return 0
    bad = [r for r in data if len(r) != len(header)]
    if not bad:
        return 0
    fixed = [repair_row(header, r) for r in data]
    for r in fixed:
        if len(r) != len(header):
            sys.exit(f"{path}: could not repair a row ({len(r)} fields)")
    if write:
        with open(path, "w", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(header)
            w.writerows(fixed)
    return len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="rewrite files in place (default: report only)")
    ap.add_argument("--root", default=".", help="repository root")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.root, "data", "**", "*.csv"),
                             recursive=True))
    total = 0
    for p in paths:
        n = process(p, write=a.write)
        if n:
            total += n
            verb = "fixed" if a.write else "would fix"
            print(f"{verb} {n:>3} row(s)  {os.path.relpath(p, a.root)}")
    if total == 0:
        print("No malformed rows found.")
    elif not a.write:
        print(f"\n{total} row(s) affected. Re-run with --write to apply.")
    else:
        print(f"\n{total} row(s) repaired. "
              "Re-run verify_tables.py to confirm 44/44 still passes.")


if __name__ == "__main__":
    main()
