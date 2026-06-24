"""Populate data/ with the 105 benchmark datasets.

103 come from the public MOOT suite (https://github.com/timm/moot); the 2 hpo
datasets are already bundled. Clones MOOT (shallow) into a cache and copies the
listed CSVs into data/<category>/, then verifies all 105 are present.

Run:  python fetch_data.py
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
MOOT_URL = "https://github.com/timm/moot"
CACHE = os.path.join(ROOT, ".moot_cache")


def wanted():
    with open(os.path.join(ROOT, "datasets_105.txt")) as f:
        return {ln.strip() for ln in f if ln.strip()}


def present():
    found = {}
    for dp, _, files in os.walk(DATA):
        for f in files:
            if f.endswith(".csv"):
                found[f] = os.path.join(dp, f)
    return found


def main():
    need = wanted()
    have = present()
    missing = need - set(have)
    print(f"want {len(need)} datasets; {len(need) - len(missing)} already in data/")
    if not missing:
        print("All 105 present. Nothing to fetch.")
        return

    if not os.path.isdir(CACHE):
        print(f"Cloning MOOT (shallow) into {CACHE} ...")
        subprocess.run(["git", "clone", "--depth", "1", MOOT_URL, CACHE], check=True)

    moot = {}
    for dp, _, files in os.walk(os.path.join(CACHE, "optimize")):
        for f in files:
            if f.endswith(".csv"):
                moot.setdefault(f, os.path.join(dp, f))

    copied, still_missing = 0, []
    for name in sorted(missing):
        src = moot.get(name)
        if not src:
            still_missing.append(name)
            continue
        cat = os.path.basename(os.path.dirname(src))   # preserve category subdir
        dst_dir = os.path.join(DATA, cat)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, name))
        copied += 1
    print(f"copied {copied} datasets from MOOT")

    if still_missing:
        print(f"\nERROR: {len(still_missing)} datasets not found in MOOT:")
        for n in still_missing:
            print("  ", n)
        sys.exit(1)
    print(f"\nDone. data/ now has all {len(need)} datasets. "
          f"(You may delete {CACHE}.)")


if __name__ == "__main__":
    main()
