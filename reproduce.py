"""Full reproduction: 20 repeats x 105 datasets x all methods, then analysis.

Pipeline:
  1. run_benchmark.py over the 105 datasets (LLM, SynthCore, OPRO, SNAP2 +
     EZR/Random/RRP baselines run automatically), writing results/status_tracker.json.
     The runner is crash-resilient: re-running resumes from the tracker.
  2. analysis/compute_wins.py   -> reports/bestranks.md
     analysis/compute_delta.py
     analysis/plot_relative.py  -> reports/snap2_relative_performance.png

Requires OPENROUTER_API_KEY in .env and the datasets in data/ (run fetch_data.py).
This is a large run (~thousands of LLM calls); use --repeats for a smaller pass.

Run:  python reproduce.py
      python reproduce.py --repeats 3            # quick partial pass
      python reproduce.py --datasets data/misc/auto93.csv   # single dataset
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
PY = sys.executable


def resolve_datasets():
    """Map the 105 names in datasets_105.txt to their paths under data/."""
    names = [ln.strip() for ln in open(os.path.join(ROOT, "datasets_105.txt")) if ln.strip()]
    found = {}
    for dp, _, files in os.walk(os.path.join(ROOT, "data")):
        for f in files:
            if f.endswith(".csv"):
                found.setdefault(f, os.path.join(dp, f))
    paths, missing = [], []
    for n in names:
        (paths.append(found[n]) if n in found else missing.append(n))
    return paths, missing


def run(cmd):
    print(f"\n$ {' '.join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd)}",
          flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="openrouter/openai/gpt-oss-120b")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Override the 105-dataset list (paths).")
    ap.add_argument("--skip-benchmark", action="store_true",
                    help="Only (re)run the analysis on an existing tracker.")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(ROOT, ".env")):
        print("ERROR: no .env found. Copy .env.example to .env and set OPENROUTER_API_KEY.")
        sys.exit(1)

    if not args.skip_benchmark:
        if args.datasets:
            datasets = args.datasets
        else:
            datasets, missing = resolve_datasets()
            if missing:
                print(f"ERROR: {len(missing)} of 105 datasets missing from data/. "
                      f"Run:  python fetch_data.py\n  e.g. {missing[:3]} ...")
                sys.exit(1)
        print(f"# Benchmark: {len(datasets)} datasets x {args.repeats} repeats x 7 methods")
        run([PY, "run_benchmark.py",
             "--datasets", *datasets,
             "--llm", args.llm,
             "--strategies", "LLM", "SynthCore", "OPRO", "SNAP2",
             "--repeats", str(args.repeats)])

    print("\n# Analysis")
    run([PY, "analysis/compute_wins.py"])
    run([PY, "analysis/compute_delta.py"])
    run([PY, "analysis/plot_relative.py"])

    print("\nDONE. See reports/bestranks.md and reports/snap2_relative_performance.png")
    print("Cost/latency: python analysis/perf_benchmark.py && python analysis/perf_report.py")


if __name__ == "__main__":
    main()
