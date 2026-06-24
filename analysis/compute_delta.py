# -*- coding: utf-8 -*-
"""compute_delta.py — Δ (normalized validation regret) per dataset/method.

Δ = 1 - (score - baseline_min) / (baseline_median - baseline_min), computed
from d2h_scores. baseline_{min,median} = empirical optimum / no-tuning median
over every row of the dataset CSV. Higher Δ = better (100% = optimum recovered).

Reads one tracker (default results/status_tracker.json, written by
run_benchmark.py) holding all 7 finalized methods. Exploit is shown as EZR.
Provides importable helpers used by plot_relative.py.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.ezr import DATA, csv as ezcsv, d2h  # noqa: E402

LLM_KEY = os.environ.get("REPRO_LLM_KEY", "openrouter/openai/gpt-oss-120b")
TRACKER = os.environ.get(
    "REPRO_TRACKER",
    os.path.join(PROJECT_ROOT, "results", "status_tracker.json"))

# Finalized methods only. (tracker_top, tracker_key, display_label). EZR == exploit.
METHOD_KEYS = [
    ("local", "rrp", "RRP"),
    ("local", "random", "Random"),
    ("local", "exploit", "EZR"),
    (LLM_KEY, "LLM", "LLM"),
    (LLM_KEY, "SynthCore", "SynthCore"),
    (LLM_KEY, "OPRO", "SNAP"),   # paper name for our snapping OPRO
    (LLM_KEY, "SNAP2", "SNAP2"),
]
LABELS = [m[2] for m in METHOD_KEYS]


def index_csvs():
    idx = {}
    for dirpath, _, files in os.walk(os.path.join(PROJECT_ROOT, "data")):
        for f in files:
            if f.endswith(".csv") and f not in idx:
                idx[f] = os.path.join(dirpath, f)
    return idx


def select_datasets(tracker, n_required=20):
    common = set(tracker.get("local", {}).keys()) & set(tracker.get(LLM_KEY, {}).keys())
    out = []
    for ds in sorted(common):
        ok = True
        for top, key, _ in METHOD_KEYS:
            n = len(tracker[top].get(ds, {}).get(key, {}).get("d2h_scores") or [])
            if n < n_required:
                ok = False
                break
        if ok:
            out.append(ds)
    return out


def baselines_for(datasets, csv_index):
    out = {}
    dims = {}
    for ds in datasets:
        d = DATA(ezcsv(csv_index[ds]))
        dists = [d2h(d, r) for r in d.rows]
        out[ds] = {
            "min": min(dists),
            "median": statistics.median(dists),
            "n": len(dists),
        }
        dims[ds] = len(d.cols.x)
    return out, dims


def compute_delta(score, bmin, bmedian):
    denom = bmedian - bmin
    if denom <= 0:
        return None
    return 1.0 - ((score - bmin) / denom)


def dim_bin(nx):
    if nx is None:
        return None
    if nx < 6:
        return "low"
    if nx <= 11:
        return "medium"
    return "high"


def main():
    with open(TRACKER) as f:
        tracker = json.load(f)
    csv_index = index_csvs()
    datasets = select_datasets(tracker, n_required=19)
    print(f"# datasets with >=19 d2h reps on all 7 finalized methods: {len(datasets)}")

    baselines, dims = baselines_for(datasets, csv_index)
    pathological = sorted(
        ds for ds in datasets if baselines[ds]["median"] - baselines[ds]["min"] <= 1e-12
    )
    print(f"Pathological (baseline median == min, Δ undefined): {pathological}")

    results = defaultdict(dict)
    for top, key, label in METHOD_KEYS:
        for ds in datasets:
            bmin = baselines[ds]["min"]
            bmed = baselines[ds]["median"]
            raw = tracker[top].get(ds, {}).get(key, {}).get("d2h_scores") or []
            if len(raw) < 19:
                continue
            deltas = [compute_delta(s, bmin, bmed) for s in raw]
            deltas = [d for d in deltas if d is not None]
            if not deltas:
                continue
            results[label][ds] = {
                "mean": statistics.mean(deltas),
                "median": statistics.median(deltas),
                "best": max(deltas),
                "n": len(deltas),
            }

    return tracker, datasets, baselines, dims, pathological, results


def pct(x):
    return f"{x * 100:+6.1f}%"


def summarize(datasets, dims, pathological, results):
    bins = {ds: dim_bin(dims[ds]) for ds in datasets}
    clean = [ds for ds in datasets if ds not in set(pathological)]

    def table(name, sub):
        print(f"\n=== {name} ({len(sub)} datasets) ===")
        print(
            f"{'Method':<12} {'Median Δ':>10} {'Mean Δ':>10} "
            f"{'Best Δ':>10} {'% > 0':>10} {'n':>4}"
        )
        rows = []
        for m in LABELS:
            vals = [results[m][ds]["mean"] for ds in sub if ds in results[m]]
            best = [results[m][ds]["best"] for ds in sub if ds in results[m]]
            if not vals:
                continue
            pos = sum(1 for v in vals if v > 0) / len(vals)
            rows.append(
                (m, statistics.median(vals), statistics.mean(vals),
                 statistics.mean(best), pos, len(vals))
            )
        rows.sort(key=lambda r: -r[1])
        for m, med, mean, bm, pos, n in rows:
            print(
                f"{m:<12} {pct(med):>10} {pct(mean):>10} "
                f"{pct(bm):>10} {pos*100:>9.1f}% {n:>4}"
            )

    table("All", datasets)
    table(f"Excl pathological {pathological}", clean)
    for g in ["low", "medium", "high"]:
        sub = [ds for ds in clean if bins[ds] == g]
        table(f"{g}-dim", sub)

    print("\n=== Pairwise eq-or-better-vs-strict-loss on per-dataset mean Δ (clean) ===")
    print(",".join([""] + LABELS))
    for a in LABELS:
        row = [a]
        for b in LABELS:
            if a == b:
                row.append("—")
                continue
            eb = w = 0
            for ds in clean:
                if ds in results[a] and ds in results[b]:
                    da, db = results[a][ds]["mean"], results[b][ds]["mean"]
                    if da >= db - 1e-6:
                        eb += 1
                    else:
                        w += 1
            row.append(f"{eb}-{w}")
        print(",".join(row))

    print("\n=== Best method per dataset (clean) ===")
    counter = Counter()
    for ds in clean:
        means = {m: results[m][ds]["mean"] for m in LABELS if ds in results[m]}
        if not means:
            continue
        counter[max(means, key=means.get)] += 1
    for m, c in counter.most_common():
        print(f"  {m:<10} {c:>3}  ({c/len(clean)*100:.1f}%)")

    print("\n=== Per-dataset mean Δ (%) ===")
    print(",".join(["Dataset", "dim"] + LABELS))
    for ds in datasets:
        row = [ds + ("*" if ds in set(pathological) else ""), str(dims[ds])]
        for m in LABELS:
            if ds in results[m]:
                row.append(f"{results[m][ds]['mean']*100:.1f}")
            else:
                row.append("NA")
        print(",".join(row))


if __name__ == "__main__":
    _, datasets, _, dims, pathological, results = main()
    summarize(datasets, dims, pathological, results)
