# -*- coding: utf-8 -*-
"""compute_wins.py — bestRanks (statistical wins) on the finalized methods.

Per dataset, sort methods by median d2h, take the leader, and pull in every
method whose distribution is statistically the same per the `same()` test
(cliff's delta < 0.195 AND KS gap <= 1.36*sqrt((n+m)/nm), gated by
eps = 0.35 * sd of all d2h scores pooled across methods on that dataset).
Every method in that top group "wins" the dataset.

Reads one tracker (default results/status_tracker.json) holding all 7 finalized
methods. AMP & DAPR are not included; Exploit is shown as EZR. Writes a markdown
report to reports/bestranks.md and prints the tables.
"""
from __future__ import annotations

import bisect
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.ezr import DATA, csv as ezcsv  # noqa: E402

LLM_KEY = os.environ.get("REPRO_LLM_KEY", "openrouter/openai/gpt-oss-120b")
TRACKER = os.environ.get(
    "REPRO_TRACKER",
    os.path.join(PROJECT_ROOT, "results", "status_tracker.json"))

# (tracker_top, tracker_key, display_label). EZR == exploit. AMP/DAPR excluded.
METHODS = [
    ("local", "rrp", "RRP"),
    ("local", "random", "Random"),
    ("local", "exploit", "EZR"),
    (LLM_KEY, "LLM", "LLM"),
    (LLM_KEY, "SynthCore", "SynthCore"),
    (LLM_KEY, "OPRO", "SNAP"),   # paper name for our snapping OPRO
    (LLM_KEY, "SNAP2", "SNAP2"),
]
LABELS = [m[2] for m in METHODS]

STATS_CLIFFS = 0.195
STATS_CONF = 1.36
STATS_EPS = 0.35


class Num:
    def __init__(self):
        self.n = self.mu = self.m2 = self.sd = 0.0

    def add(self, v):
        self.n += 1
        d = v - self.mu
        self.mu += d / self.n
        self.m2 += d * (v - self.mu)
        self.sd = ((max(0.0, self.m2) / (self.n - 1)) ** 0.5) if self.n > 1 else 0.0


def same(xs, ys, eps):
    xs, ys = sorted(xs), sorted(ys)
    n, m = len(xs), len(ys)
    if abs(xs[n // 2] - ys[m // 2]) <= eps:
        return True
    gt = sum(bisect.bisect_left(ys, a) for a in xs)
    lt = sum(m - bisect.bisect_right(ys, a) for a in xs)
    if abs(gt - lt) / (n * m) > STATS_CLIFFS:
        return False
    ks = lambda v: abs(bisect.bisect_right(xs, v) / n - bisect.bisect_right(ys, v) / m)
    return max(max(map(ks, xs)), max(map(ks, ys))) <= STATS_CONF * ((n + m) / (n * m)) ** 0.5


def best_ranks(d):
    pool = Num()
    for lst in d.values():
        for v in lst:
            pool.add(v)
    eps = pool.sd * STATS_EPS
    items = sorted(d.items(), key=lambda kv: sorted(kv[1])[len(kv[1]) // 2])
    _, lst0 = items[0]
    best = {items[0][0]}
    for k, lst in items[1:]:
        if same(lst0, lst, eps):
            best.add(k)
        else:
            break
    return best


def index_csvs():
    idx = {}
    for dirpath, _, files in os.walk(os.path.join(PROJECT_ROOT, "data")):
        for f in files:
            if f.endswith(".csv") and f not in idx:
                idx[f] = os.path.join(dirpath, f)
    return idx


def select_datasets(tracker, n_required=19):
    common = set(tracker.get("local", {}).keys()) & set(tracker.get(LLM_KEY, {}).keys())
    out = []
    for ds in sorted(common):
        if all(len(tracker[t].get(ds, {}).get(k, {}).get("d2h_scores") or []) >= n_required
               for t, k, _ in METHODS):
            out.append(ds)
    return out


def dim_bin(nx):
    return "low" if nx < 6 else "medium" if nx <= 11 else "high"


def wins_block(name, sub, winners, present):
    rows = []
    for m in LABELS:
        elig = [ds for ds in sub if m in present.get(ds, set())]
        n = len(elig)
        w = sum(1 for ds in elig if m in winners.get(ds, set()))
        rows.append((m, w, n, (w / n * 100) if n else 0))
    rows.sort(key=lambda r: -r[3])
    out = [f"\n### {name} ({len(sub)} datasets)", "",
           "| rank | method | wins | N | rate |", "|--:|---|--:|--:|--:|"]
    for i, (m, w, n, p) in enumerate(rows, 1):
        mark = f"**{m}**" if m == "SNAP2" else m
        out.append(f"| {i} | {mark} | {w} | {n} | {p:.1f}% |")
    return "\n".join(out)


def main():
    tracker = json.load(open(TRACKER))
    csv_index = index_csvs()
    datasets = select_datasets(tracker)
    dims = {ds: len(DATA(ezcsv(csv_index[ds])).cols.x) for ds in datasets}

    winners, present = {}, {}
    for ds in datasets:
        d = {}
        for top, key, label in METHODS:
            s = tracker[top].get(ds, {}).get(key, {}).get("d2h_scores") or []
            if len(s) >= 19:
                d[label] = list(s)
        present[ds] = set(d.keys())
        if d:
            winners[ds] = best_ranks(d)

    out = ["# bestRanks (statistical wins) — finalized methods",
           "",
           f"Model `{LLM_KEY}` · metric d2h (lower=better) · {len(datasets)} datasets. "
           "A method wins a dataset if it sits in the statistically-tied top group "
           "(cliff's delta + KS `same()` test). AMP & DAPR excluded; Exploit shown as EZR."]
    out.append(wins_block("Overall", datasets, winners, present))
    for g, lbl in [("low", "Small (<6 feat)"), ("medium", "Medium (6-11)"),
                   ("high", "Large (>=12)")]:
        sub = [ds for ds in datasets if dim_bin(dims[ds]) == g]
        if sub:
            out.append(wins_block(lbl, sub, winners, present))

    txt = "\n".join(out)
    os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
    open(os.path.join(PROJECT_ROOT, "reports", "bestranks.md"), "w").write(txt + "\n")
    print(txt)
    print("\nwrote reports/bestranks.md")


if __name__ == "__main__":
    main()
