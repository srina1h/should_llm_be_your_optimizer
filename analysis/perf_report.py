"""Generate /performance_analysis_snap2.md from perf_repeats.json + perf_calls.jsonl.

Same instrumented 5-dataset x 10-repeat run as performance_analysis.md, but:
  - SNAP2 added (run via run_perf_benchmark.py --methods SNAP2)
  - AMP and DAPR excluded
  - Exploit renamed EZR

Reports cost / tokens / latency per method (mean per repeat), per-call
distribution, and per-dataset latency. Cache was disabled so timings are real
provider time.
"""
import json
import os
from statistics import mean, median, pstdev

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PERF = os.path.join(ROOT, "results", "perf")
REPEATS = os.path.join(PERF, "perf_repeats.json")
CALLS = os.path.join(PERF, "perf_calls.jsonl")

# display order; exclude AMP, DAPR. record-name -> display-name.
# OPRO is shown as SNAP (paper name); Exploit as EZR.
INCLUDE = [("SNAP2", "SNAP2"), ("OPRO", "SNAP"), ("LLM", "LLM"),
           ("SynthCore", "SynthCore"), ("Exploit", "EZR"),
           ("Random", "Random"), ("RRP", "RRP")]
LLM_METHODS = {"SNAP2", "OPRO", "LLM", "SynthCore"}


def load():
    reps = json.load(open(REPEATS))
    calls = []
    if os.path.exists(CALLS):
        for line in open(CALLS):
            if line.strip():
                calls.append(json.loads(line))
    return reps, calls


def m(xs):
    return mean(xs) if xs else float("nan")


def main():
    reps, calls = load()
    clean = [r for r in reps if not r.get("error")]
    datasets = sorted({r["dataset"] for r in clean})

    lines = ["# Performance Analysis (with SNAP2): Cost, Tokens & Latency by Method",
             "",
             "> Operational cost of each warm-start strategy — tokens, USD, wall-clock "
             "latency per repeat. Instrumented run, **cache disabled** (real provider "
             "time). 5 datasets x 10 repeats per method. **AMP & DAPR excluded; "
             "Exploit shown as EZR.** Quality ranking lives in `reports/bestranks.md`.",
             ""]

    # dataset shapes
    lines += ["## Datasets", "",
              "| dataset | bin |", "|---|---|",
              "| auto93.csv | Low (4 feat) |",
              "| pom3a.csv | Medium (9) |",
              "| Apache_AllMeasurements.csv | Medium (9) |",
              "| SS-N.csv | High (17) |",
              "| xomo_flight.csv | High (23) |", ""]

    # per-method aggregate (mean per repeat)
    lines += ["## Per-method cost / tokens / latency (mean per repeat)", "",
              "| method | n | LLM calls | total tok | reasoning tok | cost $ | "
              "API s | wall s |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    agg = {}
    for key, disp in INCLUDE:
        recs = [r for r in clean if r["method"] == key]
        if not recs:
            continue
        a = {f: m([r[f] for r in recs]) for f in
             ["n_calls", "total_tokens", "reasoning_tokens", "cost_usd",
              "api_time_s", "wallclock_s"]}
        a["n"] = len(recs)
        agg[disp] = a
        mark = f"**{disp}**" if disp == "SNAP2" else disp
        lines.append(f"| {mark} | {a['n']} | {a['n_calls']:.1f} | {a['total_tokens']:.0f} | "
                     f"{a['reasoning_tokens']:.0f} | {a['cost_usd']:.5f} | "
                     f"{a['api_time_s']:.1f} | {a['wallclock_s']:.1f} |")

    # headline SNAP2 vs SNAP (=OPRO)
    if "SNAP2" in agg and "SNAP" in agg:
        s, o = agg["SNAP2"], agg["SNAP"]
        def red(k):
            return (1 - s[k] / o[k]) * 100 if o[k] else float("nan")
        lines += ["",
                  f"**SNAP2 vs SNAP (per repeat):** "
                  f"LLM calls {s['n_calls']:.1f} vs {o['n_calls']:.1f} ({red('n_calls'):.0f}% fewer) · "
                  f"tokens {s['total_tokens']:.0f} vs {o['total_tokens']:.0f} ({red('total_tokens'):.0f}% fewer) · "
                  f"cost ${s['cost_usd']:.5f} vs ${o['cost_usd']:.5f} ({red('cost_usd'):.0f}% cheaper) · "
                  f"latency {s['api_time_s']:.1f}s vs {o['api_time_s']:.1f}s ({red('api_time_s'):.0f}% faster).",
                  ""]

    # per-call distribution (LLM methods only)
    lines += ["## Per-call distribution (LLM methods)", "",
              "| method | n calls | tok/call (med) | tok/call (p95) | "
              "latency s/call (med) | cost $/call (med) |",
              "|---|--:|--:|--:|--:|--:|"]
    for key, disp in INCLUDE:
        if key not in LLM_METHODS:
            continue
        mc = [c for c in calls if c["method"] == key]
        if not mc:
            continue
        tok = [c["total_tokens"] for c in mc]
        lat = [c["api_elapsed_s"] for c in mc]
        cost = [c["cost_usd"] for c in mc]
        mark = f"**{disp}**" if disp == "SNAP2" else disp
        lines.append(f"| {mark} | {len(mc)} | {median(tok):.0f} | "
                     f"{sorted(tok)[int(0.95*(len(tok)-1))]:.0f} | "
                     f"{median(lat):.2f} | {median(cost):.6f} |")

    # per-dataset wall-clock per repeat (SNAP2 vs OPRO)
    lines += ["", "## Per-dataset latency (mean wall-clock s/repeat)", "",
              "| dataset | SNAP2 | SNAP | LLM | SynthCore |", "|---|--:|--:|--:|--:|"]
    for ds in datasets:
        row = [ds.replace(".csv", "")]
        for key in ["SNAP2", "OPRO", "LLM", "SynthCore"]:
            sub = [r["wallclock_s"] for r in clean if r["method"] == key and r["dataset"] == ds]
            row.append(f"{m(sub):.1f}" if sub else "-")
        lines.append("| " + " | ".join(row) + " |")

    # totals
    lines += ["", "## Totals (SNAP2 only, this run)"]
    s2 = [r for r in clean if r["method"] == "SNAP2"]
    if s2:
        lines.append(f"\n{len(s2)} repeats · {sum(r['cost_usd'] for r in s2):.4f} USD · "
                     f"{sum(r['total_tokens'] for r in s2):,} tokens · "
                     f"{sum(r['api_time_s'] for r in s2)/60:.1f} min API time. "
                     f"Errors: {len([r for r in reps if r['method']=='SNAP2' and r.get('error')])}.")

    txt = "\n".join(lines)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    open(os.path.join(ROOT, "reports", "performance_analysis.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
