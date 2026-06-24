"""
run_perf_benchmark.py — Per-method performance instrumentation.

Captures token counts, cost, and wall-clock latency for every LLM call across
methods (LLM, AMP, SynthCore, DAPR). Cache is DISABLED so latency reflects
real provider time. Runs 20 repeats × 5 datasets per method = 100 total per
method. Writes per-call records (JSONL) and per-repeat aggregates (JSON).
"""

import os
import sys
import json
import time
import random
import argparse
import traceback
from datetime import datetime
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import litellm
litellm.cache = None  # ensure real API timing

from src.utils.ezr import DATA, csv, chebyshev, the, smo, branch, clone
from src.models.warm_start import warm_smo
from src.models.cluster import density
from src.language import llms as _llms_module

LOCAL_METHODS = {"Random", "RRP", "Exploit"}


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------
_ctx = {"method": None, "dataset": None, "repeat": None, "call_idx": 0}
_call_records = []
_repeat_records = []


def _patch_wrapper():
    """Monkey-patch LiteLLM_Wrapper.invoke to record per-call metrics."""
    Wrapper = _llms_module.LiteLLM_Wrapper
    orig = Wrapper.invoke

    def instrumented_invoke(self, input_messages):
        t0 = time.perf_counter()
        # Replicate the wrapper's body so we can grab the raw response object
        # (the original strips usage off and only returns content).
        messages = []
        if isinstance(input_messages, str):
            messages = [{"role": "user", "content": input_messages}]
        elif isinstance(input_messages, list):
            for msg in input_messages:
                if isinstance(msg, dict):
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    })
                elif isinstance(msg, tuple):
                    role = msg[0].replace("human", "user")
                    messages.append({"role": role, "content": msg[1]})
                else:
                    messages.append({"role": "user", "content": str(msg)})

        from litellm import completion
        response = completion(model=self.model, messages=messages, **self.kwargs)
        elapsed = time.perf_counter() - t0

        usage = getattr(response, "usage", None) or {}
        # litellm's ModelResponse usage may be a pydantic obj; coerce
        if hasattr(usage, "model_dump"):
            usage_d = usage.model_dump()
        elif hasattr(usage, "__dict__"):
            usage_d = dict(usage.__dict__)
        else:
            usage_d = dict(usage)

        prompt_t = int(usage_d.get("prompt_tokens", 0) or 0)
        completion_t = int(usage_d.get("completion_tokens", 0) or 0)
        total_t = int(usage_d.get("total_tokens", prompt_t + completion_t) or 0)
        reasoning_t = 0
        ctd = usage_d.get("completion_tokens_details") or {}
        if hasattr(ctd, "model_dump"):
            ctd = ctd.model_dump()
        elif hasattr(ctd, "__dict__"):
            ctd = dict(ctd.__dict__)
        if isinstance(ctd, dict):
            reasoning_t = int(ctd.get("reasoning_tokens") or 0)

        # cost: litellm sometimes attaches `_hidden_params['response_cost']`
        cost = 0.0
        hp = getattr(response, "_hidden_params", None) or {}
        if isinstance(hp, dict):
            cost = float(hp.get("response_cost") or 0.0)
        if not cost:
            try:
                from litellm import completion_cost
                cost = float(completion_cost(completion_response=response) or 0.0)
            except Exception:
                cost = 0.0

        rec = {
            "method": _ctx["method"],
            "dataset": _ctx["dataset"],
            "repeat": _ctx["repeat"],
            "call_idx": _ctx["call_idx"],
            "prompt_tokens": prompt_t,
            "completion_tokens": completion_t,
            "reasoning_tokens": reasoning_t,
            "total_tokens": total_t,
            "cost_usd": cost,
            "api_elapsed_s": elapsed,
        }
        _call_records.append(rec)
        _ctx["call_idx"] += 1

        content = response.choices[0].message.content
        return type("Response", (object,), {"content": content})()

    Wrapper.invoke = instrumented_invoke


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_one(method, dataset_path, repeat, base_seed, args_template):
    if method == "SNAP":      # paper alias for our snapping OPRO
        method = "OPRO"
    ds_name = os.path.basename(dataset_path)
    _ctx["method"] = method
    _ctx["dataset"] = ds_name
    _ctx["repeat"] = repeat
    _ctx["call_idx"] = 0

    n_before = len(_call_records)
    random.seed(base_seed + repeat + (hash(ds_name) % 1000000))
    the.Last = args_template["last"]
    the.label = args_template["label"]
    the.GuessFaster = True

    # Build args namespace
    class A:
        pass
    a = A()
    for k, v in args_template.items():
        setattr(a, k, v)
    a.dataset = dataset_path

    d = DATA(csv(dataset_path))
    score_fn = lambda B, R, I, N: B - R

    t0 = time.perf_counter()
    err = None
    score = None
    try:
        if method == "SynthCore":
            res = density(a, method="LLM")
            score = chebyshev(d, res[0])
        elif method == "RRP":
            best, _, _ = branch(d, d.rows, 4)
            score = chebyshev(d, best[0])
        elif method == "Random":
            guess = clone(d, random.choices(d.rows, k=args_template["last"]), rank=True).rows[0]
            score = chebyshev(d, guess)
        elif method == "Exploit":
            res, _ = smo(d, score_fn)
            score = chebyshev(d, res[0])
        elif method == "OPRO":
            from methods.opro import OPRO_Strategy
            a._gen_log_synth = {}
            a.opro_seed = "random"
            strategy = OPRO_Strategy(d, a)
            results, _ = strategy.run()
            score = chebyshev(d, results[0])
        elif method == "SNAP2":
            from methods.opro import OPRO_Strategy
            a._gen_log_synth = {}
            a.opro_seed = "exploit"
            a.opro_seed_size = None
            strategy = OPRO_Strategy(d, a)
            results, _ = strategy.run()
            score = chebyshev(d, results[0])
        elif method == "LLM":
            res, _ = warm_smo(a, score_fn, method="LLM")
            score = chebyshev(d, res[0])
        else:
            raise ValueError(f"Unknown method '{method}'. Finalized: "
                             "LLM, SynthCore, OPRO, SNAP2, Exploit, Random, RRP.")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    total_elapsed = time.perf_counter() - t0

    new_calls = _call_records[n_before:]
    rec = {
        "method": method,
        "dataset": ds_name,
        "repeat": repeat,
        "n_calls": len(new_calls),
        "prompt_tokens": sum(c["prompt_tokens"] for c in new_calls),
        "completion_tokens": sum(c["completion_tokens"] for c in new_calls),
        "reasoning_tokens": sum(c["reasoning_tokens"] for c in new_calls),
        "total_tokens": sum(c["total_tokens"] for c in new_calls),
        "cost_usd": sum(c["cost_usd"] for c in new_calls),
        "api_time_s": sum(c["api_elapsed_s"] for c in new_calls),
        "wallclock_s": total_elapsed,
        "score": score,
        "error": err,
    }
    _repeat_records.append(rec)
    return rec


def save(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "perf_calls.jsonl"), "w") as f:
        for r in _call_records:
            f.write(json.dumps(r, default=str) + "\n")
    with open(os.path.join(out_dir, "perf_repeats.json"), "w") as f:
        json.dump(_repeat_records, f, indent=2, default=str)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="openrouter/openai/gpt-oss-120b")
    parser.add_argument("--repeats", type=int, default=20,
                        help="repeats per (method, dataset); 5 datasets * 20 = 100/method")
    parser.add_argument("--last", type=int, default=20)
    parser.add_argument("--label", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-dir", default="results/perf")
    parser.add_argument("--methods", nargs="+",
                        default=["LLM", "SynthCore", "OPRO", "SNAP2"])
    parser.add_argument("--datasets", nargs="+", default=[
        "data/misc/auto93.csv",
        "data/process/pom3a.csv",
        "data/config/Apache_AllMeasurements.csv",
        "data/config/SS-N.csv",
        "data/process/xomo_flight.csv",
    ])
    parser.add_argument("--resume", action="store_true",
                        help="skip (method,dataset,repeat) tuples already in perf_repeats.json")
    args = parser.parse_args()

    _patch_wrapper()

    args_template = {
        "llm": args.llm,
        "temperature": 0.7,
        "max_tokens": 4000,
        "top_p": 1.0,
        "amp_mode": "AMP-4",
        "cache": True,
        "last": args.last,
        "label": args.label,
        "dapr_k": None,
        "dapr_s": 2,
        "dapr_candidates": 4,
    }

    # Resume support
    done = set()
    repeats_path = os.path.join(args.out_dir, "perf_repeats.json")
    calls_path = os.path.join(args.out_dir, "perf_calls.jsonl")
    if args.resume and os.path.exists(repeats_path):
        with open(repeats_path) as f:
            prev = json.load(f)
        for r in prev:
            done.add((r["method"], r["dataset"], r["repeat"]))
            _repeat_records.append(r)
        if os.path.exists(calls_path):
            with open(calls_path) as f:
                for line in f:
                    if line.strip():
                        _call_records.append(json.loads(line))
        print(f"[resume] loaded {len(prev)} prior repeats, {len(_call_records)} prior calls")

    print(f"[start] {datetime.now().isoformat()} | model={args.llm}")
    print(f"[plan]  {len(args.methods)} methods × {len(args.datasets)} datasets × {args.repeats} reps "
          f"= {len(args.methods) * len(args.datasets) * args.repeats} runs")

    t_start = time.perf_counter()
    for method in args.methods:
        for ds in args.datasets:
            for rep in range(args.repeats):
                if (method, os.path.basename(ds), rep) in done:
                    continue
                try:
                    rec = run_one(method, ds, rep, args.seed, args_template)
                    print(f"  [{method}/{os.path.basename(ds)}/r{rep}] "
                          f"calls={rec['n_calls']} tok={rec['total_tokens']} "
                          f"cost=${rec['cost_usd']:.5f} "
                          f"api={rec['api_time_s']:.1f}s wall={rec['wallclock_s']:.1f}s "
                          f"score={rec['score']}", flush=True)
                except Exception as e:
                    print(f"  [{method}/{os.path.basename(ds)}/r{rep}] CRITICAL: {e}",
                          flush=True)
                    traceback.print_exc()
                # incremental save every iteration so a crash doesn't lose data
                save(args.out_dir)

    elapsed = time.perf_counter() - t_start
    print(f"[done] elapsed={elapsed/60:.1f} min, {len(_repeat_records)} repeats, "
          f"{len(_call_records)} calls")


if __name__ == "__main__":
    main()
