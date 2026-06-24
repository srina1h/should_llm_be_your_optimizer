"""
run_benchmark.py — Proper 20-repeat benchmark runner for Beyond the Prompt.

Mirrors lazy.py's warms() pipeline exactly, but wrapped in a robust,
crash-resilient JSON status tracker to allow pausing/resuming over large datasets.

All scores use chebyshev distance (L-inf). Results aggregated via SOME + Scott-Knott.
"""

import os
import sys
import json
import math
import time
import random
import argparse
import traceback
from datetime import datetime
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import litellm
litellm.cache = litellm.Cache(type="disk") 

from src.utils.ezr import (
    DATA, csv, chebyshev, d2h, smo, branch, clone, SOME, report, sk, the, btw
)
from src.models.warm_start import warm_smo
from src.models.cluster import density


def exploit(B, R):
    return B - R

def _stable_hash(s):
    # Python's builtin hash() randomizes per-process via PYTHONHASHSEED, which
    # makes seeds non-reproducible across runs. Use md5 for a stable digest.
    import hashlib
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

def set_deterministic_seed(base_seed, dataset_name, rep_index):
    hashed = _stable_hash(dataset_name) % 1000000
    random.seed(base_seed + rep_index + hashed)


def load_tracker(tracker_path):
    if os.path.exists(tracker_path):
        with open(tracker_path, "r") as f:
            return json.load(f)
    return {}

def save_tracker(tracker, tracker_path):
    os.makedirs(os.path.dirname(tracker_path), exist_ok=True)
    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)


def _llm_safe(llm_name):
    return llm_name.replace("/", "__")


def _gen_log_path(gen_log_dir, llm_name, ds_name, method, rep):
    return os.path.join(gen_log_dir, _llm_safe(llm_name), ds_name,
                        f"{method}__rep{rep:02d}.json")


def _write_gen_log(gen_log_dir, llm_name, ds_name, method, rep, seed, payload):
    path = _gen_log_path(gen_log_dir, llm_name, ds_name, method, rep)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {
        "llm": llm_name,
        "dataset": ds_name,
        "method": method,
        "rep": rep,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "selected_row": payload.get("selected_row"),
        "chebyshev": payload.get("chebyshev"),
        "d2h": payload.get("d2h"),
        "synth_rows": payload.get("synth_rows"),
        "warm_done_x": payload.get("warm_done_x"),
        "notes": payload.get("notes"),
        "opro_collision_rate": payload.get("opro_collision_rate"),
        "opro_best_d2h_per_call": payload.get("opro_best_d2h_per_call"),
        "opro_trajectory": payload.get("opro_trajectory"),
        "opro_prompt_hashes": payload.get("opro_prompt_hashes"),
        "opro_value_clamps": payload.get("opro_value_clamps"),
        "opro_todo_exhausted": payload.get("opro_todo_exhausted"),
        "opro_failed_calls": payload.get("opro_failed_calls"),
        "opro_calls": payload.get("opro_calls"),
        "opro_retries_per_call": payload.get("opro_retries_per_call"),
        "opro_total_retries": payload.get("opro_total_retries"),
        "opro_raw_responses": payload.get("opro_raw_responses"),
        "opro_num_out_of_range": payload.get("opro_num_out_of_range"),
        "opro_num_uncoercible": payload.get("opro_num_uncoercible"),
        "opro_sym_out_of_vocab": payload.get("opro_sym_out_of_vocab"),
        "opro_sym_oov_rate": payload.get("opro_sym_oov_rate"),
        "opro_num_oor_rate": payload.get("opro_num_oor_rate"),
        "snap2_seed_mode": payload.get("snap2_seed_mode"),
        "snap2_seed_size": payload.get("snap2_seed_size"),
        "snap2_handoff_d2h": payload.get("snap2_handoff_d2h"),
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)


def _all_gen_logs_present(gen_log_dir, llm_name, ds_name, method, n):
    return all(os.path.exists(_gen_log_path(gen_log_dir, llm_name, ds_name, method, r))
               for r in range(n))


def _seed_for(base_seed, ds_name, rep):
    return base_seed + rep + (_stable_hash(ds_name) % 1000000)


def track_and_run(tracker, tracker_path, llm_name, ds_name, method, rxs, rx_key,
                  repeats, run_func, runtime):
    """
    Manages the resilient running of a repeated function.
    Reads prior runs from the JSON ledger. Evaluates remaining runs. Saves instantly.
    On error: logs it, skips that repeat, and continues to the next.

    `runtime` carries cross-cutting flags: gen_log_dir, metric, backfill, base_seed.
    """
    if llm_name not in tracker: tracker[llm_name] = {}
    if ds_name not in tracker[llm_name]: tracker[llm_name][ds_name] = {}
    if method not in tracker[llm_name][ds_name]:
        tracker[llm_name][ds_name][method] = {
            "completed": 0, "scores": [], "d2h_scores": [], "errors": 0
        }

    entry = tracker[llm_name][ds_name][method]
    entry.setdefault("d2h_scores", [])
    completed = entry["completed"]
    scores = entry["scores"]
    d2h_scores = entry["d2h_scores"]
    metric = runtime["metric"]

    def _scalar_for_sk(idx):
        if metric == "d2h" and idx < len(d2h_scores):
            return d2h_scores[idx]
        return scores[idx] if idx < len(scores) else None

    n_success = 0
    n_errors = 0

    if completed >= repeats:
        gen_logs_ok = _all_gen_logs_present(
            runtime["gen_log_dir"], llm_name, ds_name, method, completed)
        d2h_ok = len(d2h_scores) >= completed
        needs_backfill = runtime["backfill"] and (not gen_logs_ok or not d2h_ok)
        if not needs_backfill:
            for i in range(completed):
                v = _scalar_for_sk(i)
                if v is not None: rxs[rx_key].add(v)
            print(f"  [{rx_key}] SKIP: {completed}/{repeats} runs loaded from tracker", flush=True)
            return True

        # Backfill path: re-run reps 0..completed-1 with same seed and overwrite
        # both scores and d2h_scores. The LLM disk cache isn't deterministic across
        # sessions (sampling at temperature>0 can vary even with cache), so we don't
        # block on chebyshev mismatch — we just record an `entry["drift"]` flag for
        # transparency. The new (chebyshev, d2h) pair always comes from the same
        # selected_row, so they're internally consistent.
        n_drift = 0
        print(f"  [{rx_key}] BACKFILL: re-running {completed} repeats to fill d2h + gen logs", flush=True)
        for rep in range(completed):
            btw(".")
            try:
                result = run_func(rep)
                old_cheb = scores[rep] if rep < len(scores) else None
                new_cheb = result["chebyshev"]
                new_d2h = result["d2h"]
                if old_cheb is not None and abs(new_cheb - old_cheb) > 1e-9:
                    n_drift += 1
                    entry.setdefault("drift_reps", []).append(rep)
                    entry["drift"] = True
                if rep < len(scores): scores[rep] = new_cheb
                else: scores.append(new_cheb)
                if rep < len(d2h_scores): d2h_scores[rep] = new_d2h
                else: d2h_scores.append(new_d2h)
                seed = _seed_for(runtime["base_seed"], ds_name, rep)
                _write_gen_log(runtime["gen_log_dir"], llm_name, ds_name,
                               method, rep, seed, result)
                save_tracker(tracker, tracker_path)
                rxs[rx_key].add(new_d2h if metric == "d2h" else new_cheb)
                n_success += 1
            except Exception as e:
                n_errors += 1
                print(f"\n  [{method}] backfill rep {rep} ERROR (skipping): {e}", flush=True)
                traceback.print_exc()
                continue
        btw("\n")
        msg = f"  [{rx_key}] backfill done ({n_success} filled, {n_errors} failed"
        if n_drift: msg += f", {n_drift} drifted from prior chebyshev"
        msg += ")"
        print(msg, flush=True)
        return True

    # Preload previous scalars (partial-progress resume case).
    for i in range(completed):
        v = _scalar_for_sk(i)
        if v is not None: rxs[rx_key].add(v)

    # Run remaining iterations — skip-and-continue on errors
    for rep in range(completed, repeats):
        btw(".")
        try:
            result = run_func(rep)
            cheb = result["chebyshev"]
            dh = result["d2h"]

            seed = _seed_for(runtime["base_seed"], ds_name, rep)
            _write_gen_log(runtime["gen_log_dir"], llm_name, ds_name,
                           method, rep, seed, result)

            entry["scores"].append(cheb)
            d2h_scores.append(dh)
            entry["completed"] += 1
            n_success += 1
            rxs[rx_key].add(dh if metric == "d2h" else cheb)
            save_tracker(tracker, tracker_path)

        except Exception as e:
            n_errors += 1
            entry["errors"] = entry.get("errors", 0) + 1
            entry["completed"] += 1
            save_tracker(tracker, tracker_path)
            print(f"\n  [{method}] repeat {rep} ERROR (skipping): {e}", flush=True)
            traceback.print_exc()
            continue

    btw("\n")
    print(f"  [{rx_key}] done ({n_success} succeeded, {n_errors} failed, {entry['completed']}/{repeats} total)", flush=True)
    return True


def run_one_dataset(dataset_path, llm_name, repeats, last, label,
                    strategies_to_run, base_seed, tracker, tracker_path,
                    skip_baselines=False, runtime=None):
    d = DATA(csv(dataset_path))
    ds_name = os.path.basename(dataset_path)
    n_rows = len(d.rows)
    print(f"\n{'='*70}", flush=True)
    print(f"Dataset: {ds_name}  ({n_rows} rows, {len(d.cols.x)} features, {len(d.cols.y)} objectives)", flush=True)
    print(f"Budget: last={last}, label={label}, repeats={repeats}", flush=True)
    print(f"{'='*70}", flush=True)

    the.Last = last
    the.label = label
    rxs = {}

    runtime = runtime or {"metric": "d2h", "gen_log_dir": "results/gen_logs",
                          "backfill": False, "base_seed": base_seed}

    # `score_fn` is required for the LLM strategies below (warm_smo's exploit scoring),
    # and `the.GuessFaster` must be flipped before the first warm_smo call.
    score_fn = lambda B, R, I, N: exploit(B, R)
    the.GuessFaster = True

    if skip_baselines:
        print("  [baselines] SKIP (--skip-baselines): baseline/rrp/random/exploit not run", flush=True)
    else:
        # 1. Baseline
        rx = f"baseline,{n_rows}"
        baseline_metric = d2h if runtime["metric"] == "d2h" else chebyshev
        rxs[rx] = SOME(txt=rx, inits=[baseline_metric(d, row) for row in d.rows])
        print(f"  [{rx}] done (n={n_rows})", flush=True)

        # 2. RRP
        rx = f"rrp,{int(0.5 + math.log(n_rows, 2) + 1)}"
        rxs[rx] = SOME(txt=rx)
        def run_rrp(rep):
            set_deterministic_seed(base_seed, ds_name, rep)
            best, _, _ = branch(d, d.rows, 4)
            row = best[0]
            return {"selected_row": list(row),
                    "chebyshev": chebyshev(d, row), "d2h": d2h(d, row),
                    "synth_rows": None, "warm_done_x": None}
        track_and_run(tracker, tracker_path, "local", ds_name, "rrp", rxs, rx, repeats, run_rrp, runtime)

        # 3. Random
        rx = f"random,{last}"
        rxs[rx] = SOME(txt=rx)
        def run_random(rep):
            set_deterministic_seed(base_seed, ds_name, rep)
            row = clone(d, random.choices(d.rows, k=last), rank=True).rows[0]
            return {"selected_row": list(row),
                    "chebyshev": chebyshev(d, row), "d2h": d2h(d, row),
                    "synth_rows": None, "warm_done_x": None}
        track_and_run(tracker, tracker_path, "local", ds_name, "random", rxs, rx, repeats, run_random, runtime)

        # 4. Exploit
        rx = f"exploit,{last}"
        rxs[rx] = SOME(txt=rx)
        def run_exploit(rep):
            set_deterministic_seed(base_seed, ds_name, rep)
            res, _ = smo(d, score_fn)
            row = res[0]
            return {"selected_row": list(row),
                    "chebyshev": chebyshev(d, row), "d2h": d2h(d, row),
                    "synth_rows": None, "warm_done_x": None}
        track_and_run(tracker, tracker_path, "local", ds_name, "exploit", rxs, rx, repeats, run_exploit, runtime)

    # 5. LLM strategies (LLM, SynthCore, OPRO, SNAP2)
    class Args: pass
    args = Args()
    args.llm = llm_name
    args.dataset = dataset_path
    args.temperature = 0.7
    args.max_tokens = 4000
    args.top_p = 1.0
    args.cache = True
    args.last = last
    args.label = label
    args.opro_seed = "random"
    args.opro_seed_size = None

    for method in (strategies_to_run or ['LLM', 'SynthCore', 'OPRO', 'SNAP2']):
        # Paper alias: SNAP == our snapping OPRO. Internal key stays "OPRO".
        if method == "SNAP":
            method = "OPRO"
        # SynthCore uses its own budget (30) and explore scoring — leave its internals untouched
        if method == "SynthCore":
            import copy
            rx = f"{llm_name}/SynthCore,30"
            rxs[rx] = SOME(txt=rx)
            def run_synth(rep):
                set_deterministic_seed(base_seed, ds_name, rep)
                # Deep-copy args so density()'s internal args.last=30 doesn't corrupt other strategies
                synth_args = copy.deepcopy(args)
                res = density(synth_args, method='LLM')
                row = res[0]
                return {"selected_row": list(row),
                        "chebyshev": chebyshev(d, row), "d2h": d2h(d, row),
                        "synth_rows": None, "warm_done_x": None}
            track_and_run(tracker, tracker_path, llm_name, ds_name, method, rxs, rx, repeats, run_synth, runtime)
        elif method == "OPRO":
            from methods.opro import OPRO_Strategy
            rx = f"{llm_name}/OPRO,{last}"
            rxs[rx] = SOME(txt=rx)
            def run_opro(rep):
                set_deterministic_seed(base_seed, ds_name, rep)
                args._gen_log_synth = {}
                args.opro_seed = "random"  # guard: SNAP2 in a prior rep mutates shared args
                strategy = OPRO_Strategy(d, args)
                results, _ = strategy.run()
                time.sleep(2)
                row = results[0]
                sink = args._gen_log_synth
                return {"selected_row": list(row),
                        "chebyshev": chebyshev(d, row), "d2h": d2h(d, row),
                        "synth_rows": sink.get("synth_rows"),
                        "warm_done_x": sink.get("new_done_x"),
                        "opro_collision_rate": sink.get("opro_collision_rate"),
                        "opro_best_d2h_per_call": sink.get("opro_best_d2h_per_call"),
                        "opro_trajectory": sink.get("opro_trajectory"),
                        "opro_prompt_hashes": sink.get("opro_prompt_hashes"),
                        "opro_value_clamps": sink.get("opro_value_clamps"),
                        "opro_todo_exhausted": sink.get("opro_todo_exhausted"),
                        "opro_failed_calls": sink.get("opro_failed_calls"),
                        "opro_calls": sink.get("opro_calls"),
                        "opro_retries_per_call": sink.get("opro_retries_per_call"),
                        "opro_total_retries": sink.get("opro_total_retries"),
                        "opro_raw_responses": sink.get("opro_raw_responses"),
                        "opro_num_out_of_range": sink.get("opro_num_out_of_range"),
                        "opro_num_uncoercible": sink.get("opro_num_uncoercible"),
                        "opro_sym_out_of_vocab": sink.get("opro_sym_out_of_vocab"),
                        "opro_sym_oov_rate": sink.get("opro_sym_oov_rate"),
                        "opro_num_oor_rate": sink.get("opro_num_oor_rate")}
            track_and_run(tracker, tracker_path, llm_name, ds_name, method, rxs, rx, repeats, run_opro, runtime)
        elif method == "SNAP2":
            from methods.opro import OPRO_Strategy
            rx = f"{llm_name}/SNAP2,{last}"
            rxs[rx] = SOME(txt=rx)
            def run_snap2(rep):
                set_deterministic_seed(base_seed, ds_name, rep)
                args._gen_log_synth = {}
                args.opro_seed = "exploit"   # the only difference vs OPRO
                args.opro_seed_size = None   # build_opro_seed -> round(last/2)
                strategy = OPRO_Strategy(d, args)
                results, _ = strategy.run()
                time.sleep(2)
                row = results[0]
                sink = args._gen_log_synth
                return {"selected_row": list(row),
                        "chebyshev": chebyshev(d, row), "d2h": d2h(d, row),
                        "synth_rows": sink.get("synth_rows"),
                        "warm_done_x": sink.get("new_done_x"),
                        "opro_collision_rate": sink.get("opro_collision_rate"),
                        "opro_best_d2h_per_call": sink.get("opro_best_d2h_per_call"),
                        "opro_trajectory": sink.get("opro_trajectory"),
                        "opro_prompt_hashes": sink.get("opro_prompt_hashes"),
                        "opro_value_clamps": sink.get("opro_value_clamps"),
                        "opro_todo_exhausted": sink.get("opro_todo_exhausted"),
                        "opro_failed_calls": sink.get("opro_failed_calls"),
                        "opro_calls": sink.get("opro_calls"),
                        "opro_retries_per_call": sink.get("opro_retries_per_call"),
                        "opro_total_retries": sink.get("opro_total_retries"),
                        "opro_raw_responses": sink.get("opro_raw_responses"),
                        "opro_num_out_of_range": sink.get("opro_num_out_of_range"),
                        "opro_num_uncoercible": sink.get("opro_num_uncoercible"),
                        "opro_sym_out_of_vocab": sink.get("opro_sym_out_of_vocab"),
                        "opro_sym_oov_rate": sink.get("opro_sym_oov_rate"),
                        "opro_num_oor_rate": sink.get("opro_num_oor_rate"),
                        "snap2_seed_mode": sink.get("snap2_seed_mode"),
                        "snap2_seed_size": sink.get("snap2_seed_size"),
                        "snap2_handoff_d2h": sink.get("snap2_handoff_d2h")}
            track_and_run(tracker, tracker_path, llm_name, ds_name, method, rxs, rx, repeats, run_snap2, runtime)
        elif method == "LLM":
            # BS_LLM: single-shot few-shot warm-start (warm_smo, method='LLM').
            rx = f"{llm_name}/LLM/exploit,{last}"
            rxs[rx] = SOME(txt=rx)
            def run_warm(rep):
                set_deterministic_seed(base_seed, ds_name, rep)
                args._gen_log_synth = {}
                res, _ = warm_smo(args, score_fn, method="LLM")
                time.sleep(2)  # Small rate limit
                row = res[0]
                return {"selected_row": list(row),
                        "chebyshev": chebyshev(d, row), "d2h": d2h(d, row),
                        "synth_rows": args._gen_log_synth.get("synth_rows"),
                        "warm_done_x": args._gen_log_synth.get("new_done_x")}
            track_and_run(tracker, tracker_path, llm_name, ds_name, method, rxs, rx, repeats, run_warm, runtime)
        else:
            raise ValueError(
                f"Unknown strategy '{method}'. Finalized methods: "
                "LLM, SynthCore, OPRO, SNAP2 (+ baselines exploit/random/rrp run automatically).")

    return rxs


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Beyond the Prompt benchmark runner")
    parser.add_argument("--datasets", nargs="+", default=["data/misc/auto93.csv"], help="Dataset CSV paths")
    parser.add_argument("--llm", type=str, default="openrouter/openai/gpt-oss-120b", help="LLM string")
    parser.add_argument("--repeats", type=int, default=20, help="Number of repeats")
    parser.add_argument("--last", type=int, default=20, help="Budget")
    parser.add_argument("--label", type=int, default=4, help="Initial random labels")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--strategies", nargs="+",
                        default=["LLM", "SynthCore", "OPRO", "SNAP2"],
                        help="Finalized methods: LLM, SynthCore, SNAP (=OPRO), SNAP2 "
                             "(exploit/random/rrp baselines run automatically)")
    parser.add_argument("--output", type=str, default="results/benchmark_results.json")
    parser.add_argument("--tracker", type=str, default="results/status_tracker.json")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip local baselines (baseline/rrp/random/exploit). Use when you only want to benchmark an LLM strategy and already have baseline scores in a prior tracker.")
    parser.add_argument("--metric", choices=["chebyshev", "d2h"], default="d2h",
                        help="Reporting metric for live SK report and analysis. Tracker stores both.")
    parser.add_argument("--gen-log-dir", type=str, default="results/gen_logs",
                        help="Per-repeat generation artifacts (selected_row, synth_rows, etc.) write here.")
    parser.add_argument("--backfill", action="store_true",
                        help="Re-run completed repeats with same seed to fill missing d2h_scores + gen logs. Asserts chebyshev matches.")
    args = parser.parse_args()

    runtime = {
        "metric": args.metric,
        "gen_log_dir": args.gen_log_dir,
        "backfill": args.backfill,
        "base_seed": args.seed,
    }

    print(f"{'#'*70}", flush=True)
    print(f"# Beyond the Prompt — Resilient Benchmark Runner", flush=True)
    print(f"# Cache/Tracker: ENABLED ({args.tracker})", flush=True)
    print(f"# Model: {args.llm}", flush=True)
    print(f"# Datasets: {len(args.datasets)}", flush=True)
    print(f"# Strategies: {args.strategies}", flush=True)
    print(f"# Repeats: {args.repeats}", flush=True)
    print(f"{'#'*70}", flush=True)

    all_rxs = {}  
    per_dataset_results = {}
    tracker = load_tracker(args.tracker)

    for ds_path in args.datasets:
        if not os.path.exists(ds_path):
            print(f"SKIP: {ds_path} not found", flush=True)
            continue

        try:
            rxs = run_one_dataset(
                dataset_path=ds_path,
                llm_name=args.llm,
                repeats=args.repeats,
                last=args.last,
                label=args.label,
                strategies_to_run=args.strategies,
                base_seed=args.seed,
                tracker=tracker,
                tracker_path=args.tracker,
                skip_baselines=args.skip_baselines,
                runtime=runtime,
            )

            # Filter out strategies that had 0 successes to avoid IndexError in Scott-Knott
            valid_rxs = [some for some in rxs.values() if some.n > 0]

            if not valid_rxs:
                print(f"WARNING: No strategies succeeded for {ds_path}", flush=True)
                continue

            print(f"\n--- Scott-Knott Report: {os.path.basename(ds_path)} ---", flush=True)
            try:
                report(valid_rxs)
            except ZeroDivisionError:
                # Scott-Knott needs >=2 repeats per group; skip the live report
                # for tiny smoke runs. Scores are already saved to the tracker.
                print("  (skipped: need >=2 repeats for the Scott-Knott report)", flush=True)

            ds_name = os.path.basename(ds_path)
            per_dataset_results[ds_name] = {}
            for rx_name, some_obj in rxs.items():
                if some_obj.n > 0:
                    per_dataset_results[ds_name][rx_name] = {
                        "rank": some_obj.rank,
                        "n": some_obj.n,
                        "median": some_obj.mid(),
                        "iqr": some_obj.div(),
                        "lo": some_obj.lo,
                        "hi": some_obj.hi,
                    }
                else:
                    per_dataset_results[ds_name][rx_name] = {"rank": -1, "n": 0}

            for rx_name, some_obj in rxs.items():
                if some_obj.n > 0:
                    if rx_name not in all_rxs:
                        all_rxs[rx_name] = SOME(txt=rx_name)
                    for val in some_obj.has():
                        all_rxs[rx_name].add(val)
        except Exception as e:
            print(f"CRITICAL ERROR processing dataset {ds_path}: {e}", flush=True)
            traceback.print_exc()
            continue

    if len(args.datasets) > 1:
        print(f"\n{'='*70}\nCROSS-DATASET Scott-Knott Report\n{'='*70}")
        valid_all_rxs = [some for some in all_rxs.values() if some.n > 0]
        if valid_all_rxs:
            try:
                report(valid_all_rxs)
            except ZeroDivisionError:
                print("  (skipped: need >=2 repeats for the Scott-Knott report)", flush=True)
        else:
            print("No valid results for cross-dataset report.")

    output = {
        "meta": {
            "model": args.llm,
            "datasets": args.datasets,
            "strategies": args.strategies,
            "repeats": args.repeats,
            "timestamp": datetime.now().isoformat(),
        },
        "per_dataset": per_dataset_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
