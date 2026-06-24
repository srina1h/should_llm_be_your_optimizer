"""OPRO (Yang et al., 2023) — LLM-as-Optimizer.

Iterative LLM-in-loop search. The LLM receives the running (x, d2h) trajectory
each iteration and proposes the next k candidates. Snap to nearest unevaluated
real row in the dataset; evaluate; append; repeat.

Compare against the BS_LLM (single-shot) and TPE-based strategies in this repo.

Lenient parsing: out-of-vocab SYMs and out-of-range NUMs are accepted (counted),
because the snap step uses dist over real rows so invalid values just contribute
max distance for that feature. Trajectory only ever contains real rows.
Retries trigger only on hard structural failures (no JSON, missing features).
"""

import hashlib
import json
import random
import re

from src.language import load_model
from src.utils.ezr import dists, d2h, clone, loglikes

from methods.base import BaseStrategy


N_RETRIES = 2
K_PER_CALL = 2


class OPROParseError(Exception):
    """Per-call structural parse failure. Retryable within an iteration."""


class OPROParseFailure(Exception):
    """Raised when LLM fails to produce parseable JSON after N_RETRIES retries.

    Triggered only by hard structural issues (no JSON, missing features, wrong
    shape). Out-of-vocab SYM and out-of-range NUM are NOT structural failures
    in this implementation — they are accepted by the parser and resolved at
    the snap step (dist=1.0 for that feature), with counts recorded for
    transparency. Failed runs are recorded in tracker.errors[].
    """


def _stable_hash(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _coerce_num(v):
    if isinstance(v, bool):
        raise ValueError(f"expected numeric, got bool: {v}")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return float(v.strip())
    raise ValueError(f"cannot coerce {type(v).__name__} to numeric")


def parse_opro_response(raw, expected_k, data):
    """Lenient OPRO response parser.

    Returns (candidates, stats) where:
      - candidates: list of feature-dicts. NUM coerced to float (no clamp);
        out-of-range values pass through. SYM passed as string; out-of-vocab
        values pass through. Caller's snap step handles invalids via dist().
      - stats: dict with n_num_oor, n_sym_oov, n_num_uncoercible counters.

    Raises OPROParseError on STRUCTURAL failures only:
      - no parseable JSON
      - missing 'candidates' key or wrong type
      - fewer than expected_k candidates
      - candidate not a dict, or missing required features

    Caller retries on these. Per-feature value validity is tracked in stats
    rather than triggering a retry.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise OPROParseError("empty response")

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise OPROParseError("no JSON object found")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise OPROParseError(f"json decode failed: {e}")

    if not isinstance(obj, dict) or "candidates" not in obj:
        raise OPROParseError("missing 'candidates' key")

    cands = obj["candidates"]
    if not isinstance(cands, list) or len(cands) < expected_k:
        raise OPROParseError(
            f"expected {expected_k} candidates, got {len(cands) if isinstance(cands, list) else 'non-list'}"
        )

    feature_names = [c.txt for c in data.cols.x]
    feature_set = set(feature_names)
    sym_vocab = {
        c.txt: set(map(str, (c.has or {}).keys()))
        for c in data.cols.x
        if not hasattr(c, "lo")
    }

    stats = {"n_num_oor": 0, "n_sym_oov": 0, "n_num_uncoercible": 0}
    parsed = []
    for idx, cand in enumerate(cands[:expected_k]):
        if not isinstance(cand, dict):
            raise OPROParseError(f"candidate {idx} is not a dict")
        missing = feature_set - set(cand.keys())
        if missing:
            raise OPROParseError(
                f"candidate {idx} missing features: {sorted(missing)[:5]}"
                f"{' ...' if len(missing) > 5 else ''}"
            )

        clean = {}
        for c in data.cols.x:
            v = cand[c.txt]
            if hasattr(c, "lo"):
                try:
                    fv = _coerce_num(v)
                except ValueError:
                    # Uncoercible NUM (e.g. LLM returned a string non-number).
                    # Pass through original; dists() will treat as "?" mismatch.
                    stats["n_num_uncoercible"] += 1
                    clean[c.txt] = v
                    continue
                if fv < c.lo or fv > c.hi:
                    stats["n_num_oor"] += 1
                clean[c.txt] = fv
            else:
                sv = str(v)
                if sym_vocab[c.txt] and sv not in sym_vocab[c.txt]:
                    stats["n_sym_oov"] += 1
                clean[c.txt] = sv
        parsed.append(clean)

    return parsed, stats


EXPLOIT_SORT_CAP = 100  # mirror the.any: only sort a random subset of todo


def _exploit_rank(data, todo, done):
    """One SMO exploit acquisition step: reorder `todo` by B - R, best first.

    B/R are Naive-Bayes log-likelihoods under the best/rest split of `done`
    (`cut = int(.5 + len(done) ** 0.5)`, matching warm_start.py). No LLM.
    """
    cut = int(.5 + len(done) ** 0.5)
    best = clone(data, done[:cut])
    rest = clone(data, done[cut:])
    n = len(done)
    key = lambda r: (loglikes(best, r, n, 2) - loglikes(rest, r, n, 2))
    random.shuffle(todo)
    return sorted(todo[:EXPLOIT_SORT_CAP], key=key, reverse=True) + todo[EXPLOIT_SORT_CAP:]


def build_opro_seed(data, args):
    """Construct OPRO's initial (done, todo, trajectory).

    Modes (via args.opro_seed, default "random"):
      - "random":  args.label random rows. Byte-for-byte the legacy OPRO seed.
      - "exploit": grow from args.label random rows to S = round(args.last / 2)
                   rows using the cheap exploit acquisition (no LLM), then hand
                   off. S clamped to [args.label, args.last]; override via
                   args.opro_seed_size.
    """
    pool = list(data.rows)
    random.shuffle(pool)
    done = pool[: args.label]
    todo = pool[args.label:]

    if getattr(args, "opro_seed", "random") == "exploit":
        s = getattr(args, "opro_seed_size", None) or round(args.last / 2)
        s = max(args.label, min(int(s), args.last))
        done = sorted(done, key=lambda r: d2h(data, r))
        while len(done) < s and len(todo) >= 3:
            top, *todo = _exploit_rank(data, todo, done)
            done.append(top)
            done = sorted(done, key=lambda r: d2h(data, r))

    trajectory = [(r, d2h(data, r)) for r in done]
    return done, todo, trajectory


class OPRO_Strategy(BaseStrategy):
    """Iterative LLM-as-optimizer (OPRO).

    Unlike BaseStrategy subclasses that implement `generate_examples()` for
    one-shot warm-starting, OPRO owns the entire post-init search loop and
    exposes `run() -> (results, metadata)`.
    """

    def __init__(self, data, args):
        super().__init__(data, args)
        self.model = load_model(args, name=args.llm).get_pipeline()
        # Wide datasets (77+ features × 2 candidates) hit 4000 max_tokens and
        # truncate the second candidate mid-JSON. Bump per-call without
        # mutating shared args so other strategies keep their default.
        if hasattr(self.model, "kwargs"):
            self.model.kwargs["max_tokens"] = max(
                self.model.kwargs.get("max_tokens", 4000), 8000
            )

    def _x_vec_to_row(self, x_dict):
        row = ["?"] * len(self.data.cols.names)
        for c in self.data.cols.x:
            row[c.at] = x_dict[c.txt]
        return row

    def _nearest_with_dist(self, pool, x_row):
        if not pool:
            return None, float("inf")
        best_row = None
        best_d = float("inf")
        for r in pool:
            d = dists(self.data, x_row, r)
            if d < best_d:
                best_d = d
                best_row = r
        return best_row, best_d

    def run(self):
        args = self.args
        data = self.data

        sink = getattr(args, "_gen_log_synth", None)
        if not isinstance(sink, dict):
            sink = {}
            args._gen_log_synth = sink

        done, todo, trajectory = build_opro_seed(data, args)
        seed_mode = getattr(args, "opro_seed", "random")
        seed_size = len(done)
        handoff_d2h = min(s for _, s in trajectory)

        synth_log = []
        new_done_log = [list(r) for r in done]
        prompt_hashes = []
        best_d2h_per_call = []
        retries_per_call = []
        raw_responses = []        # list[list[str]] — one inner list per iter (incl retries)
        n_num_oor = 0
        n_sym_oov = 0
        n_num_uncoercible = 0
        collisions = 0
        calls = 0
        failed_calls = 0
        todo_exhausted = False
        last_iter_collisions = []

        while len(done) < args.last:
            k = min(K_PER_CALL, args.last - len(done))
            sorted_traj = sorted(trajectory, key=lambda t: t[1], reverse=True)

            from prompts.opro_prompts import build_opro_prompt
            messages = build_opro_prompt(data, sorted_traj, k, last_iter_collisions)
            prompt_hashes.append(
                _stable_hash(json.dumps(messages, sort_keys=True))
            )

            candidates = None
            iter_raws = []
            n_retries_this_call = 0
            for attempt in range(1 + N_RETRIES):
                raw = self.model.invoke(messages).content
                iter_raws.append(raw)
                try:
                    candidates, stats = parse_opro_response(raw, expected_k=k, data=data)
                    n_num_oor += stats["n_num_oor"]
                    n_sym_oov += stats["n_sym_oov"]
                    n_num_uncoercible += stats["n_num_uncoercible"]
                    break
                except OPROParseError as e:
                    n_retries_this_call = attempt + 1
                    messages = build_opro_prompt(
                        data, sorted_traj, k, last_iter_collisions,
                        retry_error=str(e), prior_response=raw,
                    )

            raw_responses.append(iter_raws)

            if candidates is None:
                failed_calls += 1
                retries_per_call.append(n_retries_this_call)
                raise OPROParseFailure(
                    f"LLM returned unparseable JSON after {N_RETRIES} retries"
                )

            retries_per_call.append(n_retries_this_call)
            calls += 1
            last_iter_collisions = []
            for x_dict in candidates[:k]:
                x_row = self._x_vec_to_row(x_dict)
                synth_log.append([x_row[c.at] for c in data.cols.x])

                nn_todo, d_todo = self._nearest_with_dist(todo, x_row)
                nn_done, d_done = self._nearest_with_dist(done, x_row)

                if nn_done is not None and d_done < d_todo:
                    collisions += 1
                    last_iter_collisions.append((nn_done, d2h(data, nn_done)))

                if nn_todo is None:
                    todo_exhausted = True
                    break

                todo.remove(nn_todo)
                done.append(nn_todo)
                trajectory.append((nn_todo, d2h(data, nn_todo)))
                new_done_log.append(list(nn_todo))

            best_d2h_per_call.append(min(t[1] for t in trajectory))

            if todo_exhausted:
                break

        # Audit assertions — fire on every repeat
        assert len(done) == args.last or todo_exhausted, (
            f"OPRO budget accounting: done={len(done)} expected={args.last} "
            f"todo_exhausted={todo_exhausted}"
        )
        assert all(
            isinstance(x, (int, float)) and 0.0 <= x <= 1.0
            for x in best_d2h_per_call
        ), f"best_d2h_per_call out of [0,1]: {best_d2h_per_call}"
        assert all(
            best_d2h_per_call[i] >= best_d2h_per_call[i + 1]
            for i in range(len(best_d2h_per_call) - 1)
        ), f"best_d2h_per_call not non-increasing: {best_d2h_per_call}"
        # prompt_hashes.append fires once per iteration BEFORE the retry loop,
        # so duplicates here mean trajectory unchanged or cache scoping broken.
        assert len(set(prompt_hashes)) == len(prompt_hashes), (
            "duplicate meta-prompt across iterations — trajectory not changing "
            "or LiteLLM cache scoping broken"
        )

        proposals_per_call = K_PER_CALL  # candidates per LLM call (k)
        total_proposals = calls * proposals_per_call
        sink["synth_rows"] = synth_log
        sink["snap2_seed_mode"] = seed_mode
        sink["snap2_seed_size"] = seed_size
        sink["snap2_handoff_d2h"] = handoff_d2h
        sink["new_done_x"] = new_done_log
        sink["opro_collision_rate"] = (
            collisions / total_proposals if total_proposals else 0.0
        )
        sink["opro_best_d2h_per_call"] = best_d2h_per_call
        sink["opro_trajectory"] = [(list(r), s) for r, s in trajectory]
        sink["opro_prompt_hashes"] = prompt_hashes
        sink["opro_todo_exhausted"] = todo_exhausted
        sink["opro_failed_calls"] = failed_calls
        sink["opro_calls"] = calls
        sink["opro_retries_per_call"] = retries_per_call
        sink["opro_total_retries"] = sum(retries_per_call)
        sink["opro_raw_responses"] = raw_responses
        sink["opro_num_out_of_range"] = n_num_oor
        sink["opro_num_uncoercible"] = n_num_uncoercible
        sink["opro_sym_out_of_vocab"] = n_sym_oov
        sink["opro_sym_oov_rate"] = (
            n_sym_oov / total_proposals if total_proposals else 0.0
        )
        sink["opro_num_oor_rate"] = (
            n_num_oor / total_proposals if total_proposals else 0.0
        )

        results = sorted(done, key=lambda r: d2h(data, r))
        metadata = {
            "calls": calls,
            "collisions": collisions,
            "n_num_oor": n_num_oor,
            "n_sym_oov": n_sym_oov,
            "n_num_uncoercible": n_num_uncoercible,
            "todo_exhausted": todo_exhausted,
        }
        return results, metadata
