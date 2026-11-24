"""The seven search methods, with the prompt and parsing conventions used in the study."""

import json
import random
import re

from . import prompts
from .ezr import NUM, branch, clone, d2h, dists, loglikes


def exploit(data, done, todo, target):
    done = sorted(done, key=lambda row: d2h(data, row))
    while len(done) < target and len(todo) >= 3:
        n = len(done)
        cut = int(0.5 + n**0.5)
        best, rest = clone(data, done[:cut]), clone(data, done[cut:])
        random.shuffle(todo)

        def acquisition(row):
            return loglikes(best, row, n, 2) - loglikes(rest, row, n, 2)

        todo = sorted(todo[:100], key=acquisition, reverse=True) + todo[100:]
        done.append(todo.pop(0))
        done.sort(key=lambda row: d2h(data, row))
    return done, todo


def json_object(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    obj = json.loads(match.group())
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj


def snap_proposals(data, text, k):
    if not text or not text.strip():
        raise ValueError("empty response")
    try:
        obj = json_object(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"json decode failed: {error}") from error
    if "candidates" not in obj:
        raise ValueError("missing 'candidates' key")
    candidates = obj["candidates"]
    if not isinstance(candidates, list) or len(candidates) < k:
        count = len(candidates) if isinstance(candidates, list) else "non-list"
        raise ValueError(f"expected {k} candidates, got {count}")
    rows = []
    for index, candidate in enumerate(candidates[:k]):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {index} is not a dict")
        missing = {c.txt for c in data.cols.x} - candidate.keys()
        if missing:
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(f"candidate {index} missing features: {sorted(missing)[:5]}{suffix}")
        row = ["?"] * len(data.cols.names)
        for col in data.cols.x:
            value = candidate[col.txt]
            if col.this is NUM:
                try:
                    row[col.at] = float(value)
                except (TypeError, ValueError):
                    row[col.at] = "?"
            else:
                row[col.at] = str(value)
        rows.append(row)
    return rows


def snap(data, done, todo, model):
    trajectory = [(r, d2h(data, r)) for r in done]
    collisions = []
    while len(done) < 20 and todo:
        k = min(2, 20 - len(done), len(todo))
        trajectory.sort(key=lambda item: item[1], reverse=True)
        messages = prompts.build_opro_prompt(data, trajectory, k, collisions)
        for attempt in range(3):
            raw = model(messages)
            try:
                proposals = snap_proposals(data, raw, k)
                break
            except ValueError as error:
                if attempt == 2:
                    raise
                messages = prompts.build_opro_prompt(
                    data, trajectory, k, collisions, retry_error=str(error), prior_response=raw
                )
        collisions = []
        for proposal in proposals:

            def distance(row):
                return dists(data, proposal, row)

            nearest = min(todo, key=distance)
            previous = min(done, key=distance)
            if distance(previous) < distance(nearest):
                collisions.append((previous, d2h(data, previous)))
            todo.remove(nearest)
            done.append(nearest)
            trajectory.append((nearest, d2h(data, nearest)))
    return min(done, key=lambda row: d2h(data, row))


def bs_proposals(data, done, model):
    raw = model(prompts.bs_prompt(data, done[:2], done[2:]))
    # Historical BS_LLM asked for a list but parsed rows of a Markdown table.
    lines = raw.strip().splitlines()
    rows = [line.strip("|").split("|") for line in lines[2:]]
    return [
        [float(v) for v in row[:-1]]
        for label in ("Best", "Rest")
        for row in rows
        if row[-1].strip() == label
    ]


def amp_proposals(data, done, model):
    features = []
    for c in data.cols.x:
        if c.this is NUM:
            domain = f"{c.lo} to {c.hi}"
        else:
            values = list(c.has)
            domain = f"Categorical {values}" if len(values) < 10 else "Categorical (many values)"
        features.append(f"- {c.txt} (Feature): {domain}")
    features = "\n".join(features)
    objectives = []
    for c in data.cols.y:
        # The original AMP prompt builder used the column's w attribute.
        direction = "Maximize" if getattr(c, "w", 0) > 0 else "Minimize"
        objectives.append(f"- {c.txt} (Objective - {direction}): {c.lo} to {c.hi}")
    description = (
        features
        + "\n\nObjectives (for context, do NOT generate values for these):\n"
        + "\n".join(objectives)
    )
    analysis = model(prompts.get_analysis_prompt(description, done[:2], done[2:]))
    rules = model(prompts.get_constraint_prompt(analysis))
    raw = model(prompts.get_generation_prompt(analysis, rules, done[:2], features))
    if "better_examples" in raw:
        draft = json_object(raw)
        proposals = [
            ex["features"]
            for key in ("better_examples", "poorer_examples")
            for ex in draft.get(key, [])
        ]
    else:
        proposals = []
        for line in raw.splitlines():
            if "|" in line and all(s not in line for s in ("---", "Best", "Rest")):
                fields = [x.strip() for x in line.split("|") if x.strip()]
                try:
                    proposals.append([float(x) for x in fields[:-1]])
                except ValueError:
                    continue
    checked = []
    for proposal in proposals:
        reply = model(prompts.get_validation_prompt(rules, proposal))
        try:
            verdict = json_object(reply)
        except ValueError:
            verdict = {}
        if verdict.get("status") == "INVALID":
            proposal = verdict.get("revised_features", proposal)
        checked.append(proposal)
    return checked


def warm_start(data, done, todo, method, model):
    done.sort(key=lambda row: d2h(data, row))
    generate = amp_proposals if method == "AMP-4" else bs_proposals
    width = len(data.cols.names) - len(data.cols.y)
    min_size = max(width, 1 + max(c.at for c in data.cols.x))
    for proposal in generate(data, done, model):
        if not isinstance(proposal, list) or len(proposal) < min_size:
            continue
        random.shuffle(todo)
        todo.sort(key=lambda row: dists(data, proposal, row))
        done.append(todo.pop(0))
    done.sort(key=lambda row: d2h(data, row))
    seed = [done[0], done[1], done[-1], done[-2]]
    return exploit(data, seed, todo, 20)[0][0]


def optimize(data, method, model):
    if method == "Random":
        return clone(data, random.choices(data.rows, k=20), rank=True).rows[0]
    if method == "RRP":
        return branch(data, data.rows, 4)[0][0]
    pool = list(data.rows)
    random.shuffle(pool)
    done, todo = pool[:4], pool[4:]
    if method in ("BS_LLM", "AMP-4"):
        return warm_start(data, done, todo, method, model)
    if method == "EZR":
        return exploit(data, done, todo, 20)[0][0]
    if method == "SNAP2":
        done, todo = exploit(data, done, todo, 10)
    return snap(data, done, todo, model)
