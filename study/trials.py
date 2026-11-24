"""Fresh OpenRouter trials. Completed trials can be reused across RQs."""

import sqlite3
import hashlib
import json
import os
import random
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen
from zipfile import ZipFile

from .datasets import ARCHIVE, load_dataset
from .ezr import d2h
from .search import optimize


class OpenRouter:
    def __init__(self, model, max_tokens, probe):
        self.options = dict(model=model, max_tokens=max_tokens, temperature=0.7, top_p=1.0)
        if probe:
            self.options["provider"] = {"sort": "throughput", "require_parameters": True}
            if model == "z-ai/glm-5.3-flash":
                self.options["reasoning"] = {"effort": "low"}
        self.calls = 0
        self.tokens = 0
        self.usd = 0.0

    def __call__(self, messages):
        messages = [
            m if isinstance(m, dict) else {"role": m[0].replace("human", "user"), "content": m[1]}
            for m in messages
        ]
        request = Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(dict(self.options, messages=messages)).encode(),
            headers={
                "Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(result["error"].get("message", "OpenRouter request failed"))
        usage = result["usage"]
        self.calls += 1
        self.tokens += usage["total_tokens"]
        if self.usd is not None and usage.get("cost") is not None:
            self.usd += usage["cost"]
        else:
            self.usd = None
        return result["choices"][0]["message"]["content"] or ""


REPEATS = 20


def trials(experiment, model, datasets, methods):
    """Checkpoint each completed trial, and keep the three experiment cohorts separate."""
    folder = Path("results")
    folder.mkdir(parents=True, exist_ok=True)
    stem = experiment + "-" + model.replace("/", "-")
    with (
        ZipFile(ARCHIVE) as archive,
        sqlite3.connect(folder / (stem + ".sqlite"), timeout=900) as db,
    ):
        db.execute("""CREATE TABLE IF NOT EXISTS trials (
            dataset TEXT, method TEXT, repeat INTEGER, features INTEGER, d2h REAL, calls INTEGER,
            tokens INTEGER, usd REAL, seconds REAL,
            PRIMARY KEY (dataset, method, repeat))""")
        for name in datasets:
            existing = set(
                db.execute("SELECT method, repeat FROM trials WHERE dataset = ?", (name,))
            )
            pending = [(m, i) for m in methods for i in range(REPEATS) if (m, i) not in existing]
            if not pending:
                continue
            data = load_dataset(archive, name, reference=experiment != "quality")
            probe = experiment == "probe"
            if probe:
                seed = 20260924 + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            else:
                seed = 12345 + int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 1000000
            print(f"Running {experiment}: {name} ({len(pending)} trials)", flush=True)
            for method, rep in pending:
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    job = (name, method, rep)
                    if db.execute(
                        "SELECT 1 FROM trials WHERE dataset=? AND method=? AND repeat=?", job
                    ).fetchone():
                        continue
                    random.seed(seed + rep)
                    wide = len(data.cols.x) >= 77
                    cap = 8000 if method in ("SNAP", "SNAP2") and (probe or wide) else 4000
                    client = OpenRouter(model, cap, probe)
                    start = perf_counter()
                    try:
                        winner = optimize(data, method, client)
                    except ValueError as error:
                        raise SystemExit(
                            f"{name}, {method}, repeat {rep + 1}: {error}. Re-run to resume."
                        ) from error
                    metrics = (
                        len(data.cols.x),
                        d2h(data, winner),
                        client.calls,
                        client.tokens,
                        client.usd,
                        perf_counter() - start,
                    )
                    db.execute(
                        "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (*job, *metrics)
                    )
