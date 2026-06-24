# SNAP2: Exploit-Seeded LLM Optimization

Reproduction package for **SNAP2**, an LLM-as-optimizer that seeds the SNAP
search loop with a cheap, no-LLM active-learning pre-pass (EZR). At a fixed
evaluation budget SNAP2 matches or beats every baseline on validation regret
(d2h) while issuing **~38% fewer LLM calls** than SNAP.

On the 105-dataset suite (gpt-oss-120b, 20 repeats), SNAP2 ranks #1 by bestRanks
(statistically-tied-top-group rate 84.8%) and by mean d2h (0.1740).

> **Naming.** **SNAP** is our snapping LLM-as-optimizer — OPRO (Yang et al.,
> 2023) with candidates snapped to the nearest real row. **SNAP2** is SNAP with
> the EZR seed. The code keeps the identifier `OPRO` for this method (class
> `OPRO_Strategy`, tracker key `OPRO`); `SNAP` is accepted as a CLI alias and is
> the label shown in all reports and figures.

## Methods

| Method | LLM | Description |
|---|:--:|---|
| **SNAP2** | ✓ | SNAP seeded by an EZR pre-pass (grows `label`→`last/2` rows, no LLM, then hands off) |
| SNAP | ✓ | Our snapping OPRO (Yang et al., 2023); proposes candidates from the running trajectory |
| LLM | ✓ | Single-shot few-shot warm-start (BS_LLM) |
| SynthCore | ✓ | Density-based synthetic warm-start |
| EZR | ✗ | Sequential model optimization, `B−R` acquisition (a.k.a. exploit) |
| Random | ✗ | Random sampling |
| RRP | ✗ | Recursive random projections |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your OpenRouter key (get one at
<https://openrouter.ai/keys>):

```
OPENROUTER_API_KEY=sk-or-...
```

All LLM methods call the model named by `--llm` (default
`openrouter/openai/gpt-oss-120b`) through this key; responses are disk-cached in
`.litellm_cache/`. The three non-LLM methods (EZR, Random, RRP) need no key.

## Full reproduction (20 reps × 105 datasets × all methods)

```bash
python fetch_data.py     # pull the 105 datasets (MOOT) into data/
python reproduce.py      # benchmark all methods, then build the reports
```

`reproduce.py` runs the benchmark over every dataset in `datasets_105.txt`
(20 repeats, all 7 methods) and then the analysis, producing:

- `reports/bestranks.md` — statistical win-rate ranking (the headline result)
- `reports/snap2_relative_performance.png` — per-dataset regret vs every method

This is a large run (~10k LLM calls). It is **resumable** — the runner persists
each repeat to `results/status_tracker.json` and skips completed work, so you can
stop and re-run `python reproduce.py` at any time. Useful flags:

```bash
python reproduce.py --repeats 3                      # quick partial pass
python reproduce.py --datasets data/misc/auto93.csv  # one dataset
python reproduce.py --skip-benchmark                 # re-run analysis only
```

To run a subset by hand, `SNAP` and `OPRO` are interchangeable:

```bash
python run_benchmark.py --datasets data/misc/auto93.csv \
  --strategies LLM SynthCore SNAP SNAP2 --repeats 20
```

Cost / token / latency table (separate, cache-disabled, 5-dataset profile):

```bash
python analysis/perf_benchmark.py    # -> results/perf/
python analysis/perf_report.py       # -> reports/performance_analysis.md
```

## Layout

```
reproduce.py         one-command full reproduction (benchmark + analysis)
fetch_data.py        download the 105 datasets from MOOT into data/
datasets_105.txt     canonical list of the 105 benchmark datasets
run_benchmark.py     benchmark runner (7 methods + crash-resilient tracker)
methods/             OPRO + SNAP2 (opro.py) + seed unit test
prompts/             all LLM prompt construction (kept separate)
src/                 infra: ezr (DATA/smo/branch/d2h), warm_start, cluster, llms
analysis/            compute_wins, compute_delta, plot_relative, perf_*
data/                5 samples + 2 hpo datasets bundled; rest fetched
```

## Test

```bash
python methods/tests/test_opro_seed.py   # LLM-free unit test: SNAP2 seed construction
```

## Citation

```bibtex
@misc{snap2,
  title  = {SNAP2: Exploit-Seeded LLM Optimization},
  author = {TODO},
  year   = {2026}
}
```
