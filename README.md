# Better Together, in the Right Order

## Requirements

Python 3.10+ on macOS or Linux and an OpenRouter API key. All datasets are included. The scripts use the Python standard library.

```sh
export OPENROUTER_API_KEY="your-key"
```

## How to run

```sh
./rq1.sh   # Run all seven methods on 104 datasets
./rq2.sh   # Run the main benchmark and compare SNAP with SNAP2 on 20 datasets using GPT and GLM
./rq3.sh   # Run the main benchmark with feature counts for the dimensionality comparison
./rq4.sh   # Measure calls, tokens, cost, and elapsed time for all seven methods on five datasets
```

Each experiment runs **20 repeats per method and dataset**. The seven methods are SNAP2, SNAP, BS_LLM, AMP-4, EZR, Random, and RRP. Each script saves raw trials in SQLite files under `results/`. RQ1–RQ3 share completed main-benchmark trials. Running a script again resumes missing repeats.
