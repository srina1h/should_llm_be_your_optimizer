import argparse
import os
from pathlib import Path
from zipfile import ZipFile

from .datasets import ARCHIVE
from .trials import trials

METHODS = ("SNAP2", "SNAP", "BS_LLM", "AMP-4", "EZR", "Random", "RRP")
GPT = "openai/gpt-oss-120b"
GLM = "z-ai/glm-5.3-flash"
PROFILE = ("Apache_AllMeasurements.csv", "SS-N.csv", "auto93.csv", "pom3a.csv", "xomo_flight.csv")


def main():
    parser = argparse.ArgumentParser(description="Run the experiments for a research question.")
    parser.add_argument("rq", type=int, choices=range(1, 5))
    rq = parser.parse_args().rq
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.exit(2, "Set OPENROUTER_API_KEY before running the experiment.\n")

    if rq == 4:
        trials("cost", GPT, PROFILE, METHODS)
        return

    with ZipFile(ARCHIVE) as archive:
        datasets = sorted(n for n in archive.namelist() if n.endswith(".csv") and "/" not in n)
    trials("quality", GPT, datasets, METHODS)
    if rq == 2:
        probe = Path("data/probe.txt").read_text().splitlines()
        for model in (GPT, GLM):
            trials("probe", model, probe, ("SNAP", "SNAP2"))


if __name__ == "__main__":
    main()
