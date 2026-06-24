"""snap2_relative_performance.png — SNAP2 as the black reference line.

Two stacked panels of mean validation-regret Δ (%) per dataset (higher=better),
Δ = 1 - (score - baseline_min)/(baseline_median - baseline_min), computed from
d2h_scores (same as compute_delta_106). AMP & DAPR excluded. Exploit shown EZR.

Base (black) : SNAP2.
Top panel    : weaker methods  (SynthCore, Random, RRP).
Bottom panel : stronger methods (SNAP [=OPRO], BS_LLM, EZR [=Exploit]).
X: datasets sorted worst->best by SNAP2 Δ. Pathological (baseline med==min) dropped.
"""
import json
import os
import statistics
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from analysis.compute_delta import (  # noqa: E402
    index_csvs, select_datasets, baselines_for, compute_delta, LLM_KEY, TRACKER,
)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["font.size"] = 16
matplotlib.rcParams["axes.unicode_minus"] = False


def mean_delta(raw, bmin, bmed):
    ds = [compute_delta(s, bmin, bmed) for s in raw]
    ds = [d for d in ds if d is not None]
    return statistics.mean(ds) * 100 if ds else None  # Δ as percent (display scale)


def main():
    tracker = json.load(open(TRACKER))
    csv_index = index_csvs()
    datasets = select_datasets(tracker)
    baselines, dims = baselines_for(datasets, csv_index)

    # method -> (tracker_top, tracker_key). SNAP2 read from the same tracker.
    SRC = {"OPRO": (LLM_KEY, "OPRO"), "LLM": (LLM_KEY, "LLM"),
           "Exploit": ("local", "exploit"), "SynthCore": (LLM_KEY, "SynthCore"),
           "Random": ("local", "random"), "RRP": ("local", "rrp")}

    rows = []
    for ds in datasets:
        bmin, bmed = baselines[ds]["min"], baselines[ds]["median"]
        if bmed - bmin <= 1e-12:          # pathological -> drop
            continue
        s2 = tracker[LLM_KEY].get(ds, {}).get("SNAP2", {}).get("d2h_scores") or []
        if len(s2) < 19:                  # need SNAP2 coverage
            continue
        rec = {"ds": ds, "SNAP2": mean_delta(s2, bmin, bmed)}
        ok = rec["SNAP2"] is not None
        for label, (top, key) in SRC.items():
            raw = tracker[top].get(ds, {}).get(key, {}).get("d2h_scores") or []
            rec[label] = mean_delta(raw, bmin, bmed)
            ok = ok and rec[label] is not None
        if ok:
            rows.append(rec)

    if not rows:
        print("No datasets with >=19 reps on all methods yet — run the full "
              "benchmark (20 repeats) before plotting. Skipping figure.")
        return

    rows.sort(key=lambda r: r["SNAP2"])
    x = list(range(len(rows)))
    base = [r["SNAP2"] for r in rows]

    def col(m):
        return [r[m] for r in rows]

    BASE_COLOR = "#000000"
    PALETTE = {"SynthCore": "#1f77b4", "Random": "#2ca02c", "RRP": "#d62728",
               "OPRO": "#9467bd", "LLM": "#17becf", "Exploit": "#8c564b"}
    LABEL = {"LLM": "BS_LLM", "OPRO": "SNAP", "Exploit": "EZR"}

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)

    def panel(ax, others, title):
        for m in others:
            ax.plot(x, col(m), color=PALETTE[m], lw=1.0, alpha=0.55,
                    marker="o", ms=2.5, label=LABEL.get(m, m))
        ax.plot(x, base, color=BASE_COLOR, lw=2.6, marker="o", ms=3.5,
                label="SNAP2", zorder=5)
        ax.set_ylabel("Mean validation regret Δ (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        ax.legend(ncol=5, loc="lower right", framealpha=0.9)
        ax.set_ylim(-15, 112)

    panel(ax_top, ["SynthCore", "Random", "RRP"],
          "SNAP2 vs weaker methods  (SynthCore, Random, RRP)")
    panel(ax_bot, ["OPRO", "LLM", "Exploit"],
          "SNAP2 vs stronger methods  (SNAP, BS_LLM, EZR)")

    ax_bot.set_xlabel(
        f"Datasets sorted by SNAP2 performance: worst → best  (n={len(rows)})")
    ax_bot.set_xlim(-1, len(rows))
    fig.suptitle(
        "SNAP2 relative performance: mean Δ per dataset (higher = better)",
        fontweight="bold")
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    out = os.path.join(ROOT, "reports", "snap2_relative_performance.png")
    fig.savefig(out, dpi=200)
    print("wrote", out, " datasets:", len(rows))
    print("SNAP2 Δ range: %.1f .. %.1f" % (base[0], base[-1]))


if __name__ == "__main__":
    main()
