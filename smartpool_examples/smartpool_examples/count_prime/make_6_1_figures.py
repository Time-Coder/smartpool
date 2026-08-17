"""Generate Section 6.1 publication figures from the benchmark results JSON.

Reads `6_1_count_prime_results.json` and produces:
  - fig6_1a_throughput.svg   : throughput (tasks/s) vs worker count
  - fig6_1b_time_speedup.svg : wall-clock time and speedup vs worker count

Uses the Open Science publication style.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STYLE = Path(r"C:\Users\bingh\AppData\Roaming\com.ai4s.workbench\runtime\xdg-config\opencode\skills\publication-figures\openscience.mplstyle")
if STYLE.exists():
    plt.style.use(str(STYLE))

HERE = Path(__file__).resolve().parent
RESULTS = HERE / ".." / ".." / ".." / "doc" / "experiments" / "6_1_count_prime_results.json"
OUT_DIR = (HERE / ".." / ".." / ".." / "doc" / "experiments").resolve()


def median(seq):
    return statistics.median(seq)


def main():
    with open(RESULTS, "r", encoding="utf-8") as f:
        data = json.load(f)

    workers = data["workers"]
    n_tasks = data["n_tasks"]
    seq_time = median(data["sequential_sec"])

    series = []
    for name, info in data["pools"].items():
        times = {int(w): median(runs) for w, runs in info["times"].items()}
        ws = sorted(times.keys())
        t = np.array([times[w] for w in ws])
        series.append({
            "label": name,
            "workers": np.array(ws),
            "time": t,
            "throughput": n_tasks / t,
            "speedup": seq_time / t,
        })

    # ---- Figure (a): throughput vs worker count ----
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for s in series:
        ax.plot(s["workers"], s["throughput"], marker="o", markersize=5, label=s["label"])
    ax.set_xlabel("max_workers")
    ax.set_ylabel("Throughput (tasks / s)")
    ax.legend(loc="lower right")
    fig.savefig(OUT_DIR / "fig6_1a_throughput.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig6_1a_throughput.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure (b): wall-clock time (left) and speedup (right) ----
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for s in series:
        ax_t.plot(s["workers"], s["time"], marker="o", markersize=5, label=s["label"])
        ax_s.plot(s["workers"], s["speedup"], marker="o", markersize=5, label=s["label"])
    ax_t.set_xlabel("max_workers")
    ax_t.set_ylabel("Batch wall-clock time (s)")
    ax_t.legend(loc="upper right")
    ax_s.set_xlabel("max_workers")
    ax_s.set_ylabel("Speedup vs. sequential")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_1b_time_speedup.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig6_1b_time_speedup.png", bbox_inches="tight")
    plt.close(fig)

    print(f"figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()