"""
Make evaluation summary artifacts from manual judgment files.

Inputs (expected):
  - outputs/eval/final_eval_judgments.json
  - outputs/eval/final_eval_L15_s1_judgments.json

Outputs:
  - outputs/eval/final_eval_metrics.json
  - plots/final_eval_bars.png
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "outputs", "eval")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


FILES = {
    "L15_s1": os.path.join(EVAL_DIR, "final_eval_L15_s1_judgments.json"),
    "L15_s5": os.path.join(EVAL_DIR, "final_eval_judgments.json"),
}


def main() -> None:
    sets = ["syco_lied", "honest_control"]

    metrics = {}
    for label, path in FILES.items():
        data = json.load(open(path, "r"))
        by_set = defaultdict(list)
        for d in data:
            by_set[d["set"]].append(d)

        metrics[label] = {}
        for s in sets:
            subset = by_set[s]

            def rate(k: str) -> float:
                return sum(x[k] for x in subset) / len(subset)

            metrics[label][s] = {
                "n": len(subset),
                "baseline_refuse_rate": rate("baseline_refuse"),
                "steered_refuse_rate": rate("steered_refuse"),
                "baseline_correct_rate": rate("baseline_correct"),
                "steered_correct_rate": rate("steered_correct"),
            }

    metrics_path = os.path.join(EVAL_DIR, "final_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot: refusal + correctness for syco_lied and honest_control, comparing s1 vs s5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200)

    for ax, set_name in zip(axes, sets):
        labels = list(FILES.keys())
        x = range(len(labels))

        base_ref = [metrics[l][set_name]["baseline_refuse_rate"] for l in labels]
        steer_ref = [metrics[l][set_name]["steered_refuse_rate"] for l in labels]
        base_cor = [metrics[l][set_name]["baseline_correct_rate"] for l in labels]
        steer_cor = [metrics[l][set_name]["steered_correct_rate"] for l in labels]

        width = 0.18
        ax.bar([i - 1.5 * width for i in x], base_ref, width, label="baseline_refuse")
        ax.bar([i - 0.5 * width for i in x], steer_ref, width, label="steered_refuse")
        ax.bar([i + 0.5 * width for i in x], base_cor, width, label="baseline_correct")
        ax.bar([i + 1.5 * width for i in x], steer_cor, width, label="steered_correct")

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title(set_name)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=False)
    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, "final_eval_bars.png")
    plt.savefig(out_png)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()


