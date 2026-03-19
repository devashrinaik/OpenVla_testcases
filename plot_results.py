"""
Plot degradation curve from aggregate results.

Usage:
    python plot_results.py                         # from results/aggregate.json
    python plot_results.py --results results/aggregate.json --out results/degradation.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_degradation_curve(results: dict, out_path: str):
    """Bar chart: TSR per suite (increasing horizon)."""
    # Order suites by expected horizon
    suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    suites = [s for s in suite_order if s in results]
    labels = [results[s]["horizon"] for s in suites]
    tsrs = [results[s]["success_rate"] * 100 for s in suites]
    mean_steps = [results[s]["mean_steps"] for s in suites]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # TSR bars
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"][:len(suites)]
    bars = ax1.bar(range(len(suites)), tsrs, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(suites)))
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Task Success Rate (%)", fontsize=11)
    ax1.set_title("Frozen OpenVLA: TSR vs Task Horizon", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 105)
    for bar, val in zip(bars, tsrs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    # Mean steps bars
    bars2 = ax2.bar(range(len(suites)), mean_steps, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(suites)))
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Mean Episode Steps", fontsize=11)
    ax2.set_title("Mean Episode Length (capped at max_steps)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close()


def plot_per_task_breakdown(results: dict, out_path: str):
    """Per-task TSR within each suite."""
    suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    suites = [s for s in suite_order if s in results and "per_task" in results[s]]

    if not suites:
        print("No per-task data available.")
        return

    fig, axes = plt.subplots(len(suites), 1, figsize=(14, 3 * len(suites)))
    if len(suites) == 1:
        axes = [axes]

    for ax, suite in zip(axes, suites):
        tasks = results[suite]["per_task"]
        names = [t["task"].replace("_", " ")[:40] for t in tasks]
        rates = [t["success_rate"] * 100 for t in tasks]

        colors = ["#2ecc71" if r > 50 else "#e67e22" if r > 0 else "#e74c3c" for r in rates]
        ax.barh(range(len(names)), rates, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlim(0, 105)
        ax.set_xlabel("TSR (%)")
        ax.set_title(f"{suite} ({results[suite]['horizon']})", fontsize=10, fontweight="bold")
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved per-task plot to {out_path}")
    plt.close()


def plot_entropy_traces(results_dir: str, out_path: str):
    """Per-step entropy traces for representative tasks (one per suite)."""
    import glob
    import os

    suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"libero_spatial": "#2ecc71", "libero_object": "#f1c40f",
              "libero_goal": "#e67e22", "libero_10": "#e74c3c"}

    for suite in suite_order:
        # Load task 0 results
        path = f"{results_dir}/{suite}_task0.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        # Average entropy across episodes
        all_ent = [ep["entropies"] for ep in data["episodes"]]
        max_len = max(len(e) for e in all_ent)
        padded = np.full((len(all_ent), max_len), np.nan)
        for i, e in enumerate(all_ent):
            padded[i, :len(e)] = e
        mean_ent = np.nanmean(padded, axis=0)

        ax.plot(range(len(mean_ent)), mean_ent, label=f"{suite} (task 0)",
                color=colors.get(suite, "gray"), linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Mean Token Entropy", fontsize=11)
    ax.set_title("Action Entropy Over Episode (Frozen OpenVLA, No Memory)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved entropy traces to {out_path}")
    plt.close()


def main():
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/aggregate.json")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.results):
        with open(args.results) as f:
            results = json.load(f)

        plot_degradation_curve(results, os.path.join(args.out_dir, "degradation_curve.png"))
        plot_per_task_breakdown(results, os.path.join(args.out_dir, "per_task_breakdown.png"))
    else:
        print(f"No aggregate results at {args.results}. Run run_all.py first.")

    plot_entropy_traces(args.results_dir, os.path.join(args.out_dir, "entropy_traces.png"))


if __name__ == "__main__":
    main()
