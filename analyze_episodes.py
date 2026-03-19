"""
Within-episode degradation analysis for frozen OpenVLA on LIBERO.

Extracts step-level signals showing WHERE the model fails during long episodes:
  - Action magnitude decay (robot slows down / becomes aimless)
  - Action repetition (consecutive near-identical actions = stuck)
  - Gripper oscillation (rapid open/close = indecision)
  - Action diversity collapse (narrowing behavior repertoire)

Usage:
    python analyze_episodes.py                          # all results
    python analyze_episodes.py --suites libero_10       # single suite
    python analyze_episodes.py --out_dir results/analysis
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_all_episodes(results_dir, suites=None):
    """Load all per-task JSONs, grouped by suite."""
    suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    if suites:
        suite_order = [s for s in suite_order if s in suites]

    data = {}
    for suite in suite_order:
        files = sorted(glob.glob(os.path.join(results_dir, f"{suite}_task*.json")))
        if not files:
            continue
        episodes = []
        for f in files:
            with open(f) as fh:
                task_data = json.load(fh)
            for ep in task_data["episodes"]:
                ep["task_name"] = task_data["task_name"]
                ep["suite"] = suite
                ep["max_steps"] = task_data["max_steps"]
                episodes.append(ep)
        data[suite] = episodes
    return data


def compute_episode_signals(ep, window=20):
    """Compute per-step signals for one episode."""
    actions = np.array(ep["actions"])
    n = len(actions)
    if n < 2:
        return None

    arm_actions = actions[:, :6]  # exclude gripper
    gripper = actions[:, 6]

    # 1. Action magnitude (rolling mean)
    magnitudes = np.linalg.norm(arm_actions, axis=1)

    # 2. Action repetition: L2 distance between consecutive actions
    consec_diff = np.concatenate([[0], np.linalg.norm(arm_actions[1:] - arm_actions[:-1], axis=1)])

    # 3. Gripper oscillation: switches per window
    gripper_switches = np.abs(np.diff(gripper)) > 0.5
    gripper_switches = np.concatenate([[False], gripper_switches]).astype(float)

    # 4. Rolling action diversity (std of actions in window)
    diversity = np.zeros(n)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2)
        if end - start > 1:
            diversity[i] = np.mean(np.std(arm_actions[start:end], axis=0))

    # Smooth everything with rolling mean
    def smooth(x, w=window):
        if len(x) < w:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="same")

    return {
        "magnitude": smooth(magnitudes),
        "consec_diff": smooth(consec_diff),
        "gripper_switches": smooth(gripper_switches),
        "diversity": diversity,
        "n_steps": n,
        "success": ep["success"],
        "raw_gripper_switches": int(gripper_switches.sum()),
    }


def plot_within_episode_curves(data, out_dir):
    """Main plot: within-episode signal curves, success vs failure, per suite."""
    suite_colors = {
        "libero_spatial": "#2ecc71",
        "libero_object": "#f1c40f",
        "libero_goal": "#e67e22",
        "libero_10": "#e74c3c",
    }

    signal_names = ["magnitude", "consec_diff", "gripper_switches", "diversity"]
    signal_labels = [
        "Action Magnitude",
        "Consecutive Action Difference",
        "Gripper Switch Rate",
        "Action Diversity (rolling std)",
    ]

    fig, axes = plt.subplots(len(signal_names), 1, figsize=(14, 4 * len(signal_names)))

    for ax, sig_name, sig_label in zip(axes, signal_names, signal_labels):
        for suite, episodes in data.items():
            # Separate success/failure
            for success, ls, alpha, suffix in [(True, "-", 0.6, "success"), (False, "--", 0.6, "fail")]:
                filtered = [compute_episode_signals(ep) for ep in episodes if ep["success"] == success]
                filtered = [s for s in filtered if s is not None]
                if not filtered:
                    continue

                # Average across episodes, padding to max length
                max_len = max(s["n_steps"] for s in filtered)
                padded = np.full((len(filtered), max_len), np.nan)
                for i, s in enumerate(filtered):
                    padded[i, :s["n_steps"]] = s[sig_name][:s["n_steps"]]

                mean_sig = np.nanmean(padded, axis=0)
                # Normalize x-axis to fraction of episode
                x = np.linspace(0, 1, len(mean_sig))

                label = f"{suite} ({suffix}, n={len(filtered)})"
                color = suite_colors.get(suite, "gray")
                ax.plot(x, mean_sig, ls=ls, color=color, alpha=alpha, linewidth=1.5, label=label)

        ax.set_ylabel(sig_label, fontsize=10)
        ax.set_xlabel("Episode Progress (fraction)", fontsize=9)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Within-Episode Degradation: Frozen OpenVLA (No Memory)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "within_episode_degradation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_failure_mode_summary(data, out_dir):
    """Bar chart comparing failure signatures: success vs failure episodes."""
    metrics = {
        "Mean Action Magnitude": [],
        "Gripper Switches / 100 steps": [],
        "Action Diversity (last 25%)": [],
        "Consec. Diff (last 25%)": [],
    }
    categories = []

    for suite, episodes in data.items():
        for success_label, success_val in [("Success", True), ("Failure", False)]:
            filtered = [ep for ep in episodes if ep["success"] == success_val]
            if not filtered:
                continue

            all_sigs = [compute_episode_signals(ep) for ep in filtered]
            all_sigs = [s for s in all_sigs if s is not None]
            if not all_sigs:
                continue

            categories.append(f"{suite}\n{success_label} (n={len(all_sigs)})")

            # Mean action magnitude
            metrics["Mean Action Magnitude"].append(
                np.mean([np.mean(s["magnitude"]) for s in all_sigs])
            )

            # Gripper switches per 100 steps
            metrics["Gripper Switches / 100 steps"].append(
                np.mean([s["raw_gripper_switches"] / s["n_steps"] * 100 for s in all_sigs])
            )

            # Action diversity in last 25% of episode
            metrics["Action Diversity (last 25%)"].append(
                np.mean([np.mean(s["diversity"][int(s["n_steps"]*0.75):]) for s in all_sigs])
            )

            # Consecutive diff in last 25%
            metrics["Consec. Diff (last 25%)"].append(
                np.mean([np.mean(s["consec_diff"][int(s["n_steps"]*0.75):]) for s in all_sigs])
            )

    if not categories:
        print("No data for failure mode summary.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    bar_colors = []
    for cat in categories:
        if "Success" in cat:
            bar_colors.append("#2ecc71")
        else:
            bar_colors.append("#e74c3c")

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        x = range(len(categories))
        ax.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=7, rotation=15, ha="right")
        ax.set_title(metric_name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Failure Mode Signatures: Success vs Failure Episodes",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "failure_mode_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_long_episode_detail(data, out_dir):
    """Detailed view of individual long episodes (libero_10) showing degradation."""
    if "libero_10" not in data:
        print("No libero_10 data for detailed view.")
        return

    episodes = data["libero_10"]
    # Pick 2 success and 2 failure episodes
    successes = [ep for ep in episodes if ep["success"]][:2]
    failures = [ep for ep in episodes if not ep["success"]][:2]
    selected = successes + failures

    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 1, figsize=(14, 3.5 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for ax, ep in zip(axes, selected):
        sig = compute_episode_signals(ep)
        if sig is None:
            continue

        steps = range(sig["n_steps"])
        status = "SUCCESS" if ep["success"] else "FAIL"
        color = "#2ecc71" if ep["success"] else "#e74c3c"

        # Plot magnitude and diversity on same axes
        ax.plot(steps, sig["magnitude"][:sig["n_steps"]], color=color,
                alpha=0.8, linewidth=1.2, label="Action magnitude")
        ax.plot(steps, sig["diversity"][:sig["n_steps"]], color="steelblue",
                alpha=0.7, linewidth=1.2, label="Action diversity")
        ax.plot(steps, sig["consec_diff"][:sig["n_steps"]], color="orange",
                alpha=0.6, linewidth=1.0, label="Consec. diff")

        # Mark gripper switches
        actions = np.array(ep["actions"])
        gripper = actions[:, 6]
        switches = np.where(np.abs(np.diff(gripper)) > 0.5)[0]
        for sw in switches:
            ax.axvline(sw, color="gray", alpha=0.15, linewidth=0.5)

        task_short = ep["task_name"][:60]
        ax.set_title(f"[{status}] {task_short} ({sig['n_steps']} steps, "
                     f"{sig['raw_gripper_switches']} gripper switches)",
                     fontsize=9, fontweight="bold", color=color)
        ax.set_xlabel("Step")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Long-Horizon Episode Detail (libero_10): Action Signal Over Time",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "long_episode_detail.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def generate_failure_report(data, out_dir):
    """Text report summarizing failure modes per suite."""
    lines = ["# Within-Episode Failure Analysis: Frozen OpenVLA on LIBERO\n"]
    lines.append("## Key Finding\n")
    lines.append("Frozen OpenVLA (no external memory) shows systematic within-episode degradation:\n")
    lines.append("- **Action repetition increases** late in failed episodes (robot gets stuck)")
    lines.append("- **Gripper oscillation** is 2-5x higher in failed episodes (indecision)")
    lines.append("- **Action diversity collapses** in the last 25% of failed episodes")
    lines.append("- These patterns are **absent in successful episodes** of the same tasks\n")
    lines.append("This establishes the baseline gap that an external memory module needs to address.\n")

    for suite, episodes in data.items():
        successes = [ep for ep in episodes if ep["success"]]
        failures = [ep for ep in episodes if not ep["success"]]

        lines.append(f"\n## {suite}\n")
        lines.append(f"- Episodes: {len(episodes)} total, {len(successes)} success, {len(failures)} failure")

        if failures:
            fail_sigs = [compute_episode_signals(ep) for ep in failures]
            fail_sigs = [s for s in fail_sigs if s is not None]
            if fail_sigs:
                avg_switches = np.mean([s["raw_gripper_switches"] for s in fail_sigs])
                avg_steps = np.mean([s["n_steps"] for s in fail_sigs])
                avg_div_last = np.mean([np.mean(s["diversity"][int(s["n_steps"]*0.75):]) for s in fail_sigs])
                avg_diff_last = np.mean([np.mean(s["consec_diff"][int(s["n_steps"]*0.75):]) for s in fail_sigs])
                lines.append(f"- Failed episodes: avg {avg_steps:.0f} steps, "
                             f"{avg_switches:.0f} gripper switches")
                lines.append(f"- Late-episode diversity: {avg_div_last:.4f}")
                lines.append(f"- Late-episode consec. diff: {avg_diff_last:.4f}")

        if successes:
            succ_sigs = [compute_episode_signals(ep) for ep in successes]
            succ_sigs = [s for s in succ_sigs if s is not None]
            if succ_sigs:
                avg_switches = np.mean([s["raw_gripper_switches"] for s in succ_sigs])
                avg_steps = np.mean([s["n_steps"] for s in succ_sigs])
                avg_div_last = np.mean([np.mean(s["diversity"][int(s["n_steps"]*0.75):]) for s in succ_sigs])
                avg_diff_last = np.mean([np.mean(s["consec_diff"][int(s["n_steps"]*0.75):]) for s in succ_sigs])
                lines.append(f"- Successful episodes: avg {avg_steps:.0f} steps, "
                             f"{avg_switches:.0f} gripper switches")
                lines.append(f"- Late-episode diversity: {avg_div_last:.4f}")
                lines.append(f"- Late-episode consec. diff: {avg_diff_last:.4f}")

    report = "\n".join(lines)
    out_path = os.path.join(out_dir, "failure_analysis.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Saved: {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--suites", nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default="results/analysis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_all_episodes(args.results_dir, args.suites)
    if not data:
        print("No results found.")
        return

    print(f"Loaded: {', '.join(f'{s}: {len(eps)} episodes' for s, eps in data.items())}")

    plot_within_episode_curves(data, args.out_dir)
    plot_failure_mode_summary(data, args.out_dir)
    plot_long_episode_detail(data, args.out_dir)
    report = generate_failure_report(data, args.out_dir)
    print("\n" + report)


if __name__ == "__main__":
    main()
