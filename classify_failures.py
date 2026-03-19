"""
Failure mode classifier for frozen OpenVLA episodes on LIBERO.

Classifies each failed episode into one or more failure categories:
  - STUCK_LOOP: repeating cyclic action patterns (autocorrelation peaks)
  - GRASP_LOST: gripper closed then permanently opened (had object, lost it)
  - AIMLESS_WANDERING: high total movement but low net displacement (going nowhere)
  - GRIPPER_INDECISION: excessive gripper open/close oscillation
  - STALLED: very low action magnitude (robot stops moving)

Usage:
    python classify_failures.py
    python classify_failures.py --suites libero_10 --out_dir results/analysis
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Failure detectors
# ---------------------------------------------------------------------------

def detect_stuck_loop(actions, window=100, min_period=4, corr_threshold=0.35):
    """Detect cyclic repetition via autocorrelation on late-episode actions."""
    arm = actions[:, :6]
    n = len(arm)
    if n < window:
        return False, {}

    # Analyze last `window` steps
    chunk = arm[-window:]
    sig = np.linalg.norm(chunk, axis=1)
    sig = sig - sig.mean()
    if np.std(sig) < 1e-6:
        return False, {"reason": "constant signal"}

    autocorr = np.correlate(sig, sig, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / (autocorr[0] + 1e-8)

    # Find peaks above threshold
    peaks = []
    for j in range(min_period, len(autocorr) - 1):
        if (autocorr[j] > autocorr[j - 1] and autocorr[j] > autocorr[j + 1]
                and autocorr[j] > corr_threshold):
            peaks.append((j, float(autocorr[j])))

    is_loop = len(peaks) >= 2  # multiple periodic peaks = cyclic
    return is_loop, {"n_peaks": len(peaks), "top_peaks": peaks[:3]}


def detect_grasp_lost(actions, min_close_duration=5):
    """Detect grasp-then-lose: sustained gripper close followed by permanent open."""
    gripper = actions[:, 6]
    n = len(gripper)

    # Find sustained close regions (>= min_close_duration consecutive close)
    close_mask = gripper > 0  # +1 = close
    close_regions = []
    start = None
    for i in range(n):
        if close_mask[i]:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_close_duration:
                close_regions.append((start, i - 1))
            start = None

    if not close_regions:
        return False, {}

    # Check if after the last close region, gripper stays open
    last_close_end = close_regions[-1][1]
    remaining = gripper[last_close_end + 1:]
    if len(remaining) > 20 and np.mean(remaining < 0) > 0.9:
        # Gripper mostly open after last grasp — lost it
        steps_after = n - last_close_end - 1
        return True, {
            "last_grasp_end": int(last_close_end),
            "steps_after_loss": int(steps_after),
            "n_close_regions": len(close_regions),
        }

    return False, {}


def detect_aimless_wandering(actions, threshold=0.15):
    """Detect high movement but low net displacement (going in circles)."""
    arm_pos = actions[:, :3]  # xyz translation commands
    n = len(arm_pos)
    if n < 20:
        return False, {}

    # Cumulative displacement (treating actions as delta positions)
    cumulative = np.cumsum(arm_pos, axis=0)
    net_disp = np.linalg.norm(cumulative[-1])
    total_dist = np.sum(np.linalg.norm(arm_pos, axis=1))

    if total_dist < 0.1:
        return False, {"reason": "barely moved"}

    efficiency = net_disp / total_dist

    # Also check last 50% specifically
    half = n // 2
    late_cumulative = np.cumsum(arm_pos[half:], axis=0)
    late_net = np.linalg.norm(late_cumulative[-1]) if len(late_cumulative) > 0 else 0
    late_total = np.sum(np.linalg.norm(arm_pos[half:], axis=1))
    late_efficiency = late_net / (late_total + 1e-8)

    is_aimless = late_efficiency < threshold and late_total > 0.5
    return is_aimless, {
        "overall_efficiency": round(float(efficiency), 4),
        "late_efficiency": round(float(late_efficiency), 4),
        "total_distance": round(float(total_dist), 4),
    }


def detect_gripper_indecision(actions, rate_threshold=6.0):
    """Detect excessive gripper oscillation (switches per 100 steps)."""
    gripper = actions[:, 6]
    n = len(gripper)
    if n < 10:
        return False, {}

    switches = int(np.sum(np.abs(np.diff(gripper)) > 0.5))
    rate = switches / n * 100

    return rate > rate_threshold, {
        "total_switches": switches,
        "rate_per_100": round(rate, 1),
    }


def detect_stalled(actions, magnitude_threshold=0.02, frac_threshold=0.3):
    """Detect episodes where the robot mostly stops moving."""
    arm = actions[:, :6]
    magnitudes = np.linalg.norm(arm, axis=1)

    # Fraction of steps with very low action magnitude
    stall_frac = float(np.mean(magnitudes < magnitude_threshold))

    # Check if last 25% is mostly stalled
    n = len(magnitudes)
    last_quarter = magnitudes[int(n * 0.75):]
    late_stall = float(np.mean(last_quarter < magnitude_threshold)) if len(last_quarter) > 0 else 0

    is_stalled = stall_frac > frac_threshold or late_stall > 0.5
    return is_stalled, {
        "stall_fraction": round(stall_frac, 3),
        "late_stall_fraction": round(late_stall, 3),
    }


# ---------------------------------------------------------------------------
# Classify all failures
# ---------------------------------------------------------------------------

DETECTORS = {
    "STUCK_LOOP": detect_stuck_loop,
    "GRASP_LOST": detect_grasp_lost,
    "AIMLESS_WANDERING": detect_aimless_wandering,
    "GRIPPER_INDECISION": detect_gripper_indecision,
    "STALLED": detect_stalled,
}


def classify_episode(ep):
    """Run all detectors on one episode. Returns list of (label, details)."""
    actions = np.array(ep["actions"])
    if len(actions) < 5:
        return [("TOO_SHORT", {})]

    labels = []
    for name, detector in DETECTORS.items():
        detected, details = detector(actions)
        if detected:
            labels.append((name, details))

    if not labels:
        labels.append(("UNCLASSIFIED", {}))

    return labels


def load_all_failures(results_dir, suites=None):
    """Load all failed episodes from result JSONs."""
    suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    if suites:
        suite_order = [s for s in suite_order if s in suites]

    failures = []
    for suite in suite_order:
        files = sorted(glob.glob(os.path.join(results_dir, f"{suite}_task*.json")))
        for f in files:
            with open(f) as fh:
                data = json.load(fh)
            for i, ep in enumerate(data["episodes"]):
                if not ep["success"]:
                    ep["suite"] = suite
                    ep["task_name"] = data["task_name"]
                    ep["episode_idx"] = i
                    failures.append(ep)

    return failures


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_failure_taxonomy(classifications, out_dir):
    """Pie chart + bar chart of failure mode distribution."""
    # Count labels across all failures (episodes can have multiple labels)
    all_labels = []
    for cls in classifications:
        all_labels.extend([label for label, _ in cls["labels"]])

    counts = Counter(all_labels)
    # Also count per-suite
    suite_counts = defaultdict(Counter)
    for cls in classifications:
        for label, _ in cls["labels"]:
            suite_counts[cls["suite"]][label] += 1

    # --- Pie chart: overall distribution ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    label_colors = {
        "STUCK_LOOP": "#e74c3c",
        "GRASP_LOST": "#e67e22",
        "AIMLESS_WANDERING": "#f1c40f",
        "GRIPPER_INDECISION": "#9b59b6",
        "STALLED": "#3498db",
        "UNCLASSIFIED": "#95a5a6",
    }

    labels_sorted = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    sizes = [counts[l] for l in labels_sorted]
    colors = [label_colors.get(l, "#95a5a6") for l in labels_sorted]
    total = sum(sizes)

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=[f"{l}\n({c}/{total})" for l, c in zip(labels_sorted, sizes)],
        colors=colors, autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 9},
    )
    ax1.set_title(f"Failure Mode Distribution\n({total} failure labels across "
                  f"{len(classifications)} failed episodes)", fontsize=12, fontweight="bold")

    # --- Stacked bar: per-suite breakdown ---
    suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    suites_present = [s for s in suite_order if s in suite_counts]

    x = range(len(suites_present))
    bottom = np.zeros(len(suites_present))

    for label in labels_sorted:
        values = [suite_counts[s].get(label, 0) for s in suites_present]
        ax2.bar(x, values, bottom=bottom, label=label,
                color=label_colors.get(label, "#95a5a6"), edgecolor="black", linewidth=0.3)
        bottom += values

    ax2.set_xticks(x)
    ax2.set_xticklabels(suites_present, fontsize=10)
    ax2.set_ylabel("Count")
    ax2.set_title("Failure Modes per Suite", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "failure_taxonomy.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_failure_examples(classifications, out_dir):
    """Show one representative episode per failure mode with annotations."""
    # Pick best example per label (highest confidence = most detectors triggered)
    examples = {}
    for cls in classifications:
        for label, details in cls["labels"]:
            if label == "UNCLASSIFIED":
                continue
            if label not in examples:
                examples[label] = cls

    if not examples:
        return

    fig, axes = plt.subplots(len(examples), 1, figsize=(14, 3.5 * len(examples)))
    if len(examples) == 1:
        axes = [axes]

    label_colors = {
        "STUCK_LOOP": "#e74c3c",
        "GRASP_LOST": "#e67e22",
        "AIMLESS_WANDERING": "#f1c40f",
        "GRIPPER_INDECISION": "#9b59b6",
        "STALLED": "#3498db",
    }

    for ax, (label, cls) in zip(axes, examples.items()):
        actions = np.array(cls["actions"])
        arm = actions[:, :6]
        gripper = actions[:, 6]
        n = len(actions)
        steps = range(n)

        mag = np.linalg.norm(arm, axis=1)
        # Smooth
        w = 20
        if n > w:
            kernel = np.ones(w) / w
            mag_smooth = np.convolve(mag, kernel, mode="same")
        else:
            mag_smooth = mag

        color = label_colors.get(label, "gray")
        ax.plot(steps, mag_smooth, color=color, linewidth=1.5, label="Action magnitude")

        # Gripper state as background shading
        for i in range(n - 1):
            if gripper[i] > 0:
                ax.axvspan(i, i + 1, alpha=0.05, color="red")  # closed
            # Mark switches
            if i > 0 and abs(gripper[i] - gripper[i - 1]) > 0.5:
                ax.axvline(i, color="gray", alpha=0.2, linewidth=0.5)

        # Details annotation
        detail_str = ""
        for lbl, det in cls["labels"]:
            if lbl == label:
                detail_str = ", ".join(f"{k}={v}" for k, v in det.items()
                                       if not isinstance(v, list))
                break

        task_short = cls["task_name"][:50]
        all_labels = "+".join(l for l, _ in cls["labels"])
        ax.set_title(f"[{label}] {cls['suite']} — {task_short} "
                     f"({n} steps) [{all_labels}]\n{detail_str}",
                     fontsize=9, fontweight="bold", color=color)
        ax.set_xlabel("Step")
        ax.set_ylabel("Action Magnitude")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Representative Failure Mode Examples",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "failure_examples.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def generate_classification_report(classifications, out_dir):
    """Detailed text report with per-episode classification."""
    lines = ["# Failure Mode Classification Report\n"]
    lines.append("## Methodology\n")
    lines.append("Each failed episode is analyzed with 5 automatic detectors:")
    lines.append("- **STUCK_LOOP**: Autocorrelation on late-episode actions detects cyclic repetition")
    lines.append("- **GRASP_LOST**: Sustained gripper close followed by permanent open (had object, lost it)")
    lines.append("- **AIMLESS_WANDERING**: High total movement but low net displacement (efficiency < 0.15)")
    lines.append("- **GRIPPER_INDECISION**: Gripper switch rate > 6 per 100 steps")
    lines.append("- **STALLED**: > 30% of steps with near-zero action magnitude\n")
    lines.append("Episodes can have multiple labels (e.g., STUCK_LOOP + GRIPPER_INDECISION).\n")

    # Summary stats
    all_labels = []
    for cls in classifications:
        all_labels.extend([label for label, _ in cls["labels"]])
    counts = Counter(all_labels)
    total_failures = len(classifications)

    lines.append("## Summary\n")
    lines.append(f"Total failed episodes: {total_failures}\n")
    lines.append("| Failure Mode | Count | % of Failed Episodes |")
    lines.append("|---|---|---|")
    for label, count in counts.most_common():
        pct = count / total_failures * 100
        lines.append(f"| {label} | {count} | {pct:.0f}% |")

    # Per-suite breakdown
    suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    for suite in suite_order:
        suite_cls = [c for c in classifications if c["suite"] == suite]
        if not suite_cls:
            continue
        lines.append(f"\n## {suite} ({len(suite_cls)} failures)\n")
        for cls in suite_cls:
            labels_str = ", ".join(f"{l}" for l, _ in cls["labels"])
            task_short = cls["task_name"][:50]
            lines.append(f"- **Ep{cls['episode_idx']}** ({cls['n_steps']} steps) "
                         f"[{labels_str}] — {task_short}")

    # Implications for memory module
    lines.append("\n## Implications for External Memory Module\n")
    if "STUCK_LOOP" in counts:
        lines.append(f"- **{counts['STUCK_LOOP']} STUCK_LOOP failures**: Memory should detect "
                     f"action repetition and trigger re-planning or exploration")
    if "GRASP_LOST" in counts:
        lines.append(f"- **{counts['GRASP_LOST']} GRASP_LOST failures**: Memory should track "
                     f"grasp state and trigger recovery sub-routine")
    if "AIMLESS_WANDERING" in counts:
        lines.append(f"- **{counts['AIMLESS_WANDERING']} AIMLESS_WANDERING failures**: Memory should "
                     f"maintain goal representation to prevent drift")
    if "GRIPPER_INDECISION" in counts:
        lines.append(f"- **{counts['GRIPPER_INDECISION']} GRIPPER_INDECISION failures**: Memory should "
                     f"commit to grasp/release decisions (reduce oscillation)")
    if "STALLED" in counts:
        lines.append(f"- **{counts['STALLED']} STALLED failures**: Memory should detect inactivity "
                     f"and inject exploratory actions")

    report = "\n".join(lines)
    out_path = os.path.join(out_dir, "failure_classification.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Saved: {out_path}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--suites", nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default="results/analysis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    failures = load_all_failures(args.results_dir, args.suites)
    print(f"Loaded {len(failures)} failed episodes")

    if not failures:
        print("No failures to classify.")
        return

    # Classify each episode
    classifications = []
    for ep in failures:
        labels = classify_episode(ep)
        classifications.append({
            "suite": ep["suite"],
            "task_name": ep["task_name"],
            "episode_idx": ep["episode_idx"],
            "n_steps": ep["n_steps"],
            "labels": labels,
            "actions": ep["actions"],  # keep for plotting
        })

    # Print quick summary
    all_labels = []
    for cls in classifications:
        all_labels.extend([label for label, _ in cls["labels"]])
    counts = Counter(all_labels)
    print(f"\nFailure mode counts:")
    for label, count in counts.most_common():
        print(f"  {label}: {count} ({count/len(classifications)*100:.0f}%)")

    # Generate outputs
    plot_failure_taxonomy(classifications, args.out_dir)
    plot_failure_examples(classifications, args.out_dir)
    report = generate_classification_report(classifications, args.out_dir)
    print("\n" + report)

    # Save classification JSON (without raw actions)
    cls_json = []
    for cls in classifications:
        cls_json.append({
            "suite": cls["suite"],
            "task_name": cls["task_name"],
            "episode_idx": cls["episode_idx"],
            "n_steps": cls["n_steps"],
            "labels": [(l, d) for l, d in cls["labels"]],
        })
    out_path = os.path.join(args.out_dir, "failure_classifications.json")
    with open(out_path, "w") as f:
        json.dump(cls_json, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
