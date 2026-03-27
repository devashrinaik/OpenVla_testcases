"""
Run frozen OpenVLA across all LIBERO suites to measure horizon-dependent degradation.

Each suite uses its own finetuned checkpoint (official OpenVLA releases).
The unnorm_key auto-matches the suite name.

Usage:
    python run_all.py                            # all suites, 10 ep each
    python run_all.py --suites libero_spatial     # single suite
    python run_all.py --n_episodes 5 --device cuda:1
"""

import argparse
import json
import os
import subprocess
import sys

PYTHON = sys.executable

# Suite configs with official finetuned checkpoints
SUITES = {
    "libero_spatial": {
        "n_tasks": 10,
        "horizon": "short (10-20)",
        "model_id": "openvla/openvla-7b-finetuned-libero-spatial",
    },
    "libero_object": {
        "n_tasks": 10,
        "horizon": "medium (20-50)",
        "model_id": "openvla/openvla-7b-finetuned-libero-object",
    },
    "libero_goal": {
        "n_tasks": 10,
        "horizon": "medium-long (30-80)",
        "model_id": "openvla/openvla-7b-finetuned-libero-goal",
    },
    "libero_10": {
        "n_tasks": 10,
        "horizon": "long (50-100+)",
        "model_id": "openvla/openvla-7b-finetuned-libero-10",
    },
}


def run_task(suite, task_id, n_episodes, model_id, device, gpu_id, out_dir, precision="bf16"):
    """Run libero_runner.py for one task, return results dict or None on failure."""
    cmd = [
        PYTHON, "libero_runner.py",
        "--suite", suite,
        "--task_id", str(task_id),
        "--n_episodes", str(n_episodes),
        "--model_id", model_id,
        "--device", device,
        "--gpu_id", str(gpu_id),
        "--out_dir", out_dir,
        "--precision", precision,
    ]
    print(f"\n{'='*60}")
    print(f"Running: {suite} task {task_id}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: {suite} task {task_id}")
        return None

    out_path = os.path.join(out_dir, f"{suite}_task{task_id}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", default=list(SUITES.keys()))
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "8bit", "4bit"],
                        help="Model precision: bf16 (default), 8bit, or 4bit quantization")
    parser.add_argument("--task_ids", nargs="+", type=int, default=None,
                        help="Specific task IDs to run (default: all)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}

    for suite in args.suites:
        if suite not in SUITES:
            print(f"Unknown suite: {suite}, skipping")
            continue

        cfg = SUITES[suite]
        task_ids = args.task_ids if args.task_ids else list(range(cfg["n_tasks"]))

        suite_results = []
        for tid in task_ids:
            res = run_task(
                suite, tid, args.n_episodes, cfg["model_id"],
                args.device, args.gpu_id, args.out_dir, args.precision,
            )
            if res:
                suite_results.append(res)

        if suite_results:
            successes = sum(r["summary"]["successes"] for r in suite_results)
            total = sum(r["summary"]["n_episodes"] for r in suite_results)
            mean_steps = sum(r["summary"]["mean_steps"] * r["summary"]["n_episodes"]
                            for r in suite_results) / max(total, 1)

            all_results[suite] = {
                "horizon": cfg["horizon"],
                "model_id": cfg["model_id"],
                "n_tasks_run": len(suite_results),
                "total_episodes": total,
                "total_successes": successes,
                "success_rate": successes / max(total, 1),
                "mean_steps": round(mean_steps, 1),
                "per_task": [
                    {
                        "task": r["task_name"],
                        "success_rate": r["summary"]["success_rate"],
                        "mean_steps": r["summary"]["mean_steps"],
                    }
                    for r in suite_results
                ],
            }

    # Save aggregate
    agg_path = os.path.join(args.out_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Suite':<20} {'Horizon':<25} {'TSR':>6} {'Steps':>7}")
    print(f"{'-'*70}")
    for suite, r in all_results.items():
        print(f"{suite:<20} {r['horizon']:<25} {r['success_rate']:>5.0%} "
              f"{r['mean_steps']:>7.1f}")
    print(f"{'='*70}")
    print(f"\nAggregate results saved to {agg_path}")


if __name__ == "__main__":
    main()
