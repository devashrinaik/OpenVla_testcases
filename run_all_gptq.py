"""
Run GPTQ-quantized OpenVLA across all LIBERO suites using both GPUs in parallel.

Distributes tasks across available GPUs (2 tasks running simultaneously).
Each GPU loads one model, runs all tasks for that suite, then moves to the next.

Usage:
    # All suites, both GPUs in parallel
    python run_all_gptq.py

    # Single suite
    python run_all_gptq.py --suites libero_spatial

    # Fewer episodes for quick test
    python run_all_gptq.py --n_episodes 3

    # Specific tasks only
    python run_all_gptq.py --task_ids 0 1 2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

PYTHON = sys.executable
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libero_runner_gptq.py")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

SUITES = {
    "libero_spatial": {
        "n_tasks": 10,
        "horizon": "short (10-20)",
        "checkpoint": "openvla-7b-finetuned-libero-spatial-gptq-4bit",
    },
    "libero_object": {
        "n_tasks": 10,
        "horizon": "medium (20-50)",
        "checkpoint": "openvla-7b-finetuned-libero-object-gptq-4bit",
    },
    "libero_goal": {
        "n_tasks": 10,
        "horizon": "medium-long (30-80)",
        "checkpoint": "openvla-7b-finetuned-libero-goal-gptq-4bit",
    },
    "libero_10": {
        "n_tasks": 10,
        "horizon": "long (50-100+)",
        "checkpoint": "openvla-7b-finetuned-libero-10-gptq-4bit",
    },
}


def run_suite_tasks(suite, task_ids, n_episodes, gpu_idx, out_dir):
    """Run all tasks for a suite on a specific GPU. Called in a subprocess."""
    cfg = SUITES[suite]
    checkpoint = os.path.join(CHECKPOINT_DIR, cfg["checkpoint"])
    device = f"cuda:{gpu_idx}"
    results = []

    for tid in task_ids:
        out_path = os.path.join(out_dir, f"{suite}_task{tid}.json")

        # Skip if already done
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                if "summary" in existing:
                    print(f"  [GPU {gpu_idx}] {suite} task {tid}: already done, skipping")
                    results.append(existing)
                    continue
            except Exception:
                pass

        cmd = [
            PYTHON, "-u", RUNNER,
            "--checkpoint", checkpoint,
            "--suite", suite,
            "--task_id", str(tid),
            "--n_episodes", str(n_episodes),
            "--device", device,
            "--gpu_id", str(gpu_idx),
            "--out_dir", out_dir,
        ]

        print(f"\n{'='*60}")
        print(f"[GPU {gpu_idx}] {suite} task {tid}/{len(task_ids)-1}")
        print(f"{'='*60}")
        sys.stdout.flush()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        # Override device to cuda:0 since CUDA_VISIBLE_DEVICES remaps
        cmd_adjusted = [c if c != device else "cuda:0" for c in cmd]
        cmd_adjusted = [c if c != str(gpu_idx) else "0" for c in cmd_adjusted]

        result = subprocess.run(cmd_adjusted, capture_output=False, env=env)

        if result.returncode != 0:
            print(f"  [GPU {gpu_idx}] FAILED: {suite} task {tid}")
            continue

        if os.path.exists(out_path):
            with open(out_path) as f:
                results.append(json.load(f))

    return suite, results


def main():
    parser = argparse.ArgumentParser(description="Parallel GPTQ OpenVLA eval on LIBERO")
    parser.add_argument("--suites", nargs="+", default=list(SUITES.keys()))
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="results/gptq_4bit")
    parser.add_argument("--task_ids", nargs="+", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of GPUs to use in parallel")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Override checkpoint directory")
    args = parser.parse_args()

    global CHECKPOINT_DIR
    if args.checkpoint_dir:
        CHECKPOINT_DIR = args.checkpoint_dir

    os.makedirs(args.out_dir, exist_ok=True)

    # Build work items: list of (suite, task_ids, gpu_idx)
    work = []
    for suite in args.suites:
        if suite not in SUITES:
            print(f"Unknown suite: {suite}, skipping")
            continue
        cfg = SUITES[suite]
        task_ids = args.task_ids if args.task_ids else list(range(cfg["n_tasks"]))
        work.append((suite, task_ids))

    if not work:
        print("No work to do.")
        return

    # Print plan
    total_tasks = sum(len(tids) for _, tids in work)
    total_episodes = total_tasks * args.n_episodes
    print(f"Execution plan:")
    print(f"  Suites: {[s for s, _ in work]}")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Episodes per task: {args.n_episodes}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  GPUs: {args.num_gpus}")
    print(f"  Output: {os.path.abspath(args.out_dir)}")
    print(f"  Strategy: {args.num_gpus} suites in parallel (1 per GPU)")
    print()
    sys.stdout.flush()

    t0 = time.time()
    all_results = {}

    if args.num_gpus >= 2 and len(work) >= 2:
        # Parallel execution: distribute suites across GPUs
        with ProcessPoolExecutor(max_workers=args.num_gpus) as executor:
            futures = {}
            for i, (suite, task_ids) in enumerate(work):
                gpu_idx = i % args.num_gpus
                future = executor.submit(
                    run_suite_tasks, suite, task_ids, args.n_episodes,
                    gpu_idx, args.out_dir
                )
                futures[future] = suite

            for future in as_completed(futures):
                suite = futures[future]
                try:
                    suite_name, suite_results = future.result()
                    all_results[suite_name] = suite_results
                except Exception as e:
                    print(f"Suite {suite} failed: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # Sequential execution on GPU 0
        for suite, task_ids in work:
            suite_name, suite_results = run_suite_tasks(
                suite, task_ids, args.n_episodes, 0, args.out_dir
            )
            all_results[suite_name] = suite_results

    elapsed = time.time() - t0

    # Aggregate results
    aggregate = {}
    for suite, results_list in all_results.items():
        if not results_list:
            continue
        cfg = SUITES[suite]
        successes = sum(r["summary"]["successes"] for r in results_list if "summary" in r)
        total = sum(r["summary"]["n_episodes"] for r in results_list if "summary" in r)
        mean_steps = np.mean([r["summary"]["mean_steps"] for r in results_list if "summary" in r]) if results_list else 0
        mean_wall = np.mean([r["summary"].get("mean_wall_time", 0) for r in results_list if "summary" in r]) if results_list else 0

        aggregate[suite] = {
            "horizon": cfg["horizon"],
            "checkpoint": cfg["checkpoint"],
            "n_tasks_run": len(results_list),
            "total_episodes": total,
            "total_successes": successes,
            "success_rate": successes / max(total, 1),
            "mean_steps": round(mean_steps, 1),
            "mean_wall_time_per_episode": round(mean_wall, 1),
            "per_task": [
                {
                    "task": r.get("task_name", "?"),
                    "success_rate": r["summary"]["success_rate"],
                    "mean_steps": r["summary"]["mean_steps"],
                }
                for r in results_list if "summary" in r
            ],
        }

    # Save aggregate
    agg_path = os.path.join(args.out_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"GPTQ 4-bit Results (total time: {elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"{'Suite':<20} {'Horizon':<25} {'TSR':>6} {'Steps':>7} {'Time/ep':>8}")
    print(f"{'-'*70}")
    for suite, r in aggregate.items():
        print(f"{suite:<20} {r['horizon']:<25} {r['success_rate']:>5.0%} "
              f"{r['mean_steps']:>7.1f} {r['mean_wall_time_per_episode']:>7.1f}s")
    print(f"{'='*70}")
    print(f"Aggregate saved to {agg_path}")


# Need numpy for aggregate stats
import numpy as np

if __name__ == "__main__":
    main()
