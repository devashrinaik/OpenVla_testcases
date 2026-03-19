# OpenVLA Memory Benchmark

Evaluation pipeline for measuring where frozen OpenVLA fails on long-horizon tasks in LIBERO — establishing the baseline that external memory modules need to beat.

## Key Results

### Degradation Curve (across suites)

| Suite | Horizon | TSR | Mean Steps | Failed Episodes |
|-------|---------|-----|------------|-----------------|
| `libero_spatial` | Short (10-20) | **75%** | 137 | 25 |
| `libero_object` | Medium (20-50) | **67%** | 190 | 10 |
| `libero_goal` | Medium-long (30-80) | **80%** | 163 | 8 |
| `libero_10` | Long (50-100+) | **62%** | 367 | 15 |

### Failure Mode Taxonomy (58 failed episodes)

| Failure Mode | % of Failures | What Memory Should Fix |
|---|---|---|
| **GRIPPER_INDECISION** | 45% | Commit to grasp/release decisions, stop oscillating |
| **GRASP_LOST** | 31% | Track grasp state, trigger recovery when object dropped |
| **STALLED** | 28% | Detect inactivity, inject exploratory actions |
| **STUCK_LOOP** | 14% | Detect cyclic repetition, trigger re-planning |
| **AIMLESS_WANDERING** | 12% | Maintain goal representation to prevent drift |

Episodes can have multiple labels. Short-horizon tasks fail on execution (grasping); long-horizon tasks fail on persistence (stalling out mid-episode).

### Within-Episode Degradation

Failed episodes show systematic mid-episode degradation compared to successful ones:
- **4-5x more gripper switches** per 100 steps (indecision)
- **Action diversity collapses** to 0.03-0.06 in late failure vs 0.09-0.13 in success
- **Consecutive action diff drops 2x** in failures (robot gets stuck repeating)

These patterns are absent in successful episodes of the same tasks — the model "forgets" what it was doing.

## Setup

```bash
# Create conda environment
conda create -n vla_bench python=3.10 -y
conda activate vla_bench

# Install dependencies
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.1 timm==0.9.10 accelerate
pip install libero

# Install OpenVLA (from source)
pip install git+https://github.com/openvla/openvla.git

# LIBERO config (run once — creates ~/.libero/config.yaml on first import)
python -c "from libero.libero import benchmark"
```

**Note**: The `openvla` path in `libero_runner.py` (line 33) points to a local clone. Update this to match your installation.

## Usage

```bash
# Run one suite
python run_all.py --suites libero_spatial --n_episodes 10

# Run all suites (4 suites x 10 tasks x 10 episodes = 400 episodes, ~4-6 hours)
python run_all.py --n_episodes 10

# Run a single task for quick testing
python libero_runner.py --suite libero_spatial --task_id 0 --n_episodes 5 \
    --model_id openvla/openvla-7b-finetuned-libero-spatial

# Generate plots from results
python plot_results.py

# Run within-episode degradation analysis
python analyze_episodes.py

# Run failure mode classification
python classify_failures.py
```

Finetuned checkpoints (~15GB each) download automatically from HuggingFace on first run.

## Files

| File | Purpose |
|------|---------|
| `libero_runner.py` | Single-task closed-loop eval. Handles image preprocessing (180-deg rotation, JPEG encode-decode, Lanczos3 resize, center crop), action decoding, gripper normalization. |
| `run_all.py` | Batch runner across suites. Auto-selects the correct finetuned checkpoint per suite. |
| `plot_results.py` | Generates degradation curve (TSR vs horizon), per-task breakdown, and entropy trace plots. |
| `analyze_episodes.py` | Within-episode degradation analysis. Extracts step-level signals (action magnitude, diversity, repetition, gripper oscillation) comparing success vs failure episodes. |
| `classify_failures.py` | Failure mode classifier. Categorizes each failed episode into named failure types with actionable implications for memory module design. |

## Output

Results in `results/`:
- `results/{suite}_task{id}.json` — per-task episode data with full trajectories
- `results/aggregate.json` — summary across all suites
- `results/degradation_curve.png` — TSR vs horizon bar chart
- `results/per_task_breakdown.png` — per-task TSR within each suite
- `results/entropy_traces.png` — action entropy over episode steps

Analysis in `results/analysis/`:
- `within_episode_degradation.png` — per-suite signal curves (success vs failure)
- `failure_mode_summary.png` — success vs failure comparison across 4 metrics
- `long_episode_detail.png` — individual libero_10 episode traces
- `failure_taxonomy.png` — pie chart + per-suite stacked bar of failure modes
- `failure_examples.png` — representative episode per failure type
- `failure_classification.md` — full text report with per-episode labels
- `failure_classifications.json` — machine-readable classification data

## Key Technical Details

- **Attention**: Uses `sdpa` (PyTorch scaled dot-product attention). The `eager` implementation has a causal mask bug with transformers 4.40.1 + OpenVLA's custom model code. `flash_attention_2` works but requires CUDA toolkit for compilation.
- **Image preprocessing**: LIBERO renders images 180-degrees rotated. The pipeline applies rotation, JPEG encode-decode (matching RLDS training), Lanczos3 resize to 224x224, and center crop (90% area) to match training augmentation.
- **Gripper**: OpenVLA outputs gripper in [0,1] (0=close, 1=open). Pipeline normalizes to [-1,+1] and inverts sign for LIBERO convention (-1=open, +1=close).
- **Stabilization**: 10 no-op steps at episode start for objects to settle in simulation.
